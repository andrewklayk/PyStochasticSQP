import numpy as np
from numpy.linalg import norm
import scipy
from scipy.linalg import solve
import scipy.optimize
from scipy.optimize import minimize, LinearConstraint
from numpy import random

from problem import Problem

def _estimate_lipschitz(x, g, g_k, j, j_k, rng, lo=-1, lc=-1, batch_size=10, displacement=1e-4):
    lipschitz_constraint = lo
    lipschitz_objective = lc

    sampleDirection = rng.normal(size=x.shape)
    for _ in range(batch_size):
        samplePoint = x + displacement * sampleDirection/norm(sampleDirection, ord=2)
        lip_con, lip_obj = _compute_lipschitz_estimates(samplePoint, x, g, g_k, j, j_k)
        lipschitz_constraint = np.max([lipschitz_constraint,lip_con])
        # elementwise max
        lipschitz_objective = np.maximum(lipschitz_objective,lip_obj)
    return lip_con, lip_obj

def _compute_lipschitz_estimates(samplePoint, x_k, g, g_k, j, j_k):
    sample_jacobian = j(samplePoint)
    lip_con = norm(sample_jacobian-j_k)/norm(samplePoint-x_k)
    sample_grad = g(samplePoint)
    lip_obj = norm(sample_grad - g_k)/norm(samplePoint - x_k)
    return lip_obj, lip_con

def _compute_direction(_x, _g, _c, _j, _h, reg_par):
    n = len(_x)
    m = len(_c)
    # solve first subproblem
    def constraint_subproblem(val):
        u = val[:n]
        w = val[n:]
        return 0.5*norm(_c + _j.T@_j*w)**2 + 0.5*reg_par*norm(u)**2
    constr = [
        LinearConstraint(np.hstack([_j, np.zeros((m, m))]), np.zeros_like(_c), np.zeros_like(_c)),
        LinearConstraint(np.hstack([np.eye(n, n), _j.T]), -_x, np.ones_like(_x)*np.inf),
    ]
    res = minimize(
        constraint_subproblem,
        x0=np.zeros(shape=(n+m)),
        constraints=constr
        # jac=
        )
    v = res.x[:n] + _j.T@res.x[n:]

    #solve second subproblem
    def dir_subproblem(d):
        return _g.T@d + 0.5*d.T@_h@d
    constr = [
        LinearConstraint(_j, _j@v, _j@v),
        LinearConstraint(np.eye(n),-_x, np.inf)
    ]
    res = minimize(
        dir_subproblem,
        x0=np.zeros_like(_x),
        constraints=constr,
        # jac=
        )
    return res.x
    

def _sqp_stoch_iter(
    k,
    x_k,
    obj_grad_k,
    c,
    j_k,
    h_k,
    delta_q,
    L_obj, L_con,
    model_red_factor=0.1,
    merit_par=0.1, merit_par_red_factor = 1e-2,
    ratio_par=1, ratio_par_red_factor=1e-2,
    suf_dec_par=0.5, stepsize_scaling=1,
    subproblem_reg_par=1e-7,
    proj_width = 1e4,
    lengthening_ratio=1.1
    ):

    # set parameter values
    ## ADD INEQ
    c_k = c(x_k)
    ce_k = c(x_k)
    ci_k = []

    n = len(x_k)
    m = len(ce_k) + len(ci_k)

    _merit_par = merit_par
    _ratio_par = ratio_par

    d = _compute_direction(_x=x_k,
                           _g=obj_grad_k,
                           _c=ce_k if len(ci_k) == 0 else np.hstack(ce_k, ci_k),
                           _j=j_k,
                           _h=h_k,
                           reg_par=subproblem_reg_par)

    # check stopping condition
    #TODO: ADD NUMERICAL STOPPING CONDITION BASED ON INITIAL VALUES
    # if np.max(obj_grad_k + j_k[np.newaxis].T @ y_k) <= 1e-6*np.max((1,))
    if np.all(np.abs(obj_grad_k) < 1e-6) and np.all(np.abs([ce_k, ci_k]) < 1e-6):
        return (x_k, _merit_par, _ratio_par, True)

    # calc some terms to use later
    _quad_term = np.max(d.T @ h_k @ d, 0)
    _c_l2 = norm(ce_k)
    _d_l2 = norm(d)

    ########################
    # CALCULATE PARAMETERS #
    ########################
    
    # merit parameter
    if obj_grad_k.T @ d + _quad_term <= 0:
        merit_par_trial = np.inf
    else:
        merit_par_trial = ((1-model_red_factor)*(_c_l2 - norm(c(x_k) + j_k@d)))/((obj_grad_k.T@d + _quad_term))
    if _merit_par > merit_par_trial:
        _merit_par = min((1-merit_par_red_factor)*_merit_par, merit_par_trial)
    model_reduction = delta_q(x_k, _merit_par, obj_grad_k, j_k, d, c_k)
    # model_reduction = delta_q(x_k, _merit_par, obj_grad_k, _quad_term, d, _c_l1)
    
    # ratio parameter
    if _d_l2 >= 1e-6:
        ratio_par_trial = model_reduction/(_merit_par*_d_l2)
        if ratio_par > ratio_par_trial:
            _ratio_par = min((1-ratio_par_red_factor)*_ratio_par, ratio_par_trial)

    ######################
    # CALCULATE STEPSIZE #
    ######################
    
    denominator = (_merit_par * L_obj + L_con) * norm(d)**2
    alpha_suff = min(1, 2*(1 - suf_dec_par) * stepsize_scaling * model_reduction / denominator)
    if model_reduction <= 0:
        _alpha = 0
    else:
        alpha_min = 2*(1-suf_dec_par)*stepsize_scaling*_ratio_par*_merit_par/(_merit_par*L_obj+L_con)
        alpha_max = alpha_min + proj_width*stepsize_scaling**2
        _alpha = min(alpha_suff, alpha_min)
        while _alpha < alpha_max:
            alpha_trial = min(alpha_max, _alpha*lengthening_ratio)
            reduction = ((suf_dec_par-1)*alpha_trial*stepsize_scaling*model_reduction
                         + norm(c_k + alpha_trial*j_k@d)
                         - norm(c_k)
                         + alpha_trial*(norm(c_k) - norm(c_k + j_k@d))
                         + 0.5 * (_merit_par*L_obj + L_con)*(alpha_trial**2)*(norm(d)**2)
                         )
            if reduction > 0:
                break
            else:
                _alpha = alpha_trial

        _alpha = max(alpha_min, min(_alpha, alpha_max))

    ##################
    # UPDATE ITERATE #
    ##################
    
    x_next = x_k + _alpha*d
    return x_next, _merit_par, _ratio_par, False


def optimize_st(p: Problem, x0,L_obj=0, L_con=[],  merit_par=0.5, ratio_par=1, iter_limit=100):
    
    # def merit_fn(x, merit_par):
    #     return merit_par*p.f(x)+norm(p.c(x), ord=2)
    def delta_q(x, merit_par, _g, _j, _d, _c, _c_l2=None):
        return -merit_par*_g.T@_d + norm(_c, ord=2) - norm(_c+_j@_d)
    
    x_k = x0
    rng = np.random.default_rng(42)

    if L_con == [] or L_con is None:
        L_con = np.ones(len(p.c(x_k)))

    _L_obj = L_obj
    _L_con = L_con
    _merit_par = merit_par
    _ratio_par = ratio_par
    
    
    print(f'Starting conditions: x_0: {x0}, f(x_{0})={p.f(x0)}, c={p.c(x0)}')
    for k in range(iter_limit):
        #########################
        # ESTIMATE L. CONSTANTS #
        #########################
        # once every {some} iterations
        if k % 10 == 0:
            _L_obj, l_con = _estimate_lipschitz(x_k, p.g, p.g(x_k), p.J, p.J(x_k), rng, lo=_L_obj, lc=_L_con)
            _L_con = np.sum(l_con)

        x_k, _merit_par, _ratio_par, is_finished = _sqp_stoch_iter(
            k=k, x_k=x_k, obj_grad_k=p.g(x_k), c=p.c, j_k=p.J(x_k), h_k=p.H(x_k),
            delta_q=delta_q,
            L_obj=_L_obj, L_con=_L_con,
            merit_par=_merit_par, ratio_par=_ratio_par)
        print(f'Iteration: {k+1}, x_{k+1}: {x_k}, f(x_{k+1})={p.f(x_k)}, c={p.c(x_k)}, mp={_merit_par}, rp={_ratio_par}')
        if is_finished:
            return x_k


