import numpy as np
import scipy
from scipy.linalg import solve
import scipy.optimize

class Problem:
    def __init__(self, f, g, c, J, H):
        self.f = f
        self.g = g
        self.c = c
        self.J = J
        self.H = H

def sqp_det_iter(k,
    x_k,
    obj_grad_k,
    c,
    ce_k,
    ci_k,
    j_k,
    h_k,
    merit_fn,
    delta_q,
    adaptive=False, model_red_factor=0.5,merit_par=1, merit_par_red_factor=1e-6, alpha=1, v=0.5, nu=1e-4, rho=3, L=1, gamma=None
    ):

    # set parameter values
    if ci_k is None:
        ci_k = []
    if ce_k is None:
        ce_k = []
    if adaptive and gamma is None:
        gamma = np.ones(len(ce_k) + len(ci_k))

    n = len(x_k)
    m = len(ce_k) + len(ci_k)

    # construct and solve the SQP linear problem
    sqp_A = np.vstack([
        np.hstack([h_k, j_k[np.newaxis].T]),
        np.hstack([j_k[np.newaxis],np.zeros((m,m))])
    ])
    sqp_b = -np.hstack([obj_grad_k,ce_k])
    qp_sol = solve(sqp_A, sqp_b)
    d_k, y_k = qp_sol[:n], qp_sol[n:]

    # check stopping condition
    #TODO: ADD NUMERICAL STOPPING CONDITION
    if np.all(obj_grad_k + j_k[np.newaxis].T @ y_k) == 0 and np.all(ce_k) == 0:
        return (x_k, merit_par, L, True)

    # calc some terms to use later
    quad_term = np.max(d_k.T @ h_k @ d_k, 0)
    constraint_l1_k = np.sum(np.abs(ce_k))

    # calculate merit parameter
    if obj_grad_k.T @ d_k + quad_term <= 0:
        merit_par_trial = np.inf
    else:
        merit_par_trial = ((1-model_red_factor)*constraint_l1_k)/(obj_grad_k.T@d_k + quad_term)
    if merit_par > merit_par_trial:
        merit_par = (1-merit_par_red_factor)*merit_par_trial
    
    if adaptive:
        # set Lipschitz estimates as 1/2 of previous iteration
        if k > 0:
            L *= 0.5
            gamma *= 0.5
        while True:
            d_k_l2 = np.sum(d_k**2)
            # calculate stepsize
            alpha_denom = merit_par*L+np.sum(gamma)*d_k_l2
            alpha_k_hat = 2*(1-nu)*delta_q(x_k, merit_par, obj_grad_k, quad_term, d_k, ce_k)/alpha_denom
            alpha_k_tilde = alpha_k_hat - 4*constraint_l1_k/alpha_denom
            if alpha_k_hat < 1:
                alpha_k = alpha_k_hat
            elif alpha_k_tilde <= 1:
                alpha_k = 1
            else:
                alpha_k = alpha_k_tilde
            
            reduction_term = nu*alpha_k*delta_q(x_k, merit_par, obj_grad_k, quad_term, d_k, ce_k)
            c_212a = f(x_k + alpha_k*d_k) <= f(x_k) + alpha_k*obj_grad_k.T@d_k + 0.5*L*(alpha_k**2)*(d_k_l2)
            c_212b = np.abs(c(x_k + alpha_k*d_k)) <= np.abs(ce_k + alpha_k*j_k.T@d_k) + 0.5*gamma*(alpha_k**2)*d_k_l2
            if merit_fn(x_k+alpha_k*d_k, merit_par) <= merit_fn(x_k, merit_par) - reduction_term or (c_212a and c_212b):
                x_next = x_k + alpha_k*d_k
                break
            elif not c_212a:
                L *= rho
            elif not np.all(c_212b):
                gamma[not c_212b] *= rho
    else:
        # calculate step size (linesearch)
        alpha_k = alpha
        # check sufficient reduction
        while True:
            reduction_term = nu*alpha_k*delta_q(x_k, merit_par, obj_grad_k, quad_term, d_k, ce_k)
            if merit_fn(x_k+alpha_k*d_k, merit_par) <= merit_fn(x_k, merit_par) - reduction_term:
                break
            alpha_k *= v
        x_next = x_k + alpha_k*d_k
    return x_next, merit_par, L, False

def optimize_det(p: Problem, x0, sigma=0.5,merit_par=0.5,eps=1e-6, alpha=1, v=0.5, nu=1e-4, L=1):
    # sigma: model reduction factor
    # eps: parameter reduction factor
    # build merit fn
    def merit_fn(x, merit_par): return merit_par*p.f(x)+np.sum(np.abs(p.c(x)))
    def delta_q(x, merit_par, g, quad_term, d, c_l1=None): return -merit_par*(g.T@d + 0.5*quad_term) + np.sum(np.abs(p.c(x)))
    x_k = x0
    print(f'Starting conditions: x_0: {x0}, f(x_{0})={f(x0)}, c={c(x0)}')
    for k in range(100):
        x_k, merit_par, L, is_finished = sqp_det_iter(
            adaptive=True,
            k=k, x_k=x_k, obj_grad_k=p.g(x_k), c=p.c, ce_k=p.c(x_k), ci_k=None, j_k=p.J(x_k), h_k=p.H(x_k),
            merit_fn=merit_fn,
            delta_q=delta_q,
            merit_par=merit_par,
            L=L)
        print(f'Iteration: {k+1}, x_{k+1}: {x_k}, f(x_{k+1})={f(x_k)}, c={c(x_k)}, tau={merit_par}')
        if is_finished:
            return x_k


# x0 = np.array([4,2])
# def f(x): return 0.5*x[0]**2 + 0.5*x[1]**2
# def g(x): return np.array([x[0], x[1]])
# def c(x): return np.array([x[0] + x[1] - 1])
# def J(x): return np.ones(2)
# def H(x): return np.eye(2)

x0 = np.array([4,2])
def f(x): return x[0]**3 + x[1]**3
def g(x): return np.array([3*x[0]**2, 3*x[1]**2])
def c(x): return np.array([x[0]**2 + x[1]**2-1])
def J(x): return np.array([2*x[0], 2*x[1]])
def H(x): return np.array([[6*x[0],0],[0,6*x[1]]])

p = Problem(f, g, c, J, H)

optimize_det(p, x0, L=1)