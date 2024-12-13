from problem import Problem


def _sqp_det_iter(
    k,
    x_k,
    obj_fun,
    obj_grad_k,
    c, ce_k, ci_k,
    j_k,
    h_k,
    merit_fn,
    delta_q,
    L,
    gamma,
    adaptive, model_red_factor=0.5,merit_par=1, merit_par_red_factor=1e-6, alpha=1, v=0.5, nu=1e-4, rho=3
    ):

    # set parameter values
    if ci_k is None:
        ci_k = []
    if ce_k is None:
        ce_k = []
    if adaptive:
        _gamma = gamma
    else:
        _gamma = None

    n = len(x_k)
    m = len(ce_k) + len(ci_k)

    _merit_par = merit_par
    _L = L

    # construct and solve the SQP linear problem
    lp_A = np.vstack([
        np.hstack([h_k, j_k[np.newaxis].T]),
        np.hstack([j_k[np.newaxis],np.zeros((m,m))])
    ])
    lp_b = -np.hstack([obj_grad_k,ce_k])
    lp_sol = solve(lp_A, lp_b)
    d, y = lp_sol[:n], lp_sol[n:]

    # check stopping condition
    #TODO: ADD NUMERICAL STOPPING CONDITION BASED ON INITIAL VALUES
    # if np.max(obj_grad_k + j_k[np.newaxis].T @ y_k) <= 1e-6*np.max((1,))
    if np.all(obj_grad_k + j_k[np.newaxis].T @ y < 1e-6) and np.all(ce_k < 1e-6):
        return (x_k, _merit_par, _L, _gamma, True)

    # calc some terms to use later
    _quad_term = np.max(d.T @ h_k @ d, 0)
    _constr_l1 = norm(ce_k, ord=1)

    # calculate merit parameter
    if obj_grad_k.T @ d + _quad_term <= 0:
        merit_par_trial = np.inf
    else:
        merit_par_trial = ((1-model_red_factor)*_constr_l1)/(obj_grad_k.T@d + _quad_term)
    if _merit_par > merit_par_trial:
        _merit_par = (1-merit_par_red_factor)*merit_par_trial
    
    model_reduction = delta_q(x_k, _merit_par, obj_grad_k, _quad_term, d, _constr_l1)

    if adaptive:
        # set Lipschitz estimates as 1/2 of previous iteration
        if k > 0:
            _L *= 0.5
            # numpy doesnt like *= here for some reason
            _gamma = _gamma*0.5
        while True:
            d_k_l2 = np.sum(d**2)
            # calculate stepsize
            denom = _merit_par*_L+np.sum(_gamma)*d_k_l2
            alpha_k_hat = 2*(1-nu)*model_reduction/denom
            alpha_k_tilde = alpha_k_hat - 4*_constr_l1/denom
            if alpha_k_hat < 1:
                _alpha = alpha_k_hat
            elif alpha_k_tilde <= 1:
                _alpha = 1
            else:
                _alpha = alpha_k_tilde
            
            # sufficient descent of merit function (2.10)
            c_210 = merit_fn(x_k+_alpha*d, _merit_par) <= merit_fn(x_k, _merit_par) - nu*_alpha*model_reduction
            # check if lipschitz estimates for objective satisfy lipschitz continuity
            c_212a = obj_fun(x_k + _alpha*d) <= obj_fun(x_k) + _alpha*obj_grad_k.T@d + 0.5*_L*(_alpha**2)*(d_k_l2)
            # same for constraints
            c_212b = np.abs(c(x_k + _alpha*d)) <= np.abs(ce_k + _alpha*j_k.T@d) + 0.5*_gamma*(_alpha**2)*d_k_l2
            # if achieved sufficient descent w.r.t. merit fn or according to lipschitz
            if c_210 or (c_212a and c_212b):
                x_next = x_k + _alpha*d
                break
            else:
                # if lipschitz estimate for obj fn is bad
                if not c_212a:
                    _L *= rho
                # same for constraints
                if not np.all(c_212b):
                    _gamma[not c_212b] *= rho
    else:
        # calculate step size (linesearch)
        _alpha = alpha
        # check sufficient reduction
        while True:
            reduction_term = nu*_alpha*delta_q(x_k, _merit_par, obj_grad_k, _quad_term, d, _constr_l1)
            if merit_fn(x_k+_alpha*d, _merit_par) <= merit_fn(x_k, _merit_par) - reduction_term:
                break
            _alpha *= v
        x_next = x_k + _alpha*d
    return x_next, _merit_par, _L, _gamma, False

def optimize_det(p: Problem, x0, merit_par=0.5, alpha=1, L=1, gamma=[], eps=1e-6,sigma=0.5, v=0.5, nu=1e-4, adaptive=True):
    # sigma: model reduction factor
    # eps: parameter reduction factor
    # build merit fn
    def merit_fn(x, merit_par):
        return merit_par*p.f(x)+norm(p.c(x), ord=2)
    def delta_q(x, merit_par, _g, _j, _d, _c, _c_l2=None):
        return -merit_par*_g.T@_d + norm(_c, ord=2) - norm(_c+_j@_d)
    
    x_k = x0
    rng = np.random.default_rng(42)

    if adaptive and (gamma == [] or gamma is None):
        gamma = np.ones(len(p.c(x_k)))
    print(f'Starting conditions: x_0: {x0}, f(x_{0})={p.f(x0)}, c={p.c(x0)}')
    for k in range(100):

        x_k, merit_par, L, gamma, is_finished = _sqp_det_iter(
            adaptive=adaptive,
            k=k, obj_fun=f, x_k=x_k, obj_grad_k=p.g(x_k), c=p.c, ce_k=p.c(x_k), ci_k=None, j_k=p.J(x_k), h_k=p.H(x_k),
            merit_fn=merit_fn,
            delta_q=delta_q,
            L=L, gamma=gamma,
            merit_par=merit_par)
        print(f'Iteration: {k+1}, x_{k+1}: {x_k}, f(x_{k+1})={p.f(x_k)}, c={p.c(x_k)}, tau={merit_par}')
        if is_finished:
            return x_k