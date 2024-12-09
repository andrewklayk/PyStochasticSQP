import numpy as np
import scipy
from scipy.linalg import solve
import scipy.optimize

def sqp_det_iter(k, 
    x_k, obj_grad_k, ce_k, ci_k, j_k, h_k, phi, delta_q,
    adaptive=False, sigma=0.5,tau=1, eps=1e-6, alpha=1, v=0.5, nu=1e-4, rho=3, L=1, gamma=None
    ):

    if adaptive and gamma is None:
        gamma = np.ones(len(ce_k) + len(ci_k))
    if ci_k is None:
        ci_k = []
    m = np.shape(ce_k)[0] + np.shape(ci_k)[0]

    # construct and solve the SQP linear problem
    sqp_A = np.vstack([
        np.hstack([h_k, j_k[np.newaxis].T]),
        np.hstack([j_k[np.newaxis],np.zeros((m,m))])
    ])
    sqp_b = -np.hstack([obj_grad_k,ce_k])
    qp_sol = solve(sqp_A, sqp_b)
    d_k, y_k = qp_sol[:len(x_k)], qp_sol[len(x_k):]

    # check stopping condition
    #TODO: ADD NUMERICAL STOPPING CONDITION
    if np.all(obj_grad_k + j_k[np.newaxis].T @ y_k) == 0 and np.all(ce_k) == 0:
        return (x_k, tau, L, True)

    # calc the quadratic term to use later
    qt = np.max(d_k.T @ h_k @ d_k, 0)

    # calculate merit parameter
    if obj_grad_k.T @ d_k + qt <= 0:
        t_k_trial = np.inf
    else:
        t_k_trial = ((1-sigma)*np.sum(np.abs(ce_k) + np.abs(ci_k)))/(obj_grad_k.T@d_k + qt)
    if tau > t_k_trial:
        tau = (1-eps)*t_k_trial
    if adaptive:
        # set Lipschitz estimates as 1/2 of previous iteration
        if not k == 0:
            L *= 0.5
            gamma *= 0.5
        while True:
            # calculate stepsize
            alpha_denom = tau*L+np.sum(gamma)*np.sum(d_k**2)
            alpha_k_hat = 2*(1-nu)*delta_q(x_k, tau, obj_grad_k, qt, d_k, ce_k)/alpha_denom
            alpha_k_tilde = alpha_k_hat - 4*np.sum(np.abs(ce_k) + np.abs(ci_k))/alpha_denom
            if alpha_k_hat < 1:
                alpha_k = alpha_k_hat
            elif alpha_k_tilde <= 1:
                alpha_k = 1
            else:
                alpha_k = alpha_k_tilde
            
            reduction_term = nu*alpha_k*delta_q(x_k, tau, obj_grad_k, qt, d_k, ce_k)
            phi(x_k+alpha_k*d_k, tau) > phi(x_k, tau) - reduction_term
        raise NotImplementedError
    else:
        # calculate step size (linesearch)
        alpha_k = alpha
        # check sufficient reduction
        while True:
            reduction_term = nu*alpha_k*delta_q(x_k, tau, obj_grad_k, qt, d_k, ce_k)
            if phi(x_k+alpha_k*d_k, tau) <= phi(x_k, tau) - reduction_term:
                break
            alpha_k *= v
        x_next = x_k + alpha_k*d_k
    return x_next, tau, L, False

def det_sqp_ls(f, g, J, H, c, x0, sigma=0.5,tau=1,eps=1e-6, alpha=1, v=0.5, nu=1e-4, L=1):
    # build merit fn
    def merit(x, tau_k): return tau_k*f(x)+np.sum(np.abs(c(x)))
    def delta_q(x_k, tau_k, g_k, qt, d_k, c_l1=None): return -tau_k*(g_k.T@d_k + 0.5*qt) + np.sum(np.abs(c(x_k)))
    x_k = x0
    print(f'Starting conditions: x_0: {x0}, f(x_{0})={f(x0)}, c={c(x0)}')
    for k in range(100):
        x_k, tau, L, is_finished = sqp_det_iter(k, x_k, g(x_k), c(x_k), None, J(x_k), H(x_k), merit, delta_q, tau=tau, L=L*2)
        print(f'Iteration: {k+1}, x_{k+1}: {x_k}, f(x_{k+1})={f(x_k)}, c={c(x_k)}, tau={tau}')
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
def H(x): return np.array([[2,0],[0,2]])

det_sqp_ls(f, g, J, H, c, x0)