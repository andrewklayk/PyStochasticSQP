from stoch_sqp import optimize_st
from problem import Problem
import numpy as np

def main():
    x0 = np.array([4,2])

    # x0 = np.array([4,2])
    # def f(x): return 0.5*x[0]**2 + 0.5*x[1]**2
    # def g(x): return np.array([x[0], x[1]])
    # def c(x): return np.array([x[0] + x[1] - 1])
    # def J(x): return np.ones(2)
    # def H(x): return np.eye(2)

    def f(x): return x[0]**3 + x[1]**3
    def g(x): return np.array([3*x[0]**2, 3*x[1]**2])
    def c(x): return np.array([x[0]**2 + x[1]**2-1])
    def J(x): return np.array([[2*x[0], 2*x[1]]])
    def H(x): return np.array([[6*x[0],0],[0,6*x[1]]])

    p = Problem(f, g, c, J, H)

    optimize_st(p, x0,iter_limit=100)

if __name__ == "__main__":
    main()