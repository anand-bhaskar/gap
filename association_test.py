# coding=utf-8
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
import common


# K is 1 X p
# R is 1 X p
# Y is n X 1

def compute_smoothing_kernel(loc, threshold=0):
    n, _ = loc.shape
    S = np.cov(loc.T)
    K = np.linalg.cholesky(S)   # K * K' = S
    y = np.dot(loc, np.linalg.inv(K.T)) * n**(1./6)

    DP = cdist(y, y, metric='sqeuclidean')
    kernel = np.exp(- DP/2.)
    kernelSum = np.sum(kernel, axis=1)
    kernel = kernel / kernelSum[:, np.newaxis]

    if threshold > 0:
        kernel[kernel < threshold] = 0.
        kernelSum = np.sum(kernel, axis=1)
        kernel = kernel / kernelSum[:, np.newaxis]

    # make sparse matrix from the kernel
    kernel = sparse.csr_matrix(kernel)
    return kernel


def estimate_Q(X, kernel):
    n, p = X.shape

    q_hat = (X == 1)/2. + (X == 2)
    # q_hat = np.dot(kernel, q_hat)
    q_hat = kernel.dot(q_hat)
    q_hat[np.isnan(X)] = np.nan

    return q_hat


def estimate_K(q_hat, R_pow_Y, X):
    n, p = X.shape
    TOL = 1e-13
    EPS = 1.e-10

    K = np.ones((1, p))
    R_pow_Y_q_hat = R_pow_Y * q_hat
    for iters in range(100):
        K_RtoY_qHat = K * R_pow_Y_q_hat
        Z = 1. - q_hat + K_RtoY_qHat
        s1 = 2. * K_RtoY_qHat
        T1 = np.nansum(s1/Z - X, axis=0)

        s1 = 2. * (1.-q_hat) * R_pow_Y_q_hat
        T2 = np.nansum(s1/(Z**2), axis=0)

        K_new = K - T1/T2
        K_new = np.fmax(EPS, K_new)
        test = np.max(abs(K_new-K))
        K = K_new
        if test < TOL:
            break

    common.print_log("Num K iterations = {0}, err = {1}".format(iters, test))
    return K


def likelihood(R_pow_Y, K, X, q_hat):
    n, p = X.shape

    K_R_pow_Y_q_hat = K * R_pow_Y * q_hat
    Z = 1. - q_hat + K_R_pow_Y_q_hat
    theta = K_R_pow_Y_q_hat / Z
    T = X * np.log(theta) + (2. - X) * np.log(1. - theta)
    lmbda = np.nansum(T, axis=0)

    return lmbda


def optimize_R_K(q_hat, R, K, X, Y):
    # optimize K and R simultaneously by Newton's method
    n, p = X.shape
    TOL = 1e-6
    EPS = 1.e-10

    for iters in range(100):
        R_pow_Y_q_hat = R**Y * q_hat
        K_R_pow_Y_q_hat = K * R_pow_Y_q_hat

        Z = 1. - q_hat + K_R_pow_Y_q_hat
        s1 = 2. * K_R_pow_Y_q_hat
        s2 = s1/Z - X
        F1 = np.nansum(s2 * Y, axis=0)
        F2 = np.nansum(s2, axis=0)

        tmp = 2. * R_pow_Y_q_hat * (1. - q_hat) / Z**2
        J22 = np.nansum(tmp, axis=0)
        
        tmp = Y * tmp
        J12 = np.nansum(tmp, axis=0)

        tmp = K * tmp / R
        J21 = np.nansum(tmp, axis=0)

        tmp = Y * tmp
        J11 = np.nansum(tmp, axis=0)
        
        det = J11 * J22 - J12 * J21
        R_new = R - (J22 * F1 - J12 * F2) / det
        K_new = K - (- J21 * F1 + J11 * F2) / det

        R_new = np.fmax(EPS, R_new)
        K_new = np.fmax(EPS, K_new)

        test_R = np.max(abs((R_new - R) / np.fmax(R_new, R)))
        test_K = np.max(abs((K_new - K) / np.fmax(K_new, K)))
        
        R = R_new
        K = K_new

        if iters % 10 == 0:
            common.print_log("Iteration {0} of joint R, K optimization".format(iters))
            common.print_log("Max rel err in R = {0}".format(test_R))
            common.print_log("Max rel err in K = {0}".format(test_K))

        if test_R < TOL and test_K < TOL:
            break

    common.print_log("Num iterations of joint R, K optimization", iters)
    common.print_log("Max rel err in R", test_R)
    common.print_log("Max rel err in K", test_K)
    return R, K


def association_test(X, Y, loc):
    """ performs SCGAP association test
    """
    n, p = X.shape
    Y = Y[:, np.newaxis]

    kernel = compute_smoothing_kernel(loc, threshold=1e-4)

    # estimation under null hypothesis
    Rn = np.ones((1, p))
    Kn = np.ones((1, p))
    Rn_pow_y = Rn**Y

    common.print_log("Null hypothesis optimization")
    q_hat = estimate_Q(X, kernel)
    Kn = estimate_K(q_hat, Rn_pow_y, X)

    loglik_null = likelihood(Rn_pow_y, Kn, X, q_hat)

    # estimation under alternate hypothesis
    max_num_restarts = 5
    R = np.ones((1, p))
    K = np.ones((1, p))
    best_loglik_alt = - np.ones(p) * np.inf
    best_R_alt = np.ones(p)
    best_K_alt = np.ones(p)

    X_cur = X
    q_cur = q_hat
    neg_llr_inds = range(p)
    restart_idx = 0
    while True:
        if restart_idx > 0:     # random restart
            R = np.random.uniform(1.2**(-restart_idx), 1.2**restart_idx, (1, len(neg_llr_inds)))
            K = np.random.uniform(1.2**(-restart_idx), 1.2**restart_idx, (1, len(neg_llr_inds)))
            X_cur = X[:, neg_llr_inds]
            q_cur = q_hat[:, neg_llr_inds]

        R_pow_Y = R**Y
        common.print_log("Alternate hypothesis optimization")
        R, K = optimize_R_K(q_cur, R, K, X_cur, Y)

        R_pow_Y = R**Y
        loglik_alt = likelihood(R_pow_Y, K, X_cur, q_cur)
        best_loglik_alt[neg_llr_inds] = np.fmax(loglik_alt, best_loglik_alt[neg_llr_inds])
        best_R_alt[neg_llr_inds] = R.flatten()
        best_K_alt[neg_llr_inds] = K.flatten()

        neg_llr_inds, = np.where(best_loglik_alt < loglik_null)
        if len(neg_llr_inds) == 0:
            common.print_log("")
            common.print_log("Alternate hypotheses optimization needed", restart_idx, "restarts")
            break

        if restart_idx == max_num_restarts:
            common.print_log("Terminating restart procedure after", restart_idx, "restarts")
            break

        restart_idx = restart_idx + 1

        common.print_log("")
        common.print_log("Restart idx", restart_idx)
        common.print_log("Num SNPs", len(neg_llr_inds))

    llr = 2.*(best_loglik_alt - loglik_null)

    return llr, best_K_alt, best_R_alt
