# coding=utf-8
import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse import csr_matrix
import pandas as pd
import common


def compute_genetic_distance(X, **kwargs):
    """Given genotype matrix X, returns pairwise genetic distance between individuals
    using the estimator described in Theorem 1.

    Args:
        X: n * p matrix of 0/1/2/nan, n is #individuals, p is #SNPs
    """

    n, p = X.shape
    missing = np.isnan(X)
    col_sums = np.nansum(X, axis=0)
    col_counts = np.sum(~missing, axis=0)
    mu_hat = col_sums / 2. / col_counts     # p dimensional

    eta0_hat = np.nansum(X**2 - X, axis=0) / 2. / col_counts - mu_hat**2

    X_tmp = X/2.
    X_tmp[missing] = 0
    non_missing = np.array(~missing, dtype=float)

    X_shifted = X_tmp - mu_hat
    gdm_squared = 2. * np.mean(eta0_hat) - 2. * np.dot(X_shifted, X_shifted.T) / np.dot(non_missing, non_missing.T)
    gdm_squared[np.diag_indices(n)] = 0.

    if len(gdm_squared[gdm_squared < 0]) > 0:
        # shift all entries by the smallest amount that makes them non-negative
        shift = - np.min(gdm_squared[gdm_squared < 0])
        gdm_squared += shift
        gdm_squared[np.diag_indices(n)] = 0.

    gdm = np.sqrt(np.maximum(gdm_squared, 0.))

    return gdm


def compute_connectivity_threshold(Ds, **kwargs):
    n, _ = Ds.shape

    Ds_nz = Ds[Ds > 0]
    # find the range of values of the distance for which the graph is connected
    lo = np.min(Ds_nz)
    hi = np.max(Ds_nz)

    eps = 0.01*(hi - lo)
    while (hi - lo) >= eps:
        mid = (lo + hi) / 2.
        sparse_graph = csr_matrix(Ds <= mid)
        n_components, _ = connected_components(sparse_graph)
        if n_components != 1:   # graph is not connected
            lo = mid
        else:
            hi = mid

    thresh = hi
    common.print_log("Smallest threshold tau which makes graph connected = {0}".format(thresh))
    return thresh


def get_candidate_taus(Ds, **kwargs):
    thresh = compute_connectivity_threshold(Ds, **kwargs)
    return get_candidate_taus_above_threshold(Ds, thresh, **kwargs)


def get_candidate_taus_above_threshold(Ds, thresh, **kwargs):
    upper_tri_Ds = Ds[np.triu_indices_from(Ds, k=1)]
    if "nz_frac" in kwargs:
        nz_frac = float(kwargs["nz_frac"])
        common.print_log("Setting tau so that fraction of distances below threshold = {0}".format(nz_frac))

        all_taus = np.array(sorted(upper_tri_Ds))
        n_all_taus = len(all_taus)
        idx = min(max(int(nz_frac*n_all_taus), 0), n_all_taus-1)
        tau = all_taus[idx]
        if tau < thresh:
            common.print_log("Parameter tau was set below the minimum value which makes the graph connected. Changing it to {0}".format(thresh))
            tau = thresh
        candidate_taus = np.array([tau])
    else:
        grid_size = int(kwargs.get("grid_size", 20))

        linspace_tau = bool(kwargs.get("linspace_tau", False))
        if linspace_tau:
            candidate_taus = np.linspace(thresh, np.max(Ds[Ds > 0]), grid_size)
        else:
            all_taus = np.array(sorted(upper_tri_Ds[upper_tri_Ds > thresh]))
            n_all_taus = len(all_taus)
            tau_indices = np.asarray(np.concatenate([np.linspace(0, 1, grid_size)]) * (n_all_taus - 1), dtype=int)
            candidate_taus = sorted(all_taus[tau_indices])

    nz_fracs = [100. * np.sum(upper_tri_Ds <= tau) / len(upper_tri_Ds) for tau in candidate_taus]

    common.print_log("Found {0} candidate thresholds:".format(len(candidate_taus)), candidate_taus)
    common.print_log("Percentage of distances below threshold:", nz_fracs)
    return candidate_taus


def pca(X, **kwargs):
    col_sums = np.nansum(X, axis=0)
    n_inds = np.sum(~ np.isnan(X), axis=0)
    mu_hat = col_sums / n_inds      # dimension p (or n if rowNormalize is true)

    # normalization by estimated std deviation
    sd_hat = (1. + col_sums) / (2. + 2.*n_inds)
    sd_hat = np.sqrt(sd_hat * (1. - sd_hat))

    Xn = X.copy()
    Xn -= mu_hat
    Xn[np.isnan(Xn)] = 0.

    Xn /= sd_hat

    grm = np.dot(Xn, Xn.T)

    eig_indices = kwargs.get("eig_indices", [1, 2])
    eig_indices = np.array(sorted([int(x) for x in eig_indices]))

    common.print_log("Computing principal components:", eig_indices)

    S, U = np.linalg.eigh(grm)
    S_PCA, U_PCA = S[-eig_indices], U[:, -eig_indices]
    assert np.all(S_PCA > 0.)

    loc_PCA = np.sqrt(S_PCA) * U_PCA
    variance_explained = np.sum(S_PCA) / np.trace(grm) * 100.

    reconstruction_proportion = np.sum(S_PCA) / np.sum(np.abs(S))

    # (||LD'L||_F / ||LDL||_F)
    common.print_log("Percent variance explained by PCA projection = {0}".format(variance_explained))
    common.print_log("Distance matrix reconstruction proportion = {0}".format(reconstruction_proportion))

    return loc_PCA, variance_explained, reconstruction_proportion


def mds(dist_mat, verbose=True):
    n = dist_mat.shape[0]

    sparse_graph = csr_matrix(dist_mat)
    n_components, _ = connected_components(sparse_graph)
    if n_components != 1:
        common.print_log("Choose larger threshold tau!")
        return None, None, None

    shortest_path_dist = shortest_path(sparse_graph, method='D')
    if verbose:
        common.print_log("Shortest path distance matrix entries, mean = %f, std dev = %f, max = %f" % (np.mean(shortest_path_dist), np.std(shortest_path_dist), np.max(shortest_path_dist)))

    shortest_path_dist_sq = shortest_path_dist**2

    C = np.eye(n) - 1./n * np.ones((n, n))
    tmp = - 0.5 * np.dot(np.dot(C, shortest_path_dist_sq), C)
    tmp = (tmp + tmp.T) / 2.

    S, U = np.linalg.eigh(tmp)
    S_MDS, U_MDS = S[[-1, -2]], U[:, [-1, -2]]
    assert np.all(S_MDS > 0.)

    loc_MDS = np.sqrt(S_MDS) * U_MDS

    if verbose:
        common.print_log("Num positive eigenvalues of MDS matrix =", np.sum(S >= 0.))
        common.print_log("Num negative eigenvalues of MDS matrix =", np.sum(S < 0.))

    reconstruction_proportion = np.sum(S_MDS) / np.sum(np.abs(S))

    if verbose:
        # (||LD'L||_F / ||LDL||_F)
        common.print_log("Distance matrix reconstruction proportion = {0}".format(reconstruction_proportion))

    return loc_MDS, reconstruction_proportion


def optimal_rescaling(true_loc, estimated_loc):
    """return RMSE of the error between the true location and the estimated location,
    after applying the best affine transformation to the estimated locations
    """
    true_loc = true_loc.copy()
    estimated_loc = estimated_loc.copy()

    n = true_loc.shape[0]
    shift = np.mean(true_loc, axis=0)
    true_loc = true_loc - np.tile(shift, (n, 1))
    estimated_loc = estimated_loc - np.tile(np.mean(estimated_loc, axis=0), (n, 1))

    T1 = np.dot(estimated_loc.T, true_loc)
    T2 = np.dot(estimated_loc.T, estimated_loc)
    opt_rescaling = np.dot(np.linalg.inv(T2), T1)

    tmp = true_loc - np.dot(estimated_loc, opt_rescaling)
    tmp = np.sum(tmp**2, axis=1)
    rmse = np.sqrt(np.sum(tmp, axis=0) / n)

    return rmse, opt_rescaling, shift
