# coding=utf-8
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import argparse
import sys
import common


def isotropic_decay_allele_freq_fn(z, p, **kwargs):
    """Isotropic covariance decay model, where the spatial allele frequencies are
    inverse-logit functions applied to the sample paths of a Gaussian process. The
    kernel of the Gaussian process is exp(-(alpha_1 ||z - z'||)^alpha_2)/alpha_0.
    """
    n = z.shape[0]
    alpha0 = float(kwargs.get("alpha0", 50.))
    alpha1 = float(kwargs.get("alpha1", 1.))
    alpha2 = float(kwargs.get("alpha2", 1.))
    cov_mat = np.exp(- (alpha1 * cdist(z, z, "euclidean"))**alpha2) / alpha0
    # sample from zero mean Gaussian process with covariance matrix cov_mat
    L = np.linalg.cholesky(cov_mat)
    F = np.dot(L, np.random.randn(n, p))
    Q = 1. / (1. + np.exp(F))
    return Q


def directional_decay_allele_freq_fn(z, p, **kwargs):
    """Directional covariance decay model, where the spatial allele frequencies are
    inverse-logit functions applied to the sample paths of a Gaussian process. The
    kernel of the Gaussian process is exp(-(alpha_1 |<z - z', u>|)^alpha_2)/alpha_0,
    where u is a vector from a von-Mises distribution with mean vmu and concentration vkappa.
    """
    n = z.shape[0]
    alpha0 = float(kwargs.get("alpha0", 50.))
    alpha1 = float(kwargs.get("alpha1", 1.))
    alpha2 = float(kwargs.get("alpha2", 1.))
    vmu = float(kwargs.get("vmu", 0.))
    vkappa = float(kwargs.get("vkappa", 1.))
    n_dirs = int(kwargs.get("n_dirs", 100))

    n_snps_per_dir = p / n_dirs
    theta = np.random.vonmises(mu=vmu, kappa=vkappa, size=(n_dirs,))
    u = np.array([np.cos(theta), np.sin(theta)])
    projections = np.dot(z, u)

    Q = np.zeros((n, p))

    for i in range(n_dirs):
        projection = projections[:, i][:, np.newaxis]
        cov_mat = np.exp(- (alpha1 * cdist(projection, projection, "euclidean"))**alpha2) / alpha0
        L = np.linalg.cholesky(cov_mat)
        F = np.dot(L, np.random.randn(n, n_snps_per_dir))
        Q[:, (i*n_snps_per_dir):min(p, (i+1)*n_snps_per_dir)] = 1. / (1. + np.exp(F))

    return Q


def generate_genotypes(Q):
    """generate genotype matrix X given n X p allele frequency matrix Q
    """
    n, p = Q.shape
    tmp = np.random.rand(n, p)
    q0 = (1-Q)**2
    q1 = 2*Q*(1-Q)
    ind1 = (q0 < tmp) & (tmp <= (q0 + q1))
    ind2 = (q0 + q1) < tmp
    X = np.array(ind1 + 2*ind2, dtype=np.uint8)
    return X


def simulate_square(allele_freq_fn, n=1000, p=50000, **kwargs):
    """simulate individual locations in a unit square with coordinates drawn
    independently from a Beta(b, b) distribution, using allele frequencies drawn from
    the stochastic process encoded in function allele_freq_fn
    """
    beta = float(kwargs.get("beta", 1.0))

    common.print_log("Simulating from the unit square, n = {0}, p = {1}".format(n, p))
    common.print_log("Coordinate distribution, beta =", beta)

    loc = np.random.beta(beta, beta, size=(n, 2)) - 0.5
    Q = allele_freq_fn(loc, p, **kwargs)
    X = generate_genotypes(Q)
    return loc, Q, X


allele_freq_fns = {"isotropic": isotropic_decay_allele_freq_fn,
                   "directional": directional_decay_allele_freq_fn}


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--out_prefix", required=True,
        help="file (without extension) to write output")
    parser.add_argument("-f", "--allele_freq_model", required=False, default="isotropic",
        help="Either 'isotropic' or 'directional'. Default, 'isotropic'")
    parser.add_argument("-n", "--n", required=True, type=int,
        help="Number of individuals n to simulate")
    parser.add_argument("-p", "--p", required=True, type=int,
        help="Number of SNPs p to simulate")
    parser.add_argument("args", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv[1:])
    kwargs = common.make_kwargs(args.args)

    common.print_log(" ".join(argv))
    common.print_log("args: ", args)
    common.print_log("kwargs: ", kwargs)

    out_prefix = args.out_prefix
    allele_freq_model_name = args.allele_freq_model
    n = args.n
    p = args.p

    allele_freq_fn = allele_freq_fns[allele_freq_model_name]

    loc, Q, X = simulate_square(allele_freq_fn, n, p, **kwargs)
    assert X.shape == (n, p)

    # output true locations to file
    output_locations_file = "{0}.loc".format(out_prefix)
    df = pd.DataFrame(loc)
    df.to_csv(output_locations_file, sep="\t", header=False)
    common.print_log("Wrote ind locations to {0}".format(output_locations_file))

    # output allele frequencies to file if needed
    if "save_allele_frequencies" in kwargs:
        allele_frequency_file = "{0}.allelefreq.npy".format(out_prefix)
        np.save(allele_frequency_file, Q)
        common.print_log("Wrote allele frequencies to binary file {0}".format(allele_frequency_file))

    # output genotype data to bed/fam/bim file
    bed_file = "{0}.bed".format(out_prefix)
    common.write_bed_file_dims(X, bed_file, n, p)
    common.print_log("Wrote genotypes to file {0}".format(bed_file))

    fam_file = "{0}.fam".format(out_prefix)
    df = pd.DataFrame({"fam_id": range(n), "ind_id": range(n), "pat_id": [0]*n, "mat_id": [0]*n, "sex": [0]*n, "status": [0]*n})
    df.to_csv(fam_file, sep="\t", header=False, index=False, columns=["fam_id", "ind_id", "pat_id", "mat_id", "sex", "status"])
    common.print_log("Wrote ind list to file {0}".format(fam_file))

    bim_file = "{0}.bim".format(out_prefix)
    df = pd.DataFrame({"chr": [1]*p, "snp": range(p), "dist": range(p), "pos": range(p), "allele1": [0]*p, "allele2": [1]*p})
    df.to_csv(bim_file, sep="\t", header=False, index=False, columns=["chr", "snp", "dist", "pos", "allele1", "allele2"])
    common.print_log("Wrote snp list to file {0}".format(bim_file))


if __name__ == "__main__":
    main(sys.argv)
