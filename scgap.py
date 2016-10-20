# coding=utf-8
import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
import argparse
import association_test
import common


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--prefix", required=True,
            help="genotype file (without extension) in plink bed file format")
    parser.add_argument("-o", "--out_prefix", required=True,
        help="prefix for association test output file")
    parser.add_argument('-l', '--locations_file', required=True,
        help="PCA or GAP coordinates that will be used for allele frequency \
             estimation and smoothing")
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv[1:])

    genotype_files_prefix = args.prefix
    out_prefix = args.out_prefix
    locations_file = args.locations_file

    kwargs = common.make_kwargs(args.args)

    common.print_log(" ".join(argv))
    common.print_log("args: ", args)
    common.print_log("kwargs: ", kwargs)

    bed_file_path = "{0}.bed".format(genotype_files_prefix)
    X = common.read_bed_file(bed_file_path)
    X = np.asarray(X, dtype=float)
    X[(X < 0) | (X > 2)] = np.nan

    fam_file_path = "{0}.fam".format(genotype_files_prefix)
    phenotype_df = pd.read_table(fam_file_path, header=None, delim_whitespace=True, names=["ind_id", "phenotype"], usecols=[1, 5])
    assert X.shape[0] == len(phenotype_df)

    loc_df = pd.read_table(locations_file, delim_whitespace=True, header=None, names=["ind_id", "coord1", "coord2"])
    loc = np.array(loc_df[["coord1", "coord2"]])
    assert X.shape[0] == loc.shape[0]

    bim_file_path = "{0}.bim".format(genotype_files_prefix)
    snp_df = common.read_bim_file(bim_file_path)
    assert X.shape[1] == len(snp_df)

    n, p = X.shape
    common.print_log("Input genotype matrix dimensions, n = {0}, p = {1}".format(n, p))

    Y = np.array(phenotype_df["phenotype"])

    # remove all inds with nan phenotypes
    non_missing_pheno_inds = ~np.isnan(Y)
    Y = Y[non_missing_pheno_inds]
    X = X[non_missing_pheno_inds, :]
    loc = loc[non_missing_pheno_inds, :]
    common.print_log("Found {0} individuals with phenotype".format(np.sum(non_missing_pheno_inds)))

    llr, K, R = association_test.association_test(X, Y, loc)
    llr[llr < 0] = np.nan
    p_vals = 1. - stats.chi2.cdf(llr, df=1)
    output_df = pd.DataFrame({"snp": snp_df["snp"], "llr": llr, "K": K, "R": R, "p": p_vals})

    output_file_path = "{0}.scgap".format(out_prefix)
    output_df.to_csv(output_file_path, sep="\t", header=False, index=False, na_rep="nan", columns=["snp", "llr", "K", "R", "p"])
    common.print_log("Output of association test written to {0}".format(output_file_path))


if __name__ == "__main__":
    main(sys.argv)
