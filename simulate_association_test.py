# coding=utf-8
import numpy as np
import pandas as pd
import sys
import argparse
import simulate_localization
import common


def simulate_square(allele_freq_fn, n, p, pc, **kwargs):
    loc, Q, X = simulate_localization.simulate_square(allele_freq_fn, n, p, **kwargs)

    pr = p - pc
    common.print_log("Number of non-zero effect size SNPs = {0}".format(pc))
    common.print_log("Total number of SNPs = {0}".format(p))

    null_snps = np.arange(0, pr)
    causal_snps = np.arange(pr, p)

    geno_h = float(kwargs.get("geno_h", 0.90))
    anc_h = float(kwargs.get("anc_h", 0.05))
    env_h = float(kwargs.get("env_h", 0.05))

    is_discrete = "discrete" in kwargs

    common.print_log("Genetic heritability proportion = {0}".format(geno_h))
    common.print_log("Ancestry heritability proportion = {0}".format(anc_h))
    if not is_discrete:
        common.print_log("Environment heritability proportion = {0}".format(env_h))

    # generate genotype contribution to phenotype
    betas = np.zeros(p)
    betas[causal_snps] = np.random.normal(0, 1, pc)
    geno_contribution = np.dot(X, betas)
    geno_contribution = geno_contribution * np.sqrt(geno_h) / np.std(geno_contribution)

    # generate location-dependent contribution to phenotype
    # if alleleFreqFn == simulate.logisticDirectionalExpDecayCovAlleleFreqFn:
    theta = float(kwargs.get("theta", 0.))
    u = np.array([np.cos(theta), np.sin(theta)])
    anc_contribution = np.dot(loc, u)
    anc_contribution = anc_contribution * np.sqrt(anc_h) / np.std(anc_contribution)

    # phenotype is the sum of genotype contribution and env contribution
    Y = geno_contribution + anc_contribution

    if is_discrete:     # phenotype is discrete
        prob = 1. / (1. + np.exp(- Y))
        rnd = np.random.rand(n)
        Y = np.array(rnd <= prob, dtype=int)
    else:
        # generate independent env/noise contribution to phenotype for continuous phenotype
        env_contribution = np.random.normal(0, 1, n)
        env_contribution = env_contribution * np.sqrt(env_h) / np.std(env_contribution)
        Y = Y + env_contribution

    if is_discrete:
        return loc, Q, X, Y, null_snps, causal_snps, geno_contribution, anc_contribution
    else:
        return loc, Q, X, Y, null_snps, causal_snps, geno_contribution, anc_contribution, env_contribution


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
    parser.add_argument("--pc", required=False, type=int, default=10,
        help="Number of SNPs with non-zero effect sizes")
    parser.add_argument("-g", "--geno_h", required=False, type=float, default=0.95,
        help="Genetic heritability contribution (fraction between 0 and 1)")
    parser.add_argument("-a", "--anc_h", required=False, type=float, default=0.05,
        help="Ancestry heritability contribution (fraction between 0 and 1)")
    parser.add_argument("-e", "--env_h", required=False, type=float, default=0.05,
        help="Environment heritability contribution (fraction between 0 and 1)")
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
    pc = args.pc

    allele_freq_fn = simulate_localization.allele_freq_fns[allele_freq_model_name]

    is_discrete = "discrete" in kwargs

    if is_discrete:
        loc, Q, X, Y, null_snps, causal_snps, geno_contribution, anc_contribution = simulate_square(allele_freq_fn, n, p, pc, geno_h=args.geno_h, anc_h=args.anc_h, **kwargs)
    else:
        loc, Q, X, Y, null_snps, causal_snps, geno_contribution, anc_contribution, env_contribution = simulate_square(allele_freq_fn, n, p, pc, geno_h=args.geno_h, anc_h=args.anc_h, env_h=args.env_h, **kwargs)

    assert X.shape == (n, p)

    # output true locations to file
    output_locations_file = "{0}.loc".format(out_prefix)
    df = pd.DataFrame(loc)
    df.to_csv(output_locations_file, sep="\t", header=False)
    common.print_log("Wrote ancestry information to {0}".format(output_locations_file))

    # output allele frequencies to file if needed
    if "save_allele_frequencies" in kwargs:
        allele_frequency_file = "{0}.allelefreq.npy".format(out_prefix)
        np.save(allele_frequency_file, Q)
        common.print_log("Wrote allele frequencies to binary file {0}".format(allele_frequency_file))

    # output genotype data to bed/fam/bim file
    bed_file = "{0}.bed".format(out_prefix)
    common.write_bed_file_dims(X, bed_file, n, p)
    common.print_log("Wrote genotypes to file {0}".format(bed_file))

    # output phenotype contribution from different components
    fam_file = "{0}.fam".format(out_prefix)
    pheno_dict = {"fam_id": range(n), "ind_id": range(n), "pat_id": [0]*n, "mat_id": [0]*n, "sex": [0]*n, "phenotype": Y, "geno_contribution": geno_contribution, "anc_contribution": anc_contribution}
    if not is_discrete:
        pheno_dict["env_contribution"] = env_contribution
    pheno_df = pd.DataFrame(pheno_dict)
    if is_discrete:
        pheno_df.to_csv(fam_file, sep="\t", header=False, index=False, columns=["fam_id", "ind_id", "pat_id", "mat_id", "sex", "phenotype", "geno_contribution", "anc_contribution"])
    else:
        pheno_df.to_csv(fam_file, sep="\t", header=False, index=False, columns=["fam_id", "ind_id", "pat_id", "mat_id", "sex", "phenotype", "geno_contribution", "anc_contribution", "env_contribution"])
    common.print_log("Wrote phenotype to file {0}".format(fam_file))

    bim_file = "{0}.bim".format(out_prefix)
    # null SNPs are simulated to be on chr 1, causal on chr 2
    null_snps = set(null_snps)
    chrs = [1 if snp_idx in null_snps else 2 for snp_idx in range(p)]
    df = pd.DataFrame({"chr": chrs, "snp": range(p), "dist": range(p), "pos": range(p), "allele1": [0]*p, "allele2": [1]*p})
    df.to_csv(bim_file, sep="\t", header=False, index=False, columns=["chr", "snp", "dist", "pos", "allele1", "allele2"])
    common.print_log("Wrote snp list to file {0}".format(bim_file))


if __name__ == "__main__":
    main(sys.argv)
