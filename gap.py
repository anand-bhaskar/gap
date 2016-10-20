# coding=utf-8
import numpy as np
import pandas as pd
import sys
import argparse
import localization
import common


def rescale_locations(df_inferred, df_training):
    df_training = pd.merge(df_inferred, df_training, on="ind_id", suffixes=["_inferred", "_true"])

    loc_inferred_training = np.array(df_training[["coord1_inferred", "coord2_inferred"]])
    loc_true_training = np.array(df_training[["coord1_true", "coord2_true"]])
    training_rmse, rescaling, shift = localization.optimal_rescaling(loc_true_training, loc_inferred_training)

    loc_inferred = np.array(df_inferred[["coord1", "coord2"]])
    n = len(loc_inferred)
    loc_inferred = loc_inferred - np.tile(np.mean(loc_inferred, axis=0), (n, 1))
    loc_inferred = np.dot(loc_inferred, rescaling) + np.tile(shift, (n, 1))

    df_inferred.loc[:, ["coord1", "coord2"]] = loc_inferred
    return df_inferred, training_rmse


def localize_pca(X, out_prefix, inds_df, inds_training_df, **kwargs):
    n, p = X.shape

    common.print_log()
    common.print_log("Running PCA")
    loc_PCA, variance_explained, reconstruction_proportion = localization.pca(X, **kwargs)

    df_inferred = pd.DataFrame({"ind_id": inds_df.ind_id, "coord1": loc_PCA[:, 0], "coord2": loc_PCA[:, 1]})
    if inds_training_df is not None:
        df_inferred, training_rmse = rescale_locations(df_inferred, inds_training_df)
        common.print_log("RMSE on training data: {0}".format(training_rmse))

    pca_output_path = "{0}.pca".format(out_prefix)
    df_inferred.to_csv(pca_output_path, sep="\t", header=False, index=False, columns=["ind_id", "coord1", "coord2"])
    common.print_log("Wrote PCA inferred locations to {0}".format(pca_output_path))


def split_folds(training_df, cv_folds):
    if training_df is None:
        return None
    n = training_df.shape[0]
    cv_folds = min(cv_folds, n)
    size_fold = int((n + cv_folds - 1) / cv_folds)
    training_df_folds = [None]*cv_folds
    for fold_idx in range(cv_folds):
        training_df_folds[fold_idx] = training_df[fold_idx*size_fold:min(n, (fold_idx+1)*size_fold)]
    return training_df_folds


def localize_gap(X, out_prefix, inds_df, inds_training_df, cv_folds, **kwargs):
    gdm = localization.compute_genetic_distance(X, **kwargs)

    common.print_log()
    common.print_log("Running GAP")
    cand_taus = localization.get_candidate_taus(gdm, **kwargs)

    upper_tri_Ds = gdm[np.triu_indices_from(gdm, k=1)]

    training_df_folds = None
    if inds_training_df is not None:
        training_df_folds = split_folds(inds_training_df, cv_folds)
        common.print_log("Using {0}-fold cross-validation to optimize threshold tau".format(cv_folds))
    else:
        common.print_log("No training data provided. Using distance matrix reconstruction proportion to optimize threshold tau.")

    best_cv_rmse = np.inf
    best_reconstruction_proportion = 0.0
    for tau_idx, tau in enumerate(cand_taus):
        thresholded_gdm = gdm * (gdm <= tau)
        common.print_log()
        common.print_log("tau_idx = {0}, tau = {1}, percentage of distances <= tau = {2}".format(tau_idx, tau, 100.*np.sum(upper_tri_Ds <= tau)/len(upper_tri_Ds)))

        loc_GAP, reconstruction_proportion = localization.mds(thresholded_gdm)
        if loc_GAP is None:
            continue

        df_inferred = pd.DataFrame({"ind_id": inds_df.ind_id, "coord1": loc_GAP[:, 0], "coord2": loc_GAP[:, 1]})
        if training_df_folds is not None:
            cv_rmse = 0.
            for fold_idx in range(cv_folds):
                _, fold_rmse = rescale_locations(df_inferred, training_df_folds[fold_idx])
                cv_rmse += fold_rmse
            cv_rmse /= cv_folds
            common.print_log("Cross-validation RMSE = {0}".format(cv_rmse))

            if cv_rmse <= best_cv_rmse:
                best_tau_idx, best_tau, best_cv_rmse = tau_idx, tau, cv_rmse
        else:
            if reconstruction_proportion >= best_reconstruction_proportion:
                best_tau_idx, best_tau, best_reconstruction_proportion = tau_idx, tau, reconstruction_proportion
                output_df = df_inferred

    common.print_log()
    common.print_log("Optimal tau_idx = {0}, tau = {1}, percentage of distances <= tau = {2}".format 
        (best_tau_idx, best_tau, 100.*np.sum(upper_tri_Ds <= best_tau)/len(upper_tri_Ds)))

    if training_df_folds is not None:
        # compute RMSE on all training data
        thresholded_gdm = gdm * (gdm <= best_tau)
        loc_GAP, reconstruction_proportion = localization.mds(thresholded_gdm, verbose=False)
        df_inferred = pd.DataFrame({"ind_id": inds_df.ind_id, "coord1": loc_GAP[:, 0], "coord2": loc_GAP[:, 1]})
        output_df, training_rmse = rescale_locations(df_inferred, inds_training_df)

        common.print_log("Best cross-validation RMSE = {0}".format(best_cv_rmse))
        common.print_log("RMSE on training data = {0}".format(training_rmse))
    else:
        common.print_log("Best reconstruction proportion = {0}".format(best_reconstruction_proportion))

    gap_output_path = "{0}.gap".format(out_prefix)
    output_df.to_csv(gap_output_path, sep="\t", header=False, index=False, columns=["ind_id", "coord1", "coord2"])
    common.print_log()
    common.print_log("Wrote GAP locations to {0}".format(gap_output_path))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--prefix", required=True,
        help="genotype file (without extension) in plink bed file format")
    parser.add_argument("-l", "--out_prefix", required=True,
        help="prefix for file names where localization outputs will be stored")
    parser.add_argument("-t", "--training_file", required=False, default=None,
        help="file containing a subset of the individuals with known locations")
    parser.add_argument("-c", "--cv_folds", required=False, type=int, default=1,
        help="number of folds for cross-validation")
    parser.add_argument("args", nargs=argparse.REMAINDER,
        help="specify either gap or pca (or both) for the localization algorithm to run")

    args = parser.parse_args(argv[1:])

    genotype_files_prefix = args.prefix
    out_prefix = args.out_prefix
    training_file = args.training_file if args.training_file else None
    cv_folds = args.cv_folds

    kwargs = common.make_kwargs(args.args)

    common.print_log(" ".join(argv))
    common.print_log("args: ", args)
    common.print_log("kwargs: ", kwargs)

    fam_file_path = "{0}.fam".format(genotype_files_prefix)
    inds_df = pd.read_table(fam_file_path, header=None, delim_whitespace=True, names=["ind_id"], usecols=[1])

    inds_training_df = None
    if training_file:
        inds_training_df = pd.read_table(training_file, delim_whitespace=True, header=None, names=["ind_id", "coord1", "coord2"])
        inds_training_df = pd.merge(inds_df, inds_training_df, how="inner", on=["ind_id"])
        cv_folds = min(cv_folds, inds_training_df.shape[0])

    bed_file_path = "{0}.bed".format(genotype_files_prefix)
    X = common.read_bed_file(bed_file_path)

    n, p = X.shape
    assert len(inds_df) == n

    X = np.asarray(X, dtype=float)
    X[(X < 0) | (X > 2)] = np.nan

    common.print_log("Input matrix dimensions, n = {0}, p = {1}".format(n, p))

    if "pca" in kwargs:
        localize_pca(X, out_prefix, inds_df, inds_training_df, **kwargs)

    if "gap" in kwargs:
        localize_gap(X, out_prefix, inds_df, inds_training_df, cv_folds, **kwargs)


if __name__ == "__main__":
    main(sys.argv)
