# coding=utf-8
from __future__ import print_function
import numpy as np
import sys
import os
import pandas as pd


def print_log(*s, **kwargs):
    log_handle = kwargs.get("log_handle", sys.stderr)
    print(*s, file=log_handle)
    log_handle.flush()


def make_kwargs(args):
    kwargs = {}
    for kv in args:
        arr = kv.split("=")
        if len(arr) == 2:
            key, val = arr
        else:
            key = arr[0]
            val = None
        kwargs[key] = val
    return kwargs


def count_lines(fpath):
    line_count = 0
    with open(fpath, "r") as fin:
        for line_count, _ in enumerate(fin, 1):
            pass
    return line_count


PLINK_BED_MAGIC_NUMBERS = [0b01101100, 0b00011011]
PLINK_SNP_MAJOR_MODE = 1


def read_bed_file(bed_file_path, missing=255):
    """read genotypes in plink bed file format with sample size n and number of SNPs p
        missing genotypes are set to missing
    """
    bed_file_dirname, bed_file_basename = os.path.split(bed_file_path)
    arr = bed_file_basename.split(".")
    if len(arr) >= 2:
        bed_file_prefix = ".".join(arr[:-1])
    else:
        bed_file_prefix = ".".join(arr)
    n = count_lines(os.path.join(bed_file_dirname, bed_file_prefix + ".fam"))
    p = count_lines(os.path.join(bed_file_dirname, bed_file_prefix + ".bim"))
    print_log("Reading bed file {0} with n = {1}, p = {2}".format(bed_file_path, n, p))
    return read_bed_file_dims(bed_file_path, n, p, missing)


def read_bed_file_dims(bed_file_path, n, p, missing=255):
    """read genotypes in plink bed file format with sample size n and number of SNPs p
        missing genotypes are output as 255 by default
    """
    # reads bed file format described at http://pngu.mgh.harvard.edu/~purcell/plink/binary.shtml
    tmp_X = np.fromfile(bed_file_path, dtype=np.uint8)

    assert all(tmp_X[0:2] == PLINK_BED_MAGIC_NUMBERS), "magic numbers in bed file don't match expected values"
    assert tmp_X[2] == PLINK_SNP_MAJOR_MODE, "bed file needs to be in SNP-major mode"
    assert 4*(tmp_X.shape[0] - 3) % p == 0, "mismatch in bed file dimensions"

    n_tilde = int((tmp_X.shape[0] - 3) * 4 / p)     # n_tilde = 4 * ceil (n / 4)
    
    X = np.zeros(n_tilde * p, dtype=np.uint8)

    X[0::4] = (tmp_X[3:] & 0b11      )
    X[1::4] = (tmp_X[3:] & 0b1100    ) >> 2
    X[2::4] = (tmp_X[3:] & 0b110000  ) >> 4
    X[3::4] = (tmp_X[3:] & 0b11000000) >> 6

    # 00  Homozygote "1"/"1"
    # 01  Heterozygote
    # 11  Homozygote "2"/"2"
    # 10  Missing genotype
    X[X == 2] = missing   # missing value
    X[X == 3] = 2

    X = np.reshape(X, (p, n_tilde))

    return X[:, 0:n].T


def write_bed_file_dims(X, file_path, n, p):
    X = np.array(X, dtype=np.uint8)
    n_tilde = 4 * int((n + 3) / 4)
    X = np.vstack([X, np.zeros((n_tilde - n, p), dtype=np.uint8)])
    X = X.T.flatten()
    X[X == 2] = 3
    tmp_X = np.zeros((n_tilde * p) / 4 + 3, dtype=np.uint8)
    # print tmp_
    tmp_X[0:2] = PLINK_BED_MAGIC_NUMBERS
    tmp_X[2] = PLINK_SNP_MAJOR_MODE
    tmp_X[3:] = X[0::4] | (X[1::4] << 2) | (X[2::4] << 4) | (X[3::4] << 6)
    tmp_X.tofile(file_path)


def read_bim_file(bim_file_path):
    snp_df = pd.read_table(bim_file_path, delim_whitespace=True, header=None, names=["chr", "snp", "dist", "pos", "allele1", "allele2"])
    return snp_df
