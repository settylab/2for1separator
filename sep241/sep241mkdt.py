#! /usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
import scipy

from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
from sep241util import logger, setup_logging, detect_cores
import sep241prep as spr
import sep241peakcalling as sep241peakcalling


desc = "Use 2for1seperator peak calling based on KDE of two single antibody CUT&Tag files."
parser = argparse.ArgumentParser(description=desc)
parser.add_argument(
    "c1_bed_file",
    metavar="reads_of_c1.bed",
    type=str,
    help="Bed file with single antibody reads of component 1.",
)
parser.add_argument(
    "c2_bed_file",
    metavar="reads_of_c2.bed",
    type=str,
    help="Bed file with single antibody reads of component 2.",
)
parser.add_argument(
    "out", help="Output directory.", type=str, metavar="out_dir",
)
parser.add_argument(
    "-l",
    "--log",
    dest="logLevel",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default="INFO",
    help="Set the logging level (default=INFO).",
    metavar="LEVEL",
)
parser.add_argument(
    "--logfile",
    help="Write detailed log to this file.",
    type=str,
    metavar="logfile",
)
parser.add_argument(
    "--blacklisted-seqs",
    help="Sequences to exclude from the deconvolution (default=chrM).",
    type=str,
    default=["chrM"],
    nargs="+",
    metavar="chrN",
)
parser.add_argument(
    "--kde-bw",
    help="Bandwidth (sigma) for kernel density estimate (KDE) used for interval selection (default=200).",
    type=float,
    default=200,
    metavar="float",
)
parser.add_argument(
    "--c1-min-peak-size",
    help="Minimal number of bases per peak for component 1 (default=100).",
    type=int,
    default=100,
    metavar="int",
)
parser.add_argument(
    "--c2-min-peak-size",
    help="Minimal number of bases per peak for component 2 (default=400).",
    type=int,
    default=400,
    metavar="int",
)
parser.add_argument(
    "--c1-smooth",
    help="Apply gaussian filter with this standard deviation to component 1 (default=0).",
    type=float,
    default=0,
    metavar="float",
)
parser.add_argument(
    "--c2-smooth",
    help="Apply gaussian filter with this standard deviation to component 2 (default=2000).",
    type=float,
    default=2000,
    metavar="float",
)
parser.add_argument(
    "--fraction-in-peaks",
    help="Fraction of reads that are expected to be in peaks (default=0.5).",
    type=float,
    default=0.5,
    metavar="float",
)
parser.add_argument(
    "--fraction-overlap",
    help="""If more than this fraction of a peak overlaps with a peak of the
    other target than it is considered an overlapping region (default=0.5).""",
    type=float,
    default=0.5,
    metavar="float",
)
parser.add_argument(
    "--uncorrected",
    help="Do not correct cut ratio estimate with Bayesian prior.",
    action="store_true",
)
parser.add_argument(
    "--span",
    help="Resolution in number of base pairs (default=10).",
    type=int,
    default=10,
    metavar="int",
)
parser.add_argument(
    "--no-progress", help="Do not show progress.", action="store_true",
)
parser.add_argument(
    "--cores", help="Number of CPUs to use.", type=int, default=0, metavar="int",
)


def read_bed(file_path):
    header = {0: "seqname", 1: "start", 2: "end"}
    bed_content = pd.read_csv(file_path, delimiter="\t", header=None).iloc[:, :3]
    bed_content.columns = ["seqname", "start", "end"]
    return bed_content


def write_bed(bed_df, out_path):
    bed_df.to_csv(out_path, sep="\t", header=False, index=False)


def envents_from_intervals(interval_df):
    id_vars = set(interval_df.columns) - {"start", "end"}
    df = interval_df.melt(id_vars=id_vars, value_name="location")
    return df


def name_peaks(df, prefix):
    name_dat = pd.DataFrame(
        list(df.index.str.split("_")),
        columns=["seqname", "interval", "is_peak"],
        index=df.index,
    )
    df["name"] = (
        str(prefix) + "_"
        + name_dat["seqname"] + "_"
        + name_dat["interval"]
    )
    return df


def events_dict_from_file(c1_file, c2_file):
    df_c1 = read_bed(c1_file)
    df_c2 = read_bed(c2_file)

    events_c1 = envents_from_intervals(df_c1)
    events_c2 = envents_from_intervals(df_c2)

    ebs_c1 = {
        seqname: seq_events for seqname, seq_events in events_c1.groupby("seqname")
    }
    ebs_c2 = {
        seqname: seq_events for seqname, seq_events in events_c2.groupby("seqname")
    }

    return ebs_c1, ebs_c2


def deconv_like_kde(
    ebs_c1,
    ebs_c2,
    c1_sigma=None,
    c2_sigma=2000,
    step=10,
    kde_bw=200,
    blacklisted=list(),
):

    sequences = set(ebs_c1.keys()).union(ebs_c2.keys()) - set(blacklisted)
    result_dfs = list()
    ntotal = len(sequences)
    signal_c1_list_global = list()
    signal_c2_list_global = list()

    for i, seqname in enumerate(sequences):
        logger.info(f"Making KDE of {seqname} [{i+1}/{ntotal}].")

        events_c1 = ebs_c1.get(seqname, pd.DataFrame())
        events_c2 = ebs_c2.get(seqname, pd.DataFrame())

        events_c1["true_origin"] = "c1"
        events_c2["true_origin"] = "c2"

        cuts_c1 = events_c1["location"].values
        cuts_c2 = events_c2["location"].values

        all_cuts = np.concatenate([cuts_c1, cuts_c2])
        grid = spr.full_kde_grid(all_cuts)
        cut_idx = all_cuts - grid.min()

        _, density_c1 = spr.get_kde(cuts_c1, kde_bw=kde_bw, grid=grid)
        density_c1 *= len(events_c1)
        _, density_c2 = spr.get_kde(cuts_c2, kde_bw=kde_bw, grid=grid)
        density_c2 *= len(events_c2)

        comb_df = pd.DataFrame(
            {
                "seqname": seqname,
                "interval": seqname,
                "location": grid[::step],
                "c1": density_c1[::step],
                "c2": density_c2[::step],
            }
        )

        if c1_sigma:
            comb_df["c1 smooth"] = scipy.ndimage.gaussian_filter1d(
                comb_df["c1"], int(c1_sigma / step)
            )
        if c2_sigma:
            comb_df["c2 smooth"] = scipy.ndimage.gaussian_filter1d(
                comb_df["c2"], int(c2_sigma / step)
            )

        signal_c1_list_global.append(density_c1[cut_idx])
        signal_c2_list_global.append(density_c2[cut_idx])
        result_dfs.append(comb_df)
    comb_data = pd.concat(result_dfs, axis=0)

    return comb_data, signal_c1_list_global, signal_c2_list_global


def main():
    args = parser.parse_args()
    setup_logging(args.logLevel, args.logfile)
    logger.debug("Loglevel is on DEBUG.")
    cores = detect_cores(args.cores)
    logger.info("Limiting computation to %i cores.", cores)
    threadpool_limits(limits=int(cores))
    os.environ["NUMEXPR_MAX_THREADS"] = str(cores)

    logger.info("Reading %s and %s", args.c1_bed_file, args.c2_bed_file)
    ebs_c1, ebs_c2 = events_dict_from_file(args.c1_bed_file, args.c2_bed_file)

    comb_data, signal_c1_list_global, signal_c2_list_global = deconv_like_kde(
        ebs_c1,
        ebs_c2,
        c1_sigma=args.c1_smooth,
        c2_sigma=args.c2_smooth,
        step=args.span,
        kde_bw=args.kde_bw,
        blacklisted=args.blacklisted_seqs,
    )

    logger.info("Calling peaks.")
    peaks_c1, peaks_c2, overlaps = sep241peakcalling.call_peaks(
        comb_data,
        signal_c1_list_global,
        signal_c2_list_global,
        work_coverage=1.0,
        fraction_in_peaks=args.fraction_in_peaks,
        fraction_overlap=args.fraction_overlap,
        c1_min_peak_size=args.c1_min_peak_size,
        c2_min_peak_size=args.c2_min_peak_size,
        uncorrected=args.uncorrected,
        span=args.span,
        cores=cores,
        progress=not args.no_progress,
    )

    bed_c1 = sep241peakcalling.inlude_auc(name_peaks(peaks_c1, "c1"))
    bed_c2 = sep241peakcalling.inlude_auc(name_peaks(peaks_c2, "c2"))

    out_path = args.out
    try:
        os.mkdir(out_path)
    except FileExistsError:
        pass

    out_base = os.path.join(out_path, "peaks_")
    logger.info("Writing results to %s...", out_base)
    sep241peakcalling.write_peaks(bed_c1, bed_c2, overlaps, out_path)
    logger.info("Finished sucesfully.")


if __name__ == "__main__":
    main()
