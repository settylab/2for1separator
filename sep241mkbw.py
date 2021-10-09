#! /usr/bin/env python
import os
import argparse
import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pyBigWig

from dcbackend import logger, setup_logging
from dcbackend import read_job_data, read_results, read_region_string
from dcbackend import check_length_distribution_flip
from dcbackend import MissingData

def parse_args():
    desc = "Prepair CUT&TAG 2for1 deconvolution."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "jobdata",
        metavar="jobdata-file",
        type=str,
        nargs="?",
        help="Jobdata with cuts per intervall and workchunk ids.",
    )
    parser.add_argument(
        "chrom_sizes_file",
        help="""Chromosome sizes file with the two columns seqname and size.
        You can use a script like
        https://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/fetchChromSizes
        to fetch the file.
        """,
        type=str,
        nargs="?",
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
        "-o",
        "--out",
        help="Output directory (default is the path of the jobdata).",
        type=str,
        metavar="dir",
    )
    parser.add_argument(
        "--unit",
        help="Output will have the unit `cuts per {unit} base pairs` (default=100).",
        type=int,
        default=100,
        metavar="int",
    )
    parser.add_argument(
        "--region",
        help="Limit output to this genomic region.",
        type=str,
        metavar="chr:start-end",
    )
    parser.add_argument(
        "--span",
        help="The span of base pairs with the same value in the ouptut bigwig (default=10).",
        type=int,
        default=10,
        metavar="int",
    )
    parser.add_argument(
        "--no-check",
        help="Do not test for flipped length distributions.",
        action="store_true",
    )
    parser.add_argument(
        "--no-progress",
        help="Do not show progress.",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        help="Make bigwigs even if some results are missing.",
        action="store_true",
    )
    parser.add_argument(
        "--cores",
        help="Number of CPUs to use for the preparation.",
        type=int,
        default=0,
        metavar="int",
    )
    return parser.parse_args()


def generate_entry(
    seqname,
    locations,
    signal_c1,
    signal_c2,
    bw_c1,
    bw_c2,
    span=10,
    unit=100,
    seq_size=np.inf,
):
    span_multiple = np.max(locations) - np.min(locations)
    span_multiple = min(span_multiple, seq_size - np.min(locations))
    factor, rest = divmod(span_multiple, span)
    span_multiple -= rest
    grid = np.arange(np.min(locations), np.min(locations) + span_multiple)
    interpolated_c1 = np.interp(grid, locations, signal_c1)
    interpolated_c2 = np.interp(grid, locations, signal_c2)
    if span != 1:
        grid = grid[::span]
        interpolated_c1 = interpolated_c1.reshape(factor, span).mean(axis=1)
        interpolated_c2 = interpolated_c2.reshape(factor, span).mean(axis=1)
    bw_c1.addEntries(
        seqname,
        int(np.min(grid)),
        span=span,
        step=span,
        values=interpolated_c1 * unit,
    )
    bw_c2.addEntries(
        seqname, int(np.min(grid)), span=span, step=span, values=interpolated_c2 * unit
    )


def make_bigwigs(
    workdata,
    map_results,
    sizes_file,
    out_base_filename,
    span=10,
    unit=100,
    max_log_value=50,
    region=None,
    progress=True,
):
    if region:
        sel_seq, sel_start, sel_end = read_region_string(region, short_seqname=False)
        all_seqs = [sel_seq, ]
    else:
        all_seqs = workdata["seqname"].unique()
    chrom_sizes = dict()
    with open(sizes_file, "r") as fl:
        for line in fl:
            seq, size = line.split()
            if seq in all_seqs:
                chrom_sizes[seq] = int(size)

    bw_c1_file = out_base_filename + "_c1.bw"
    bw_c1 = pyBigWig.open(bw_c1_file, "w")
    bw_c1.addHeader(list(sorted(chrom_sizes.items())))

    bw_c2_file = out_base_filename + "_c2.bw"
    bw_c2 = pyBigWig.open(bw_c2_file, "w")
    bw_c2.addHeader(list(sorted(chrom_sizes.items())))

    results_per_chr = dict()
    signal_c1_list = list()
    signal_c2_list = list()
    location_list = list()
    last_name = ""

    for name, dat in tqdm(
        workdata.iterrows(), total=len(workdata), desc="intervals", disable=~progress
    ):
        wg = dat["workchunk"]
        maxlle = map_results.get(dat["workchunk"], None)
        if maxlle is None:
            continue
        if f"f_c1_{name}" not in maxlle.keys():
            continue
        start_idx = dat["stich_start"][0]
        end_idx = dat["stich_end"][0]
        seq = dat["seqname"]

        if region is not None:
            # consider skipping this interval
            if(
                (seq != sel_seq)
                or (dat["end"] < sel_start)
                or (dat["start"] > sel_end)
            ):
                continue

        locations = dat["cuts"]["location"][start_idx:end_idx].values.astype(int)
        idx = dat["cuts"]["location"].rank(method="dense").astype(int) - 1
        log_signal_c1 = maxlle[f"f_c1_{name}"][idx][start_idx:end_idx]
        log_signal_c2 = maxlle[f"f_c2_{name}"][idx][start_idx:end_idx]
        if np.max(log_signal_c1) > max_log_value:
            logger.warn(
                f"The c1 track of intervall {name} in work chunk {wg} contains "
                "values that are too large. The interval will be skipped."
            )
            continue
        if np.max(log_signal_c2) > max_log_value:
            logger.warn(
                f"The c2 track of intervall {name} in work chunk {wg} contains "
                "values that are too large. The interval will be skipped."
            )
            continue
        signal_c1 = np.exp(log_signal_c1)
        signal_c2 = np.exp(log_signal_c2)
        base_name, _ = name.split(".")
        if location_list and last_name != base_name:
            generate_entry(
                last_seq,
                np.hstack(location_list),
                np.hstack(signal_c1_list),
                np.hstack(signal_c2_list),
                bw_c1,
                bw_c2,
                span=span,
                unit=unit,
                seq_size=chrom_sizes[last_seq],
            )
            location_list = list()
            signal_c1_list = list()
            signal_c2_list = list()
        if dat["is_subdivide"]:
            location_list.append(locations)
            signal_c1_list.append(signal_c1)
            signal_c2_list.append(signal_c2)
            last_seq = seq
            last_name = base_name
        else:
            generate_entry(
                seq,
                locations,
                signal_c1,
                signal_c2,
                bw_c1,
                bw_c2,
                span=span,
                unit=unit,
                seq_size=chrom_sizes[seq],
            )

    logger.info("Closing bigwig files %s and %s.", bw_c1_file, bw_c2_file)
    bw_c1.close()
    bw_c2.close()


def main():
    args = parse_args()
    setup_logging(args.logLevel, args.logfile)

    logger.info("Reading jobdata.")
    workdata = read_job_data(args.jobdata)
    logger.info("Reading deconvolution results.")
    try:
        map_results = read_results(args.jobdata, workdata, progress=~args.no_progress, error=~args.force)
    except MissingData:
        raise MissingData(
            "Results are missing. Complete missing work chunks or pass --force to "
            "prodce bigwig files regardless."
        )

    if not args.no_check:
        logger.info("Checking length distribution flip.")
        check_length_distribution_flip(workdata, map_results)

    if args.out is None:
        out_path, old_file = os.path.split(args.jobdata)
    else:
        out_path = args.out
    out_base = os.path.join(out_path, f"deconv_cuts-per-{args.unit}bp")
    logger.info("Writing results to %s", out_base)
    make_bigwigs(
        workdata,
        map_results,
        args.chrom_sizes_file,
        out_base,
        span=args.span,
        unit=args.unit,
        max_log_value=50,
        region=args.region,
        progress=~args.no_progress,
    )
    logger.info("Finished sucesfully.")


if __name__ == "__main__":
    main()
