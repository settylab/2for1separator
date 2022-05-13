#! /usr/bin/env python
import os
import argparse
import pickle
import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import pymc3 as pm

from sep241util import logger, setup_logging
from sep241util import read_job_data, read_results, write_bed
from sep241util import MissingData
from sep241util import posterior_mode_weights
from sep241util import check_length_distribution_flip
from sep241model import get_length_dist_modes


desc = """Export deconvolution results for each individual event in the data.
Each end of a fragment is considered an event. The output format is one
bed-like tab-separated text table per channel: ``c1.bed`` and ``c2.bed``.
The columns of the two tables correspond to:

    chrom, chromStart, chromEnd, name, likelihood, location-marginal likelihood

Where *likelihood* corresponds to the likelihood of the event to originate
from the respective target and *location marginal likelihood* ignores the
length of the associated fragment and only uses the genomic-location-specific
target likelihood. The one-dimensional
genomic-location-specific target likelihood can also be exported with
sep241mkbw.
"""
parser = argparse.ArgumentParser(description=desc)
parser.add_argument(
    "jobdata",
    metavar="jobdata-file",
    type=str,
    nargs="?",
    help="Jobdata with cuts per interval and workchunk ids.",
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
    metavar="out_dir",
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
    "--exclude-flipped",
    help="Exclude results from workchunks with flipped length distribution.",
    action="store_true",
)


def likelihoods_per_cut(workdata, map_results, progress=True):
    df_list = list()
    last_name = ""

    max_log_value = None
    missing_data_wg = set()

    for name, dat in tqdm(
        workdata.iterrows(), total=len(workdata), desc="intervals", disable=not progress
    ):
        wg = dat["workchunk"]
        maxlle = map_results.get(dat["workchunk"], None)
        if maxlle is None:
            if wg not in missing_data_wg:
                logger.warning("Data of workchunk %d is None.", wg)
                missing_data_wg.add(wg)
            continue
        if f"f_c1_{name}" not in maxlle.keys():
            logger.warning("Marginal likelihood results not found in workchunk %d.", wg)
            continue
        if max_log_value is None:
            max_log_value = np.log(np.finfo(maxlle[f"f_c1_{name}"].dtype).max)
        start_idx = dat["stich_start"][0]
        end_idx = dat["stich_end"][0]
        seq = dat["seqname"]

        df = dat["cuts"].iloc[start_idx:end_idx, :].copy()
        locations = df["location"].values.astype(int)

        idx = dat["cuts"]["location"].rank(method="dense").astype(int) - 1

        df["log_location_marginal_c1"] = maxlle[f"f_c1_{name}"][idx][start_idx:end_idx]
        df["log_location_marginal_c2"] = maxlle[f"f_c2_{name}"][idx][start_idx:end_idx]

        df["workchunk"] = wg
        df["name"] = name
        df_list.append(df)

    return pd.concat(df_list, axis=0)


def main():
    args = parser.parse_args()
    logging.getLogger("filelock").setLevel(logging.ERROR) # aesara produces a lot
    setup_logging(args.logLevel, args.logfile)
    logger.debug("Loglevel is on DEBUG.")

    logger.info("Reading jobdata.")
    workdata = read_job_data(args.jobdata)

    logger.info("Reading deconvolution results.")
    try:
        map_results = read_results(
            args.jobdata, workdata, progress=not args.no_progress, error=not args.force
        )
    except MissingData:
        raise MissingData(
            "Results are missing. Complete missing work chunks or pass --force to "
            "call peaks regardless."
        )
    if not args.no_check:
        bad_wgs = check_length_distribution_flip(workdata, map_results)
        if args.exclude_flipped:
            logger.info("Removing flipped workgroups from results.")
            for idx in bad_wgs:
                del map_results[idx]

    logger.info("Getting location marginal likelihoods.")
    events = likelihoods_per_cut(workdata, map_results, progress=not args.no_progress)

    logger.info("Producing posterior length marginal likelihood distributions.")
    dc_args = next(iter(map_results.values()))["arguments"]
    length_comps = get_length_dist_modes(
        dc_args.length_dist_modes, dc_args.length_dist_mode_sds
    )()
    w_c1_posterior, w_c2_posterior = posterior_mode_weights(workdata, map_results)

    logger.info("Computing length marginal likelihoods for component 1.")
    mdist_c1 = pm.Mixture.dist(
        w=w_c1_posterior / np.sum(w_c1_posterior), comp_dists=length_comps
    )
    events["log_length_marginal_c1"] = mdist_c1.logp(events["length"]).eval()

    logger.info("Computing length marginal likelihoods for component 2.")
    mdist_c2 = pm.Mixture.dist(
        w=w_c2_posterior / np.sum(w_c2_posterior), comp_dists=length_comps
    )
    events["log_length_marginal_c2"] = mdist_c2.logp(events["length"]).eval()

    logger.info("Postprocessing.")
    events["likelihood_c1"] = np.exp(
        events["log_length_marginal_c1"] + events["log_location_marginal_c1"]
    )
    events["likelihood_c2"] = np.exp(
        events["log_length_marginal_c2"] + events["log_location_marginal_c2"]
    )
    events["marginal_c1"] = np.exp(events["log_location_marginal_c1"])
    events["marginal_c2"] = np.exp(events["log_location_marginal_c2"])
    events["bed_end"] = events["location"] + 1

    if args.out is None:
        out_path, old_file = os.path.split(args.jobdata)
    else:
        out_path = args.out
        try:
            os.mkdir(out_path)
        except FileExistsError:
            pass

    out_base = os.path.join(out_path, "events_deconv_")

    out_c1 = out_base + "c1.bed"
    columns = ["seqname", "location", "bed_end", "barcode"]
    if "barcode" not in events.columns:
        logger.info("Generating event names.")
        n = len(events)
        nchars = int(np.ceil(np.log10(n)))
        event_numbers = pd.Series(range(n)).astype(str).str.zfill(nchars)
        events["barcode"] = "event_" + event_numbers.values
    logger.info("Writing results to %s", out_c1)
    write_bed(
        events[columns + ["likelihood_c1", "marginal_c1"]], out_c1,
    )
    out_c2 = out_base + "c2.bed"
    logger.info("Writing results to %s", out_c2)
    write_bed(
        events[columns + ["likelihood_c2", "marginal_c2"]], out_c2,
    )
    logger.info("Finished sucesfully.")


if __name__ == "__main__":
    main()
