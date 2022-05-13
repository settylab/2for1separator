#! /usr/bin/env python
import os
import argparse
import pickle

import numpy as np
import scipy
import pandas as pd
from tqdm.auto import tqdm
from threadpoolctl import threadpool_limits
import multiprocessing

from sep241util import logger, setup_logging
from sep241util import read_job_data, read_results, write_bed, detect_cores
from sep241util import check_length_distribution_flip
from sep241util import MissingData


desc = """Peak calling after CUT&TAG 2for1 deconvolution.
We use the positional marginal likelihoods to originate from one of the two
channels to identify peaks of high cut likelihood for the respective channels
according to
`CUT&Tag2for1: a modified method
for simultaneous profiling of the accessible and silenced regulome in single
cells <https://doi.org/10.1186/s13059-022-02642-w>`_.

By default, channel 1 is assumed to contain narrow, sharp peaks. Channel 2
is assumed to contain broad domains of a more gradually changing signal. The minimum
peak size can be controlled by ``--c1-min-peak-size`` and
``--c2-min-peak-size``. Additionally, signals can be
smoothed through ``--c1-smooth`` and ``--c2-smooth`` before peak calling
to reflect a more gradual change.
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
    help="Output directory (default is the directory of the jobdata input file).",
    type=str,
    metavar="out_dir",
)
parser.add_argument(
    "--c1-min-peak-size",
    help="Minimal number of bases per peak for channel 1 (default=100).",
    type=int,
    default=100,
    metavar="int",
)
parser.add_argument(
    "--c2-min-peak-size",
    help="Minimal number of bases per peak for channel 2 (default=400).",
    type=int,
    default=400,
    metavar="int",
)
parser.add_argument(
    "--c1-smooth",
    help="Apply gaussian filter with this standard deviation to channel 1 (default=0).",
    type=float,
    default=0,
    metavar="float",
)
parser.add_argument(
    "--c2-smooth",
    help="Apply gaussian filter with this standard deviation to channel 2 (default=2000).",
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
    other target, then it is considered an overlapping region (default=0.5).""",
    type=float,
    default=0.5,
    metavar="float",
)
parser.add_argument(
    "--span",
    help="Resolution in number of base pairs (default=10).",
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
    "--only-check", 
    help="Stop after initial checking.",
    action="store_true",
)
parser.add_argument(
    "--uncorrected",
    help="Do not correct cut ratio estimate with Bayesian prior.",
    action="store_true",
)
parser.add_argument(
    "--force",
    help="Call peaks even if some results are missing.",
    action="store_true",
)
parser.add_argument(
    "--no-progress", help="Do not show progress.", action="store_true",
)
parser.add_argument(
    "--cores",
    help="Number of CPUs to use for the preparation.",
    type=int,
    default=0,
    metavar="int",
) 


def interpolate_entry(job):
    (
        interval_name,
        seqname,
        locations,
        signal_c1,
        c1_sigma,
        signal_c2,
        c2_sigma,
        step,
    ) = job
    grid = np.arange(np.min(locations), np.max(locations) + 1)
    interpolated_c1 = np.interp(grid, locations, signal_c1)[::step]
    interpolated_c2 = np.interp(grid, locations, signal_c2)[::step]
    df = pd.DataFrame(
        {
            "interval": interval_name,
            "seqname": seqname,
            "location": grid[::step],
            "c1": interpolated_c1,
            "c2": interpolated_c2,
        }
    )
    if c1_sigma:
        df["c1 smooth"] = scipy.ndimage.gaussian_filter1d(
            interpolated_c1, int(c1_sigma / step)
        )
    if c2_sigma:
        df["c2 smooth"] = scipy.ndimage.gaussian_filter1d(
            interpolated_c2, int(c2_sigma / step)
        )
    return df


def make_interpolation_jobs(workdata, map_results, c1_sigma, c2_sigma, step_size):
    jobs = list()
    signal_c1_list = list()
    signal_c2_list = list()
    signal_c1_list_global = list()
    signal_c2_list_global = list()
    location_list = list()
    last_name = ""

    max_log_value = None
    missing_data_wg = set()

    for name, dat in tqdm(workdata.iterrows(), total=len(workdata), desc="intervals"):
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
        locations = dat["cuts"]["location"][start_idx:end_idx].values.astype(int)

        idx = dat["cuts"]["location"].rank(method="dense").astype(int) - 1

        log_signal_c1 = maxlle[f"f_c1_{name}"][idx][start_idx:end_idx]
        log_signal_c2 = maxlle[f"f_c2_{name}"][idx][start_idx:end_idx]
        if np.max(log_signal_c1) > max_log_value:
            logger.warning(
                f"The c1 track of interval {name} in work chunk {wg} contains "
                "values that are too large. The interval will be skipped."
            )
            continue
        if np.max(log_signal_c2) > max_log_value:
            logger.warning(
                f"The c2 track of interval {name} in work chunk {wg} contains "
                "values that are too large. The interval will be skipped."
            )
            continue
        signal_c1 = np.exp(log_signal_c1)
        signal_c2 = np.exp(log_signal_c2)

        signal_c1_list_global.append(signal_c1)
        signal_c2_list_global.append(signal_c2)
        base_name, _ = name.split(".")
        if location_list and last_name != base_name:
            jobs.append(
                (
                    last_name,
                    last_seq,
                    np.hstack(location_list),
                    np.hstack(signal_c1_list),
                    c1_sigma,
                    np.hstack(signal_c2_list),
                    c2_sigma,
                    step_size,
                )
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
            jobs.append(
                (
                    base_name,
                    seq,
                    locations,
                    signal_c1,
                    c1_sigma,
                    signal_c2,
                    c2_sigma,
                    step_size,
                )
            )
    return jobs, signal_c1_list_global, signal_c2_list_global


def interpolate(
    workdata, map_results, c1_sigma, c2_sigma, step_size, cores=8, progress=False
):
    logger.info("Splitting data into intervals.")
    jobs, signal_c1_list_global, signal_c2_list_global = make_interpolation_jobs(
        workdata, map_results, c1_sigma, c2_sigma, step_size
    )
    results_list = list()
    if c1_sigma or c2_sigma:
        if not c1_sigma:
            smooth_msg = "c2"
        elif not c1_sigma:
            smooth_msg = "c1"
        else:
            smooth_msg = "c1 and c2"
    else:
        smooth_msg = "none"
    logger.info("Interpolating + smoothening %s.", smooth_msg)
    n_jobs = len(jobs)
    mutliprocessing_chunksize = min(10, int(np.ceil(n_jobs/cores)))
    with threadpool_limits(limits=1):
        with multiprocessing.Pool(cores) as pool:
            for df in tqdm(
                pool.imap(interpolate_entry, jobs, mutliprocessing_chunksize),
                total=n_jobs,
                disable=not progress,
                desc="intervals",
            ):
                results_list.append(df)
    comb_data = pd.concat(results_list, axis=0)
    comb_data.index = range(len(comb_data))
    return comb_data, signal_c1_list_global, signal_c2_list_global


def target_fractions(comb_data, fraction_in_peaks, uncorrected=False):
    total_c1 = comb_data["c1"].sum()
    total_c2 = comb_data["c2"].sum()
    total = total_c1 + total_c2
    fraction_c1_uncorrected = total_c1 / total
    fraction_c2_uncorrected = total_c2 / total
    logger.info(
        f"Integral ratio: {fraction_c1_uncorrected:.1%} c1 and {fraction_c2_uncorrected:.1%} c2 cuts."
    )
    if uncorrected:
        fraction_c1 = fraction_in_peaks * total_c1 / total
        fraction_c2 = fraction_in_peaks * total_c2 / total

    equal_dist_cuts = total
    total_c1 = comb_data["c1"].sum() + (equal_dist_cuts / 2)
    total_c2 = comb_data["c2"].sum() + (equal_dist_cuts / 2)
    total = total_c1 + total_c2
    fraction_c1_permissive = total_c1 / total
    fraction_c2_permissive = total_c2 / total
    logger.info(
        f"Bayesian estimation: {fraction_c1_permissive:.1%} c1 and {fraction_c2_permissive:.1%} c2 cuts."
    )

    if uncorrected:
        logger.info(
            f"Estimating an uncorrected total of {fraction_c1:.1%} c1 and {fraction_c2:.1%} c2 cuts in peaks."
        )
        return (
            fraction_c1,
            fraction_c2,
            fraction_c1_uncorrected,
            fraction_c2_uncorrected,
        )

    fraction_c1 = fraction_in_peaks * total_c1 / total
    fraction_c2 = fraction_in_peaks * total_c2 / total
    logger.info(
        f"Estimating a total of {fraction_c1:.1%} c1 and {fraction_c2:.1%} c2 cuts in peaks."
    )
    return fraction_c1, fraction_c2, fraction_c1_permissive, fraction_c2_permissive


def estimated_deconvolved_fraction(workdata):
    total = 0
    covered = 0
    for seqname, data in workdata.groupby("seqname"):
        total += np.max(data["end"])
        covered += np.sum(data["end"] - data["start"])
    return covered / total


def mark_peaks(
    comb_data,
    signal_c1_list_global,
    signal_c2_list_global,
    fraction_c1,
    fraction_c2,
    work_coverage=None,
):

    bound_c1 = np.quantile(np.hstack(signal_c1_list_global), 1 - fraction_c1)
    if "c1 smooth" in comb_data.keys():
        logger.info("Averaging original and smoothed signal of c1.")
        comb_data["c1 mean"] = np.mean(
            comb_data.loc[:, ["c1", "c1 smooth"]].values, axis=1
        )
        comb_data["c1_peak"] = (comb_data["c1 mean"] > bound_c1) | (
            comb_data["c1 smooth"] > bound_c1
        )
    else:
        comb_data["c1_peak"] = comb_data["c1"] > bound_c1
    fraction_selected = comb_data["c1_peak"].sum() / len(comb_data["c1_peak"])
    msg = f"c1: {fraction_selected:.2%} of intervals"
    if work_coverage:
        msg += f" and {work_coverage*fraction_selected:.2%} of genome selected."
    logger.info(msg)

    bound_c2 = np.quantile(np.hstack(signal_c2_list_global), 1 - fraction_c2)
    if "c2 smooth" in comb_data.keys():
        logger.info("Averaging original and smoothed signal of c2.")
        comb_data["c2 mean"] = np.mean(
            comb_data.loc[:, ["c2", "c2 smooth"]].values, axis=1
        )
        comb_data["c2_peak"] = (comb_data["c2 mean"] > bound_c2) | (
            comb_data["c2 smooth"] > bound_c2
        )
    else:
        comb_data["c2_peak"] = comb_data["c2"] > bound_c2
    fraction_selected = comb_data["c2_peak"].sum() / len(comb_data["c2_peak"])
    msg = f"c2: {fraction_selected:.2%} of intervals"
    if work_coverage:
        msg += f" and {work_coverage*fraction_selected:.2%} of genome selected."
    logger.info(msg)
    return comb_data


def track_to_interval(loc_comb_data, comp_name, step_size, seqname):
    idx = np.insert(loc_comb_data[f"{comp_name}_peak"].values.astype(int), 0, 0)
    indicator = np.diff(idx)
    loc_comb_data[f"{comp_name}_peak_number"] = np.cumsum(indicator == 1)
    loc_comb_data[f"{comp_name}_peak_name"] = (
        loc_comb_data["interval"]
        + "_"
        + loc_comb_data[f"{comp_name}_peak_number"].astype(str)
        + "_"
        + loc_comb_data[f"{comp_name}_peak"].astype(str)
    )

    peak_location_df = loc_comb_data[
        loc_comb_data[f"{comp_name}_peak"].values
    ].set_index("location")
    locations = peak_location_df.groupby(f"{comp_name}_peak_name")[comp_name].idxmax()
    peak_groups = peak_location_df.reset_index().groupby(f"{comp_name}_peak_name")
    start = peak_groups["location"].min().values - (step_size / 2)
    end = peak_groups["location"].max().values + (step_size / 2)
    means = peak_groups[comp_name].mean()
    peaks = pd.DataFrame(
        {
            "seqname": seqname,
            "start": start.astype(int),
            "end": end.astype(int),
            "mean": means,
            "summit": locations,
        }
    ).sort_values("start")
    return peaks


def both_tracks_to_intervlas(job):
    loc_comb_data, step_size, seqname = job
    peaks_c1 = track_to_interval(loc_comb_data, "c1", step_size, seqname)
    peaks_c2 = track_to_interval(loc_comb_data, "c2", step_size, seqname)
    return peaks_c1, peaks_c2


def tracks_to_intervals(comb_data, step_size, cores=8, progress=True):
    peaks_list_c1 = list()
    peaks_list_c2 = list()
    jobs = [
        (loc_comb_data, step_size, seqname)
        for seqname, loc_comb_data in comb_data.groupby("seqname")
    ]
    lcores = min(cores, len(jobs))
    logger.debug("Using %d cores for conversion.", lcores)
    with threadpool_limits(limits=1):
        with multiprocessing.Pool(lcores) as pool:
            for peaks_c1, peaks_c2 in tqdm(
                pool.imap(both_tracks_to_intervlas, jobs),
                total=len(jobs),
                disable=not progress,
                desc="sequence",
            ):
                peaks_list_c1.append(peaks_c1)
                peaks_list_c2.append(peaks_c2)
    peaks_c1 = pd.concat(peaks_list_c1, axis=0)
    peaks_c1["length"] = peaks_c1["end"] - peaks_c1["start"]
    peaks_c2 = pd.concat(peaks_list_c2, axis=0)
    peaks_c2["length"] = peaks_c2["end"] - peaks_c2["start"]
    return peaks_c1, peaks_c2


def make_overlap_report(
    comb_data,
    peaks_c1,
    peaks_c2,
    fraction_c1_permissive,
    fraction_c2_permissive,
    progress=True,
):
    sel_comb_data = comb_data[comb_data["c1_peak"] | comb_data["c2_peak"]]
    df_entries = list()
    df_header = [
        "seqname",
        "start",
        "end",
        "length",
        "peak_c1",
        "peak_c2",
        "frac_c1",
        "frac_c2",
        "mean_c1",
        "mean_c2",
        "cosine_distance",
    ]
    for seqname, loc_comb_data in tqdm(
        sel_comb_data.groupby("seqname"), disable=not progress, desc="sequence"
    ):
        loc_peaks_c1 = peaks_c1[peaks_c1["seqname"] == seqname].sort_values("start")
        loc_peaks_c2 = peaks_c2[peaks_c2["seqname"] == seqname].sort_values("start")
        max_pos_p = len(loc_peaks_c1)
        max_pos_k = len(loc_peaks_c2)
        if max_pos_p == 0:
            logger.info(f"No c1 peaks in sequence {seqname}.")
            continue
        if max_pos_k == 0:
            logger.info(f"No c2 peaks in sequence {seqname}.")
            continue
        rows_p = loc_peaks_c1.iterrows()
        name_p, dat_p = next(rows_p)
        rows_k = loc_peaks_c2.iterrows()
        name_k, dat_k = next(rows_k)
        rows_values = loc_comb_data.iterrows()
        _, values = next(rows_values)
        try:
            for _ in range(max_pos_p * max_pos_k):
                if dat_p["start"] > dat_k["end"]:
                    name_k, dat_k = next(rows_k)
                    continue
                if dat_k["start"] < dat_p["end"]:
                    # overlap
                    start = max(dat_k["start"], dat_p["start"])
                    end = min(dat_k["end"], dat_p["end"])
                    while values["location"] < start:
                        # move up towards start of overlap
                        _, values = next(rows_values)
                    values_p = list()
                    values_k = list()
                    while values["location"] < end:
                        # move up towards end of overlap
                        values_p.append(values["c1"])
                        values_k.append(values["c2"])
                        _, values = next(rows_values)
                    if len(values_p) > 1:
                        length = end - start
                        values_p = np.array(values_p)
                        values_k = np.array(values_k)
                        mean_p = np.mean(values_p)
                        mean_k = np.mean(values_k)
                        frac_p = length / (dat_p["end"] - dat_p["start"])
                        frac_k = length / (dat_k["end"] - dat_k["start"])
                        cosine_distance = scipy.spatial.distance.cosine(
                            values_p - mean_p, values_k - mean_k
                        )
                        df_entries.append(
                            (
                                seqname,
                                start,
                                end,
                                length,
                                name_p,
                                name_k,
                                frac_p,
                                frac_k,
                                mean_p,
                                mean_k,
                                cosine_distance,
                            )
                        )
                name_p, dat_p = next(rows_p)
        except StopIteration:
            pass
    overlap_df = pd.DataFrame(df_entries, columns=df_header)
    overlap_df["log_ratio"] = np.log(overlap_df["mean_c1"]) - np.log(
        overlap_df["mean_c2"]
    )
    overlap_df["log_ratio_corrected"] = (
        overlap_df["log_ratio"]
        - np.log(fraction_c1_permissive)
        + np.log(fraction_c2_permissive)
    )
    overlap_df["callability"] = (
        np.abs(overlap_df["log_ratio_corrected"]) * overlap_df["cosine_distance"]
    )
    overlap_df["ambiguous"] = overlap_df["callability"] < 1
    return overlap_df


def merge_overlapping_intervals(interval_df):
    merged_df_list = list()
    for seqname, data in interval_df.groupby("seqname"):
        data = data.sort_values("start")
        last_stop = None
        new_starts = list()
        new_stops = list()
        new_means = list()
        for _, d in data.iterrows():
            start = d["start"]
            stop = d["end"]
            if last_stop is None:  # init
                new_starts.append(start)
                means = [d["mean"]]
                last_stop = stop
                continue
            means.append(d["mean"])
            if start < last_stop:  # combine
                last_stop = max(last_stop, stop)
                continue
            # finish interval
            new_stops.append(last_stop)
            new_means.append(np.mean(means))
            # start new interval
            new_starts.append(start)
            last_stop = stop
        new_stops.append(last_stop)
        new_means.append(np.mean(means))
        merged_df_list.append(
            pd.DataFrame(
                {
                    "seqname": seqname,
                    "start": np.array(new_starts),
                    "end": np.array(new_stops),
                    "mean": np.array(new_means),
                    "name": [f"{seqname}_{i}" for i in range(len(new_starts))],
                }
            )
        )
    return pd.concat(merged_df_list, axis=0)


def fiter_overlaps(peaks_c1, peaks_c2, overlap_df, threshold=0.5):
    fractions_c1 = overlap_df.groupby("peak_c1", sort=False)["frac_c1"].sum()
    fractions_c2 = overlap_df.groupby("peak_c2", sort=False)["frac_c2"].sum()
    bad_idx_c1 = fractions_c1.index[fractions_c1 > threshold]
    overlapping_c1 = peaks_c1.loc[bad_idx_c1, :]
    bad_idx_c2 = fractions_c2.index[fractions_c2 > threshold]
    overlapping_c2 = peaks_c2.loc[bad_idx_c2, :]
    overlap_df = merge_overlapping_intervals(
        pd.concat([overlapping_c1, overlapping_c2])
    )

    peaks_c1 = peaks_c1.loc[peaks_c1.index.difference(bad_idx_c1), :].sort_values(
        ["seqname", "start"]
    )
    peaks_c2 = peaks_c2.loc[peaks_c2.index.difference(bad_idx_c2), :].sort_values(
        ["seqname", "start"]
    )

    return peaks_c1, peaks_c2, overlap_df


def name_peaks(df, prefix):
    name_dat = pd.DataFrame(
        list(df.index.str.split("_")),
        columns=["seqname", "interval", "subinterval", "is_peak"],
        index=df.index,
    )
    df["name"] = (
        str(prefix) + "_"
        + name_dat["seqname"] + "_"
        + name_dat["interval"] + "_"
        + name_dat["subinterval"]
    )
    return df


def inlude_auc(df):
    df["auc"] = df["mean"] * df["length"]
    return df


def bed_to_summit_bed(df):
    summits = df["summit"].astype(int)
    summit_df = pd.DataFrame(
        {
            "seqname": df["seqname"],
            "start": summits,
            "end": summits + 1,
            "name": df["name"],
            "signal": df["mean"],
        }
    )
    return summit_df


def call_peaks(
    comb_data,
    signal_c1_list_global,
    signal_c2_list_global,
    work_coverage=1.0,
    fraction_in_peaks=0.5,
    fraction_overlap=0.5,
    uncorrected=False,
    c1_min_peak_size=100,
    c2_min_peak_size=400,
    span=10,
    cores=16,
    progress=False,
):
    (
        fraction_c1,
        fraction_c2,
        fraction_c1_permissive,
        fraction_c2_permissive,
    ) = target_fractions(comb_data, fraction_in_peaks, uncorrected=uncorrected)
    logger.info("Selecting peak candidates.")
    comb_data = mark_peaks(
        comb_data,
        signal_c1_list_global,
        signal_c2_list_global,
        fraction_c1,
        fraction_c2,
        work_coverage,
    )
    logger.info("Converting peak signal to intervals.")
    peaks_c1, peaks_c2 = tracks_to_intervals(comb_data, span, cores, progress=progress)
    peaks_c1 = peaks_c1[peaks_c1["length"] > c1_min_peak_size]
    peaks_c2 = peaks_c2[peaks_c2["length"] > c2_min_peak_size]

    logger.info("Analyzing overlaps.")
    overlap_df = make_overlap_report(
        comb_data,
        peaks_c1,
        peaks_c2,
        fraction_c1_permissive,
        fraction_c2_permissive,
        progress=progress,
    )

    logger.info("Filtering overlaps.")
    peaks_c1, peaks_c2, overlaps = fiter_overlaps(
        peaks_c1, peaks_c2, overlap_df, threshold=fraction_overlap
    )
    logger.info("Peaks for c1: %d", len(peaks_c1))
    logger.info("Peaks for c2: %d", len(peaks_c2))
    return peaks_c1, peaks_c2, overlaps


def write_peaks(bed_c1, bed_c2, overlaps, out_path):

    out_base = os.path.join(out_path, "peaks_")

    write_bed(
        bed_c1[["seqname", "start", "end", "name", "auc"]], out_base + "deconv_c1.bed"
    )
    write_bed(
        bed_c2[["seqname", "start", "end", "name", "auc"]], out_base + "deconv_c2.bed"
    )

    write_bed(bed_to_summit_bed(bed_c1), out_base + "deconv_c1.summits.bed")
    write_bed(bed_to_summit_bed(bed_c2), out_base + "deconv_c2.summits.bed")

    write_bed(
        overlaps[["seqname", "start", "end", "name", "mean"]], out_base + "overlap.bed",
    )
    return


def main():
    args = parser.parse_args()
    setup_logging(args.logLevel, args.logfile)
    logger.debug("Loglevel is on DEBUG.")
    cores = detect_cores(args.cores)
    logger.info("Limiting computation to %i cores.", cores)
    threadpool_limits(limits=int(cores))
    os.environ["NUMEXPR_MAX_THREADS"] = str(cores)

    logger.info("Reading jobdata.")
    workdata = read_job_data(args.jobdata)
    work_coverage = estimated_deconvolved_fraction(workdata)
    logger.info(f"At most {work_coverage:.2%} of the genome is deconvolved.")
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
        check_length_distribution_flip(workdata, map_results)

    if args.only_check:
        return

    logger.info("Processing deconvolved tracks.")
    comb_data, signal_c1_list_global, signal_c2_list_global = interpolate(
        workdata,
        map_results,
        args.c1_smooth,
        args.c2_smooth,
        args.span,
        cores=cores,
        progress=not args.no_progress,
    )

    logger.info("Calling peaks.")
    peaks_c1, peaks_c2, overlaps = call_peaks(
        comb_data,
        signal_c1_list_global,
        signal_c2_list_global,
        work_coverage=work_coverage,
        fraction_in_peaks=args.fraction_in_peaks,
        fraction_overlap=args.fraction_overlap,
        uncorrected=args.uncorrected,
        c1_min_peak_size=args.c1_min_peak_size,
        c2_min_peak_size=args.c2_min_peak_size,
        span=args.span,
        cores=cores,
        progress=not args.no_progress,
    )

    bed_c1 = inlude_auc(name_peaks(peaks_c1, "c1"))
    bed_c2 = inlude_auc(name_peaks(peaks_c2, "c2"))

    if args.out is None:
        out_path, old_file = os.path.split(args.jobdata)
    else:
        out_path = args.out
        try:
            os.mkdir(out_path)
        except FileExistsError:
            pass

    logger.info("Writing results to %s/...", out_path)
    write_peaks(bed_c1, bed_c2, overlaps, out_path)
    logger.info("Finished sucesfully.")


if __name__ == "__main__":
    main()
