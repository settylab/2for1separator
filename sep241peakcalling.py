#! /usr/bin/env python
import os
import argparse
import pickle

import numpy as np
import scipy
import pandas as pd
from tqdm.auto import tqdm

from dcbackend import logger, setup_logging
from dcbackend import read_job_data, read_results
from dcbackend import check_length_distribution_flip


def parse_args():
    desc = "Peak calling after CUT&TAG 2for1 deconvolution."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "jobdata",
        metavar="jobdata-file",
        type=str,
        nargs="?",
        help="Jobdata with cuts per intervall and workchunk ids.",
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
        default=100,
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
        type=int,
        default=0.5,
        metavar="int",
    )
    parser.add_argument(
        "--fraction-overlap",
        help="""If more than this fraction of a peak overlaps with a peak of the
        other target than it is considered an overlapping region (default=0.5).""",
        type=int,
        default=0.5,
        metavar="int",
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
        "--no-progress",
        help="Do not show progress.",
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

    for name, dat in tqdm(workdata.iterrows(), total=len(workdata), desc="intervals"):
        wg = dat["workchunk"]
        maxlle = map_results.get(dat["workchunk"], None)
        if maxlle is None:
            continue
        if f"f_c1_{name}" not in maxlle.keys():
            continue
        start_idx = dat["stich_start"][0]
        end_idx = dat["stich_end"][0]
        seq = dat["seqname"]
        locations = dat["cuts"]["location"][start_idx:end_idx].values.astype(int)

        idx = dat["cuts"]["location"].rank(method="dense").astype(int) - 1
        signal_c1 = np.exp(maxlle[f"f_c1_{name}"][idx][start_idx:end_idx])
        signal_c2 = np.exp(maxlle[f"f_c2_{name}"][idx][start_idx:end_idx])
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
    workdata, map_results, c1_sigma, c2_sigma, step_size, cores=8, progress=True
):
    jobs, signal_c1_list_global, signal_c2_list_global = make_interpolation_jobs(
        workdata, map_results, c1_sigma, c2_sigma, step_size
    )
    results_list = list()
    with Pool(cores) as pool:
        with threadpool_limits(limits=1):
            for df in tqdm(
                pool.imap(generate_entry, jobs),
                total=len(jobs),
                disable=~progress,
            ):
                results_list.append(df)
    comb_data = pd.concat(results_list, axis=0)
    comb_data.index = range(len(comb_data))
    return comb_data, signal_c1_list_global, signal_c2_list_global


def target_fractions(comb_data):
    total_c1 = comb_data["c1"].sum()
    total_c2 = comb_data["c2"].sum()
    total = total_c1 + total_c2
    fraction_c1_uncorrected = total_c1 / total
    fraction_c2_uncorrected = total_c2 / total
    logger.info(
        f"Integral ratio: {fraction_c1_uncorrected:.1%} c1 and {fraction_c2_uncorrected:.1%} c2 cuts."
    )

    equal_dist_cuts = total
    total_c1 = comb_data["c1"].sum() + (equal_dist_cuts / 2)
    total_c2 = comb_data["c2"].sum() + (equal_dist_cuts / 2)
    total = total_c1 + total_c2
    fraction_c1_permissive = total_c1 / total
    fraction_c2_permissive = total_c2 / total
    logger.info(
        f"Bayesian estimation: {fraction_c1_permissive:.1%} c1 and {fraction_c2_permissive:.1%} c2 cuts."
    )

    fraction_c1 = (1 - fraction_bad_reads) * total_c1 / total
    fraction_c2 = (1 - fraction_bad_reads) * total_c2 / total
    logger.info(
        f"Estimated a total of {fraction_c1:.1%} c1 and {fraction_c2:.1%} c2 cuts in peaks."
    )
    return fraction_c1, fraction_c2, fraction_c1_permissive, fraction_c2_permissive

def mark_peaks(comb_data, signal_c1_list_global, signal_c2_list_global, fraction_c1, fraction_c2):
    
    bound_c1 = np.quantile(np.hstack(signal_c1_list_global), 1-fraction_c1)
    if 'c1 smooth' in comb_data.keys():
        comb_data['c1 mean'] = np.mean(comb_data.loc[:, ['c1', 'c1 smooth']], axis=1)
        comb_data['c1_peak'] = (comb_data['c1 mean'] > bound_c1) | (comb_data['c1 smooth'] > bound_c1)
    else:
        comb_data['c1_peak'] = comb_data['c1'] > bound_c1
    fraction_selected = comb_data['c1_peak'].sum() / len(comb_data['c1_peak'])
    logger.debug(
        f'Component 1: {fraction_selected:.2%} of intervalls '
        f'and {work_coverage*fraction_selected:.2%} of genome selected.'
    )
    idx = np.insert(comb_data['c1_peak'].values.astype(int), 0, 0)
    indicator = np.diff(idx)
    comb_data['c1_peak_number'] = np.cumsum(indicator==1)
    comb_data['c1_peak_name'] = (
        comb_data['interval']
        + '_' + comb_data['c1_peak_number'].astype(str)
        + '_' + comb_data['c1_peak'].astype(str)
    )
    
    bound_c2 = np.quantile(np.hstack(signal_c2_list_global), 1-fraction_c2)
    if 'c2 smooth' in comb_data.keys():
        comb_data['c2 mean'] = np.mean(comb_data.loc[:, ['c2', 'c2 smooth']], axis=1)
        comb_data['c2_peak'] = (comb_data['c2 mean'] > bound_c2) | (comb_data['c2 smooth'] > bound_c2)
    else:
        comb_data['c2_peak'] = comb_data['c2'] > bound_c2
    fraction_selected = comb_data['c2_peak'].sum() / len(comb_data['c2_peak'])
    logger.debug(
        f'Component 2: {fraction_selected:.2%} of intervalls and '
        f'{work_coverage*fraction_selected:.2%} of genome selected.'
    )
    idx = np.insert(comb_data['c2_peak'].values.astype(int), 0, 0)
    indicator = np.diff(idx)
    comb_data['c2_peak_number'] = np.cumsum(indicator==1)
    comb_data['c2_peak_name'] = (
        comb_data['interval']
        + '_' + comb_data['c2_peak_number'].astype(str)
        + '_' + comb_data['c2_peak'].astype(str)
    )
    return comb_data

def track_to_interval(loc_comb_data, comp_name, step_size):
    peak_location_df = loc_comb_data[loc_comb_data[f'{comp_name}_peak'].values].set_index('location')
    locations = peak_location_df.groupby(f'{comp_name}_peak_name')['comp_name'].idxmax()
    peak_groups = peak_location_df.reset_index().groupby(f'{comp_name}_peak_name')
    start = peak_groups['location'].min().values - (step_size/2)
    end = peak_groups['location'].max().values + (step_size/2)
    means = peak_groups[comp_name].mean()
    peaks = pd.DataFrame({
        'seqname':seqname,
        'start':start.astype(int),
        'end':end.astype(int),
        'mean':means,
        'summit':locations,
    }).sort_values('start')
    return peaks

def tracks_to_intervals(comb_data, step_size, progress=True):
    peaks_list_c1 = list()
    peaks_list_c2 = list()
    for seqname, loc_comb_data in tqdm(comb_data.groupby('seqname'), disable=~progress, desc='sequence'):
        peaks_c1 = track_to_interval(loc_comb_data, 'c1', step_size)
        peaks_list_c1.append(peaks_c1)
        peaks_c2 = track_to_interval(loc_comb_data, 'c2', step_size)
        peaks_list_c2.append(peaks_c1)
    peaks_c1 = pd.concat(peaks_list_c1, axis=0)
    peaks_c1['length'] = peaks_c1['end'] - peaks_c1['start']
    peaks_c2 = pd.concat(peaks_list_c2, axis=0)
    peaks_c2['length'] = peaks_c2['end'] - peaks_c2['start']
    return peaks_c1, peaks_c2

def make_overlap_report(comb_data, peaks_c1, peaks_c2, fraction_c1_permissive, fraction_c2_permissive, progress=True)
    sel_comb_data = comb_data[comb_data['c1_peak'] | comb_data['c2_peak']]
    df_entries = list()
    df_header = [
        'seqname',
        'start',
        'end',
        'length',
        'peak_c1',
        'peak_c2',
        'frac_c1',
        'frac_c2',
        'mean_c1',
        'mean_c2',
        'cosine_distance',
    ]
    for seqname, loc_comb_data in tqdm(sel_comb_data.groupby('seqname'), disable=~progress, desc='sequence'):
        loc_peaks_c1 = peaks_c1[peaks_c1['seqname'] == seqname].sort_values('start')
        loc_peaks_c2 = peaks_c2[peaks_c2['seqname'] == seqname].sort_values('start')
        max_pos_p = len(loc_peaks_c1)
        max_pos_k = len(loc_peaks_c2)
        if max_pos_p == 0:
            logger.info(f'No c1 peaks in sequence {seqname}.')
            continue
        if max_pos_k == 0:
            logger.info(f'No c2 peaks in sequence {seqname}.')
            continue
        rows_p = loc_peaks_c1.iterrows()
        name_p, dat_p = next(rows_p)
        rows_k = loc_peaks_c2.iterrows()
        name_k, dat_k = next(rows_k)
        rows_values = loc_comb_data.iterrows()
        _, values = next(rows_values)
        try:
            for _ in range(max_pos_p*max_pos_k):
                if dat_p['start'] > dat_k['end']:
                    name_k, dat_k = next(rows_k)
                    continue
                if dat_k['start'] < dat_p['end']:
                    # overlap
                    start = max(dat_k['start'], dat_p['start'])
                    end = min(dat_k['end'], dat_p['end'])
                    while values['location'] < start:
                        # move up towards start of overlap
                        _, values = next(rows_values)
                    values_p = list()
                    values_k = list()
                    while values['location'] < end:
                        # move up towards end of overlap
                        values_p.append(values['c1'])
                        values_k.append(values['c2'])
                        _, values = next(rows_values)
                    if len(values_p) > 1:
                        length = end - start
                        values_p = np.array(values_p)
                        values_k = np.array(values_k)
                        mean_p = np.mean(values_p)
                        mean_k = np.mean(values_k)
                        frac_p = length / (dat_p['end'] - dat_p['start'])
                        frac_k = length / (dat_k['end'] - dat_k['start'])
                        cosine_distance = scipy.spatial.distance.cosine(values_p-mean_p, values_k-mean_k)
                        df_entries.append((
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
                        ))
                name_p, dat_p = next(rows_p)
        except StopIteration:
            pass
    overlap_df = pd.DataFrame(df_entries, columns=df_header)
    overlap_df['log_ratio'] = np.log(overlap_df['mean_c1']) - np.log(overlap_df['mean_c2'])
    overlap_df['log_ratio_corrected'] = (
        overlap_df['log_ratio'] - np.log(fraction_c1_permissive) + np.log(fraction_c2_permissive)
    )
    overlap_df['callability'] = np.abs(overlap_df['log_ratio_corrected']) * overlap_df['cosine_distance']
    overlap_df['ambiguous'] = overlap_df['callability'] < 1
    return overlap_df

def merge_overlapping_intervals(interval_df):
    merged_df_list = list()
    for seqname, data in interval_df.groupby('seqname'):
        data = data.sort_values('start')
        last_stop = None
        new_starts = list()
        new_stops = list()
        new_means = list()
        for _, d in data.iterrows():
            start = d['start']
            stop = d['end']
            if last_stop is None: # init
                new_starts.append(start)
                means = [d['mean']]
                last_stop = stop
                continue
            means.append(d['mean'])
            if start < last_stop: # combine
                last_stop = max(last_stop, stop)
                continue
            # finish intervall
            new_stops.append(last_stop)
            new_means.append(np.mean(means))
            # start new intervall
            new_starts.append(start)
            last_stop = stop
        new_stops.append(last_stop)
        new_means.append(np.mean(means))
        merged_df_list.append(pd.DataFrame({
            'seqname':seqname,
            'start':np.array(new_starts),
            'end':np.array(new_stops),
            'mean':np.array(new_means),
            'name':[f'{seqname}_{i}' for i in range(len(new_starts))],
        }))
    return pd.concat(merged_df_list, axis=0)
    
def fiter_overlaps(peaks_c1, peaks_c2, overlap_df, threshold=.5)
    fractions_c1 = overlap_df.groupby('peak_c1', sort=False)['frac_c1'].sum()
    fractions_c2 = overlap_df.groupby('peak_c2', sort=False)['frac_c2'].sum()
    bad_idx_c1 = fractions_c1.index[fractions_c1 > threshold]
    overlapping_c1 = peaks_c1.loc[bad_idx_c1, :]
    bad_idx_c2 = fractions_c2.index[fractions_c2 > threshold]
    overlapping_c2 = peaks_c2.loc[bad_idx_c2, :]
    overlap_df = merged_df_list(pd.concat([overlapping_c1, overlapping_c2]))
    
    peaks_c1 = peaks_c1.loc[peaks_c1.index.difference(bad_idx_c1), :].sort_values(['seqname', 'start'])
    peaks_c2 = peaks_c1.loc[peaks_c2.index.difference(bad_idx_c2), :].sort_values(['seqname', 'start'])

    return peaks_c1, peaks_c2, overlap_df

def name_peaks(df):
    name_dat = pd.DataFrame(list(df.index.str.split('_')), columns=['seqname', 'interval', 'subinterval', 'is_peak'], index=df.index)
    df['name'] = 'PolS5P_' + name_dat['interval'] + '_' + name_dat['subinterval']
    return df

def inlude_auc(df):
    df['auc'] = df['mean'] * df['length']
    return df

def write_bed(bed_df, out_path):
    bed_df.to_csv(out_path, sep='\t', header=False, index=False)
    
def bed_to_summit_bed(df, out_path):
    summits = df['summit'].astype(int)
    summit_df = pd.DataFrame({
        'seqname':df['seqname'],
        'start':summits,
        'end':summits+1,
        'name':df['name'],
        'signal':df['mean'],
    })
    return summit_df

def main():
    args = parse_args()
    setup_logging(args.logLevel, args.logfile)
    if args.cores and args.cores > 0:
        logger.info("Limiting computation to %i cores.", args.cores)
        threadpool_limits(limits=int(args.cores))
        os.environ["NUMEXPR_MAX_THREADS"] = str(args.cores)

    logger.info("Reading jobdata.")
    workdata = read_job_data(args.jobdata)
    logger.info("Reading deconvolution results.")
    map_results = read_results(args.jobdata, workdata, progress=~args.no_progress)
    
    if not args.no_check:
        logger.info("Checking length distribution flip.")
        check_length_distribution_flip(map_results)

    logger.info("Interpolating deconvolved tracks.")
    comb_data, signal_c1_list_global, signal_c2_list_global = interpolate(
        workdata,
        map_results,
        args.c1_smooth,
        args.c2_smooth,
        args.span,
        cores=args.cores,
        progress=~args.no_progress,
    )
    
    fraction_c1, fraction_c2, fraction_c1_permissive, fraction_c2_permissive = target_fractions(comb_data)
    comb_data = mark_peaks(comb_data, signal_c1_list_global, signal_c2_list_global, fraction_c1, fraction_c2)
    logger.info('Converting peak signal to intervals.')
    peaks_c1, peaks_c2 = tracks_to_intervals(comb_data, step_size, progress=~args.no_progress)
    peaks_c1 = peaks_c1[peaks_c1['length']>args.c1_min_peak_size]
    peaks_c2 = peaks_c2[peaks_c2['length']>args.c2_min_peak_size]
    
    logger.info('Analyzing overlaps.')
    overlap_df = make_overlap_report(comb_data, peaks_c1, peaks_c2, fraction_c1_permissive, fraction_c2_permissive, progress=~args.no_progress)
    
    logger.info('Filtering overlaps.')
    peaks_c1, peaks_c2, overlaps = fiter_overlaps(peaks_c1, peaks_c2, overlap_df, threshold=args.fraction_overlap)

    if args.out is None:
        out_path, old_file = os.path.split(args.jobdata)
    else:
        out_path = args.out

    out_base = os.path.join(out_path, f"deconv_cuts-per-{args.unit}bp")
    logger.info("Writing results to %s", out_base)
    
    bed_c1 = inlude_auc(name_peaks(peaks_c1))
    bed_c2 = inlude_auc(name_peaks(peaks_c2))
    
    write_bed(bed_c1[['seqname', 'start', 'end', 'name', 'auc']], out_base + '_deconv_c1.bed')
    write_bed(bed_c2[['seqname', 'start', 'end', 'name', 'auc']], out_base + '_deconv_c2.bed')
    
    write_bed(bed_to_summit_bed(bed_c1), out_base + '_deconv_c1.summits.bed')
    write_bed(bed_to_summit_bed(bed_c2), out_base + '_deconv_c2.summits.bed')
    
    write_bed(overlap_df[['seqname', 'start', 'end', 'name', 'mean']], out_base + '_overlap.bed')

    logger.info("Finished sucesfully.")


if __name__ == "__main__":
    main()
