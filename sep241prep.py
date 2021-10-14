#! /usr/bin/env python
import os
import warnings
import argparse

from datetime import datetime
import tabix
from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from KDEpy import FFTKDE
import multiprocessing

from dcbackend import Deconvoluter, LevelSet
from dcbackend import logger, setup_logging


class RegionTooSmall(Exception):
    pass


class InvalidRegionBounds(Exception):
    pass


class SubdivisionError(Exception):
    pass


def filter_sort_events(
    events, blacklist=None, blacklisted_sequences=set(), progress=True
):
    grouped = events.groupby("seqname")
    if blacklist:
        tb = tabix.open(blacklist)

    filtered_list = list()

    logger.info("Removing blacklisted reads and sorting.")
    for seqname, local_events in tqdm(grouped, desc="sequence", disable=~progress):
        if seqname in blacklisted_sequences:
            continue
        local_events = local_events.sort_values("location")
        if not blacklist:
            filtered_list.append(local_events)
            continue
        locations = local_events["location"].values
        n = len(locations)
        blacklisted = np.zeros(n, dtype=bool)
        if blacklist:
            records = tb.querys(seqname)
            loc_idx = 0
            for seq, start, stop, stype in records:
                nstart = int(start)
                nstop = int(stop)
                for loc_idx in range(loc_idx, n):
                    location = locations[loc_idx]
                    if location > nstop:
                        break
                    if nstart <= location:
                        blacklisted[loc_idx] = True
        filtered_list.append(local_events.loc[~blacklisted, :])
    events = pd.concat(filtered_list, axis=0)
    return events


def filter_intervals(events, blacklist, blacklisted_sequences=set(), progress=True):
    grouped = events.groupby("seqname")
    tb = tabix.open(blacklist)

    filtered_list = list()

    logger.info("Removing blacklisted reads.")
    for seqname, local_events in tqdm(grouped, desc="sequence", disable=~progress):
        if seqname in blacklisted_sequences:
            continue
        records = tb.querys(seqname)
        local_events = local_events.sort_values("start")
        rows = local_events.iterrows()
        _, row = next(rows)
        try:
            for seq, start, stop, stype in records:
                nstart = int(start)
                nstop = int(stop)
                while row["start"] < nstop:
                    if row["end"] < nstart:
                        filtered_list.append(row)
                    _, row = next(rows)
            for i in range(len(local_events)):
                _, row = next(rows)
                filtered_list.append(row)
        except StopIteration:
            pass
    events = pd.DataFrame(filtered_list)
    return events


def write_bed(df, file_path):
    return df.to_csv(file_path, sep="\t", header=False, index=False)


def full_kde_grid(x, xmin=None, xmax=None):
    if xmin is None:
        xmin = np.min(x) - 1
    if xmax is None:
        xmax = np.max(x) + 1
    grid = np.arange(xmin, xmax + 1)
    return grid


def get_kde(
    cut_locations, kde_bw=500, kernel="gaussian", xmin=None, xmax=None, grid=None
):
    if grid is None:
        grid = full_kde_grid(cut_locations, xmin, xmax)
    kernel = FFTKDE(kernel=kernel, bw=kde_bw)
    kernel = kernel.fit(cut_locations)
    density = kernel.evaluate(grid)
    return grid, density


def get_intervals(boolean_selection, region_padding):
    """Convert boolean selection of a seqence with additional padding into
    a vector of interval starts and ends."""
    data_count = boolean_selection.cumsum()
    gains = np.zeros(data_count.shape, dtype=int)
    gains[:-region_padding] = data_count[region_padding:]
    gains[-region_padding:] = data_count[-1]
    looses = np.zeros(data_count.shape, dtype=int)
    looses[region_padding:] = data_count[:-region_padding]
    data_in_reach = gains - looses
    selected_data = data_in_reach > 0

    bp_selected = np.sum(selected_data)
    selected_percentage = bp_selected / len(selected_data)
    logger.info(
        f"Selected {selected_percentage:.4%} ({bp_selected:,} bp) of the sequence..."
    )

    start_end = np.zeros(selected_data.shape[0] + 1)
    start_end[:-1] = selected_data
    start_end[1:] -= start_end[:-1]
    starts = np.where(start_end == 1)[0]
    ends = np.where(start_end == -1)[0]
    if ends[-1] == len(start_end):
        # last position is outside of grid and musst be avoided
        ends[-1] -= 1
    interval_sizes = ends - starts

    logger.info(
        f"...in {len(starts):,} intervals with a maximum size of {interval_sizes.max():,} bp."
    )
    return starts, ends


def get_cuts(starts, ends, events):
    if events is None or len(events) == 0:
        return list()
    assert len(starts) == len(ends)
    if len(starts) == 0:
        return list()
    cuts = list()
    i = 0
    isin = False
    for j, c in enumerate(events["location"]):
        if not isin and c >= starts[i]:
            isin = True
            local_cuts = list()
        elif c > ends[i]:
            isin = False
            i += 1
            cuts.append(events.iloc[local_cuts, :].copy())
            if i >= len(starts):
                break
        if isin:
            local_cuts.append(j)
    if isin:
        cuts.append(events.iloc[local_cuts, :].copy())
    return cuts


def subdivide(
    events,
    start,
    end,
    region_padding,
    max_cuts_per_intervall,
    max_locations_per_intervall,
):
    """Subdevides cuts into overlapping regions. Cuts musst be sorted."""
    lcuts = events["location"].values
    region_size = end - start
    if region_size <= 2 * region_padding:
        raise RegionTooSmall(
            "Region cannot be subdevided, smaller than three paddings."
            f"{region_size: = } {3*region_padding: = }"
        )
    if start > np.min(lcuts):
        raise InvalidRegionBounds(f"First cut before region start.")
    if end < np.max(lcuts):
        raise InvalidRegionBounds(f"Last cut after region end.")
    starts = list()
    stich_starts = list()
    ends = list()
    stich_ends = list()
    cuts = list()

    # init
    start_tuple = [0, lcuts[0]]
    next_start = start_tuple.copy()
    stich_start = start_tuple.copy()
    stich_end = None
    min_strich = start + region_padding
    min_end = min_strich + region_padding
    sel_cuts = list()
    n_locs = 0
    last_loc = -np.inf
    for i, c in enumerate(lcuts):
        if c >= min_strich and not stich_end:
            stich_end = [i, c]
            stich_start = [i, c]
        if c >= min_end:
            break
        sel_cuts.append(i)
        if c != last_loc:
            n_locs += 1
        last_loc = c

    # extend
    for k, c in enumerate(lcuts[i:]):
        global_idx = k + i
        sel_cuts.append(global_idx)
        if c != last_loc:
            n_locs += 1
        last_loc = c
        while c - stich_end[1] > region_padding:
            stich_end[0] += 1
            stich_end[1] = lcuts[sel_cuts[stich_end[0]]]
        while stich_end[1] - next_start[1] > region_padding:
            next_start[0] += 1
            next_start[1] = lcuts[sel_cuts[next_start[0]]]
        exclusive_region = stich_end[1] - stich_start[1]
        if (
            (
                len(sel_cuts) >= (max_cuts_per_intervall / 2)
                or n_locs >= max_locations_per_intervall
            )  # max work
            and exclusive_region >= region_padding  # min region
            and k != len(lcuts[i:])  # not the last cut
        ):
            # end interval
            starts.append(start_tuple[1])
            stich_starts.append(stich_start.copy())
            ends.append(c)
            stich_ends.append(stich_end.copy())
            cuts.append(events.iloc[sel_cuts, :].copy())
            if len(sel_cuts) > max_cuts_per_intervall:
                perc = len(sel_cuts) / max_cuts_per_intervall
                warnings.warn(
                    f"Subdivide {len(starts)} has {perc:.2%} of the selected maximum "
                    "of cuts per interval. This could result in too large work chunks."
                )
            if n_locs > max_locations_per_intervall:
                perc = n_locs / max_locations_per_intervall
                warnings.warn(
                    f"Subdivide {len(starts)} has {perc:.2%} of the selected maximum "
                    "of cuts per interval. This could result in too large work chunks."
                )

            # start new intervall
            index_shift = next_start[0]
            sel_cuts = sel_cuts[index_shift:]
            n_locs = len(set(lcuts[sel_cuts]))
            next_start[0] = 0
            start_tuple = next_start.copy()
            stich_end[0] -= index_shift
            stich_start = stich_end.copy()
    for c_idx in sel_cuts[stich_end[0] :]:
        stich_end[0] += 1
        stich_end[1] = lcuts[c_idx]
        if end - stich_end[1] <= region_padding:
            break
    starts.append(start_tuple[1])
    stich_starts.append(stich_start.copy())
    ends.append(c)
    stich_ends.append(stich_end.copy())
    cuts.append(events.iloc[sel_cuts, :].copy())

    return starts, stich_starts, ends, stich_ends, cuts


def make_peak_weights(events, log=True):
    """Weights of peak values to use for integration along sequence."""
    peaks_per_pos = events["location"].value_counts(sort=False).sort_index()
    midpoints = (peaks_per_pos.index[:-1] + peaks_per_pos.index[1:]) / 2
    sizes = np.diff(
        np.concatenate(
            [[events["location"].min()], midpoints, [events["location"].max()]]
        )
    )
    weights_per_peak_with_pos_index = sizes / peaks_per_pos
    peak_weights = weights_per_peak_with_pos_index[events["location"]]
    if not log:
        return peak_weights.values
    return np.log(peak_weights.values)


def assigne_workchunk(result_df, max_mem_per_group=200, seed=None, progress=True):
    interval_idxs = np.arange(len(result_df))
    np.random.seed(seed=seed)
    np.random.shuffle(interval_idxs)
    work_per_group = [0]
    grouping = np.zeros_like(interval_idxs)
    for idx in interval_idxs:
        interval_work = result_df["memory"][idx]
        if interval_work > max_mem_per_group:
            seqname = result_df["seqname"][idx]
            start = result_df["start"][idx]
            end = result_df["end"][idx]
            pec = interval_work / max_mem_per_group
            group_idx = len(work_per_group) + 1
            logger.warning(
                f"{seqname}:{start:,}-{end:,} has a very high cut concentration and "
                f"work chunk {group_idx} "
                f"will consume {pec:.2%} of the configured memory. "
                "Lower --max-locs and set lower --interval-padding to allow for smaller intervals."
            )
            grouping[idx] = group_idx
            work_per_group.append(interval_work)
            continue
        fits = False
        for group_idx, work in enumerate(work_per_group):
            new_work = work + interval_work
            if new_work < max_mem_per_group:
                fits = True
                break
        if not fits:
            group_idx += 1
            work_per_group.append(0)
            new_work = interval_work
        work_per_group[group_idx] += interval_work
        grouping[idx] = group_idx
    result_df["workchunk"] = grouping
    return result_df


def read_bed(file_path):
    header = {0: "seqname", 1: "start", 2: "end", 3: "name"}
    bed_content = pd.read_csv(file_path, delimiter="\t", header=None)
    bed_content.columns = [header.get(i, i) for i in bed_content.columns]
    return bed_content


def read_bedgraph(file_path):
    header = ["seqname", "start", "end", "value"]
    bed_content = pd.read_csv(file_path, delimiter="\t", header=None, skiprows=1)
    bed_content.columns = header
    return bed_content


def to_seq_dict(dataframe, name_col=None):
    intervalls = dict()
    for seqname in dataframe["seqname"].unique():
        idx = dataframe["seqname"] == seqname
        df = dataframe[idx].sort_values("start")
        intervalls[seqname] = {
            "start": df["start"].values,
            "end": df["end"].values,
        }
        if name_col is None:
            intervalls[seqname]["name"] = [
                f"{seqname}_{i}" for i in range(len(df["start"]))
            ]
        else:
            intervalls[seqname]["name"] = df[name_col].values
    return intervalls


def coverage(intervalls, printit=True):
    """Approximate fraction of covered genome"""
    total = 0
    covered = 0
    for seqname, data in intervalls.items():
        total += np.max(data["end"])
        covered += np.sum(data["end"] - data["start"])
    percentage = covered / total
    if printit:
        print(f"{percentage:.2%} covered")
    return percentage


def extend_intervalls(intervalls, amount):
    extended_intervalls = dict()
    for seqname, data in intervalls.items():
        starts = np.where(data["start"] > amount, data["start"] - amount, 0)
        stops = data["end"] + amount
        last_stop = None
        new_starts = list()
        new_stops = list()
        for start, stop in zip(starts, stops):
            if last_stop is None:  # init
                new_starts.append(start)
                last_stop = stop
                continue
            if start < last_stop:  # combine
                last_stop = stop
                continue
            # finish intervall
            new_stops.append(last_stop)
            # start new intervall
            new_starts.append(start)
            last_stop = stop
        new_stops.append(last_stop)
        extended_intervalls[seqname] = {
            "start": np.array(new_starts),
            "end": np.array(new_stops),
            "name": [f"{seqname}_{i}" for i in range(len(new_starts))],
        }
    return extended_intervalls


def parse_args():
    desc = "Prepair CUT&TAG 2for1 deconvolution."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "fragment_files",
        metavar="fragment-file.bed.gz",
        type=str,
        nargs="+",
        help="(Compressed) bed files. All files are combined to one single track.",
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
        help="Output file path (default=work_chunks.pandas_pkl).",
        default="work_chunks.pandas_pkl",
        type=str,
        metavar="dir",
    )
    parser.add_argument(
        "--seed",
        help="Random state seed to assign intervals to work chunks.",
        type=int,
        default=242567,
        metavar="int",
    )
    parser.add_argument(
        "--kde-bw",
        help="Bandwidth (sigma) for kernel density estimate (KDE) used for interval selection (default=200).",
        type=float,
        default=200,
        metavar="float",
    )
    parser.add_argument(
        "--kde-threshold",
        help="Minimum KDE value to consider genomic section for deconvolution (default=2).",
        type=float,
        default=2,
        metavar="float",
    )
    parser.add_argument(
        "--selection-padding",
        help="Additional padding around genomic section selected based on KDE (default=10,000).",
        type=int,
        default=10_000,
        metavar="int",
    )
    parser.add_argument(
        "--selection-bed",
        help="Alternatively to KDE selection, specifiy bed file of regions to deconvolve.",
        type=str,
        default=None,
        metavar="str",
    )
    parser.add_argument(
        "--region-padding",
        help="Additional padding around intervals even if specified through bed file (default=5,000).",
        type=int,
        default=5_000,
        metavar="int",
    )
    parser.add_argument(
        "--interval-overlap",
        help="Overlap to neighboring subdevides and minimal exclusive section size (default=5,000).",
        type=int,
        default=10_000,
        metavar="int",
    )
    parser.add_argument(
        "--max-locs",
        help="Maximum of unique cuts per interval. Memory demand may increases proportinally to its quadrat (default=15,000).",
        type=int,
        default=10_000,
        metavar="int",
    )
    parser.add_argument(
        "--max-cuts",
        help="Maximum of cuts per interval. Memory demand may increases proportinally (default=inf).",
        type=int,
        default=np.inf,
        metavar="int",
    )
    parser.add_argument(
        "--memory-target",
        help="Memory demand target in GBs for individual work chunks (default=300).",
        type=float,
        default=300,
        metavar="float",
    )
    parser.add_argument(
        "--blacklist",
        help="Bed file of genomic regions to exlcude from the deconvolution.",
        type=str,
        default=None,
        metavar="file.bed.gz2",
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


def get_intervals_by_kde(events, min_density, selection_padding, kde_bw=200):
    sequences = events["seqname"].unique()
    intervalls_small = dict()
    other_data = dict()

    for i, (seqname, seq_events) in enumerate(events.groupby("seqname")):
        if min_density == 0:
            grid = full_kde_grid(seq_events["location"].values)
            data_idx = np.ones_like(grid)
        else:
            logger.info(f"Making KDE of {seqname} [{i+1}/{len(sequences)}].")
            grid, density = get_kde(seq_events["location"].values, kde_bw=kde_bw)
            density *= len(seq_events["location"]) * 100
            logger.info(f"Finding intervals based on KDE.")
            data_idx = density > min_density
        starts, ends = get_intervals(data_idx, selection_padding)
        gstarts = grid[starts]
        gends = grid[ends - 1]
        intervalls_small[seqname] = {
            "start": gstarts,
            "end": gends,
            "name": [f"{seqname}_{i}" for i in range(len(gstarts))],
        }
    return intervalls_small


def make_work_packages(
    intervalls,
    cut_df,
    interval_padding,
    max_cuts_per_intervall,
    max_locations_per_intervall,
):
    result_df_list = list()
    cuts_per_seq = {seq: df for seq, df in cut_df.groupby("seqname")}
    for i, (seqname, data) in enumerate(intervalls.items()):
        logger.info(
            f"Subdeviding intervals in sequence {seqname} [{i+1}/{len(intervalls)}]."
        )
        starts, ends = data["start"], data["end"]
        all_cuts = cuts_per_seq.get(seqname)
        if all_cuts is None or len(all_cuts) == 0:
            logger.warn('No cuts in sequence "%s".', seqname)
            continue
        cuts = get_cuts(starts, ends, all_cuts)
        data["cuts"] = cuts
        data["ncuts"] = [len(c) for c in cuts]
        data.pop("subdivides", None)
        subdivided_ints = dict()
        max_num_locs = 0
        max_num_locs_idx = None
        richest_sub_int = {"size": 0}
        for i, lcuts in enumerate(cuts):
            num_locs = len(lcuts["location"].unique())
            if num_locs > max_num_locs:
                max_num_locs = num_locs
                max_num_locs_idx = i
            start = data["start"][i]
            if i == 0:
                # avoid removing overlap at interval start
                start -= 1.5 * interval_padding
            end = data["end"][i]
            if i == len(cuts) - 1:
                # avoid removing overlap at interval end
                end += 1.5 * interval_padding
            with warnings.catch_warnings(record=True) as warns:
                try:
                    r = subdivide(
                        lcuts,
                        start,
                        end,
                        interval_padding,
                        max_cuts_per_intervall,
                        max_locations_per_intervall,
                    )
                except Exception as e:
                    raise SubdivisionError(
                        f"In sequence {seqname} intervall {i} cannot be subdivided."
                    )
            for w in warns:
                logger.warning(
                    "%s:%s-%s: %s",
                    seqname,
                    "{:,d}".format(start),
                    "{:,d}".format(end),
                    w.message,
                )
            subdivided_ints[i] = {
                "start": r[0],
                "stich_start": r[1],
                "end": r[2],
                "stich_end": r[3],
                "cuts": r[4],
                "ncuts": [len(c) for c in r[4]],
                "nlocs": [len(c["location"].unique()) for c in r[4]],
                "name": [data["name"][i] + f".{k}" for k in range(len(r[0]))],
            }
            result_df = pd.DataFrame(subdivided_ints[i])
            if len(r[0]) > 1:
                result_df["is_subdivide"] = True
            else:
                result_df["is_subdivide"] = False
            result_df["seqname"] = seqname
            result_df_list.append(result_df)

            largest_size = 0
            largest_idx = None
            for j, nlc in enumerate(subdivided_ints[i]["nlocs"]):
                if nlc > largest_size:
                    largest_size = nlc
                    largest_idx = j
            subdivided_ints[i]["largest"] = largest_idx
            if largest_size > richest_sub_int["size"]:
                richest_sub_int = {
                    "size": largest_size,
                    "interval index": i,
                    "subintval index": largest_idx,
                }
        message = f"Richest interval was cut in {max_num_locs:,} locations"
        idx = subdivided_ints[max_num_locs_idx]["largest"]
        size = subdivided_ints[max_num_locs_idx]["nlocs"][idx]
        message += f" and was subdivided with maximal {size:,} locations."
        logger.debug(message)
        nc = richest_sub_int["size"]
        logger.debug(f"Richest subintervall was cut in {nc:,} locations.")
    result_df = pd.concat(result_df_list, axis=0).set_index("name")
    return result_df


def approximate_memory_demand(result_df):
    result_df["memory"] = (result_df["nlocs"] ** 2) * 1e-6 + 5e-2
    return result_df


def main():
    args = parse_args()
    setup_logging(args.logLevel, args.logfile)
    out_path, _ = os.path.split(args.out)
    if not os.path.isdir(out_path):
        logger.info("Making directory %s", out_path)
        os.mkdir(out_path)
    if not args.cores:
        cores = multiprocessing.cpu_count()
        logger.info("Detecting %s compute cores.", cores)
    else:
        cores = args.cores
    logger.info("Limiting computation to %i cores.", cores)
    threadpool_limits(limits=int(cores))
    os.environ["NUMEXPR_MAX_THREADS"] = str(cores)
    fragments_files = {f"file {i}": file for i, file in enumerate(args.fragment_files)}
    dc = Deconvoluter(fragments_files=fragments_files, show_progress=~args.no_progress)
    len_df = dc.all_fragments()
    all_events = dc.envents_from_intervals(len_df)
    events = filter_sort_events(
        all_events, args.blacklist, args.blacklisted_seqs, progress=~args.no_progress
    )
    if args.selection_bed:
        bed_intervals = read_bed(args.selection_bed)
        intervalls_small = {
            seqname: df for seqname, df in bed_intervals.groupby("seqname")
        }
    else:
        intervalls_small = get_intervals_by_kde(
            events, args.kde_threshold, args.selection_padding, args.kde_bw
        )

    cov = coverage(intervalls_small, printit=False)
    logger.info(
        f"Approximatly {cov:.2%} of genome selected for deconvolution before padding."
    )
    intervalls = extend_intervalls(intervalls_small, args.region_padding)
    cov = coverage(intervalls, printit=False)
    logger.info(
        f"Approximatly {cov:.2%} of genome selected for deconvolution after padding."
    )

    result_df = make_work_packages(
        intervalls, events, args.interval_overlap, args.max_cuts, args.max_locs
    )
    approximate_memory_demand(result_df)

    assigne_workchunk(result_df, max_mem_per_group=args.memory_target, seed=args.seed)
    logger.info("Writing work chunks to %s", args.out)
    result_df.to_pickle(args.out)
    n_wg = len(result_df["workchunk"].unique())
    print(
        f"""
    Preperation finished and there are {n_wg:,} work chunks (0-{n_wg-1}).

    Run deconvolution using SLURM:

        sbatch --array=0-{n_wg-1} --mem={args.memory_target}G sep241deconvolve.py {args.out}

    Run in bash:

        for wc in $(seq 0 {n_wg-1}); do
            ./sep241deconvolve.py {args.out} --workchunk $wc
        done

    Please consider adjusting and passing the following additional arguments:

        --c1-cov 'Matern32(500)' --c2-cov 'Matern32(2000)' \\
        --c1-dirichlet-prior 450 100 10 1 --c2-dirichlet-prior 150 300 50 10

    """
    )


if __name__ == "__main__":
    main()
