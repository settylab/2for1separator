import warnings

import os
import re
import logging
import pickle

import numpy as np
import scipy
import pandas as pd
from tqdm.auto import tqdm

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels
from sklearn.linear_model import LinearRegression
from scipy.stats import lognorm


class NoData(Exception):
    pass


class MissingData(Exception):
    pass


class BEDFormatError(Exception):
    pass


class NotARegion(Exception):
    pass

class BarcodeError(Exception):
    pass


logger = logging.getLogger("2for1seperator")


def setup_logging(level, logfile=None):
    logger.propagate = False
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)-8s] %(message)s",
        level=logging.getLevelName(level),
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.getLevelName(level))
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)-8s] %(message)s")
    ch.setFormatter(formatter)
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.addHandler(ch)


def set_flag(flag, value=None):
    """
    Sets or overwrites the theano `flag` in the environment variable 'THEANO_FLAGS'.

    :param flag: The flag name that is to be overwritten or set.
    :param value: The value to be assigned to the flag. If it is
        `None` then `flag` will be pasted as is into 'THEANO_FLAGS'.
    :return: The new value of 'THEANO_FLAGS'.
    """
    if not isinstance(flag, str):
        raise TypeError("The argument `flag` needs to be a string.")
    flagString = os.getenv("THEANO_FLAGS")

    if value is None:
        if flagString:
            flag = flagString + "," + flag
        os.environ["THEANO_FLAGS"] = flag
        return newFlagString

    if not isinstance(value, str):
        raise TypeError("The argument `value` needs to be a string or `None`.")

    if flagString:
        oldFlags = flagString.split(",")
    else:
        oldFlags = []
    flagTag = flag + "="
    newFlags = [s for s in oldFlags if flagTag not in s]
    newFlags.append(flagTag + value)
    newFlagString = ",".join(newFlags)
    os.environ["THEANO_FLAGS"] = newFlagString
    return newFlagString


def format_cuts(locations_ds):
    """Formats cuts as represented in the model."""

    interleave_index = locations_ds.rank(method="dense").astype(int) - 1

    cuts_per_pos = locations_ds.value_counts(sort=False).sort_index()
    midpoints = (cuts_per_pos.index[:-1] + cuts_per_pos.index[1:]) / 2
    rectangle_width = np.diff(
        np.concatenate([[locations_ds.min()], midpoints, [locations_ds.max()]])
    )
    # ends have only half of expected width
    rectangle_width[0] += rectangle_width[0]
    rectangle_width[-1] += rectangle_width[-1]
    log_loc_weight = np.log(rectangle_width)

    location_shift = np.median(cuts_per_pos.index).astype(int)  # for numeric ease
    unique_locations = np.array(cuts_per_pos.index - location_shift)

    return unique_locations, log_loc_weight, interleave_index.values


def write_bed(bed_df, out_path):
    bed_df.to_csv(out_path, sep="\t", header=False, index=False)


def detect_cores(cores=None):
    import multiprocessing
    if not cores:
        cores = multiprocessing.cpu_count()
        logger.info("Detecting %s compute cores.", cores)
    return cores


def read_region_string(
    feature, short_seqname=True, format_string=r"([^:]+):([0-9,_]+)-([0-9,_]+)"
):
    region_format = re.compile(format_string)
    is_region = region_format.match(feature)
    if not is_region:
        raise NotARegion(
            f"The passed feature `{feature}` does not match the region format seqname:start-end."
        )
    seq, start, end = is_region.groups()
    start = int(re.sub("[^0-9]", "", start))
    end = int(re.sub("[^0-9]", "", end))
    if short_seqname and seq.startswith("chr"):
        seq = seq[3:]
    return seq, start, end


def length_dist(
    weights,
    sigmas=[0.4, 0.18, 0.115, 0.085],
    modes=[70, 200, 400, 600],
    x=np.linspace(1, 800, 500),
):
    mus = [np.log(modes[i]) + (s ** 2) for i, s in enumerate(sigmas)]
    dists = [
        weights[i] * lognorm.pdf(x, sigmas[i], scale=np.exp(mus[i]), loc=0)
        for i in range(len(weights))
    ]
    return np.sum(np.stack(dists), axis=0)


def posterior_length_dists(
    map_results, x=np.linspace(1, 800, 500), prior=True, progress=False
):
    df_list = list()
    for workchunk, maxlle in tqdm(
        map_results.items(),
        desc="work chunk",
        total=len(map_results),
        disable=~progress,
    ):
        ndfp = pd.DataFrame(
            {
                "density": length_dist(maxlle["weight_c1"]),
                "length": x,
                "from": "Posterior c1",
                "workchunk": str(workchunk),
            }
        )
        df_list.append(ndfp)
        ndfk = pd.DataFrame(
            {
                "density": length_dist(maxlle["weight_c2"]),
                "length": x,
                "from": "Posterior c2",
                "workchunk": str(workchunk),
            }
        )
        df_list.append(ndfk)
    if prior and (args := maxlle.get("args")):
        w_c1 = args.c1_dirichlet_prior
        y = length_dist(w_c1 / np.sum(w_c1))
        prior_df = pd.DataFrame(
            {"density": y, "length": x, "from": "Prior c1", "workchunk": "none",}
        )
        df_list.append(prior_df)
        w_c2 = args.c2_dirichlet_prior
        y = length_dist(w_c2 / np.sum(w_c2))
        prior_df = pd.DataFrame(
            {"density": y, "length": x, "from": "Prior c2", "workchunk": "none",}
        )
        df_list.append(prior_df)
    result_df = pd.concat(df_list)
    result_df["group"] = result_df["from"] + result_df["workchunk"]
    return result_df


def posterior_mode_weights(workdata, map_results):
    w_c1_posterior = w_c2_posterior = False
    for name, dat in workdata.iterrows():
        wg = dat["workchunk"]
        if wg not in map_results:
            continue
        n_cuts = len(dat["cuts"])
        w_c1_inferred = map_results[wg]["weight_c1"]
        w_c2_inferred = map_results[wg]["weight_c2"]
        if w_c1_posterior is False:
            w_c1_posterior = np.zeros(len(w_c1_inferred))
            w_c2_posterior = np.zeros(len(w_c2_inferred))
        w_c1_posterior += n_cuts * w_c1_inferred
        w_c2_posterior += n_cuts * w_c2_inferred
    return w_c1_posterior, w_c2_posterior


def check_length_distribution_flip(workdata, map_results, threshold=0.9):
    """Look for flip between posterior length distributions.

    :param threshold: Pearson correlation threshold for difference
        between c1 and c2 (default=.9).
    """
    logger.info("Calculating length distributions.")
    length_df = posterior_length_dists(map_results, prior=False)
    components = length_df["from"].unique()
    logger.info("Looking for flipped distributions.")
    diffs = dict()
    for wc, df in length_df.groupby("workchunk"):
        c1 = df[df["from"] == components[0]].set_index("length")["density"]
        c2 = df[df["from"] == components[1]].set_index("length")["density"]
        diffs[wc] = c1 - c2
    diffs = pd.DataFrame(diffs)

    usual_diff = diffs.median(axis=1)
    diff_coors = diffs.corrwith(usual_diff)
    idx_flipped = diff_coors < threshold
    bad_wgs = set(idx_flipped.index[idx_flipped].astype(int))

    if any(idx_flipped):
        w_c1_posterior, w_c2_posterior = posterior_mode_weights(workdata, map_results)
        wg_string = to_grouped_string(bad_wgs)
        post_str_c1 = " ".join([str(int(w)) for w in w_c1_posterior])
        post_str_c2 = " ".join([str(int(w)) for w in w_c2_posterior])
        logger.warn(
            """The length distributions for the following workgoups are flipped: %s
            Consider rerunning these workchunks with the following Dirichlet priors:

                --c1-dirichlet-prior %s
                --c2-dirichlet-prior %s

            To do that on slurm you can run

                jobs=%s
                sbatch --mem=[max memory] --array=$jobs sep241deconvolve.py \\
                    --c1-dirichlet-prior %s \\
                    --c2-dirichlet-prior %s \\
                    [additional parameters]
                """,
            wg_string,
            post_str_c1,
            post_str_c2,
            wg_string,
            post_str_c1,
            post_str_c2,
        )
    else:
        logger.info("No flipped length distributions found.")
    return bad_wgs


def read_job_data(jobdata_file):
    return pd.read_pickle(jobdata_file)


def to_grouped_string(list_of_integers):
    grouped_str = ""
    last_m = -np.inf
    in_group = False
    for m in sorted(list_of_integers):
        if m == last_m+1:
            in_group = True
        else:
            if in_group:
                # close of last group
                grouped_str += f"-{last_m}"
                in_group = False
            grouped_str += f",{m}"
        last_m = m
    if in_group:
        grouped_str += f"-{last_m}"
    grouped_str = grouped_str[1:] # remove leading ,
    return grouped_str


def read_results(jobdata_file, workdata, progress=True, error=True):
    map_results = dict()
    misses = list()
    last_wg_data = None
    for workchunk in tqdm(
        workdata["workchunk"].unique(), disable=~progress, desc="work chunk"
    ):
        filename = os.path.splitext(jobdata_file)[0] + f"_wg-{workchunk}.pkl"
        try:
            with open(filename, "rb") as fl:
                map_results[workchunk] = pickle.load(fl)
            last_wg_data = map_results[workchunk]
        except FileNotFoundError:
            misses.append(workchunk)
    if last_wg_data is None:
        raise NoData("No deconvolution results found.")
    if misses:
        total_wc = len(workdata["workchunk"].unique())
        miss_str = to_grouped_string(misses)
        logger.warn("Results for the following %s of %s work chunks are missing: %s",
                    f'{len(misses):,}', f'{total_wc:,}', miss_str)
        if error:
            raise MissingData(
                f"Results for the following {len(misses):,} of {total_wc:,} work chunks are missing: {miss_str}"
            )
    return map_results


class LevelSet:
    """
    Used by the gtf drawing tool.

    Remembers levels occupied by intervals if called
    for all intervals sorted by their start.
    Ensures a minimal padding distance between intervals in
    same level.
    """

    def __init__(self, interval_padding, base_level=0, increments=-2):
        self.padding = interval_padding
        self.base_level = base_level
        self.increments = increments
        self.occupied_levels = dict()

    def get_free(self, interval_start, interval_end):
        now_free = set()
        for level, end in self.occupied_levels.items():
            if end + self.padding < interval_start:
                now_free.add(level)
        [self.occupied_levels.pop(l) for l in now_free]
        level = self.base_level
        while level in self.occupied_levels.keys():
            level += self.increments
        self.occupied_levels[level] = interval_end
        return level

    def reset(self):
        self.occupied_levels = dict()


def igv_plot(
    df,
    color=None,
    group=None,
    facet_scales="free_y",
    min_counts=0,
    min_density=1e-10,
    sort_by=None,
    relabeling=dict(),
    smooth_events=None,
    max_res=1000,
    border=True,
):
    """
    Draws an IGV-like plot. Used by Multitool class.

    :param df: DataFrame with reads and columns 'start', 'stop' and
        columns specified by group and color.
    :param color: Column of df to color tracks by. Must be coarser than groups.
    :param group: Column of df to draw different tracks per group. Must be finer than color.
    :param facet_scales: passed to ggplot facet_grid to scale axis of tracks.
    :param min_count: Filter out tracks with less reads (default: 0).
    :param sort_by: Column of df. Sort tracks by the median within group. (default: 'rna_pseudotime')
    :param relabeling: Dict to rename track names.
    :param smooth_events: Plot kde of event rates instead of IGV if not None.
        Use value as bw_method in scipy.stats.gaussian_kde.
    :param max_res: Maximum grid resolution when plotting smooth_events.
    """
    import plotnine as p9
    if color is not None and group is None:
        group = color
    elif color is not None:
        if (df.groupby(group)[color].nunique() > 1).any():
            raise ValueError(
                f"More than one color per group. group: {group} color: {color}"
            )
    mdf = df.copy()
    if sort_by is not None and group is not None:
        sorted_cats = mdf.groupby(group)[sort_by].median().sort_values().index
        mdf[group] = pd.Categorical(mdf[group], categories=sorted_cats)
    if smooth_events is None:
        is_density = False
        sdf = mdf.sort_values("location")
        sdf["delta"] = np.where(sdf["variable"] == "start", 1, -1)
        if group is None:
            sdf["igv"] = sdf["delta"].cumsum()
        else:
            sdf["igv"] = sdf.groupby(group)["delta"].cumsum()
            if min_counts > 0:
                maxes = sdf.groupby(group)["igv"].transform(len) / 2
            if min_counts > 0:
                idx = maxes >= min_counts
                if not any(idx):
                    raise ValueError(
                        f"No group is above the min_counts of: {min_counts}"
                    )
                sdf = sdf.loc[idx, :]
        plot_df = sdf.copy()
        if group is None:
            plot_df["igv"] = plot_df["igv"].shift(1)
        else:
            plot_df["igv"] = plot_df.groupby(group)["igv"].shift(1)
        plot_df = plot_df.append(sdf)
        pl = (
            p9.ggplot(plot_df, p9.aes("location", ymin=0, ymax="igv"))
            + p9.geom_ribbon()
        )
    else:
        is_density = True
        lower_bound = mdf["location"].min() - 1
        upper_bound = mdf["location"].max() + 1
        grid = np.arange(lower_bound, upper_bound + 1)
        if max_res < len(grid):
            grid_idx = np.linspace(0, upper_bound - lower_bound, max_res).astype(int)
        else:
            grid_idx = slice(None)

        from KDEpy import FFTKDE
        if group is None:
            kernel = FFTKDE(kernel="gaussian", bw=smooth_events)
            kernel = kernel.fit(mdf["location"].values)
            values = kernel.evaluate(grid)
            factor = len(ldf["location"].values) * 100
            plot_df = pd.DataFrame(
                {
                    "density": kernel.evaluate(grid)[grid_idx],
                    "location": grid[grid_idx] * factor,
                }
            )
            if min_density is not None:
                plot_df.loc[plot_df["density"] < min_density, "density"] = min_density
        else:
            groups = mdf[group].unique()
            signals = list()
            for g in groups:
                idx = mdf[group] == g
                ldf = mdf.loc[idx, :]
                kernel = FFTKDE(kernel="gaussian", bw=smooth_events)
                kernel = kernel.fit(ldf["location"].values)
                values = kernel.evaluate(grid)[grid_idx]
                factor = len(ldf["location"].values) * 100
                signals.append(
                    pd.DataFrame(
                        {
                            "density": values * factor,
                            "location": grid[grid_idx],
                            group: g,
                            color: ldf[color].values[0],
                        }
                    )
                )
            plot_df = pd.concat(signals)
            if min_density is not None:
                plot_df.loc[plot_df["density"] < min_density, "density"] = min_density
            plot_df[group] = plot_df[group].astype(mdf[group].dtype)
            plot_df[color] = plot_df[color].astype(mdf[color].dtype)
        pl = p9.ggplot(
            plot_df, p9.aes("location", "density", ymax="density")
        ) + p9.geom_ribbon(ymin=0)
    if color is not None:
        if border:
            pl = pl + p9.aes(fill=color, color=color)
        else:
            pl = pl + p9.aes(fill=color)
    known_celltypes = {
        "HSC": "#F780BE",
        "HMP": "#E78AC2",
        "MEP": "#A7D75D",
        "Ery": "#4DAF50",
        "CLP": "#E5C397",
        "Mono": "#E50B22",
        "cDC": "#1A7AB2",
        "pDC": "#A5CFE2",
    }
    if color == "celltype" and all(
        ct in known_celltypes for ct in plot_df["celltype"].unique()
    ):
        # use familiar coloring
        (breaks, values) = zip(*known_celltypes.items())
        pl = pl + p9.scale_fill_manual(breaks=breaks, values=values,)
    else:
        pl = pl + p9.scale_fill_hue()

    if group is not None:
        pl += p9.facet_grid(
            f"{group} ~ .", scales=facet_scales, labeller=p9.labeller(relabeling),
        )
    if is_density:
        pl += p9.labs(y="cuts/hbp")
    else:
        pl += p9.labs(y="count")
    return pl


def plot_gtf(
    gtf_dataframe,
    draw_genes=True,
    draw_exons=True,
    draw_transcripts=True,
    draw_codons=True,
    draw_utr=True,
    draw_labels=True,
    padding=200,
):
    """
    Draw features of a gtf DataFrame. Used by Multitool.

    :param gtf_dataframe: A pandas data frame from a gtf parsed by gtfparse.
    :param draw_genes: Bool, if genes should be included.
    :param draw_exons: Bool, if exons should be drwan.
    :param draw_transcripts: Bool, if transcripts should be drawn (in a seperate track).
    :param draw_codons: Bool, if start and stop codon should be marked.
    :param draw_utr: Bool, if UTR should be drawn.
    :param draw_labels: Bool, if labels with gene names should be dran on genes.
    :param padding: Minimal horizontal distance in bp between drawn genes/transcripts.
        If features are too close, they a shifted vertically.
    """
    import plotnine as p9
    arrows = list()
    boxes = list()
    labels = list()
    levels = LevelSet(padding)
    if draw_genes:
        idx = gtf_dataframe["feature"] == "gene"
        genes_df = gtf_dataframe.loc[idx, :].sort_values("start")
        for gene, gene_dat in genes_df.iterrows():
            start = gene_dat["start"]
            end = gene_dat["end"]
            gene_name = gene_dat["gene_name"]
            level = levels.get_free(start, end)
            a_df = pd.DataFrame(
                {
                    "type": "gene",
                    "level": level,
                    "name": gene_name,
                    "start": gene_dat["stranded_start"],
                    "end": gene_dat["stranded_end"],
                    "feature": "interval",
                    "gene": gene_name,
                },
                index=[gene],
            )
            arrows.append(a_df)
            lab_df = pd.DataFrame(
                {
                    "type": "gene",
                    "level": level,
                    "name": gene_name,
                    "position": (gene_dat["start"] + gene_dat["end"]) / 2,
                    "gene": gene_name,
                },
                index=[gene],
            )
            labels.append(lab_df)
            most_exons = 1
            exon_hight = 0.6
            idx = gtf_dataframe["gene_name"] == gene_name
            gene_df = gtf_dataframe.loc[idx, :]
            if draw_exons:
                exon_entries = gene_df.loc[gene_df["feature"] == "exon", :]
                count_df = pd.concat(
                    [
                        pd.DataFrame({"location": exon_entries["start"], "delta": 1}),
                        pd.DataFrame({"location": exon_entries["end"], "delta": -1}),
                    ]
                )
                count_df = count_df.sort_values("location")
                signal = count_df["delta"].cumsum()
                count_df["box_up"] = level + exon_hight
                count_df["box_down"] = level - exon_hight
                most_exons = max(most_exons, signal.max())
                box_df = count_df.copy()
                box_df["start"] = box_df["location"]
                box_df["end"] = box_df["location"].shift(-1)
                box_df = box_df.loc[signal != 0, :]
                box_df["feature"] = "exon"
                box_df["type"] = "gene"
                box_df["gene"] = gene_name
                boxes.append(box_df)
            if draw_codons:
                hight = 1.5 * exon_hight
                starts = gene_df.loc[gene_df["feature"] == "start_codon", :].index
                for entry in starts:
                    codon_dat = gene_df.loc[entry, :]
                    box_df = pd.DataFrame(
                        {
                            "start": codon_dat["start"],
                            "end": codon_dat["end"],
                            "feature": "start codon",
                            "level": level,
                            "box_up": level + hight,
                            "box_down": level - hight,
                            "type": "gene",
                            "gene": gene_name,
                        },
                        index=[entry],
                    )
                    boxes.append(box_df)
                stops = gene_df.loc[gene_df["feature"] == "stop_codon", :].index
                for entry in stops:
                    codon_dat = gene_df.loc[entry, :]
                    box_df = pd.DataFrame(
                        {
                            "start": codon_dat["start"],
                            "end": codon_dat["end"],
                            "feature": "stop codon",
                            "level": level,
                            "box_up": level + hight,
                            "box_down": level - hight,
                            "type": "gene",
                            "gene": gene_name,
                        },
                        index=[entry],
                    )
                    boxes.append(box_df)
            if draw_utr:
                hight = 0.5 * exon_hight
                starts = gene_df.loc[gene_df["feature"] == "UTR", :].index
                for entry in starts:
                    utr_dat = gene_df.loc[entry, :]
                    box_df = pd.DataFrame(
                        {
                            "start": utr_dat["start"],
                            "end": utr_dat["end"],
                            "feature": "UTR",
                            "level": level,
                            "box_up": level + hight,
                            "box_down": level - hight,
                            "type": "gene",
                            "gene": gene_name,
                        },
                        index=[entry],
                    )
                    boxes.append(box_df)
    levels.reset()
    if draw_transcripts:
        idx = gtf_dataframe["feature"] == "transcript"
        tf_df = gtf_dataframe.loc[idx, :].sort_values("start")
        tf_shift = 0
        for trans in tf_df.index:
            trans_dat = tf_df.loc[trans, :]
            tf_name = trans_dat["transcript_id"]
            start = trans_dat["start"]
            end = trans_dat["end"]
            gene = trans_dat["gene_name"]
            level = levels.get_free(start, end)
            a_df = pd.DataFrame(
                {
                    "type": "transcript",
                    "level": level,
                    "name": tf_name,
                    "start": trans_dat["stranded_start"],
                    "end": trans_dat["stranded_end"],
                    "feature": "interval",
                    "gene": gene,
                },
                index=[trans],
            )
            arrows.append(a_df)
            if draw_exons:
                idx = (gene_df["feature"] == "exon") & (
                    gene_df["transcript_id"] == tf_name
                )
                exon_entries = gene_df.loc[idx, :]
                count_df = pd.concat(
                    [
                        pd.DataFrame({"location": exon_entries["start"], "delta": 1}),
                        pd.DataFrame({"location": exon_entries["end"], "delta": -1}),
                    ]
                )
                count_df = count_df.sort_values("location")
                signal = count_df["delta"].cumsum()
                signal = 0.8 * signal / signal.max()
                count_df["box_up"] = level + signal
                count_df["box_down"] = level - signal
                box_df = count_df.copy()
                box_df["start"] = box_df["location"]
                box_df["end"] = box_df["location"].shift(-1)
                box_df = box_df.loc[signal != 0, :]
                box_df["feature"] = "exon"
                box_df["type"] = "transcript"
                box_df["gene"] = gene
                boxes.append(box_df)
    pl = (
        p9.ggplot(p9.aes(fill="feature", color="feature"))
        + p9.facet_grid(f"type ~ .", scales="y_free")
        + p9.labs(x="location", y="tracks")
        + p9.theme(axis_text_y=p9.element_blank())
    )
    if boxes:
        boxes_df = pd.concat(boxes)
        pl = pl + (
            p9.geom_rect(
                boxes_df,
                p9.aes(xmin="start", xmax="end", ymin="box_down", ymax="box_up"),
                alpha=0.5,
            )
        )
    if arrows:
        arrow_df = pd.concat(arrows)
        pl = pl + p9.geom_segment(
            arrow_df,
            p9.aes(x="start", xend="end", y="level", yend="level"),
            size=1,
            arrow=p9.arrow(length=0.2),
        )
    if labels and draw_labels:
        labels_df = pd.concat(labels)
        pl = pl + p9.geom_label(
            labels_df,
            p9.aes("position", "level", label="name"),
            inherit_aes=False,
            alpha=0.7,
        )
    return pl


class DataManager:
    """
    A utility class for sep241 deconvolution.
    """

    def __init__(
        self, fragments_files=dict(), gtf_file=None, show_progress=True, comment="#",
        remove_pcr_duplicates = True,
    ):
        self.gtf_file = gtf_file
        self.fragments_files = fragments_files
        self.max_smooth_igv_resolution = 1000
        self._cache = dict()
        self.show_progress = show_progress
        self.comment = "#"
        self.remove_pcr_duplicates = remove_pcr_duplicates


    def plot_frag_length(self, log_scale=True):
        import plotnine as p9
        alen = self.all_fragments()
        pl = (
            p9.ggplot(df, p9.aes("lengths"))
            + p9.geom_line()
            + p9.ylim(0, None)
            + p9.ggtitle("atac read length distribution")
        )
        if log_scale:
            pl = pl + p9.scale_y_log10()
        return pl

    def plot_genes(self, genes, x="pseudotime", points=True, regression=True, std=True):
        import plotnine as p9
        if isinstance(genes, str):
            genes = genes.split(",")
        if regression:
            dfs = self.prediction(genes, x=x, std=std)
            rdf = pd.concat(dfs)
        if points:
            ad = self.rna_ad
            idx = ad.obs["selected"]
            df = pd.DataFrame({x: ad.obs[x][idx],}, index=ad.obs_names[idx],)
            for gene in genes:
                df[gene] = ad.layers["MAGIC_imputed_data"][idx, ad.var_names == gene]
            pdf = df.melt(id_vars=[x], var_name="gene", value_name="expression")
        pl = p9.ggplot(p9.aes(x, "expression", color="gene"))
        if regression and std:
            pl = pl + p9.geom_ribbon(
                p9.aes(x, ymin="lb", ymax="ub", fill="gene"),
                data=rdf,
                alpha=0.15,
                inherit_aes=False,
            )
        if points:
            pl = pl + p9.geom_point(data=pdf, alpha=0.05)
        if regression:
            pl = pl + p9.geom_line(data=rdf)
        return pl

    @property
    def gtf_file(self):
        return self._gtf_file

    @gtf_file.setter
    def gtf_file(self, gtf_file):
        if gtf_file is not None:
            logger.info("Parsing gtf file...")
            from gtfparse import read_gtf
            self.gtf = read_gtf(gtf_file)
        self._gtf_file = gtf_file

    @property
    def gtf(self):
        return self._gtf

    @gtf.setter
    def gtf(self, gtf):
        gtf["stranded_start"] = np.where(gtf["strand"] == "-", gtf["end"], gtf["start"])
        gtf["stranded_end"] = np.where(gtf["strand"] == "-", gtf["start"], gtf["end"])
        if not gtf["seqname"][0].startswith("chr"):
            gtf["seqname"] = 'chr' + gtf["seqname"]
        self.gene_names = set(gtf["gene_name"].unique())
        self._gtf = gtf

    def _interpret_feature(self, feature):
        try:
            return read_region_string(feature)
        except NotARegion:
            pass
        gene_gtf = self.gtf.loc[self.gtf["feature"] == "gene", :]
        if feature in self.gene_names:
            idx = np.where(gene_gtf["gene_name"] == feature)
            dat = gene_gtf.iloc[idx[0], :][["seqname", "start", "end"]]
            return dat.values.ravel()
        idx = np.where(gene_gtf["gene_id"] == feature)[0]
        if len(idx) > 0:
            dat = gene_gtf.iloc[idx[0], :][["seqname", "start", "end"]]
            return dat.values.ravel()
        idx = self.gtf["transcript_id"] == feature
        if any(idx):
            idx = np.where(idx & self.gtf["feature"] == "transcript")
            dat = self.gtf.iloc[idx[0], :][["seqname", "start", "end"]]
            return dat.values.ravel()
        raise Exception(f"Feature not recognized: {feature}")

    def get_fragments(
        self,
        feature,
        max_dist=0,
        include_events=True,
        nuc_size=120,
        _return_bounds=False,
        with_name=False,
    ):
        seq, start, end = self._interpret_feature(feature)
        if not seq.startswith("chr"):
            seq = f"chr{seq}"
        lower_bound = start - max_dist
        upper_bound = end + max_dist
        region = "%s:%d-%d" % (seq, lower_bound, upper_bound)
        atac_intervals = self.get_intervals(region, nuc_size=nuc_size, with_name=with_name)
        atac_intervals["seqname"] = seq
        if not include_events:
            return atac_intervals
        all_evens = self.envents_from_intervals(atac_intervals)
        if _return_bounds:
            return atac_intervals, all_evens, lower_bound, upper_bound
        return atac_intervals, all_evens

    def plot_igv(
        self,
        feature,
        max_dist=0,
        nuc_size=120,
        group=None,
        color="from",
        group_nucs=True,
        sort_by=None,
        obs_from=None,
        y_scale="free",
        draw_peaks=False,
        min_counts=0,
        min_density=1e-10,
        smooth_events_bw=None,
        mark_nucs=list(),
        anno_plot=True,
        color_anno_by="gene",
        full_label=False,
        igv_border=True,
        **kwargs,
    ):
        """
        A plot similar to the IGV visualization of ATAC reads.

        :param feature: A gene, genomic interval or df of ATAC reads.
        :param max_distance: Maximal distance around the selected feature to show.
        :param nuc_size: Length of ATAC reads at which to classify as nucleosome reads.
        :param group: Used in ggplot for faceting. Must be a subset of `color`\.
        :param color: Used in ggplot for coloring of tracks. Must be a superset of `group`\.
        :param group_nucs: Put nucleosome reads in a subgroup.
        :param sort_by: Sort groups and color by this column.
        :param obs_from: Select annData object. Must be `ATAC` or `RNA`\.
        :param y_scale: Must be `fixed` or `free`\.
        :param draw_peaks: Mark peaks selected in ATAC annData object.
        :param min_counts: Tracks with less than that will not be plotted.
        :param smooth_events_bw: If set to number draw event kde with that bw instead of igv signal.
        :param mark_nucs: List of nucleosome positions to mark.
        :param anno_plot: Call plot_gtf to show genes and genomic features.
        :param color_anno_by: E.g. `features` or `genes`\.
        :param full_label: Set to True to write full label into facet headers.
        :param \*\*kwargs: Args passed to plot_gtf.
        """
        if isinstance(feature, str):
            intervals, events, lower_bound, upper_bound = self.get_fragments(
                feature, max_dist=max_dist, nuc_size=nuc_size, _return_bounds=True,
            )
        else:
            events = feature
            max_dist = 0
            lower_bound = events["location"].min()
            upper_bound = events["location"].max()
        seq = events["seqname"][0]
        if y_scale == "free":
            facet_scales = "free_y"
        elif y_scale == "fixed":
            facet_scales = "fixed"
        else:
            raise ValueError(f"Argument for y_scale unknown: {y_scale}")

        if color in events.columns and group in events.columns:
            # we do not need to merge
            obs_from = None
        if obs_from is not None:
            if obs_from == "atac":
                obs = self.atac_ad.obs
            elif obs_from == "rna":
                obs = self.rna_ad.obs
            else:
                raise ValueError(f"Unrecognized obs_from: {obs_from}")
            events = pd.merge(
                events,
                obs,
                suffixes=("", obs_from),
                left_on="name",
                right_index=True,
                how="left",
            )
        if group is None and color is not None:
            group = color
        if min_counts > 0:
            idx = events.groupby(group)[group].transform(len) >= min_counts
            events = events.loc[idx, :]
        if sort_by is not None:
            if group is not None:
                presorted_group_cats = (
                    events.groupby(group)[sort_by].median().sort_values().index
                )
                if not group_nucs:
                    events[group] = pd.Categorical(
                        events[group], categories=presorted_group_cats
                    )
            if color is not None:
                sorted_cats = (
                    events.groupby(color)[sort_by].median().sort_values().index
                )
                events[color] = pd.Categorical(events[color], categories=sorted_cats)
        elif group is not None:
            presorted_group_cats = events[group].unique()

        relabeling = dict()
        if group_nucs:
            sorted_types = sorted(events["read_type"].unique(), reverse=True)
            if group is None:
                grouped = events["read_type"]
                sorted_group_cats = sorted_types
            else:
                grouped = events[group].astype(str) + " " + events["read_type"]
                sorted_group_cats = list()
                for ct in presorted_group_cats:
                    sorted_group_cats += [f"{ct} {t}" for t in sorted_types]
                    if full_label is not True:
                        relabeling.update({f"{ct} {t}": t for t in sorted_types})
            group = "_igv_group"
            events[group] = pd.Categorical(grouped, categories=sorted_group_cats)

        import plotnine as p9
        pl = igv_plot(
            events,
            group=group,
            color=color,
            sort_by=sort_by,
            facet_scales=facet_scales,
            min_counts=min_counts,
            min_density=min_density,
            relabeling=relabeling,
            smooth_events=smooth_events_bw,
            max_res=self.max_smooth_igv_resolution,
            border=igv_border,
        )
        pl = (
            pl
            + p9.coord_cartesian(xlim=(lower_bound, upper_bound))
            + p9.ggtitle(f"{seq}:{lower_bound:,}-{upper_bound:,}")
        )

        if draw_peaks:
            atac = self.atac_ad.var
            peaks_idx = (
                (atac["midpoint"] > lower_bound)
                & (atac["midpoint"] < upper_bound)
                & (atac["seqnames"] == seq)
            )
            atac = atac.loc[peaks_idx, :]
            pl = (
                pl
                + p9.geom_vline(
                    atac, p9.aes(xintercept="start"), linetype="solid", alpha=0.5
                )
                + p9.geom_vline(
                    atac, p9.aes(xintercept="end"), linetype="dashed", alpha=0.5
                )
            )
        if mark_nucs:
            shift = np.ceil(nuc_size / 2)
            nuc_starts = [mid - shift for mid in mark_nucs]
            nuc_ends = [p + nuc_size for p in nuc_starts]
            nucs = pd.DataFrame({"start": nuc_starts, "end": nuc_ends, "null": 0})
            pl = pl + p9.geom_segment(
                nucs,
                p9.aes("start", "null", xend="end", yend="null"),
                inherit_aes=False,
            )
        if not anno_plot:
            return pl

        # if seq.startswith('chr'):
        #    seq = seq[3:]
        idx = (
            (self.gtf["end"] > lower_bound)
            & (self.gtf["start"] < upper_bound)
            & (self.gtf["seqname"] == seq)
        )
        df = self.gtf.loc[idx, :]
        padding = (upper_bound - lower_bound) * 0.05
        kwargs.setdefault("padding", padding)
        kwargs.setdefault("draw_labels", True)
        kwargs.setdefault("draw_transcripts", False)
        ofs = p9.options.figure_size
        if kwargs.get("draw_transcripts"):
            vsize = 0.6 * ofs[0]
        else:
            vsize = 0.3 * ofs[0]
        if not any(idx):
            anno_pl = p9.ggplot()
        else:
            anno_pl = plot_gtf(df, **kwargs)
        anno_pl = (
            anno_pl
            + p9.theme(figure_size=(ofs[0], vsize))
            + p9.coord_cartesian(xlim=(lower_bound, upper_bound))
            + p9.aes(color=color_anno_by, fill=color_anno_by)
        )
        if draw_peaks:
            anno_pl = (
                anno_pl
                + p9.geom_vline(
                    atac, p9.aes(xintercept="start"), linetype="solid", alpha=0.5
                )
                + p9.geom_vline(
                    atac, p9.aes(xintercept="end"), linetype="dashed", alpha=0.5
                )
            )
        if mark_nucs:
            anno_pl = anno_pl + p9.geom_segment(
                nucs,
                p9.aes("start", "null", xend="end", yend="null"),
                inherit_aes=False,
            )
        return pl, anno_pl

    file_length = {}

    def _all_of_file(self, file, rep):
        comment = self.comment
        h = (
            ("f", "_all_of_file"),
            ("file", file),
            ("rep", rep),
            ("comment", comment),
        )
        if (result := self._cache.get(h)) is not None:
            return result
        reads = list()
        if file.endswith("gz"):
            logger.debug("Assuming gzip format for file %s", file)
            import gzip
            openfn = gzip.open
        elif file.endswith("gz2") or file.endswith("bz2"):
            logger.debug("Assuming bzip2 format for file %s", file)
            import bz2
            openfn = bz2.open
        else:
            logger.debug("Assuming no compression for file %s", file)
            openfn = open
        with openfn(file, "rb") as buff:
            absfile = os.path.abspath(file)
            if absfile in self.file_length:
                lines = tqdm(
                    buff,
                    desc="reads",
                    total=self.file_length[absfile],
                    disable=not self.show_progress,
                )
            else:
                lines = tqdm(buff, desc="reads", disable=not self.show_progress)
            n_fileds = None
            en_lines = enumerate(lines)
            for i, line in enumerate(lines):
                if n_fileds is None:
                    if not line.decode().strip().startswith(comment):
                        n_fileds = min(4, len(line.decode().split("\t")))
                        logger.debug("Using %d columns of %s.", n_fileds, file)
                if n_fileds == 3:
                    seq, s, e = line.decode().strip().split("\t")[:3]
                    start = int(s)
                    end = int(e)
                    reads.append((rep, seq, start, end, f"read-{i}"))
                elif n_fileds == 4:
                    seq, s, e, bc = line.decode().strip().split("\t")[:4]
                    start = int(s)
                    end = int(e)
                    reads.append((rep, seq, start, end, bc))
                elif line.decode().strip().startswith(comment):
                    continue
                else:
                    logger.error("Line %d of %s: %s", i, file, line)
                    raise BEDFormatError(f"Less than 3 columns in bed file {file}.")
        self._cache[h] = reads
        return reads

    def all_fragments(self, barcode=None, replace_barcode=None):
        reads = list()
        all_lengths = list()
        for rep, file in self.fragments_files.items():
            logger.info(f"Reading sample {rep}")
            reads += self._all_of_file(file, rep)
        logger.info("Making data frame.")
        result_df = pd.DataFrame(
            reads, columns=["from", "seqname", "start", "end", "name"]
        )
        if barcode:
            if barcode not in ["from", "name"]:
                raise BarcodeError("Barcode '%s' must be 'from' or 'name'.", barcode)
            result_df["barcode"] = result_df[barcode]
        total_reads = len(result_df)
        if replace_barcode:
            repl = lambda m: m.group(1)
            result_df['barcode'] = result_df['barcode'].str.replace(replace_barcode, repl, regex=True)
        if self.remove_pcr_duplicates:
            if "barcode" in result_df.columns:
                result_df = result_df.drop_duplicates(subset=["barcode", "seqname", "start", "end"])
            else:
                result_df = result_df.drop_duplicates()
            n_removed = total_reads - len(result_df)
            faction = n_removed / total_reads
            logger.info(f"Removed {n_removed:,} of {total_reads:,} ({faction:.2%}) duplicate reads.")
            total_reads = len(result_df)
        result_df["start"] = result_df["start"].astype(int)
        result_df["end"] = result_df["end"].astype(int)
        logger.info(f"Calculating read lengths for all {total_reads:,} reads.")
        result_df["length"] = result_df["end"] - result_df["start"]
        return result_df

    def get_intervals(self, region, nuc_size=120, with_name=False):
        intervals = list()
        import tabix
        for rep, fragments_file in self.fragments_files.items():
            tb = tabix.open(fragments_file)
            records = tb.querys(region)
            for record in records:
                if with_name:
                    intervals.append((rep, record[0], int(record[1]), int(record[2]), record[3]))
                else:
                    intervals.append((rep, record[0], int(record[1]), int(record[2])))
        if with_name:
            intervals_df = pd.DataFrame(intervals, columns=["from", "seq", "start", "end", "barcode"])
        else:
            intervals_df = pd.DataFrame(intervals, columns=["from", "seq", "start", "end"])
        intervals_df["length"] = np.abs(intervals_df["end"] - intervals_df["start"])
        intervals_df["read_type"] = np.where(
            intervals_df["length"] > nuc_size, f">{nuc_size}", f"≤{nuc_size}"
        )
        intervals_df["long_type"] = np.where(
            intervals_df["length"] > 2 * nuc_size,
            f">{2*nuc_size}",
            intervals_df["read_type"],
        )
        return intervals_df

    def envents_from_intervals(self, interval_df):
        id_vars = set(interval_df.columns) - {"start", "end"}
        df = interval_df.melt(id_vars=id_vars, value_name="location")
        return df

    def count(self, bins, index_name=None, normalize=False, count_prior=0):
        midpoints = (bins[1:] + bins[:-1]) / 2

        def do_count(data):
            cumhist = (data.values[None, :] > bins[:, None]).sum(axis=1)
            counts = np.diff(-cumhist)
            result = pd.Series(counts, index=midpoints)
            if index_name is not None:
                result.index.name = index_name
            result += count_prior
            if normalize:
                result /= result.sum()
            return result

        return do_count

    def _length_hist_bins(self, maximum, bin_width=1, padding=0.5):
        return np.arange(padding, maximum + padding + bin_width, bin_width)

    def hist_from_df(
        self,
        len_df,
        bin_width=1,
        normalize=True,
        color="from",
        count_prior=0,
        bins=None,
    ):
        if bins is None:
            bins = self._length_hist_bins(len_df["length"].max(), bin_width)
        cf = self.count(bins, "length", normalize=normalize, count_prior=count_prior)
        gdf = len_df.groupby(color)["length"].apply(cf)
        if normalize:
            gdf.name = "density"
        else:
            gdf.name = "count"
        return gdf

    def length_histograms(
        self, bin_width=1, normalize=True, color="from", count_prior=0, bins=None
    ):
        len_df = self.all_fragments()
        return self.hist_from_df(
            len_df,
            bin_width=bin_width,
            normalize=normalize,
            color=color,
            count_prior=count_prior,
            bins=bins,
        )

    def plot_length_hist(
        self,
        gdf=None,
        bin_width=1,
        normalize=True,
        color="from",
        count_prior=0,
        bins=None,
    ):
        import plotnine as p9
        if gdf is None:
            gdf = self.length_histograms(
                bin_width=bin_width,
                normalize=normalize,
                color=color,
                count_prior=count_prior,
                bins=bins,
            )
        ylab = gdf.name
        pdata = pd.DataFrame(gdf).reset_index()
        pl = (
            p9.ggplot(pdata, p9.aes("length", ylab, ymax=ylab, color=color, fill=color))
            + p9.geom_ribbon(ymin=0, alpha=0.1)
            + p9.ggtitle(f"length bin width {bin_width}")
        )
        return pl
