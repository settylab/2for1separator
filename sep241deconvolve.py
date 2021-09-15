#! /usr/bin/env python
#SBATCH --job-name=2for1seperator
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00

import os
import re
import pickle
import warnings
import logging
import argparse

from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing

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


if __name__ == "__main__":
    desc = "Deconvolve CUT&TAG 2for1."
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
        help="Set the logging level (default=INFO)",
        metavar="LEVEL",
    )
    parser.add_argument(
        "--logfile",
        help="Write detailed log to this file.",
        type=str,
        metavar="logfile",
    )
    parser.add_argument(
        "--workchunk",
        help="""Work chunk number of jobdata to handle.
        Defaults to the environment variable SLURM_ARRAY_TASK_ID if set.""",
        type=str,
        default=os.getenv("SLURM_ARRAY_TASK_ID"),
        metavar="int",
    )
    parser.add_argument(
        "--c1-cov",
        help="""Covariance function of component 1 using pymc3
        covariance functions without the `input_dim` argument:
        https://docs.pymc.io/api/gp/cov.html
        (default=Matern32(500))
        """,
        type=str,
        default="Matern32(500)",
        metavar="'a*CovFunc(args) + ...''",
    )
    parser.add_argument(
        "--c2-cov",
        help="""Covariance function of component 2 using pymc3
        covariance functions without the `input_dim` argument:
        https://docs.pymc.io/api/gp/cov.html
        (default=Matern32(2000))
        """,
        type=str,
        default="Matern32(2000)",
        metavar="'a*CovFunc(args) + ...''",
    )
    parser.add_argument(
        "--length-dist-modes",
        help="""Modes of the mixture model for the fragment length distribution.
        (default=70 200 400 600)
        """,
        type=float,
        nargs="+",
        default=[70, 200, 400, 600],
        metavar="int",
    )
    parser.add_argument(
        "--length-dist-mode-sds",
        help="""Standard deviatuions of the log-normal modes of
        the mixture model for the fragment length distribution.
        (default=0.29 0.18 0.15 0.085)
        """,
        type=float,
        nargs="+",
        default=[0.29, 0.18, 0.15, 0.085],
        metavar="int",
    )
    parser.add_argument(
        "--c1-dirichlet-prior",
        help="""Dirichlet prior for the ratio of modes in the
        length distribution.
        (default=450 100 10 1)
        """,
        type=float,
        nargs="+",
        default=[450, 100, 10, 1],
        metavar="int",
    )
    parser.add_argument(
        "--c2-dirichlet-prior",
        help="""Dirichlet prior for the ratio of modes in the
        length distribution.
        (default=150 300 50 10)
        """,
        type=float,
        nargs="+",
        default=[150, 300, 50, 10],
        metavar="int",
    )
    parser.add_argument(
        "--constraint",
        help="Constraint average fragemnt length to be larger in one component.",
        action="store_true",
    )
    parser.add_argument(
        "--cores",
        help="""Number of CPUs to use for inference.
            Defaults to the environemnt variabe SLURM_CPUS_PER_TASK if set.""",
        type=int,
        default=os.getenv("SLURM_CPUS_PER_TASK"),
        metavar="int",
    )
    parser.add_argument(
        "--compiledir",
        help="""Directory to compile code in.
            Should be different between parallel run instances
            and can be delted after run.
            (default=sep241tmp or if he environment variable TMPDIR is set
            default=TMPDIR/sep241tmp)""",
        type=str,
        metavar="dir",
    )

    args = parser.parse_args()
    setup_logging(args.logLevel, args.logfile)
    logger.info("Running whole_genome_deconvolve_job with arguments:")
    logger.info(args)
    logging.getLogger("filelock").setLevel(logging.ERROR)  # theano/asrea prints a lot


def setFlag(flag, value=None):
    """
    Description
    -----------
    Sets or overites the theano `flag` in the envirnment variable 'THEANO_FLAGS'.

    Parameter
    ---------
    flag : The flag name that is to be overwritten or set.
    value : The value to be asigned to the flag. If it is
    `None` then `flag` will be pasted as is into 'THEANO_FLAGS'.

    Value
    -----
    The new value of 'THEANO_FLAGS'.
    """
    if not isinstance(flag, str):
        raise TypeError("The arrgument `flag` needs to be a string.")
    if "THEANO_FLAGS" in os.environ:
        flagString = os.getenv("THEANO_FLAGS")
    else:
        flagString = ""

    if value is None:
        newFlagString = flagString + "," + flag
        os.environ["THEANO_FLAGS"] = newFlagString
        return newFlagString

    if not isinstance(value, str):
        raise TypeError("The arrgument `value` needs to be a string or `None`.")

    oldFlags = flagString.split(",")
    flagTag = flag + "="
    newFlags = [s for s in oldFlags if flagTag not in s]
    newFlags.append(flagTag + value)
    newFlagString = ",".join(newFlags)
    os.environ["THEANO_FLAGS"] = newFlagString
    return newFlagString


tmpdir = os.getenv("TMPDIR")
if args.compiledir is not None:
    compileDir = args.compiledir
elif tmpdir is None:
    compileDir = "./sep241tmp"
else:
    compileDir = os.path.join(tmpdir, "sep241tmp")
setFlag("base_compiledir", compileDir)
# setFlag("blas.ldflags", "'-L/usr/lib/ -lblas'")

import theano

theano.config.mode = "FAST_COMPILE"
theano.config.optimizer = "fast_compile"

import pymc3 as pm
import theano.tensor as at


class NoData(Exception):
    pass


class ArgumentError(Exception):
    pass


class BadCovariance(Exception):
    pass


def get_cov_functions(c1_cov_string, c2_cov_string):
    p = re.compile(r"([a-zA-Z][a-zA-Z0-9]*\()")
    c1_code = p.sub(r"pm.gp.cov.\g<1>1, ", c1_cov_string)
    c2_code = p.sub(r"pm.gp.cov.\g<1>1, ", c2_cov_string)
    try:
        c1_cov = eval(c1_code)
    except Exception:
        raise BadCovariance(
            "Error parsing the covariance function string "
            f'for component 1: "{c1_cov_string}"'
        )
    try:
        c2_cov = eval(c2_code)
    except Exception:
        raise BadCovariance(
            "Error parsing the covariance function string "
            f'for component 1: "{c2_cov_string}"'
        )
    return [c1_cov, c2_cov]


def get_length_dist_modes(modes=[70, 200, 400, 600], sigmas=[0.29, 0.18, 0.15, 0.085]):
    mus = [np.log(modes[i]) + (s ** 2) for i, s in enumerate(sigmas)]
    length_comps = list()
    for i in range(len(mus)):
        mode = pm.Lognormal.dist(mus[i], sigma=sigmas[i], testval=modes[i])
        length_comps.append(mode)
    return length_comps


def format_cuts(locations_ds):
    """Fromats cuts as represented in the model."""

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

    return unique_locations, log_loc_weight, interleave_index


def make_model(
    events, cov_functions, dirichlet_priors, length_comps, mlevel=0, constraint=False
):
    logger.info("Compiling model.")
    with pm.Model() as model:
        cov_c1 = cov_functions[0]
        gp_c1 = pm.gp.Latent(mean_func=pm.gp.mean.Constant(mlevel), cov_func=cov_c1)
        w_c1 = dirichlet_priors[0]
        weights_c1 = pm.Dirichlet("weight_c1", w_c1, testval=w_c1 / np.sum(w_c1))
        l_like_c1 = pm.Mixture.dist(w=weights_c1, comp_dists=length_comps[: len(w_c1)])

        cov_c2 = cov_functions[1]
        gp_c2 = pm.gp.Latent(mean_func=pm.gp.mean.Constant(mlevel), cov_func=cov_c2)
        w_c2 = dirichlet_priors[1]
        weights_c2 = pm.Dirichlet("weight_c2", w_c2, testval=w_c2 / np.sum(w_c2))
        l_like_c2 = pm.Mixture.dist(w=weights_c2, comp_dists=length_comps[: len(w_c2)])

        if constraint:
            logger.debug("Compiling constraint.")
            means = at.stack([a.mean for a in length_comps])
            mean_prior_length_c1 = means.dot(w_c1 / np.sum(w_c1)).eval()
            mean_prior_length_c2 = means.dot(w_c2 / np.sum(w_c2)).eval()
            if mean_prior_length_c1 > mean_prior_length_c2:
                logger.info(
                    "Constraining component 1 to have larger fragments on average."
                )
                larger_weights = weights_c1
                smaller_weights = weights_c2
            else:
                logger.info(
                    "Constraining component 2 to have larger fragments on average."
                )
                larger_weights = weights_c2
                smaller_weights = weights_c1
            length_diff = (means.dot(larger_weights) - means.dot(smaller_weights),)
            const = pm.Potential(
                "length_mode_constraint",
                pm.math.switch(length_diff > 0, 0, -length_diff * 10),
            )

        for name, dat in events.iterrows():

            unique_locations, log_loc_weight, interleave_index = format_cuts(
                dat["cuts"]["location"]
            )
            lengths = dat["cuts"]["length"].values
            logger.info(
                "Compiling intervall %s with %i cuts in %i locations.",
                name,
                len(dat["cuts"]),
                len(unique_locations),
            )

            f_c1 = gp_c1.prior(f"f_c1_{name}", X=unique_locations[:, None])
            logp_c1 = f_c1[interleave_index] + l_like_c1.logp(lengths)
            f_c2 = gp_c2.prior(f"f_c2_{name}", X=unique_locations[:, None])
            logp_c2 = f_c2[interleave_index] + l_like_c2.logp(lengths)

            comb = at.stack([logp_c2, logp_c1], axis=0)
            comb_logp = pm.math.logsumexp(comb, axis=0)
            potential = pm.Potential(f"potential_{name}", comb_logp)

            weight_c1 = at.sum(at.exp(f_c1 + log_loc_weight))
            weight_c2 = at.sum(at.exp(f_c2 + log_loc_weight))
            weight_total = weight_c1 + weight_c2
            n_cuts = len(dat["cuts"])
            t_potential = pm.Potential(
                f"total_{name}",
                pm.Normal.dist(np.log(n_cuts), 0.001).logp(at.log(weight_total)),
            )

    return model


def main(args):
    if not args.cores:
        cores = multiprocessing.cpu_count()
        logger.info("Detecting %s compute cores.", cores)
    else:
        cores = args.cores
    logger.info("Limiting computation to %i cores.", cores)
    threadpool_limits(limits=int(cores))
    os.environ["NUMEXPR_MAX_THREADS"] = str(cores)
    logger.info("Prepairing dirichlet prior.")
    c1_prior = np.array(args.c1_dirichlet_prior)
    if len(c1_prior) < 2:
        raise ArgumentError("Dirichlet prior of c1 needs at least two components.")
    nmodes = len(args.length_dist_modes)
    if len(args.length_dist_mode_sds) != nmodes:
        raise ArgumentError(
            "Number of fragment length distribution mode standard deviations must "
            "equal the number of fragment length distribution modes."
        )
    if len(c1_prior) > nmodes:
        raise ArgumentError(
            "Dirichlet prior of c1 has more than components than modes defined "
            "for the fragment length distribution."
        )
    c2_prior = np.array(args.c2_dirichlet_prior)
    if len(c2_prior) < 2:
        raise ArgumentError("Dirichlet prior of c2 needs at least two components.")
    if len(c2_prior) > nmodes:
        raise ArgumentError(
            "Dirichlet prior of c2 has more than components than modes defined "
            "for the fragment length distribution."
        )
    dirichlet_priors = [c1_prior, c2_prior]
    logger.info("Making covariance functions.")
    cov_functions = get_cov_functions(args.c1_cov, args.c2_cov)
    logger.info("Loading workdata from: %s", args.jobdata)
    workdata = pd.read_pickle(args.jobdata)
    logger.info("Selecting intervalls of work chunk %s.", args.workchunk)
    idx = workdata["workchunk"].astype(str) == args.workchunk
    events = workdata[idx]
    if len(events) == 0:
        raise NoData(f"No intervalls found for work chunk {args.workchunk}.")
    workload = events["memory"].sum()
    logger.info(
        "Found %i intervalls with a target memory consumption of %s GBs.",
        len(events),
        f"{workload:,.3f}",
    )
    length_comps = get_length_dist_modes(
        args.length_dist_modes, args.length_dist_mode_sds
    )
    try:
        model = make_model(
            events,
            cov_functions,
            dirichlet_priors,
            length_comps,
            constraint=args.constraint,
        )
    except Exception as e:
        logger.info(
            "First compilition attempt failed likely due to pymc3 mixture distribution bug."
        )
        length_comps = get_length_dist_modes()
        model = make_model(
            events,
            cov_functions,
            dirichlet_priors,
            length_comps,
            constraint=args.constraint,
        )
    logger.info("Computing MAP.")
    with model:
        maxlle = pm.find_MAP()
    out_path = os.path.splitext(args.jobdata)[0] + f"_wg-{args.workchunk}.pkl"
    logger.info("Saving result in %s.", out_path)
    maxlle["arguments"] = args
    with open(out_path, "wb") as fl:
        pickle.dump(maxlle, fl, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Finished successfully.")


if __name__ == "__main__":
    main(args)
