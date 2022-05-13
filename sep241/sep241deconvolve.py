#!/usr/bin/env python

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
from inspect import stack

from sep241util import setup_logging, logger, set_flag, format_cuts


desc = """Deconvolve CUT&TAG 2for1. Uses the output of the sep241prep step.

This step computes two likelihoods for all cuts (both ends of each fragment) for
them to originate from one of two channels. By default, we expect that
channel 1 contains smaller fragments, e.g., Pol2S5p induced, than channel 2,
e.g., H3K27me3 induced. This is controlled by ``--c1-dirichlet-prior`` and
``--c2-dirichlet-prior``. The dirichlet prior specifies weights for multiple
log-normal distributions that make up a mixture distribution describing the prior
fragment-length distribution that typically asserts a ladder of distinct modes.

The mode position and width is shared between both channels and controlled
by ``--length-dist-modes`` and ``--length-dist-mode-sds``.
The default setting further assumes that co-occurrence of cuts in channel 1 is
more concentrated than channel 2, in which cuts are assumed to be more spread
out. This is controlled through the covariance functions of the respective
position marginal log-likelihood functions ``--c1-cov`` and ``--c2-cov``.

A single instance of the deconvolution process deconvolves one workchunk
specified by ``--workchunk``. Either of the following code snippets run all the
deconvolutions at once, where each variable with args. is an argument
to the deconvolution step and ``[N]`` is the highest workchunk id.

Run deconvolution using `slurm <https://slurm.schedmd.com>`_::

  sbatch --array=0-[N] --mem=[memory target] <(sep241deconvolve_sbatch) [jobdata pkl file]

Run in bash or zsh::

  for wc in $(seq 0 [N]); do
      sep241deconvolve [jobdata pkl file] --workchunk $wc
  done

The preparation step tries to create workgroups that will take the
target memory demand, but this is only an estimate. If the memory demand
of the deconvolution step is prohibitively high, try rerunning the
preparation step with a lower memory target.
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
    help="""Covariance function of component 1 using pymc
    covariance functions without the `input_dim` argument:
    https://docs.pymc.io/api/gp/cov.html
    (default=Matern32(500))
    """,
    type=str,
    default="Matern32(500)",
    metavar="'a*CovFunc(args) + ...'",
)
parser.add_argument(
    "--c2-cov",
    help="""Covariance function of component 2 using pymc
    covariance functions without the `input_dim` argument:
    https://docs.pymc.io/api/gp/cov.html
    (default=Matern32(2000))
    """,
    type=str,
    default="Matern32(2000)",
    metavar="'a*CovFunc(args) + ...'",
)
parser.add_argument(
    "--sparsity-threshold",
    help="""The Gaussian processes modeling the location distribution use a custom
    covariance matrix that saves time and memory by representing only
    covariances above this threshold.
    (default=1e-8)""",
    type=float,
    default=1e-8,
    metavar="float",
)
parser.add_argument(
    "--length-dist-modes",
    help="""Modes of the log-normal distributions used in the
    mixture model for the fragment length distribution.
    (default=70 200 400 600)
    """,
    type=float,
    nargs="+",
    default=[70, 200, 400, 600],
    metavar="floats",
)
parser.add_argument(
    "--length-dist-mode-sds",
    help="""Standard deviations of the log-normal modes of
    the mixture model for the fragment length distribution.
    (default=0.29 0.18 0.15 0.085)
    """,
    type=float,
    nargs="+",
    default=[0.29, 0.18, 0.15, 0.085],
    metavar="floats",
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
    metavar="floats",
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
    metavar="floats",
)
parser.add_argument(
    "--constrain",
    help="""Use this flag to constrain average fragment length to be larger in one component.
    (default=False)""",
    action="store_true",
)
parser.add_argument(
    "--cores",
    help="""Number of CPUs to use for inference.
    Defaults to the environment variable SLURM_CPUS_PER_TASK if set.""",
    type=int,
    default=os.getenv("SLURM_CPUS_PER_TASK"),
    metavar="int",
)
parser.add_argument(
    "--compiledir",
    help="""Directory to compile code in.
    Should be different between parallel run instances
    and can be deleted after run.
    Defaults to sep241tmp/{args.jobdata}/deconvolve/{args.workchunk} or if the environment variable TMPDIR is set, defaults to TMPDIR/sep241tmp/{args.jobdata}/deconvolve/{args.workchunk}""",
    type=str,
    metavar="dir",
)


caller = stack()[-1].filename
caller = caller[caller.rindex('/'):]

if __name__ == "__main__" or caller == "/sep241deconvolve":
    args = parser.parse_args()
    tmpdir = os.getenv("TMPDIR")
    if args.compiledir is not None:
        compileDir = args.compiledir
    elif tmpdir is None:
        compileDir = f"./sep241tmp/{args.jobdata}/deconvolve/{args.workchunk}"
    else:
        compileDir = os.path.join(tmpdir, f"sep241tmp/{args.jobdata}/deconvolve/{args.workchunk}")
    set_flag("base_compiledir", compileDir)

import theano
import pymc3 as pm
import theano.tensor as at

from sep241covariance import SparseCov
from sep241latent import SparseLatent, use_sksparse
from pymc3.theanof import inputvars
from pymc3.model import Point, modelcontext
from pymc3.blocking import DictToArrayBijection, ArrayOrdering
from scipy.optimize import Bounds

if __name__ == "__main__" or caller == "/sep241deconvolve":
    if not use_sksparse:
        logger.info("""Scikit-sparse is not found; defaulting to a more expensive
                    cholesky decomposition implementation.""")

class NoData(Exception):
    pass


class ArgumentError(Exception):
    pass


class BadCovariance(Exception):
    pass


def safety_check(code):
    """
    Checks that the code is safe to execute. If the code is dangerous, raises
    an exception. The converse is not true. If the code is a valid covariance
    function, does not raise an exception. The converse is not true.
    """
    free = "([ |\(|\)]*)"
    dec = f"({free}((\.[0-9]+)|([0-9]+(\.[0-9]+)?)){free})"
    cov = f"({free}(pm.gp.cov.[0-9a-zA-Z]*\({dec}(,{dec})*\)){free})"
    values = f"({dec}|{cov})"
    operators = "(\+|\*|\*\*)"
    expression = f"{values}({operators}{values})*"

    if not re.fullmatch(expression, code):
        raise BadCovariance()


def get_cov_functions(c1_cov_string, c2_cov_string, threshold):
    p = re.compile(r"([a-zA-Z][a-zA-Z0-9]*\()")
    c1_code = p.sub(r"pm.gp.cov.\g<1>1, ", c1_cov_string)
    c2_code = p.sub(r"pm.gp.cov.\g<1>1, ", c2_cov_string)
    covs_wo_input_dims = {
        "WhiteNoise(": r"WhiteNoise\(1, ",
        "Constant(": r"Constant\(1, ",
        "Kron(": r"Kron\(1, ",
    }
    for repl, pattern in covs_wo_input_dims.items():
        c1_code = re.sub(pattern, repl, c1_code)
        c2_code = re.sub(pattern, repl, c2_code)
    try:
        safety_check(c1_code)
        c1_cov = eval(c1_code)
    except Exception:
        raise BadCovariance(
            "Error parsing the covariance function string "
            f'for component 1: "{c1_cov_string}"'
        )
    try:
        safety_check(c2_code)
        c2_cov = eval(c2_code)
    except Exception:
        raise BadCovariance(
            "Error parsing the covariance function string "
            f'for component 2: "{c2_cov_string}"'
        )

    wrap_1 = SparseCov(c1_cov, threshold=threshold)
    wrap_2 = SparseCov(c2_cov, threshold=threshold)
    return [wrap_1, wrap_2]


def get_length_dist_modes(modes=[70, 200, 400, 600], sigmas=[0.29, 0.18, 0.15, 0.085]):
    def generator():
        mus = [np.log(modes[i]) + (s ** 2) for i, s in enumerate(sigmas)]
        length_comps = list()
        for i in range(len(mus)):
            mode = pm.Lognormal.dist(mus[i], sigma=sigmas[i], testval=modes[i])
            length_comps.append(mode)
        return length_comps

    return generator


def make_bounds():
    k = 9.458853214586302
    model = modelcontext(None)
    lpoint = Point(model.test_point, model=model)
    upoint = Point(model.test_point, model=model)
    for name, value in lpoint.items():
        if name == "weight_c1_stickbreaking__" or name == "weight_c2_stickbreaking__":
            lpoint[name] = np.full(value.shape, -k)
            upoint[name] = np.full(value.shape, k)
        else:
            lpoint[name] = np.full(value.shape, -np.inf)
            upoint[name] = np.full(value.shape, np.inf)

    vars = model.cont_vars
    vars = inputvars(vars)

    lbij = DictToArrayBijection(ArrayOrdering(vars), lpoint)
    ubij = DictToArrayBijection(ArrayOrdering(vars), upoint)

    lb = lbij.map(lpoint)
    ub = ubij.map(upoint)
    return Bounds(lb, ub, True)


def make_model(
    events,
    cov_functions,
    dirichlet_priors,
    length_comps_gen,
    mlevel=0,
    constrain=False,
):
    logger.info("Compiling model.")
    with pm.Model() as model:
        length_comps = length_comps_gen()

        cov_c1 = cov_functions[0]
        gp_c1 = SparseLatent(
            mean_func=pm.gp.mean.Constant(mlevel),
            cov_func=cov_c1,
        )
        w_c1 = dirichlet_priors[0]
        weights_c1 = pm.Dirichlet("weight_c1", w_c1, testval=w_c1 / np.sum(w_c1))
        l_like_c1 = pm.Mixture.dist(w=weights_c1, comp_dists=length_comps[: len(w_c1)])

        cov_c2 = cov_functions[1]
        gp_c2 = SparseLatent(
            mean_func=pm.gp.mean.Constant(mlevel),
            cov_func=cov_c2,
        )
        w_c2 = dirichlet_priors[1]
        weights_c2 = pm.Dirichlet("weight_c2", w_c2, testval=w_c2 / np.sum(w_c2))
        l_like_c2 = pm.Mixture.dist(w=weights_c2, comp_dists=length_comps[: len(w_c2)])

        if constrain:
            logger.debug("Compiling constrain.")
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
            length_diff = means.dot(larger_weights) - means.dot(smaller_weights)
            const = pm.Potential(
                "length_mode_constrain",
                pm.math.switch(length_diff > 0, 0, -length_diff * 1e99),
            )

        n_intervals = len(events)
        events["par_work"] = events["nlocs"].astype(float) * events["ncuts"].astype(
            float
        )
        events = events.sort_values(by="par_work", ascending=False)
        for i, (name, dat) in enumerate(events.iterrows()):
                unique_locations, log_loc_weight, interleave_index = format_cuts(
                    dat["cuts"]["location"]
                )
                lengths = dat["cuts"]["length"].values
                logger.info(
                    "Parameterizing interval (%d/%d) %s with %i cuts in %i locations.",
                    i + 1,
                    n_intervals,
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


def gradient1_custom(f, v):
    theano.config.compute_test_value = "off"
    result = at.flatten(at.grad(f, v, disconnected_inputs="warn"))
    return result


def main():
    setup_logging(args.logLevel, args.logfile)
    logger.info("Running whole_genome_deconvolve_job with arguments:")
    logger.info(args)
    logger.debug("Theano flags: %s", os.getenv("THEANO_FLAGS"))

    if not args.cores:
        cores = multiprocessing.cpu_count()
        logger.info("Detecting %s compute cores.", cores)
    else:
        cores = args.cores
    logger.info("Theano compile directory set to %s", theano.config.compiledir)
    logger.info("Limiting computation to %i cores.", cores)
    threadpool_limits(limits=int(cores))
    os.environ["NUMEXPR_MAX_THREADS"] = str(cores)
    logger.info("Preparing dirichlet prior.")
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
    cov_functions = get_cov_functions(args.c1_cov, args.c2_cov, args.sparsity_threshold)
    logger.info("Loading workdata from: %s", args.jobdata)
    workdata = pd.read_pickle(args.jobdata)
    logger.info("Selecting intervals of work chunk %s.", args.workchunk)
    idx = workdata["workchunk"].astype(str) == args.workchunk
    events = workdata.loc[idx, :]
    del workdata
    if len(events) == 0:
        raise NoData(f"No intervals found for work chunk {args.workchunk}.")
    workload = events["memory"].sum() + 9.4640381e-2
    logger.info(
        "Found %i intervals with a target memory consumption of %s GBs.",
        len(events),
        f"{workload:,.3f}",
    )
    length_comps_gen = get_length_dist_modes(
        args.length_dist_modes, args.length_dist_mode_sds
    )
    model = make_model(
        events,
        cov_functions,
        dirichlet_priors,
        length_comps_gen,
        constrain=args.constrain,
    )

    logger.info("Computing MAP.")
    with model:
        pm.theanof.gradient1 = gradient1_custom
        maxlle = pm.find_MAP(bounds=make_bounds())
    out_path = os.path.splitext(args.jobdata)[0] + f"_wg-{args.workchunk}.pkl"
    logger.info("Saving result in %s.", out_path)
    maxlle["arguments"] = args
    with open(out_path, "wb") as fl:
        pickle.dump(maxlle, fl, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Finished successfully.")


if __name__ == "__main__":
    main()
