import os
import re
import pickle

from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd
import multiprocessing

import theano
import pymc3 as pm
import theano.tensor as at

from sep241util import setup_logging, logger, format_cuts
from sep241covariance import SparseCov
from sep241latent import SparseLatent
from pymc3.theanof import inputvars
from pymc3.model import Point, modelcontext
from pymc3.blocking import DictToArrayBijection, ArrayOrdering
from scipy.optimize import Bounds


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


def deconvolve(args):
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