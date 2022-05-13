#!/usr/bin/env python
#SBATCH --job-name=2for1separator
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00


import os
import argparse

from sep241.sep241util import set_flag


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

  sbatch --array=0-[N] --mem=[memory target] sep241deconvolve [jobdata pkl file]

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


def main():
    args = parser.parse_args()
    tmpdir = os.getenv("TMPDIR")
    if args.compiledir is not None:
        compileDir = args.compiledir
    elif tmpdir is None:
        compileDir = f"./sep241tmp/{args.jobdata}/deconvolve/{args.workchunk}"
    else:
        compileDir = os.path.join(tmpdir, f"sep241tmp/{args.jobdata}/deconvolve/{args.workchunk}")
    set_flag("base_compiledir", compileDir)
    from sep241.sep241model import deconvolve
    deconvolve(args)


if __name__ == "__main__":
    main()
