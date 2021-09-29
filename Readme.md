# 2for1separator

[![DOI](https://zenodo.org/badge/402908753.svg)](https://zenodo.org/badge/latestdoi/402908753)

2for1 separator is an algorithm to deconvolve CUT&Tag2for1 data.  It uses
differences in the fragment length distributions of the two targets and the
proximity of chromatin cuts to estimate the probability for each cut to
originate from one target or the other. The result is a set of cut density
tracks that represent the estimated number of cuts induced by the two
antibodies used in the CUT&Tag2for1 experiment.

![Schmenatic](schematic.jpg?raw=true "Schematic")

## Disclaimer

This is an alpha version with very high memory demand of up to
300 GB. A new version, that aims to reduce the resource demand,
will be released soon.

## Installation

Please make sure `python` points to a Python 3.9 interpreter
and istall the requirements:
```
pip install -r requirements.txt
```

## Usage

Before the deconvolution the data has to be split up into manageable chunks:
```
./sep241prep.py [bed files] --out [jobdata pkl file] --memory [max memory target in GB]
```
It is recommended to use approximatly `300` GB or more for `--memory`.

The ouptut of the function reports the number of seperate work chunks
and suggests subsequent calls for the deconvolution.
Note that if memory resources are exhausted, deconvolution jobs
may be cancled by slurm or the operating system and subsequent
results will be missing. The downstream scripts will look for missing
results and report a comma sperated list of respective work chunk
numbers. If you are using slurm you can rerun the slurm jobs of only
the specified jobs by passing the list with the `--array=` parameter
of the `sbatch` command.

To produce bigwig files from the deconvolution results run
```
./sep241mkbw.py [jobdata pkl file] [chrom sizes file]
```
The chomosome sizes file needs to have two columns with `chromosome name` and
`size in bases`
(see [bigWIG format](https://genome.ucsc.edu/goldenPath/help/bigWig.html)).

The produced bigWIG files may be used for downstream analysis such as peak
calling. To use the 2for1seperator cut-likelyhood-based peak calling
with overlap idendification run:
```
./sep241peakcalling.py [jobdata pkl file]
```
Note, that `sep241peakcalling.py` does not required the prior conversion
to bigWIG but instead uses the raw deconvolution output.

For more information pass `--help` to the respective script.

## Visualization

Output files are bigwigs and bed-files. These can be visualised
with softwra etools such as the
[IGV Browser](https://software.broadinstitute.org/software/igv/)
or
[JBrowser 2](https://jbrowse.org/jb2/).
Vizualization of intermediate results are currently only possible
if intermediate function calls within the supplied scripts are
reproduced in a python environment.
