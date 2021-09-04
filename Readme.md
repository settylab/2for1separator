# 2for1seperator

2for1 seperator is a computatial to deconvolve data from CUT&Tag 2for 1
chromatine profiling. It uses differences in the fragment length
distributions of the two targets and teh proximity of chromatin
cuts to extimate the probability for each cut to originate from one
target or the other. The result is a set of cut density tracks
that represent the estimated number of cuts induced by the two
antibodies used in the CUT&Tag 2for1 experiment.

## Disclaimer

This is an alpha version with very high memory demand of up to
300 GB. An new version, that aims to reduce the resource demand,
may be released soon.

## Installation

Please make sure `python` points to a Python 3.9 interpreter
and istall the requirements:
```
pip install -r requirements.txt
```

## Usage

Before the deconvolution the data has to be split up into manageable chunks:
```
./sep241prep.py [bed files] --out [jobdata file] --memory [max memory target in GB]
```
It is recommendet to use approximatly 300 for `--memory`.

The ouptut of the function reports the number of seperate work chunks
and suggests susequent calls for the deconvolution.

To produce bigwig files from the deconvolution results run
```
./sep241mkbw.py [jobdata file] [chrom sizes file]
```
The chomosome sizes file needs to have two clolumns with `chromosome name` and
`size in bases`
(s. [bigWIG forma](https://genome.ucsc.edu/goldenPath/help/bigWig.html)).

To call peaks from the deconvolved tracks run:
```
./sep241peakcalling.py [jobdata file]
```

For more information pass `--help` to the respective script.
