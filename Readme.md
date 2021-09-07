# 2for1separator

2for1 separator is an algorithm to deconvolve CUT&Tag2for1 data. It uses differences in the fragment length distributions of the two targets and teh proximity of chromatin
cuts to estimate the probability for each cut to originate from one
target or the other. The result is a set of cut density tracks
that represent the estimated number of cuts induced by the two
antibodies used in the CUT&Tag2for1 experiment.

## Disclaimer

This is an alpha version with very high memory demand of up to
300 GB. An new version, that aims to reduce the resource demand,
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
It is recommended to use approximatly 300 for `--memory`.

The ouptut of the function reports the number of seperate work chunks
and suggests susequent calls for the deconvolution.

To produce bigwig files from the deconvolution results run
```
./sep241mkbw.py [jobdata pkl file] [chrom sizes file]
```
The chomosome sizes file needs to have two columns with `chromosome name` and
`size in bases`
(see [bigWIG forma](https://genome.ucsc.edu/goldenPath/help/bigWig.html)).

To call peaks from the deconvolved tracks run:
```
./sep241peakcalling.py [jobdata pkl file]
```

For more information pass `--help` to the respective script.
