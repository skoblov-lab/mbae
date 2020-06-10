# mbae - MHC Binding Affinity Estimator

Canonically pronounced as British 'ember'. The package contains yet another MHC class I affinity predictor (the publication is being written).

## Installation

If you don't want to use a GPU for inference, all you need is
```
pip install --no-cache-dir https://github.com/skoblov-lab/mbae.git
```
If you do want to utilise a GPU, you will need to uninstall `tensorflow` and install `tensorflow-gpu>=2.2`.

## Usage

The main entry point is `mbae.py`. It packs two commands
```
$ mbae.py -h
Usage: mbae.py [OPTIONS] COMMAND [ARGS]...

Options:
  -h, --help  Show this message and exit.

Commands:
  alleles  List all supported alleles
  predict  Predict binding affinity
```
The first command, `alelles`, does nothing but printing out a list of all supported alleles to let you know which alleles are supported at the moment.

The second command, `predict`, predicts binding affinity.
```
$ mbae.py predict -h
Usage: mbae.py predict [OPTIONS]

  Predict binding affinity

Options:
  -a, --allele TEXT     a supported MHC allele  [required]
  -p, --peptides FILE   a headerless tab-separated table with 1 or 2 columns;
                        the first column must contain unique single-letter
                        IUPAC peptide sequences; the optional second column
                        must contain a 1-based index for a target position: it
                        is used to filter scanning windows covering the
                        position; the second column is only supported in
                        scanning mode  [required]

  -l, --length INTEGER  length of the peptide-scanning window; if no position
                        is provided, complete peptide sequences are
                        considered;specifying this option invokes scanning
                        mode

  -h, --help            Show this message and exit.
```
You can specify multiple alleles by providing multiple `-a` options. For example,
```
$ mbae.py -a 'HLA-A*02:02' -a 'HLA-C*05:29' -p peptides.tsv -l 9
```
The command can operate in two modes: simple prediction and scanning prediction.
In the first case we take entire peptides as inputs to the affinity estimator. This mode is useful when we want to estimate binding affinity of physiologically-long peptides. To invoke simple prediction mode, you must omit the `--length` option. Bear in mind, the tool cannot predict affinity for peptides that are more than 16 amino acids long. Also, bear in mind, that long peptides are not particularly physiological. For example, [it's believed](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4744552/) that human MHC I alleles prefer peptides in the range of 8-10 amino acids.
However, in many practical cases we have to deal with therapeutic peptides of non-physiological lengths, e.g. 24. In this case we want to scan our long peptides by rolling a window of physiological width and estimate binding affinity of each window. Specifying `--length` invokes scanning mode. In this mode you can also specify target positions (e.g. position of a cancer-specific mutation) that must be covered by a scanning window for it to be evaluated. You can refer to the help-string of option `--peptides` for further details.

The command writes results to standard output. It is a tab-separated table with four columns:
1. `peptide` - original peptide as provided via `--peptides`;
2. `allele` - an MHC allele name (as specified by `--allele`);
3. `window` - a sequence that was evaluated by the model;
4. `prediction` - predicted affinity; a value in the range (0, 1]; larger values mean higher affinity
