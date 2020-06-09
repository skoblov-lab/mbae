import sys
import operator as op
import typing as t
from itertools import product

import click
import numpy as np
import pandas as pd
from tensorflow.keras import models
from Bio import SeqIO

import mbae_resources
from mbae_resources import resources, CONSURF_SCORE
from mbae.data import preprocessing as pp, encoding


def load_consurf() -> pd.DataFrame:
    with resources.path(mbae_resources, 'consurf.tsv') as path:
        return pd.read_csv(path, sep='\t')


def load_binding_regions() -> t.Mapping[str, str]:
    with resources.path(mbae_resources, 'binding_regions.fsa') as path:
        return {
            seqrec.description.split()[-1]: str(seqrec.seq) for seqrec in
            SeqIO.parse(path, 'fasta')
        }


def load_predictors() -> t.List[models.Model]:
    predictors = []
    for rsc in resources.contents(mbae_resources):
        if not rsc.startswith('model'):
            continue
        with resources.path(mbae_resources, rsc) as model_path:
            predictors.append(models.load_model(model_path))
    return predictors


def input_lengths(predictors: t.List[models.Model]) -> t.Tuple[int, int]:
    """
    :param predictors:
    :return: MHC length, peptide length
    """
    mhc_lengths = {model.inputs[0].shape[1] for model in predictors}
    if len(mhc_lengths) != 1:
        raise ValueError('...')
    peptide_lengths = {model.inputs[1].shape[1] for model in predictors}
    if len(peptide_lengths) != 1:
        raise ValueError('...')
    return mhc_lengths.pop(), peptide_lengths.pop()


def scan_peptide_target(length: int, seq: str, target_idx: int) -> t.List[str]:
    """
    :param length: scanning window length
    :param seq: peptide sequence
    :param target_idx: all scanning windows in the output must include
    target_idx
    :return:
    """
    if length < 1:
        raise ValueError()
    if target_idx < 0:
        raise ValueError('negative target indices are not allowed')
    if target_idx > len(seq) - 1:
        raise ValueError(
            f'target index {target_idx} falls beyond sequence {seq}')
    if len(seq) <= length:
        return [seq]
    start = max(target_idx - length + 1, 0)
    stop = min(len(seq) - length + 1, target_idx + 1)
    return [seq[i:i + length] for i in range(start, stop)]


def scan_peptide(length: int, seq: str) -> t.List[str]:
    """
    :param length: scanning window length
    :param seq: peptide sequence
    :return:
    """
    if len(seq) <= length:
        return [seq]
    return [seq[i:i + length] for i in range(0, len(seq) - length + 1)]


@click.group('mbae', context_settings=dict(help_option_names=['-h', '--help']),
             invoke_without_command=True)
@click.pass_context
def mbae(ctx):
    pass


@mbae.command('alleles', help='List all supported alleles')
@click.pass_context
def list_alleles(ctx):
    for allele in sorted(load_binding_regions()):
        print(allele)


@mbae.command('predict', help='Predict binding affinity')
@click.option('-a', '--allele', type=str, multiple=True, required=True,
              help='a supported MHC allele')
@click.option('-p', '--peptides', required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False,
                              resolve_path=True),
              help=('a headerless tab-separated table with 1 or 2 columns; '
                    'the first column must contain unique single-letter '
                    'IUPAC peptide sequences; the optional second column must '
                    'contain a 1-based index for a target position: it is used '
                    'to filter scanning windows covering the position; the '
                    'second column is only supported in scanning mode'))
@click.option('-l', '--length', type=int, required=False,
              help=('length of the peptide-scanning window; if no position is '
                    'provided, complete peptide sequences are considered;'
                    'specifying this option invokes scanning mode'))
@click.pass_context
def predict(ctx, allele: t.Sequence[str], peptides: str,
            length: t.Optional[int]):
    if not (length is None or length):
        raise ValueError('length of 0 is not allowed')
    binding_regions = load_binding_regions()
    consurf = load_consurf()
    predictors = load_predictors()
    mhc_len, pep_len = input_lengths(predictors)
    # make sure all alleles are available
    mhcs = list(allele)
    for mhc in mhcs:
        if mhc not in binding_regions:
            raise click.BadParameter(f'unsupported allele {mhc}')
    # parse peptides
    peptide_table = pd.read_csv(peptides, sep='\t', header=None)
    if peptide_table.shape[1] not in {1, 2}:
        raise click.BadParameter(
            '--peptides must specify a headerless table with 1 or 2 columns'
        )
    pept_seqs = peptide_table[0]
    # make sure all peptides are unique
    if pept_seqs.unique().shape[0] != pept_seqs.shape[0]:
        raise ValueError('duplicate peptides are not allowed')
    try:
        # by trying to convert targets to integers we are also checking whether
        # there are any NaNs and/or non-numeric values
        # we subtract 1 from targets to convert 1-based indices into 0-based
        pept_targets = (None if peptide_table.shape[1] == 1 else
                        peptide_table[1].astype(int) - 1)
    except ValueError:
        raise click.BadParameter(
            'the second column of --peptides must contain integers with no '
            'missing values'
        )
    if length is None and pept_targets is not None:
        raise click.BadParameter(
            f'the second column in --peptides is only supported in scanning '
            f'mode, but --length is unspecified'
        )
    # if length is specified, make sure it's not greater than model input
    if length is not None and length > pep_len:
        raise click.BadParameter(
            f'--length {length} exceeds the maximum length supported by our '
            f'models ({pep_len})'
        )
    # if length is None, make sure no peptides exceed the supported length
    if length is None and pept_seqs.apply(len).max() > pep_len:
        raise click.BadParameter(
            f'received peptides exceeding the maximum length supported by our '
            f'models; you should use scanning mode or remove sequences of more '
            f'than {pep_len} amino acids'
        )
    # create a dataset
    records = []
    if pept_targets is None:
        for peptide, mhc in product(pept_seqs, mhcs):
            windows = scan_peptide(length, peptide) if length else [peptide]
            for window in windows:
                records.append({'allele': mhc, 'peptide': peptide, 'window': window})
    else:
        for (peptide, target), mhc in product(zip(pept_seqs, pept_targets), mhcs):
            windows = scan_peptide_target(length, peptide, target)
            for window in windows:
                records.append({'allele': mhc, 'peptide': peptide, 'window': window})
    data = pd.DataFrame.from_records(records)
    # encode peptide windows

    pep_encoded, _ = pp.stack(
        [encoding.encode_protein(peptide) for peptide in data['window']],
        shape=(pep_len,), dtype=np.int32)
    # get mhc_len most variable positions (as per consurf) and encode mhc
    # sequences
    mhc_positions = (
        consurf
        .sort_values(CONSURF_SCORE, ascending=False)
        .iloc[:mhc_len, 0]
        .sort_values()
    )
    mhc_subsetter = op.itemgetter(*mhc_positions)
    mhc_seqs = [
        mhc_subsetter(binding_regions[mhc]) for mhc in data['allele']
    ]
    mhc_encoded, _ = pp.stack(
        [encoding.encode_protein(mhc) for mhc in mhc_seqs],
        shape=(mhc_len,), dtype=np.int32
    )
    # make predictions
    predictions = [predictor.predict([mhc_encoded, pep_encoded])
                   for predictor in predictors]
    average = np.array(predictions).mean(axis=0)
    # write results to stdout
    data['prediction'] = average
    data[['peptide', 'allele', 'window', 'prediction']].to_csv(sys.stdout, sep='\t', index=False)


if __name__ == '__main__':
    mbae()
