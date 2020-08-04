#! /usr/bin/env python

import logging
import operator as op
import sys
import typing as t
from itertools import product, filterfalse

import click
import numpy as np
import pandas as pd
from Bio import SeqIO
from tensorflow.keras import models

import mbae_resources
from mbae_resources import resources, CONSURF_SCORE
from mbae_src.data import preprocessing as pp, encoding
from mbae_src.data.base import Constants
from mbae_src.data.prepare import (
    obtain_mapping, read_mapping, separate_fraction, separate_abundant, IEDB, Bdata, IMGTHLAhistory, IPDMHChistory)


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


@mbae.command('prepare', help='Prepare training data')
@click.option('-d', '--download_dir', default='./', show_default=True,
              type=click.Path(
                  exists=True, file_okay=False, dir_okay=True, resolve_path=True),
              help='Path to a download directory')
@click.option('-D', '--database', multiple=True, default=['all'], show_default=True,
              help='Databases to prepare. Supports multiple values. '
                   'Use `--database iedb` or `--database bdata` to download and prepare '
                   'IEDB or Bdata, separately. '
                   'Currently, available resources are: iedb and bdata, while "all" and "none" values '
                   'are reserved for parsing all or none data sources, respectively. ')
@click.option('-m', '--mapping', default=None,
              type=click.Path(
                  exists=True, file_okay=True, dir_okay=False, resolve_path=True),
              help='Path to mapping: a headerless file with space-like separator (e.g., \\t) '
                   'holding mappings between allele names (1st column) and accessions (2nd column). '
                   'Accessions must be from IPD-MHC (for non-human alleles) or IMGT/HLA (for human alleles).'
                   'If not provided, the command will download and parse mappings '
                   'from the aforementioned resources and manually add mice alleles. ')
@click.option('-s', '--save', multiple=True, default=['final'], show_default=True,
              help='Option controls what will be saved. '
                   'Multiple values are supported: '
                   '- final (final data will be saved); '
                   '- parsed (every parsed Resource will be saved); '
                   '- mapping (the obtained mapping will be saved); '
                   '- raw (raw downloaded files will be saved). '
                   'Example: "-s parsed -s final" to save parsed data of used resources '
                   'along with the final dataset. ')
@click.option('-S', '--separate_rare', is_flag=True, default=False, show_default=True,
              help=f'Whether to separate the resulting DataFrame into two parts - "abundant" and "rare." '
                   f'If provided, each of the "abundant" and "rare" subsets will also be separated '
                   f'into a "train" and "test" subsets based on `sep_fraction` value.')
@click.option('-t', '--rare_threshold', default=Constants.rare_threshold, show_default=True,
              help='If the `separate_rare` flag is provided, use this threshold '
                   'to control the placement into "abundant" and "rare" subsets.')
@click.option('-f', '--sep_fraction', default=Constants.train_fraction, show_default=True,
              help='A fraction of training examples to separate. ')
@click.option('-M', '--sep_mode', default='observations',
              type=click.Choice(['observations', 'allotypes']), show_default=True,
              help='A separation mode. If `observation`, will separate '
                   'a `sep_fraction` of unique allotype-peptide pairs. '
                   'If `allotypes`, will separate a `sep_fraction` of unique allotypes. ')
@click.option('-v', '--verbose', is_flag=True, default=False, show_default=True,
              help='If the flag is provided, will output logging messages '
                   'with the info describing main data processing steps. '
                   'By default, outputs only warnings. ')
@click.pass_context
def prepare(ctx, download_dir, database, mapping, save,
            separate_rare, rare_threshold, sep_fraction, sep_mode, verbose):
    # Parse and validate arguments
    # -- Handle databases
    if not (database[0] in ['all', 'none']) and any(db not in Constants.available_sources for db in database):
        raise click.BadOptionUsage(
            option_name='database', message=f'Incorrect input for the --database option; got {database}')
    # -- Handle save modes
    valid_save_options = ['final', 'parsed', 'mapping', 'raw']
    if any(s not in valid_save_options for s in save):
        raise click.BadOptionUsage(
            option_name='save', message=f'Incorrect usage of the --save option; got {database}')
    save_mapping = 'mapping' in save
    save_raw = 'raw' in save
    save_parsed = 'parsed' in save
    save_raw_dir = download_dir if save_raw else None
    # -- Handle separate fraction
    sep_fraction = Constants.train_fraction or sep_fraction
    # -- Handle verbosity level
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Prepare the data
    # -- Mapping preparation
    if mapping is None:
        # If `mapping` is not provided, create one here
        imgt = IMGTHLAhistory(download_dir=save_raw_dir)
        ipd = IPDMHChistory(download_dir=save_raw_dir)
        imgt.fetch(), imgt.parse()
        ipd.fetch(), ipd.parse()
        mapping = obtain_mapping(download_dir=save_raw_dir, imgt=imgt, ipd=ipd)
        if save_mapping:
            mapping_save_path = f'{download_dir}/mapping.tsv'
            pd.DataFrame(
                list(mapping.items())
            ).to_csv(
                mapping_save_path, sep='\t', index=False, header=False)
            logging.info(f'Saved mapping to {mapping_save_path}')
    else:
        # Otherwise, read the headerless mapping with space-like separator
        imgt, ipd = None, None
        mapping = read_mapping(mapping, sep=r'\s+')
    # -- IEDB preparation
    if 'iedb' in database or 'all' in database:
        iedb = IEDB(download_dir=save_raw_dir, mapping=mapping)
        iedb.fetch(), iedb.parse()
    else:
        iedb = None
    # -- Bdata preparation
    if 'bdata' in database or 'all' in database:
        bdata = Bdata(download_dir=save_raw_dir, mapping=mapping)
        bdata.fetch(), bdata.parse()
    else:
        bdata = None

    # Save parsed resources
    if save_parsed:
        resources_to_dump = filterfalse(
            lambda x: x[1] is None,
            [('IMGTHLAhistory.tsv', imgt), ('IPDMHChistory.tsv', ipd),
             ('IEDB_parsed.tsv', iedb), ('Bdata2013_parsed.tsv', bdata)])
        for r_name, r in resources_to_dump:
            r.dump(f'{download_dir}/{r_name}', kwargs={'sep': '\t'})
        logging.info(f'Saved parsed resources to {download_dir}')

    # If the user does not intend to save the final data, stop here
    if 'final' not in save:
        return

    # Combine sources if needed
    if iedb is None and bdata is None:
        logging.info('No data sources to combine; stopping `prepare`.')
        return
    elif iedb is None and bdata is not None:
        df = bdata.parsed_data
    elif iedb is not None and bdata is None:
        df = iedb.parsed_data
    else:
        df = pd.concat([
            iedb.parsed_data,
            bdata.parsed_data[~(
                    (bdata.parsed_data['accession'].isin(iedb.parsed_data['accession'])) &
                    (bdata.parsed_data['peptide'].isin(iedb.parsed_data['peptide']))
            )]]).reset_index(drop=True)
        logging.info(f'Combined data sources; records: {len(df)}')

    # Remove the exact same measurements duplicated (due to, e.g., missing inequality signs)
    df = df[~(df[['accession', 'peptide', 'measurement', 'measurement_ord']].duplicated())]
    logging.info(f'Removed duplicated measurements; records: {len(df)}')

    # No need to separate observations into abundant and rare
    if not separate_rare:
        train, test = separate_fraction(df, sep_fraction, sep_mode)
        train.to_csv(f'{download_dir}/train_data.tsv', sep='\t', index=False)
        test.to_csv(f'{download_dir}/test_data.tsv', sep='\t', index=False)
        logging.info(f'Saved {len(train)} training and {len(test)} testing observations.')
        return

    # Separate otherwise
    abundant, rare = separate_abundant(df, rare_threshold)
    # -- Process abundant allotypes
    abundant_train, abundant_test = separate_fraction(abundant, sep_fraction, sep_mode)
    abundant_train.to_csv(f'{download_dir}/train_data_abundant.tsv', sep='\t', index=False)
    abundant_test.to_csv(f'{download_dir}/test_data_abundant.tsv', sep='\t', index=False)
    logging.info(f'Saved {len(abundant_train)} training and {len(abundant_test)} testing observations '
                 f'for the abundant subset.')
    # -- Process rare allotypes
    rare_train, rare_test = separate_fraction(rare, sep_fraction, sep_mode)
    rare_train.to_csv(f'{download_dir}/train_data_rare.tsv', sep='\t', index=False)
    rare_test.to_csv(f'{download_dir}/test_data_rare.tsv', sep='\t', index=False)
    logging.info(f'Saved {len(rare_train)} training and {len(rare_test)} testing observations '
                 f'for the rare subset.')


if __name__ == '__main__':
    mbae()
