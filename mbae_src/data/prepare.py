"""
    Module contains data preparation routines.
    Every resource (e.g., IEDB) is represented as a class,
    having three required methods: fetch, parse, and dump,
    depending on the results of each other in that exact order.

    For each of the resources, if download directory is not provided,
    the resource will store the fetched data into a temporary directory.
"""
import bisect
import logging
import typing as t
from itertools import chain, dropwhile, islice
from numbers import Number
from xml.etree import cElementTree, ElementTree

import numpy as np
import pandas as pd
import wget

from mbae_src.data.base import Resource, Constants


class IEDB(Resource):
    """
    Resource fetching and parsing IEDB data.
    """

    def __init__(
            self, download_dir: t.Optional[str] = None, download_file_name: str = 'mhc_ligand_full.zip',
            mapping: t.Union[str, t.IO, t.Mapping[str, str], None] = './mbae_resources/mapping.tsv'):
        """
        :param download_dir: Path to a directory where the resource will be downloaded.
        :param download_file_name: How to name a raw downloaded file.
        :param mapping: Initializing IEDB requires valid mapping (i.e., a dictionary) between allotypes and accessions.
        If not provided, this mapping will be obtained via IMGTHLAhistory and IPDMHChistory classes.
        """
        super().__init__(download_dir, download_file_name)
        self.resource_name = 'IEDB'

        # handle the `mapping` argument
        self.mapping = _handle_mapping(mapping, self.resource_name, download_dir)
        logging.info(f'{self.resource_name} -- successfully initialized resource')

    def fetch(self) -> str:
        """
        Downloads IEDB from the default location.
        :return: Path to a downloaded file (should be the same as {self.download_dir}/{self.download_file_name})
        """
        result = wget.download(Constants.iedb_url, str(self.download_path), None)
        logging.info(f'{self.resource_name} -- downloaded resource from {Constants.iedb_url}')
        return result

    def parse(self) -> pd.DataFrame:
        """
        Parses the downloaded IEDB resource and assigns the parsed DataFrame to self.parsed_data
        :return: Parsed IEDB data.
        :raises ValueError: In the absence of data to parse (likely due to not calling `fetch` first)
        """
        if not self.download_path.exists():
            raise ValueError(f'{self.resource_name} -- nothing to parse: {self.download_path.name} does not exist. '
                             f'Have you called `fetch` first?')
        # Load initial data
        df = pd.read_csv(
            self.download_path, low_memory=False, skiprows=1, usecols=list(Constants.iedb_renames)
        ).rename(columns=Constants.iedb_renames)
        logging.info(f'{self.resource_name} -- loaded resource; records: {len(df)}')

        # Filter records
        # -- Filter by MHC type
        df = df[df['mhc_type'] == 'I']
        logging.info(f'{self.resource_name} -- filtered class I records; records: {len(df)}')
        # -- Filter quantitative records;
        # -- TODO: we might change this behaviour later,
        # -- TODO: once we've found a way of correct MS data usage
        # -- -- Assay is a quantitative one
        df = df[df['method'].isin(Constants.iedb_quantitative_assays)]
        logging.info(f'{self.resource_name} -- filtered quantitative assays; records: {len(df)}')
        # -- -- Measurement must be present
        df = df[~df['measurement'].isna()]
        logging.info(f'{self.resource_name} -- filtered quantitative measurements; records: {len(df)}')
        # -- Filter by evidence codes
        # -- TODO: An optional step for quantitative records, but it's good being aware of eco's
        df = df[df['eco'] != 'Inferred by motifs or alleles present']
        logging.info(f'{self.resource_name} -- filtered by evidence codes; records: {len(df)}')
        # -- Filter by antigen type (some antigens may be organic molecules)
        df = df[df['antigen_type'] == 'Linear peptide']
        logging.info(f'{self.resource_name} -- filtered by antigen type; records: {len(df)}')
        df = _finalize_filtering(df, self.resource_name, self.mapping)

        # Finalize IEDB preparation
        df = _finalize_data_source(df, 'IEDB', list(Constants.data_source_final_fields))
        logging.info(f'IEDB -- completed resource preparation; records: {len(df)}')
        self.parsed_data = df
        return df

    def dump(self, dump_path: str, kwargs) -> None:
        """
        Dumps the resource to `dump_path`.
        :param dump_path: A valid path.
        :param kwargs: kwargs to be passed to `DataFrame.to_csv`
        :return:
        """
        _dump_data(
            dump_path=dump_path, dump_data=self.parsed_data,
            resource_name=self.resource_name, kwargs=kwargs)


class Bdata(Resource):
    """
    Resource fetching and parsing Bdata2013.
    """

    def __init__(
            self, download_dir: t.Optional[str] = None, download_file_name: str = 'bdata.zip',
            mapping: t.Union[str, t.IO, t.Mapping[str, str], None] = './mbae_resources/mapping.tsv'):
        """
        :param download_dir: Path to a directory where the resource will be downloaded.
        :param download_file_name: How to name a raw downloaded file.
        :param mapping: Initializing IEDB requires valid mapping (i.e., a dictionary) between allotypes and accessions.
        If not provided, this mapping will be obtained via IMGTHLAhistory and IPDMHChistory classes.
        """
        super().__init__(download_dir, download_file_name)
        self.resource_name = 'Bdata'

        # handle the `mapping` argument
        self.mapping = _handle_mapping(mapping, self.resource_name, download_dir)

        logging.info(f'{self.resource_name} -- successfully initialized resource')

    def fetch(self) -> str:
        """
        Downloads IEDB from the default location.
        :return: Path to a downloaded file (should be the same as {self.download_dir}/{self.download_file_name})
        """
        result = wget.download(Constants.bdata_url, str(self.download_path), None)
        logging.info(f'{self.resource_name} -- downloaded resource from {Constants.bdata_url}')
        return result

    def parse(self):
        """
        Parses the downloaded Bdata2013 resource and assigns the parsed DataFrame to self.parsed_data
        :return: Parsed Bdata2013.
        :raises ValueError: In the absence of data to parse (likely due to not calling `fetch` first)
        """
        if not self.download_path.exists():
            raise ValueError(f'{self.resource_name} -- nothing to parse: {self.download_path.name} does not exist. '
                             f'Have you called `fetch` first?')

        # Load initial data
        df = pd.read_csv(
            self.download_path, low_memory=False, sep='\t',
        ).rename(columns=Constants.bdata_renames)
        logging.info(f'{self.resource_name} -- loaded resource; records: {len(df)}')

        # Filter records
        df = _finalize_filtering(df, self.resource_name, self.mapping)

        # Finalize Bdata preparation
        df = _finalize_data_source(df, self.resource_name, list(Constants.data_source_final_fields))
        logging.info(f'{self.resource_name} -- completed resource preparation; records: {len(df)}')
        self.parsed_data = df
        return df

    def dump(self, dump_path: str, kwargs) -> None:
        """
        Dumps the resource to `dump_path`.
        :param dump_path: A valid path.
        :param kwargs: kwargs to be passed to `DataFrame.to_csv`
        :return:
        """
        _dump_data(
            dump_path=dump_path, dump_data=self.parsed_data,
            resource_name=self.resource_name, kwargs=kwargs)


class IPDMHChistory(Resource):
    """
    Resource fetching and parsing xml dump of the IPDMHC database.
    """

    def __init__(self, download_dir: t.Optional[str] = None, download_file_name: str = 'MHC.xml'):
        """
        :param download_dir: Path to a directory where the resource will be downloaded.
        :param download_file_name: How to name a raw downloaded file.
        """
        super().__init__(download_dir, download_file_name)
        self.resource_name = 'IPDMHC'
        logging.info(f'{self.resource_name} -- successfully initialized resource')

    def fetch(self) -> str:
        """
        Downloads IPDMHC xml from the default location.
        :return: Path to a downloaded file (should be the same as {self.download_dir}/{self.download_file_name})
        """
        result = wget.download(Constants.ipd_url, str(self.download_path), None)
        logging.info(f'{self.resource_name} -- downloaded resource from {Constants.ipd_url}')
        return result

    def parse(self) -> t.Dict[str, str]:
        """
        Uses cElementTree to parse a downloaded xml dump.
        Maps allele names (nomenclature field) to accessions and combines the mappings into a dictionary.
        The latter will be stored in `parsed_data` attribute.
        :return:
        """
        if not self.download_path.exists():
            raise ValueError(f'{self.resource_name} -- nothing to parse: {self.download_path.name} does not exist')
        tree = cElementTree.parse(self.download_path)
        logging.info(f'{self.resource_name} -- parsed xml tree')
        self.parsed_data = self._map_alleles_to_accessions(tree)
        logging.info(f'{self.resource_name} -- successfully extracted mappings')
        return self.parsed_data

    @staticmethod
    def _map_alleles_to_accessions(tree: cElementTree) -> t.Dict[str, str]:
        """
        Walks the xml tree and maps each `name` in the `nomenclature` to `id` (accession).
        :param tree: Parsed xml tree (ElementTree object).
        :return: Dict, mapping allotype names to IPDMHC accessions.
        """

        def parse_entry(entry: ElementTree.Element) -> t.Iterator[t.Tuple[str, str]]:
            allele_names = entry.find('nomenclature').findall('name')
            name_prefix = entry.find('name').text.split('-')[0]
            accession = entry.attrib['id']
            return ((f'{name_prefix}-{allele_name.text}', accession) for allele_name in allele_names)

        return dict(chain.from_iterable(map(parse_entry, tree.find('entries').findall('entry'))))

    def dump(self, dump_path: str, kwargs) -> None:
        """
        Dumps the resource to `dump_path`.
        :param dump_path: A valid path.
        :param kwargs: kwargs to be passed to `DataFrame.to_csv`
        :return:
        """
        _dump_data(
            dump_path=dump_path, dump_data=self.parsed_data,
            resource_name=self.resource_name, kwargs=kwargs)


class IMGTHLAhistory(Resource):
    """
    Resource fetching and parsing IMGT/HLA Allelelist_history.txt dump.
    """

    def __init__(self, download_dir: t.Optional[str] = None, download_file_name: str = 'IMGTHLA.txt'):
        """
        :param download_dir: Path to a directory where the resource will be downloaded.
        :param download_file_name: How to name a raw downloaded file.
        """
        super().__init__(download_dir, download_file_name)
        self.resource_name = 'IMGTHLA'
        logging.info(f'{self.resource_name} -- successfully initialized resource')

    def fetch(self) -> str:
        """
        Downloads IPDMHC xml from the default location.
        :return: Path to a downloaded file (should be the same as {self.download_dir}/{self.download_file_name})
        """
        result = wget.download(Constants.imgt_history_url, str(self.download_path), None)
        logging.info(f'{self.resource_name} -- downloaded resource from {Constants.imgt_history_url}')
        return result

    def parse(self) -> t.Dict[str, str]:
        """
        Parses the Allelelist_history.txt
        :return: Dict with mappings between allele names (first two digits in the nomenclature, e.g. HLA-A*01:01)
        and IMGT/HLA accessions.
        """

        def format_name(name: str) -> str:
            """
            Formats raw name into a commonly used one.
            :param name: A raw `name` of the allotype (e.g., A*01:01:01:01)
            :return: A formatted allotype name (e.g., HLA-A*01:01)
            """
            if ":" not in name:
                return f'HLA-{name}'
            return f'HLA-{":".join(name.split(":")[:2])}'

        def parse_line(line: str) -> t.Iterator[t.Tuple[str, str]]:
            """
            :param line: A line from the Allelelist_history.txt file
            :return: An iterator over pairs (name, accession)
            """
            line_split = line.rstrip('\n').split(',')
            accession = line_split[0]
            historical_names = map(lambda name: format_name(name), line_split[1:])
            return ((name, accession) for name in historical_names)

        if not self.download_path.exists():
            raise ValueError(f'{self.resource_name} -- nothing to parse: {self.download_path} does not exist')
        with open(self.download_path) as f:
            lines = islice(dropwhile(lambda l: l.startswith('#'), f), 1, None)
            self.parsed_data = dict(chain.from_iterable(map(parse_line, lines)))
        logging.info(f'{self.resource_name} -- successfully extracted mappings')
        return self.parsed_data

    def dump(self, dump_path: str, kwargs) -> None:
        """
        Dumps the resource to `dump_path`.
        :param dump_path: A valid path.
        :param kwargs: kwargs to be passed to `DataFrame.to_csv`
        :return:
        """
        _dump_data(
            dump_path=dump_path, dump_data=self.parsed_data,
            resource_name=self.resource_name, kwargs=kwargs)


def categorise(cutoffs: t.List[Number], x: Number) -> int:
    """
    Categorizes a number `x` into one of the `cutoffs`.
    :param cutoffs: A list of numerical boundaries between categories.
    :param x: A number.
    :return: A category corresponding to a number.
    """
    ordered = sorted(cutoffs)
    n_cat = len(ordered)
    return n_cat - bisect.bisect_left(ordered, x)


def separate_fraction(df: pd.DataFrame, fraction: float, mode: str) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divides a DataFrame into two parts based on a `fraction`.
    :param df: A DataFrame with `accession` and `peptide` columns.
    :param fraction: A fraction used to separate entries.
    :param mode: If `mode` == "observations", will separate a `fraction` of unique allotype-peptide pairs.
    If mode == "allotypes", will separate a `fraction` of unique allotypes.
    :return: Train and test fractions of the initial DataFrame
    :raises ValueError: If `mode` is invalid.
    """
    if mode not in ['observations', 'allotypes']:
        raise ValueError(f'Supported modes: observations, allotypes; got: {mode}')

    # Both approaches are less readable but much faster than using groupby
    if mode == 'observations':
        sub = df[['accession', 'peptide']].drop_duplicates().sample(frac=1.0).reset_index(drop=True)
        train = pd.merge(
            sub[:int(fraction * len(sub))],
            df, how='inner', on=['accession', 'peptide'])
        test = pd.merge(
            sub[int(fraction * len(sub)):],
            df, how='inner', on=['accession', 'peptide'])
        return train, test
    else:
        sub = df['accession'].drop_duplicates().sample(frac=1.0).reset_index(drop=True)
        train = df[df['accession'].isin(sub[:int(fraction * len(sub))])]
        test = df[df['accession'].isin(sub[int(fraction * len(sub)):])]
        return train, test


def separate_abundant(
        df: pd.DataFrame, rare_threshold: int) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separates a DataFrame into "abundant" and "rare" subsets.
    :param df: A DataFrame with `accession` and `peptide` columns.
    :param rare_threshold: If the number of (unique) observations falling under an accession
    is smaller than this number, all observations associated with this allotype will be considered "rare".
    Vice versa in case of "abundant".
    :return: A tuple of DataFrames (abundant, rare).
    """
    counts = df.groupby('accession', as_index=False).count()[['accession', 'peptide']]
    abundant_accessions = {acc for _, acc, num in counts.itertuples() if num >= rare_threshold}
    abundant = df[df['accession'].isin(abundant_accessions)]
    rare = df[~df['accession'].isin(abundant_accessions)]
    return abundant, rare


def read_mapping(file: t.Union[str, t.IO], sep: str = '\t'):
    """
    Reads a headerless mapping file and converts it into a Dict.
    :param file: Path or a buffer holding allotype to accession mappings.
    :param sep: Separator between columns (default is \t)
    :return:
    """
    df = pd.read_csv(file, sep=sep, names=['Allele', 'Accession'])
    return pd.Series(df.Accession.values, index=df.Allele).to_dict()


def obtain_mapping(
        download_dir: t.Optional[str] = None,
        imgt: t.Optional[IMGTHLAhistory] = None,
        ipd: t.Optional[IPDMHChistory] = None) -> t.Dict[str, str]:
    """
    :param download_dir: An argument to IMGTHLAhistory and IPDMHChistory objects.
    :param imgt: IMGTHLAhistory object (optional). If not provided, will be initialized internally.
    :param ipd: IPDMHChistory object (optional). If not provided, will be initialized internally.
    :return: Allotype to accession mapping as a dictionary.
    """
    # initialize and parse IMGTHLA
    if imgt is None:
        imgt = IMGTHLAhistory(download_dir=download_dir)
        imgt.fetch(), imgt.parse()
    # initialize and parse IPDMHC
    if ipd is None:
        ipd = IPDMHChistory(download_dir=download_dir)
        ipd.fetch(), ipd.parse()
    # combine mappings and return as a dictionary "allotype -> accession"
    return {**imgt.parsed_data, **ipd.parsed_data, **Constants.mapping_addition}


def _handle_mapping(
        mapping: t.Union[str, t.IO, t.Mapping[str, str], None],
        resource_name: str,
        download_dir: t.Optional[str] = None) -> t.Union[t.Mapping, t.Dict]:
    """
    A helper function handling the `mapping` argument of a Resource.
    Internally calls `obtain_mapping` which does the hard work.
    If `mapping` is a Dict object, returns untouched.
    :param mapping: A raw `mapping` argument.
    :param resource_name: A name of the resource calling the function.
    :param download_dir: An argument passed to `obtain_mapping` function.
    :return:
    """
    if isinstance(mapping, t.Dict):
        return mapping
    elif isinstance(mapping, str) or isinstance(mapping, t.IO):
        try:
            return read_mapping(mapping)
        except FileNotFoundError:
            logging.warning(f'{resource_name} -- found no "allotype->accession" mapping at {mapping}; '
                            f'attempting to create a new one by fetching IPD-MHC and IMGT/HLA history')
            return obtain_mapping(download_dir=download_dir)
    elif mapping is None:
        logging.info(f'{resource_name} -- got None for mapping; '
                     f'attempting to create a new one by fetching IPD-MHC and IMGT/HLA')
        return obtain_mapping(download_dir=download_dir)
    else:
        raise ValueError(f'{resource_name} -- failed to handle the `mapping` argument')


def _map_allotypes(df: pd.DataFrame, df_name: str, mapping: t.Mapping) -> pd.DataFrame:
    """
    A helper function helping to map allotype names in df having `accession` field to accessions.
    Logging will warn a user about unmapped allotypes.
    :param df: DataFrame with `allotype` field present.
    :param df_name: Name of the df (used for logging)
    :param mapping: Mapping between allotype names and accessions.
    :return: DataFrame with a new `accession` field.
    """
    # Prevent SettingWithCopyWarning
    df = df.copy()
    # Map allotypes accessions
    df['accession'] = df['allotype'].map(mapping)
    # Warn about unmapped allotypes
    unmapped_allotypes_loc = ~df['accession'].isin(set(mapping.values()))
    if unmapped_allotypes_loc.any():
        df.loc[unmapped_allotypes_loc, 'accession'] = np.nan
        unmapped_allotypes = df.loc[unmapped_allotypes_loc, "allotype"].sort_values().unique()
        logging.warning(f'{df_name} -- could not map {len(unmapped_allotypes)} '
                        f'allotypes {unmapped_allotypes} corresponding to '
                        f'{unmapped_allotypes_loc.sum()} records')
    # Filter out unmapped allotypes
    df = df[~df['accession'].isna()]
    logging.info(f'{df_name} -- filtered out unmapped allotypes; records: {len(df)}')
    return df


def _finalize_filtering(df: pd.DataFrame, resource_name: str, mapping: t.Mapping[str, str]) -> pd.DataFrame:
    """
    A helper function to finalize filtering of the data source encompassing common routines:
    filtering by peptide length, allotype mapping.
    :param df: A DataFrame with peptide and allotype fields.
    :param resource_name: A name of the resource.
    :param mapping: Mapping to be passed to `_map_allotypes`
    :return:
    """
    # Prevent SettingWithCopyWarning
    df = df.copy()
    # Filter by peptide length
    df = df[df['peptide'].apply(
        lambda p: Constants.peptide_length.start <= len(p) <= Constants.peptide_length.end)]
    logging.info(f'{resource_name} -- filtered by peptide length; records: {len(df)}')
    # Map allotypes to accessions and remove unmapped allotypes
    df = _map_allotypes(df, resource_name, mapping)
    return df


def _finalize_data_source(df: pd.DataFrame, df_name: str, final_fields: t.List[str]) -> pd.DataFrame:
    """
    A helper function for finalizing the Resource preparation (e.g., IEDB).
    Creates `measurement_ord` field, categorizing `measurement` values into pre-defined categories
    (defined by Constants object).
    Creates the `source` field and fills it with the `df_name` value.
    Selects only necessary columns and drops duplicated entries.
    :param df: DataFrame with the `measurement` field.
    :param df_name: The name of the DataFrame (used for to create the `source` and for logging).
    :param final_fields: Final fields to pick.
    :return: A finalized DatFrame.
    """
    # -- Categorize measurements
    df['measurement_ord'] = df['measurement'].apply(lambda x: categorise(Constants.ord_cutoffs, x))
    # -- Handle `inequality`
    df.replace({'inequality': Constants.inequality_renames})
    # -- Set source field for tracking records
    df['source'] = df_name
    # -- Select only necessary columns and drop duplicates
    df = df[final_fields].drop_duplicates()
    logging.info(f'{df_name} -- dropped unnecessary columns and removed duplicates; records {len(df)}')
    return df


def _dump_data(
        dump_path: str, resource_name: str,
        dump_data: t.Union[pd.DataFrame, t.Dict],
        kwargs: t.Dict[str, t.Any]) -> None:
    """
    A helper function to dump data of a resource.
    Currently we have two resources types: A DataFrame and a Dict.
    Both of them will be dumped using `DataFrame.to_csv`.
    :param dump_path: A path to dump to.
    :param resource_name: A name of the resource for formatting errors and logging messages.
    :param dump_data: Data to dump.
    :param kwargs: kwargs to pass to `DataFrame.to_csv`.
    """
    if isinstance(dump_data, pd.DataFrame):
        dump_data.to_csv(dump_path, **kwargs)
    elif isinstance(dump_data, t.Dict):
        pd.DataFrame(
            [list(dump_data.keys()), list(dump_data.values())]
        ).to_csv(dump_path, **kwargs)
    else:
        raise ValueError(f'{resource_name} -- dumping the input of such type is not supported')
    logging.info(f'{resource_name} -- saved parsed data to {dump_path}')


if __name__ == '__main__':
    raise RuntimeError
