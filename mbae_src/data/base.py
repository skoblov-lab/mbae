import typing as t
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PosixPath
from tempfile import TemporaryDirectory

_Tuple_mapping = t.Tuple[t.Tuple[str, str]]
Boundaries = t.NamedTuple('boundaries', [('start', int), ('end', int)])


@dataclass
class _Constants:
    """
    Data class holding constant values required for data preparation.
    """
    available_sources: t.Tuple = ('iedb', 'bdata')
    iedb_url: str = "https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip"
    bdata_url: str = "http://tools.iedb.org/static/main/binding_data_2013.zip"
    ipd_url: str = "https://raw.githubusercontent.com/ANHIG/IPDMHC/Latest/MHC.xml"
    imgt_history_url: str = "https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/Allelelist_history.txt"
    peptide_length: Boundaries = Boundaries(6, 16)
    rare_threshold: int = 100
    train_fraction: float = 0.8
    _iedb_renames: _Tuple_mapping = (
        ('Name', 'species'),
        ('Object Type', 'antigen_type'),
        ('Description', 'peptide'),
        ('Quantitative measurement', 'measurement'),
        ('Units', 'units'),
        ('Measurement Inequality', 'inequality'),
        ('Allele Name', 'allotype'),
        ('Allele Evidence Code', 'eco'),
        ('MHC allele class', 'mhc_type'),
        ('Qualitative Measure', 'measurement_qual'),
        ('Assay Group', 'assay_group'),
        ('Method/Technique', 'method'))
    _bdata_renames: _Tuple_mapping = (
        ('mhc', 'allotype'),
        ('sequence', 'peptide'),
        ('meas', 'measurement'),
    )
    _inequality_renames: _Tuple_mapping = (
        ('>=', '>'),
        ('<=', '<'),
        ('=', '='),
        ('>', '>'),
        ('<', '<'))
    iedb_quantitative_assays: t.Tuple[str] = (
        'cellular MHC/competitive/fluorescence',
        'cellular MHC/competitive/radioactivity',
        'cellular MHC/direct/fluorescence',
        'purified MHC/competitive/fluorescence',
        'purified MHC/competitive/radioactivity',
        'purified MHC/direct/fluorescence')
    data_source_final_fields: t.Tuple[str] = (
        'accession',
        'peptide',
        'measurement',
        'measurement_ord',
        'inequality',
        'source',
    )
    _mapping_addition: _Tuple_mapping = (
        ('H2-Kd', 'H-2-Kd'),
        ('H2-Kb', 'H-2-Kb'),
        ('H2-Kk', 'H-2-Kk'),
        ('H2-Dd', 'H-2-Dd'),
        ('H2-Db', 'H-2-Db'),
        ('H2-Dk', 'H-2-Dk'),
        ('H2-Ld', 'H-2-Ld'),
        ('H-2-Kd', 'H-2-Kd'),
        ('H-2-Kb', 'H-2-Kb'),
        ('H-2-Kk', 'H-2-Kk'),
        ('H-2-Dd', 'H-2-Dd'),
        ('H-2-Db', 'H-2-Db'),
        ('H-2-Dk', 'H-2-Dk'),
        ('H-2-Ld', 'H-2-Ld'),
    )
    ord_cutoffs: t.Tuple[int] = (
        10,
        100,
        500,
        2500,
        10000,
        20000,
    )

    @property
    def iedb_renames(self) -> t.Dict[str, str]:
        return dict(self._iedb_renames)

    @property
    def bdata_renames(self) -> t.Dict[str, str]:
        return dict(self._bdata_renames)

    @property
    def inequality_renames(self) -> t.Dict[str, str]:
        return dict(self._inequality_renames)

    @property
    def mapping_addition(self) -> t.Dict[str, str]:
        return dict(self._mapping_addition)


# Init Constants; this object should be imported elsewhere
Constants = _Constants()


class Resource(metaclass=ABCMeta):
    """
    Abstract base class defining a basic interface for a Resource.
    """
    def __init__(self, download_dir: t.Optional[str] = None, download_file_name: str = 'download'):
        self.parsed_data: t.Any = None
        self.download_dir = _handle_dir(download_dir)
        self.download_path = Path(f'{self.download_dir.name}/{download_file_name}')

    @abstractmethod
    def fetch(self):
        raise NotImplementedError

    @abstractmethod
    def parse(self):
        raise NotImplementedError

    @abstractmethod
    def dump(self, dump_path: str, kwargs: t.Dict[str, t.Any]):
        raise NotImplementedError


def _handle_dir(directory: t.Optional[str] = None) -> t.Union[TemporaryDirectory, PosixPath]:
    if directory is None:
        return TemporaryDirectory()
    if not isinstance(directory, str):
        raise ValueError('Wrong input for directory')
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f'The provided directory {directory.name} does not exist')
    return directory


if __name__ == '__main__':
    raise RuntimeError()
