import typing as t
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

_Tuple_mapping = t.Tuple[t.Tuple[str, str]]
Boundaries = t.NamedTuple('boundaries', [('start', int), ('end', int)])


@dataclass
class _Constants:
    """
    Data class holding constant values required for data preparation.
    """
    alignment_tool: str = 'mafft'
    alignment_command: t.Callable[[str, str], str] = (
        lambda seq, msa, threads:
        f'mafft --add {seq} --keeplength --anysymbol --thread {threads} {msa}')
    alignment_profile_path: str = './mbae_resources/binding_regions.fsa'
    available_sources: t.Tuple = ('iedb', 'bdata')
    iedb_url: str = "https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip"
    bdata_url: str = "http://tools.iedb.org/static/main/binding_data_2013.zip"
    ipd_history_url: str = "https://raw.githubusercontent.com/ANHIG/IPDMHC/Latest/MHC.xml"
    ipd_sequences_url: str = "https://raw.githubusercontent.com/ANHIG/IPDMHC/Latest/MHC_prot.fasta"
    imgt_history_url: str = "https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/Allelelist_history.txt"
    imgt_sequences_url: str = "https://raw.githubusercontent.com/ANHIG/IMGTHLA/Latest/hla_prot.fasta"
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
        ('Method/Technique', 'method')
    )
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
        ('<', '<')
    )
    iedb_quantitative_assays: t.Tuple[str, ...] = (
        'cellular MHC/competitive/fluorescence',
        'cellular MHC/competitive/radioactivity',
        'cellular MHC/direct/fluorescence',
        'purified MHC/competitive/fluorescence',
        'purified MHC/competitive/radioactivity',
        'purified MHC/direct/fluorescence'
    )
    data_source_final_fields: t.Tuple[str, ...] = (
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
    _ord_cutoffs: t.Tuple[int, ...] = (
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

    @property
    def ord_cutoffs(self) -> t.List[int]:
        return list(self._ord_cutoffs)


# Init Constants; this object should be imported elsewhere
Constants = _Constants()


class AbstractResource(metaclass=ABCMeta):
    """
    Abstract base class defining a basic interface for a Resource.
    """

    @abstractmethod
    def fetch(self, url: str):
        raise NotImplementedError

    @abstractmethod
    def parse(self):
        raise NotImplementedError

    @abstractmethod
    def dump(self, dump_path: str):
        raise NotImplementedError


if __name__ == '__main__':
    raise RuntimeError()
