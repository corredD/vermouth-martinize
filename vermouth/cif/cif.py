# -*- coding: utf-8 -*-
# Copyright 2018 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Provides functions for reading and writing CIF files.
"""

import numpy as np
import networkx as nx

# for now or use a new parser from scratch ?
import gemmi

from ..file_writer import deferred_open
from ..molecule import Molecule
from ..utils import first_alpha, distance, format_atom_string
from ..parser_utils import LineParser
from ..truncating_formatter import TruncFormatter
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))


class CIFParser():
    """
    Parser for CIF files

    Attributes
    ----------
    active_molecule: vermouth.molecule.Molecule
        The molecule/model currently being read.
    molecules: list[vermouth.molecule.Molecule]
        All complete molecules read so far.
    modelidx: int
        Which model to take.

    Parameters
    ----------
    exclude: collections.abc.Container[str]
        Container of residue names. Any atom that has a residue name that is in
        `exclude` will be skipped.
    ignh: bool
        Whether all hydrogen atoms should be skipped
    modelidx: int
        Which model to take.
    """

    def __init__(self, exclude=('SOL',), ignh=False, modelidx=1):
        self.active_molecule = Molecule()
        self.molecules = []
        self._conects = []
        self.exclude = exclude
        self.ignh = ignh
        self.modelidx = modelidx
        self._skipahead = False

    def dispatch(self, line):
        """
        Returns the appropriate method for parsing `line`. This is determined
        based on the first 6 characters of `line`.

        Parameters
        ----------
        line: str

        Returns
        -------
        collections.abc.Callable[str, int]
            The method to call with the line, and the line number.
        """
        record = line[:6].strip().lower()
        return getattr(self, record, self._unknown_line)

    def parse(self, file_handle):
        # Only CIFParser.finalize should produce a result, namely a list of
        # molecules. This means that mols is a list containing a single list of
        # molecules, which is a little silly.
        cif_block = gemmi.cif.read(file_handle)[0]
        st = gemmi.make_structure_from_block(cif_block)
        # convert to vermouth molecules through properties
        st.remove_waters()
        st.add_entity_types()
        # always use first BU
        assem = gemmi.make_assembly(st.assemblies[0], st[0], gemmi.HowToNameCopiedChain.AddNumber)
        for n_ch, chain in enumerate(assem):
            print(n_ch, chain.name)
            for n_res, res in enumerate(chain):
                for n_atom, atom in enumerate(res):
                    properties = {}
                    properties['atomid'] = atom.serial
                    properties['atomname'] = atom.name
                    properties['altloc'] = atom.altloc
                    properties['resname'] = res.name
                    properties['chain'] = chain.name
                    properties['resid'] = res.seqid.num
                    properties['insertion_code'] = res.seqid.icode
                    # Coordinates are read in Angstrom, but we want them in nm
                    properties['position'] = np.array(
                        [atom.pos.x,
                         atom.pos.y,
                         atom.pos.z], dtype=float) / 10.0
                    properties['element'] = first_alpha(atom.element.name)
                    properties['occupancy'] = atom.occ
                    properties['temp_factor'] = atom.b_iso
                    properties['charge'] = atom.charge
                    if (properties['resname'] in self.exclude or
                       (self.ignh and properties['element'] == 'H')):
                        continue
                    idx = max(self.active_molecule) + 1 if self.active_molecule else 0
                    self.active_molecule.add_node(idx, **properties)

        return self.finalize()

    def conect(self, line, lineno=0):
        """
        Parse a CONECT record. The line is stored for later processing.

        Parameters
        ----------
        line: str
            The line to parse. Should start with CONECT, but this is not checked
        lineno: int
            The line number (not used).
        """
        # We can't add edges immediately, since the molecule might not be parsed
        # yet (does the CIF file format mandate anything on the order of
        # records?). Instead, just store the lines for later use.
        self._conects.append(line)

    def _finish_molecule(self, line="", lineno=0):
        """
        Finish parsing the molecule. :attr:`active_molecule` will be appended to
        :attr:`molecules`, and a new :attr:`active_molecule` will be made.
        """
        # We kind of *want* to yield self.active_molecule here, but we can't
        # since there's a very good chance it's CONECT records have not been
        # parsed yet, and the molecule won't have any edges.
        if self.active_molecule:
            self.molecules.append(self.active_molecule)
        self.active_molecule = Molecule()

    def finalize(self, lineno=0):
        """
        Finish parsing the file. Process all CONECT records found, and returns
        a list of molecules.

        Parameters
        ----------
        lineno: int
            The line number (not used).

        Returns
        -------
        list[vermouth.molecule.Molecule]
            All molecules parsed from this file.
        """
        # TODO: cross reference number of molecules with CMPND records
        self._finish_molecule()
        # self.do_conect()
        return self.molecules


def read_cif(file_name, exclude=('SOL',), ignh=False, modelidx=1):
    """
    Parse a CIF file to create a molecule.

    Parameters
    ----------
    filename: str
        The file to read.
    exclude: collections.abc.Container[str]
        Atoms that have one of these residue names will not be included.
    ignh: bool
        Whether hydrogen atoms should be ignored.
    model: int
        If the CIF file contains multiple models, which one to select.

    Returns
    -------
    list[vermouth.molecule.Molecule]
        The parsed molecules. Will only contain edges if the CIF file has
        CONECT records. Either way, the molecules might be disconnected. Entries
        separated by TER, ENDMDL, and END records will result in separate
        molecules.
    """
    parser = CIFParser(exclude, ignh, modelidx)
    mols = list(parser.parse(file_name))
    LOGGER.info('Read {} molecules from CIF file {}', len(mols), file_name)
    return mols


def get_not_none(node, attr, default):
    """
    Returns ``node[attr]``. If it doesn't exists or is ``None``, return
    `default`.

    Parameters
    ----------
    node: collections.abc.Mapping
    attr: collections.abc.Hashable
    default
        The value to return if ``node[attr]`` is either ``None``, or does not
        exist.

    Returns
    -------
    object
        The value of ``node[attr]`` if it exists and is not ``None``, else
        `default`.
    """
    value = node.get(attr)
    if value is None:
        value = default
    return value

