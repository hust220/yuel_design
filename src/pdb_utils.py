#%%

import numpy as np
from typing import List, Tuple, Dict, Optional, Set, Iterator, Union
import os
from dataclasses import dataclass
from collections import defaultdict
import requests
import tempfile
from io import StringIO

def three_to_one_letter(res_name: str) -> str:
    """Convert three-letter amino acid code to one-letter code.
    
    Args:
        res_name: Three-letter amino acid code
        
    Returns:
        One-letter amino acid code, or 'X' if unknown
    """
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z', 'XLE': 'J',
        'XAA': 'X', 'UNK': 'X'
    }
    return three_to_one.get(res_name.upper(), 'X')

@dataclass
class Atom:
    """Represents an atom in a PDB file."""
    record: str  # ATOM or HETATM
    atom_id: int
    atom_name: str
    alt_loc: str
    res_name: str
    chain_id: str
    res_id: int
    insertion: str
    x: float
    y: float
    z: float
    occupancy: float
    temp_factor: float
    element: str
    charge: str
    pdb_line: Optional[str] = None  # Store the original PDB line if available
    
    def get_coord(self) -> np.ndarray:
        """Get atom coordinates as numpy array."""
        return np.array([self.x, self.y, self.z])
    
    def to_pdb(self) -> str:
        """Convert atom to PDB format line."""
        # if self.pdb_line is not None:
        #     return self.pdb_line
            
        record = f"{self.record:<6}"
        atom_id = f"{self.atom_id:>5}"
        atom_name = f"{self.atom_name:<4}"
        alt_loc = self.alt_loc
        res_name = f"{self.res_name:<3}"
        chain_id = self.chain_id
        res_id = f"{self.res_id:>4}"
        insertion = self.insertion
        x = f"{self.x:>8.3f}"
        y = f"{self.y:>8.3f}"
        z = f"{self.z:>8.3f}"
        occupancy = f"{self.occupancy:>6.2f}"
        temp_factor = f"{self.temp_factor:>6.2f}"
        element = f"{self.element:>2}"
        charge = f"{self.charge:>2}"
        
        return f"{record}{atom_id} {atom_name}{alt_loc}{res_name} {chain_id}{res_id}{insertion}   {x}{y}{z}{occupancy}{temp_factor}          {element}{charge}\n"

class Residue:
    """Represents a residue in a PDB file."""
    def __init__(self, res_name: str, res_id: int, chain_id: str, insertion: str = ' '):
        self.res_name = res_name
        self.res_id = res_id
        self.chain_id = chain_id
        self.insertion = insertion
        self.atoms: List[Atom] = []
        self.is_hetatm_record: bool = False  # Track if this residue is from HETATM records
        
    def add_atom(self, atom: Atom) -> None:
        """Add an atom to the residue."""
        self.atoms.append(atom)
        # If this is the first atom, set is_hetatm_record based on the atom's record type
        if len(self.atoms) == 1:
            self.is_hetatm_record = atom.record == 'HETATM'
        
    def get_atom(self, atom_name: str) -> Optional[Atom]:
        """Get atom by name."""
        for atom in self.atoms:
            if atom.atom_name == atom_name:
                return atom
        return None
    
    def get_atoms(self) -> List[Atom]:
        """Get all atoms in the residue.
        
        Returns:
            List of all atoms in the residue
        """
        return self.atoms
    
    def get_coords(self) -> np.ndarray:
        """Get coordinates of all atoms in the residue."""
        return np.array([atom.get_coord() for atom in self.atoms])
    
    def is_hetatm(self) -> bool:
        """Check if this residue is a HETATM record.
        
        Returns:
            bool: True if this residue was read from HETATM records, False otherwise
        """
        return self.is_hetatm_record
    
    def to_pdb(self) -> str:
        """Convert residue to PDB format string.
        
        Returns:
            PDB format string for all atoms in the residue
        """
        return '\n'.join(atom.to_pdb().rstrip() for atom in self.atoms) + '\n'
    
    def __iter__(self) -> Iterator[Atom]:
        """Iterate over atoms in the residue."""
        return iter(self.atoms)

class Chain:
    """Represents a chain in a PDB file."""
    def __init__(self, chain_id: str):
        self.chain_id = chain_id
        self.residues: List[Residue] = []
        self._residue_dict: Dict[Tuple[int, str], Residue] = {}  # (res_id, insertion) -> Residue
        
    def add_residue(self, residue: Residue) -> None:
        """Add a residue to the chain."""
        self.residues.append(residue)
        self._residue_dict[(residue.res_id, residue.insertion)] = residue
        
    def get_residue(self, res_id: int, insertion: str = ' ') -> Optional[Residue]:
        """Get residue by ID and insertion code."""
        return self._residue_dict.get((res_id, insertion))
    
    def get_residues(self) -> List[Residue]:
        """Get all residues in the chain.
        
        Returns:
            List of all residues in the chain, ordered by residue ID
        """
        return self.residues
    
    def get_atoms(self) -> List[Atom]:
        """Get all atoms in the chain.
        
        Returns:
            List of all atoms in the chain, ordered by residue and atom
        """
        atoms = []
        for residue in self.residues:
            atoms.extend(residue.atoms)
        return atoms
    
    def to_pdb(self) -> str:
        """Convert chain to PDB format string.
        
        Returns:
            PDB format string for all atoms in the chain
        """
        return '\n'.join(residue.to_pdb().rstrip() for residue in self.residues) + '\n'
    
    def __iter__(self) -> Iterator[Residue]:
        """Iterate over residues in the chain."""
        return iter(self.residues)

class Model:
    """Represents a model in a PDB file."""
    def __init__(self, model_id: int = 1):
        self.model_id = model_id
        self.chains: List[Chain] = []
        
    def add_chain(self, chain: Chain) -> None:
        """Add a chain to the model."""
        self.chains.append(chain)
        
    def get_residues(self) -> List[Residue]:
        """Get all residues in the model.
        
        Returns:
            List of all residues in the model, ordered by chain and residue ID
        """
        residues = []
        for chain in self.chains:
            residues.extend(chain.residues)
        return residues
    
    def get_atoms(self) -> List[Atom]:
        """Get all atoms in the model.
        
        Returns:
            List of all atoms in the model, ordered by chain, residue and atom
        """
        atoms = []
        for chain in self.chains:
            atoms.extend(chain.get_atoms())
        return atoms
    
    def to_pdb(self) -> str:
        """Convert model to PDB format string.
        
        Returns:
            PDB format string for all atoms in the model
        """
        if len(self.chains) == 0:
            return ""
        lines = [f"MODEL     {self.model_id:>4}"]
        for chain in self.chains:
            lines.append(chain.to_pdb().rstrip())
        lines.append("ENDMDL")
        return '\n'.join(lines) + '\n'
    
    def __iter__(self) -> Iterator[Chain]:
        """Iterate over chains in the model."""
        return iter(self.chains)

class Structure:
    """Represents a molecular structure from a PDB file."""
    
    def __init__(self, file_path: Optional[str] = None, **kwargs):
        """Initialize a structure.
        
        Args:
            file_path: Optional path to a PDB file to load
            **kwargs: Additional keyword arguments:
                - skip_hetatm: Whether to skip HETATM records (default: False)
                - skip_water: Whether to skip water molecules (default: True)
                - altloc: List of altloc codes to keep (default: [' ', 'A'])
        """
        self.models: List[Model] = []
        self.file_path = file_path
        self.kwargs = kwargs
        
        if file_path is not None:
            self.read(file_path, **kwargs)
    
    def __getitem__(self, index: int) -> Model:
        return self.models[index]
    
    def read(self, source, **kwargs) -> None:
        """Read a PDB file from various sources.
        
        This method automatically detects the type of input and handles it appropriately.
        It can read from:
        - File path (str or Path)
        - File handle (file-like object)
        
        Args:
            source: The source to read from. Can be:
                - str or Path: Path to a PDB file
                - file-like object: An object that supports iteration over lines
            skip_hetatm: Whether to skip HETATM records (default: False)
            skip_water: Whether to skip water molecules (default: True)
            altloc: List of altloc codes to keep (default: [' ', 'A'])
        """
        self.models = []
        
        # Handle file path (str or Path)
        if isinstance(source, (str, os.PathLike)):
            if os.path.exists(source):
                self.file_path = str(source)
                with open(source, 'r') as f:
                    self._read_from_handle(f, **kwargs)
                return
            else:
                raise FileNotFoundError(f"PDB file not found: {source}")
            
        # Handle file-like object
        self._read_from_handle(source, **kwargs)
    
    def _read_from_handle(self, handle, **kwargs) -> None:
        """Internal method to read PDB from a file handle.
        
        Args:
            handle: A file-like object that supports iteration over lines
            **kwargs: Additional keyword arguments:
                - skip_hetatm: Whether to skip HETATM records (default: False)
                - skip_water: Whether to skip water molecules (default: True)
                - altloc: List of altloc codes to keep (default: [' ', 'A'])
        """
        current_model = None
        current_chain = None
        current_residue = None
        skip_hetatm = kwargs.get('skip_hetatm', False)
        skip_water = kwargs.get('skip_water', True)
        altloc = kwargs.get('altloc', [' ', 'A'])  # Default to space and 'A'
        
        for line in handle:
            if line.startswith('MODEL'):
                try:
                    model_id = int(line[10:14])
                except ValueError:
                    model_id = len(self.models) + 1
                current_model = Model(model_id)
                self.models.append(current_model)
                current_chain = None
                current_residue = None
                
            elif line.startswith('ENDMDL'):
                # End current model when ENDMDL found
                if current_model is not None:
                    current_model = None
            elif line.startswith('ATOM') or line.startswith('HETATM'):
                # Skip HETATM records if specified
                if skip_hetatm and line.startswith('HETATM'):
                    continue
                    
                # Check altloc
                atom_altloc = line[16]
                if atom_altloc not in altloc:
                    continue
                    
                atom = Atom(
                    record=line[0:6].strip(),
                    atom_id=int(line[6:11]),
                    atom_name=line[12:16].strip(),
                    alt_loc=atom_altloc,
                    res_name=line[17:20].strip(),
                    chain_id=line[21],
                    res_id=int(line[22:26]),
                    insertion=line[26],
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                    occupancy=float(line[54:60]),
                    temp_factor=float(line[60:66]),
                    element=line[76:78].strip(),
                    charge=line[78:80].strip(),
                    pdb_line=line  # Store the original PDB line
                )
                
                # Skip water molecules if specified
                if skip_water and (atom.res_name in ['HOH', 'WAT', 'TIP3', 'H2O', 'SOL']):
                    continue

                if current_model is None:
                    current_model = Model(len(self.models) + 1)
                    self.models.append(current_model)
                    current_chain = None
                    current_residue = None

                if current_chain is None or atom.chain_id != current_chain.chain_id:
                    current_chain = Chain(atom.chain_id)
                    current_model.add_chain(current_chain)
                    current_residue = None

                if current_residue is None or atom.res_id != current_residue.res_id or atom.insertion != current_residue.insertion or atom.res_name != current_residue.res_name or atom.chain_id != current_residue.chain_id:
                    current_residue = Residue(atom.res_name, atom.res_id, atom.chain_id, atom.insertion)
                    current_chain.add_residue(current_residue)

                current_residue.add_atom(atom)
    
    def write(self, file_path: str, residue_bfactors: Optional[np.ndarray] = None) -> None:
        """Write the structure to a PDB file.
        
        Args:
            file_path: Path to write the PDB file
            residue_bfactors: Optional array of beta factors (temperature factors) for each residue
        """
        with open(file_path, 'w') as f:
            residue_idx = 0
            for model in self.models:
                if len(self.models) > 1:
                    f.write(f"MODEL     {model.model_id:>4}\n")
                
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            if residue_bfactors is not None and residue_idx < len(residue_bfactors):
                                atom.temp_factor = float(residue_bfactors[residue_idx])
                            f.write(atom.to_pdb())
                        residue_idx += 1
                
                if len(self.models) > 1:
                    f.write("ENDMDL\n")
    
    def to_pdb(self) -> str:
        """Convert structure to PDB format string.
        
        Returns:
            PDB format string for all atoms in the structure
        """
        return '\n'.join(model.to_pdb().rstrip() for model in self.models) + '\n'
    
    def get_models(self) -> List[Model]:
        """Get all models in the structure.
        
        Returns:
            List of all models, ordered by model ID
        """
        return self.models
    
    def __iter__(self) -> Iterator[Model]:
        """Iterate over models in the structure."""
        return iter(self.models)

    def get_protein_sequence(self, chain_id: Optional[str] = None) -> str:
        """Get protein sequence for a specific chain or all chains.
        
        Args:
            chain_id: Optional chain ID to get sequence for. If None, returns concatenated sequences of all chains.
            
        Returns:
            Protein sequence as a string of one-letter amino acid codes
        """
        sequence = []
        
        for model in self.models:
            for chain in model:
                if chain_id is not None and chain.chain_id != chain_id:
                    continue
                    
                for residue in chain:
                    # Skip non-protein residues (HETATM records)
                    if residue.is_hetatm():
                        continue
                        
                    # Skip water and other non-protein molecules
                    if residue.res_name in ['HOH', 'WAT', 'TIP3', 'H2O', 'SOL']:
                        continue
                        
                    # Convert 3-letter code to 1-letter code
                    one_letter = three_to_one_letter(residue.res_name)
                    if one_letter != 'X':  # Only include valid amino acids
                        sequence.append(one_letter)
        
        return ''.join(sequence)

def download_pdb(pdb_id: str, save_to_file: bool = True) -> Union[str, str]:
    """Download a PDB file from RCSB PDB.
    
    Args:
        pdb_id: 4-letter PDB ID
        save_to_file: If True, save to a temporary file and return the file path.
                     If False, return the PDB content as a string.
        
    Returns:
        If save_to_file is True:
            Path to the downloaded PDB file
        If save_to_file is False:
            String containing the PDB file content
    """
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    
    if save_to_file:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdb')
        temp_file.close()
        
        # Write the content to the temporary file
        with open(temp_file.name, 'w') as f:
            f.write(response.text)
        
        return temp_file.name
    else:
        return response.text

def test_structure():
    """Test the Structure class with real PDB files."""
    # Test proteins
    pdb_ids = ['3qdx', '1a4d']
    
    for pdb_id in pdb_ids:
        print(f"\nTesting PDB ID: {pdb_id}")
        
        # Test downloading to file
        pdb_content = download_pdb(pdb_id, save_to_file=False)
        print(f"Downloaded content (first 100 chars): {pdb_content[:100]}...")
        
        try:
            # Test default reading (skip HETATM and water)
            print("\nTesting default reading (keep HETATM and skip water):")
            structure = Structure()
            structure.read(StringIO(pdb_content))
            print(f"Number of models: {len(structure.models)}")
            print(f"Number of residues: {len(structure[0].get_residues())}")
            print(f"Number of atoms: {len(structure[0].get_atoms())}")
            
            # Test reading with HETATM
            print("\nTesting reading with HETATM:")
            structure_with_hetatm = Structure()
            structure_with_hetatm.read(StringIO(pdb_content), skip_hetatm=True)
            print(f"Number of models: {len(structure_with_hetatm.models)}")
            print(f"Number of residues: {len(structure_with_hetatm[0].get_residues())}")
            print(f"Number of atoms: {len(structure_with_hetatm[0].get_atoms())}")
            
            # Test reading with water
            print("\nTesting reading with water:")
            structure_with_water = Structure()
            structure_with_water.read(StringIO(pdb_content), skip_water=False)
            print(f"Number of models: {len(structure_with_water.models)}")
            print(f"Number of residues: {len(structure_with_water[0].get_residues())}")
            print(f"Number of atoms: {len(structure_with_water[0].get_atoms())}")

            # Test model iteration
            print("\nTesting model iteration:") 
            for i, model in enumerate(structure):
                print(f"Model {i+1}: ID={model.model_id}")
                print(f"Number of chains: {len(model.chains)}")
                
                # Test chain iteration
                for chain in model:
                    print(f"\nChain {chain.chain_id}:")
                    print(f"Number of residues: {len(chain.residues)}")
                    
                    # Test residue iteration
                    print("First 5 residues:")
                    for residue in chain.residues[:5]:
                        print(f"  {residue.res_name} {residue.res_id}")
                        print(f"  Number of atoms: {len(residue.atoms)}")
                        
                        # Test atom iteration
                        print("  First 3 atoms:")
                        for atom in residue.atoms[:3]:
                            print(f"    {atom.atom_name}: {atom.get_coord()}")
            
            # Test writing
            output_file = f"test_output_{pdb_id}.pdb"
            structure.write(output_file)
            print(f"\nWrote structure to: {output_file}")
            
            # Test reading the written file
            print("\nTesting reading the written file:")
            new_structure = Structure(output_file)
            print(f"Number of models in written file: {len(new_structure.models)}")
            
        except Exception as e:
            print(f"Error testing {pdb_id}: {str(e)}")

if __name__ == '__main__':
    test_structure() 
# %%
