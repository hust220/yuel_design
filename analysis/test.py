#%%
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SDWriter
import sys
import os
sys.path.append('../')
from db_utils import db_connection

#%%
def get_unique_ligand_names():
    """Get unique names from the ligands table."""
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT pocket FROM random_docking")
            names = [row[0] for row in cursor.fetchall()]
    return names

def save_structures_for_pockets():
    """Save ligand and protein structures for each unique pocket."""
    # Create test_structures directory if it doesn't exist
    os.makedirs('test_structures', exist_ok=True)
    
    pockets = get_unique_ligand_names()
    print(f"Found {len(pockets)} unique pockets")
    
    with db_connection() as conn:
        with conn.cursor() as cursor:
            for pocket in pockets:
                # Get ligand structure
                cursor.execute("""
                    SELECT l.mol, l.protein_name 
                    FROM ligands l 
                    WHERE l.name = %s 
                    LIMIT 1
                """, (pocket,))
                ligand_data = cursor.fetchone()
                
                if ligand_data:
                    mol_bytes, protein_name = ligand_data
                    
                    # Save ligand structure
                    ligand_path = f'test_structures/{pocket}_pocket.mol'
                    with open(ligand_path, 'wb') as f:
                        f.write(mol_bytes.tobytes())
                    
                    # Get and save protein structure
                    cursor.execute("""
                        SELECT pdb 
                        FROM proteins 
                        WHERE name = %s
                    """, (protein_name,))
                    protein_data = cursor.fetchone()
                    
                    if protein_data:
                        pdb_bytes = protein_data[0].tobytes()
                        protein_path = f'test_structures/{pocket}_protein.pdb'
                        with open(protein_path, 'wb') as f:
                            f.write(pdb_bytes)
                        print(f"Saved structures for {pocket} (protein: {protein_name})")
                    else:
                        print(f"Warning: No protein structure found for {protein_name}")
                else:
                    print(f"Warning: No ligand structure found for {pocket}")

#%%
# Test the function
save_structures_for_pockets()

#%%
# Read the first molecule from the SDF file
# supplier = Chem.SDMolSupplier('aa.sdf', sanitize=False, strictParsing=False)
supplier = Chem.SDMolSupplier('aa.sdf', strictParsing=False)
mol = next((m for m in supplier if m is not None), None)
pattern = Chem.MolFromSmarts('c1ccccc1')
# pattern = Chem.MolFromSmarts('c1:c:c:c:c:c1')
if mol is not None:
    smarts = Chem.MolToSmarts(mol)
    print(smarts)
    matches = mol.GetSubstructMatches(pattern)
    print(matches)
else:
    print('No valid molecule found in aa.sdf')


# %%
group_smarts = {
    'Benzene': Chem.MolFromSmarts('c1:c:c:c:c:c1'),
    'Pyridine': Chem.MolFromSmarts('n1aaaaa1'),
    'Pyrimidine': Chem.MolFromSmarts('c1ncncn1'),
    'Imidazole': Chem.MolFromSmarts('c1ncn[nH]1'),
    'Indole': Chem.MolFromSmarts('c1ccc2c(c1)[nH]c2'),
    'Furan': Chem.MolFromSmarts('a1occc1'),
    'Thiophene': Chem.MolFromSmarts('a1sccc1'),
    'Oxazole': Chem.MolFromSmarts('a1ocn1'),
}

for group, smarts in group_smarts.items():
    print(group)
    print(Chem.MolToSmarts(smarts))

# %%

# Embed 3D and save to SDF with group name
writer = SDWriter('group_smarts_3d.sdf')
for group, smarts in group_smarts.items():
    if smarts is not None:
        smiles = Chem.MolToSmiles(smarts)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
            AllChem.UFFOptimizeMolecule(mol)
            mol.SetProp('_Name', group)
            writer.write(mol)

# %%
# Create a dictionary with the given data
functional_groups = {
    "Ketone": 106167,
    "Alcohol": 210538,  
    "Ether": 67760, 
    "Phenol": 8883, 
    "Thioether": 13471, 
    "Benzene": 20028,  
    "Furan": 4325,  
    "Amide": 37748,  
    "Amine (Primary/Secondary)": 104979,  
    "Cyclopropane": 16213, 
    "Amine (Tertiary)": 15776, 
    "Ester": 15526,    
    "Pyridine": 12360,  
    "Pyrimidine": 3403, 
    "Thiophene": 1540,  
    "Halogen": 4801,  
    "Cyclobutane": 3058, 
    "Thiol": 6535,  
    "Epoxide": 6719,   
    "Sulfonamide": 652,  
    "Carboxylic Acid": 844,  
    "Oxazole": 1041,  
    "Imidazole": 111,  
    "Nitrile": 2355,  
    "Aldehyde": 702, 
    "Indole": 19
}

# Sort the dictionary by value in descending order
sorted_groups = sorted(functional_groups.items(), key=lambda x: x[1], reverse=True)

# Print the ranked functional groups
for group, count in sorted_groups:
    print(f"{group}: {count}")

# %%
data = {
    "Alcohol": 204618,
    "Amine (Tertiary)": 26721,
    "Thiol": 2853,
    "Benzene": 40151,
    "Pyridine": 9305,
    "Amine (Primary/Secondary)": 131028,
    "Ether": 59964,
    "Pyrimidine": 19690,
    "Imidazole": 12269,
    "Furan": 15100,
    "Thioether": 7366,
    "Halogen": 14749,
    "Thiophene": 1597,
    "Phenol": 9680,
    "Cyclopropane": 1299,
    "Epoxide": 464,
    "Cyclobutane": 1216,
    "Oxazole": 126,
    "Indole": 5
}

# Sort by values in descending order
ranked_data = sorted(data.items(), key=lambda x: x[1], reverse=True)

# Print ranked data
for rank, (name, count) in enumerate(ranked_data, start=1):
    print(f"{name}: {count}")

# %%
# First group data
group1 = {
    "Alcohol": 204618,
    "Amine (Primary/Secondary)": 131028,
    "Ether": 59964,
    "Benzene": 40151,
    "Amine (Tertiary)": 26721,
    "Pyrimidine": 19690,
    "Furan": 15100,
    "Halogen": 14749,
    "Imidazole": 12269,
    "Phenol": 9680,
    "Pyridine": 9305,
    "Thioether": 7366,
    "Thiol": 2853,
    "Thiophene": 1597,
    "Cyclopropane": 1299,
    "Cyclobutane": 1216,
    "Epoxide": 464,
    "Oxazole": 126,
    "Indole": 5
}

# Second group data
group2 = {
    "Alcohol": 210538,
    "Ketone": 106167,
    "Amine (Primary/Secondary)": 104979,
    "Ether": 67760,
    "Amide": 37748,
    "Benzene": 20028,
    "Cyclopropane": 16213,
    "Amine (Tertiary)": 15776,
    "Ester": 15526,
    "Thioether": 13471,
    "Pyridine": 12360,
    "Phenol": 8883,
    "Epoxide": 6719,
    "Thiol": 6535,
    "Halogen": 4801,
    "Furan": 4325,
    "Pyrimidine": 3403,
    "Cyclobutane": 3058,
    "Nitrile": 2355,
    "Thiophene": 1540,
    "Oxazole": 1041,
    "Carboxylic Acid": 844,
    "Aldehyde": 702,
    "Sulfonamide": 652,
    "Imidazole": 111,
    "Indole": 19
}

# Find items in group2 that are not in group1
not_in_group1 = {key: group2[key] for key in group2 if key not in group1}

# Print the result
print("Items in the second group but not in the first group:")
for key, value in not_in_group1.items():
    print(f"{key}: {value}")

# %%

# save best_pose (bytea) of table medusadock_results where molecule_id=67234 to a file 3zcw_0_design.pdb
with db_connection() as conn:
    with conn.cursor() as cursor:
        cursor.execute("SELECT best_pose FROM medusadock_results WHERE molecule_id = 67234")
        best_pose = cursor.fetchone()[0]
        with open('3zcw_0_dock.pdb', 'wb') as f:
            f.write(best_pose)

# save sdf (bytea) of table molecules where id=67234 to a file 3zcw_0.sdf
with db_connection() as conn:
    with conn.cursor() as cursor:
        cursor.execute("SELECT sdf FROM molecules WHERE id = 67234")
        sdf = cursor.fetchone()[0]
        with open('3zcw_0_design.sdf', 'wb') as f:
            f.write(sdf)
# %%



