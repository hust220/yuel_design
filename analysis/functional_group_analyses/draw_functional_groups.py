#%%

import matplotlib.pyplot as plt

# 数据
data = {
    "Alcohol": 362758,
    "Ketone": 200480,
    "Amine (Primary/Secondary)": 188790,
    "Ether": 154699,
    "Amide": 69024,
    "Amine (Tertiary)": 47347,
    "Benzene": 46937,
    "Pyridine": 40381,
    "Ester": 32602,
    "Cyclopropane": 27614,
    "Thioether": 26122,
    "Furan": 21092,
    "Phenol": 21080,
    "Epoxide": 11919,
    "Pyrimidine": 11846,
    "Thiol": 10836,
    "Halogen": 10534,
    "Cyclobutane": 6410,
    "Nitrile": 4365,
    "Thiophene": 4323,
    "Imidazole": 3357,
    "Oxazole": 3026,
    "Sulfonamide": 1967,
    "Carboxylic Acid": 1528,
    "Aldehyde": 1381,
    "Indole": 526
}

# 数据处理：除以 21000
data_normalized = {key: value / 21000 for key, value in data.items()}

# 排序
data_sorted = dict(sorted(data_normalized.items(), key=lambda x: x[1], reverse=True))

# 提取键和值
labels = list(data_sorted.keys())
values = list(data_sorted.values())

# 创建子图
fig, axs = plt.subplots(2, 1, figsize=(12, 16))

# 普通柱状图
axs[0].barh(labels, values, color='skyblue')
axs[0].set_xlabel("Count / 21000")
axs[0].set_title("Normalized Data (Divided by 21000)")
axs[0].invert_yaxis()
axs[0].grid(axis='x', linestyle='--', alpha=0.7)

# 对数刻度柱状图
axs[1].barh(labels, values, color='skyblue')
axs[1].set_xlabel("Log Scale (Count / 21000)")
axs[1].set_title("Normalized Data (Log Scale)")
axs[1].invert_yaxis()
axs[1].set_xscale('log')
axs[1].grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()



# %%
# Import necessary libraries from RDKit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import os

# Define the dictionary of functional groups and their SMARTS strings
# SMARTS (SMiles ARbitrary Target Specification) is a language for describing molecular patterns.
molecules = {
    'Carboxylic Acid': '*-C(=O)-O',  # Carbon (not in a ring) double bonded to an Oxygen, single bonded to an OH group
    'Ester': '*-C(=O)-O-*',          # Carbon (not in a ring) double bonded to an Oxygen, single bonded to an Oxygen which is bonded to a Carbon
    'Amide': '*-C(=O)-N',              # Carbon (not in a ring) double bonded to an Oxygen, single bonded to a Nitrogen
    'Ketone': '*-C(=O)-C',          # Carbon (not in a ring) double bonded to an Oxygen, single bonded to another Carbon
    'Aldehyde': '*-C(=O)C',       # Carbon (with 3 connections, one of which is a Hydrogen) double bonded to an Oxygen, single bonded to a Carbon

    # Nitrogen functional groups
    'Amine (Primary/Secondary)': '*-C-N,*-N', # Nitrogen (with 3 connections, 2 or 1 Hydrogens, not part of an amide)
    'Amine (Tertiary)': 'N(-C)(-C)(-C)',       # Nitrogen (with 3 connections) bonded to three Carbons
    'Nitrile': '*-C#N',                           # Carbon (not in a ring) triple bonded to a Nitrogen

    # Oxygen functional groups
    'Alcohol': '*-C-[OH]',                            # Carbon (with 4 single bonds) bonded to an OH group
    'Phenol': 'c1ccccc1[OH]',                         # An OH group bonded to an aromatic carbon ring (benzene ring) - simplified for typical representation
    'Ether': 'O(-C)(-C)',                        # Oxygen (with 2 single bonds) bonded to two Carbons
    'Epoxide': 'O1-C-C1',                   # Two Carbons and one Oxygen forming a three-membered ring

    # Sulfur functional groups
    'Thiol': 'S-H',                                # Sulfur (atomic number 16) bonded to one Hydrogen
    'Thioether': 'S(-C)(-C)',                  # Sulfur (atomic number 16, with 2 connections) bonded to two Carbons
    'Sulfonamide': 'S(=O)(=O)-N',                  # Sulfur (not in a ring) double bonded to two Oxygens, and single bonded to a Nitrogen

    # Halogen
    'Halogen': '[F,Cl,Br,I]',                          # Fluorine, Chlorine, Bromine, or Iodine

    # Aromatic ring structures
    'Benzene': '*-C1:C:C:C:C:C1',                             # A 6-membered aromatic ring of Carbons
    'Pyridine': '*-N1:C:C:C:C:C1',                            # A 6-membered aromatic ring with one Nitrogen
    'Pyrimidine': '*-N1:C:C:C:C:C1',                       # A 6-membered aromatic ring with two Nitrogens (positions 1 and 3)
    'Imidazole': '*-N1:C:C:C:C:C1',                         # A 5-membered aromatic ring with two Nitrogens
    'Indole': '*-C1:C:C:C:C:C1',                    # Benzene ring fused to a pyrrole ring
    'Furan': '*-C1:O:C:C:C1',                                # A 5-membered aromatic ring with one Oxygen
    'Thiophene': '*-C1:S:C:C:C:C1',                            # A 5-membered aromatic ring with one Sulfur
    'Oxazole': '*-C1=N-C=C-O1',                              # A 5-membered aromatic ring with one Nitrogen and one Oxygen

    # Special ring structures
    'Cyclopropane': '*-C1-C-C-C1',                           # A 3-membered ring of Carbons
    'Cyclobutane': '*-C1-C-C-C-C1',                           # A 4-membered ring of Carbons
}

# Create a directory to save the SVG files, if it doesn't already exist
output_directory = "figures/functional_groups"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created directory: {output_directory}")

# Loop through the molecules dictionary
for name, smarts in molecules.items():
    # Create a molecule object from the SMARTS string
    mol = Chem.MolFromSmarts(smarts)

    if mol:
        try:
            # Prepare the molecule for drawing (computes 2D coordinates)
            # AllChem.Compute2DCoords(mol) # Often not needed for MolFromSmarts if drawing simple fragments

            # Create a drawing object
            # The MolDraw2DSVG class is used to generate SVG output.
            # We specify the width and height of the SVG image.
            drawer = rdMolDraw2D.MolDraw2DSVG(300, 300) # width, height

            # Optional: Set drawing options for better visualization
            # drawer.SetFontSize(0.8 * drawer.FontSize()) # Adjust font size if needed
            # opts = drawer.drawOptions()
            # opts.addStereoAnnotation = True # Add stereo annotations (e.g., R/S)
            # opts.addAtomIndices = True # Add atom indices (numbers)

            # Draw the molecule
            # For SMARTS, it's often better to draw the query itself if it's a fragment
            # Highlight the SMARTS pattern if drawing a larger molecule that *contains* the SMARTS:
            # drawer.DrawMolecule(mol, highlightAtoms=mol.GetSubstructMatch(Chem.MolFromSmarts(smarts)))
            # For drawing the SMARTS fragment itself:
            drawer.DrawMolecule(mol)

            drawer.FinishDrawing()

            # Get the SVG string
            svg_content = drawer.GetDrawingText()

            # Define the filename for the SVG file
            # Replace spaces and special characters in the name for a valid filename
            filename = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "") + ".svg"
            filepath = os.path.join(output_directory, filename)

            # Write the SVG content to a file
            with open(filepath, 'w') as f:
                f.write(svg_content)
            print(f"Successfully saved: {filepath}")

        except Exception as e:
            print(f"Error drawing or saving {name}: {e}")
            print(f"  SMARTS: {smarts}")
    else:
        # Print a message if the SMARTS string could not be converted to a molecule
        print(f"Could not generate molecule for '{name}' from SMARTS: {smarts}")
        print("  This might be due to an invalid SMARTS string or a pattern that RDKit cannot directly visualize as a standalone structure.")
        print("  Consider if the SMARTS is intended as a query rather than a complete molecule.")

print("\nProcessing complete.")
print(f"SVG files are saved in the '{output_directory}' directory.")


# %%

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from IPython.display import display, SVG
from rdkit.Chem.Draw import rdMolDraw2D
import os
# Function to visualize a molecule from SMARTS pattern
def visualize_from_smarts(smarts, title):
    # Convert SMARTS to a molecule object
    mol = Chem.MolFromSmarts(smarts)
    
    # For better visualization, convert to SMILES and back to ensure proper hydrogen treatment
    try:
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        
        # Delete explicit hydrogens
        mol = Chem.RemoveHs(mol)
        # Add explicit hydrogens for better visualization
        # mol = Chem.AddHs(mol)
        
        # Compute 2D coordinates
        AllChem.Compute2DCoords(mol)
    except:
        # If conversion to SMILES fails, just use the original SMARTS mol
        # This can happen with some SMARTS patterns
        pass
    
    # Set the title as a property
    if mol:
        mol.SetProp("_Name", f"{title} ({smarts})")
    
    return mol

def draw_svg(mol, name):
    output_directory = "figures/functional_groups"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")

    if mol:
        try:
            # Prepare the molecule for drawing (computes 2D coordinates)
            # AllChem.Compute2DCoords(mol) # Often not needed for MolFromSmarts if drawing simple fragments

            # Create a drawing object
            # The MolDraw2DSVG class is used to generate SVG output.
            # We specify the width and height of the SVG image.
            drawer = rdMolDraw2D.MolDraw2DSVG(60, 60) # width, height

            # Optional: Set drawing options for better visualization
            drawer.SetFontSize(12) # Adjust font size if needed
            # drawer.SetFontSize(1.5 * drawer.FontSize()) # Adjust font size if needed
            # opts = drawer.drawOptions()
            # opts.addStereoAnnotation = True # Add stereo annotations (e.g., R/S)
            # opts.addAtomIndices = True # Add atom indices (numbers)

            # Draw the molecule
            # For SMARTS, it's often better to draw the query itself if it's a fragment
            # Highlight the SMARTS pattern if drawing a larger molecule that *contains* the SMARTS:
            # drawer.DrawMolecule(mol, highlightAtoms=mol.GetSubstructMatch(Chem.MolFromSmarts(smarts)))
            # For drawing the SMARTS fragment itself:
            drawer.DrawMolecule(mol)

            drawer.FinishDrawing()

            # Get the SVG string
            svg_content = drawer.GetDrawingText()

            # also display in jupyter notebook
            display(SVG(svg_content))

            # Define the filename for the SVG file
            # Replace spaces and special characters in the name for a valid filename
            filename = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "") + ".svg"
            filepath = os.path.join(output_directory, filename)

            # Write the SVG content to a file
            with open(filepath, 'w') as f:
                f.write(svg_content)
            print(f"Successfully saved: {filepath}")

        except Exception as e:
            print(f"Error drawing or saving {name}: {e}")
            print(f"  SMARTS: {smarts}")
    else:
        # Print a message if the SMARTS string could not be converted to a molecule
        print(f"Could not generate molecule for '{name}' from SMARTS: {smarts}")
        print("  This might be due to an invalid SMARTS string or a pattern that RDKit cannot directly visualize as a standalone structure.")
        print("  Consider if the SMARTS is intended as a query rather than a complete molecule.")


# Create a list of all representations of oxazole
molecules = {
    'Carboxylic Acid': '*-C(=O)-O',  # Carbon (not in a ring) double bonded to an Oxygen, single bonded to an OH group
    'Ester': '*-C(=O)-O-*',          # Carbon (not in a ring) double bonded to an Oxygen, single bonded to an Oxygen which is bonded to a Carbon
    'Amide': '*-C(=O)-N',              # Carbon (not in a ring) double bonded to an Oxygen, single bonded to a Nitrogen
    'Ketone': '*-C(=O)-*',          # Carbon (not in a ring) double bonded to an Oxygen, single bonded to another Carbon
    'Aldehyde': '*-C=O',       # Carbon (with 3 connections, one of which is a Hydrogen) double bonded to an Oxygen, single bonded to a Carbon

    # Nitrogen functional groups
    'Amine (Primary/Secondary)': '*-C-N', # Nitrogen (with 3 connections, 2 or 1 Hydrogens, not part of an amide)
    'Amine (Tertiary)': 'N(-C)(-C)(-C)',       # Nitrogen (with 3 connections) bonded to three Carbons
    'Nitrile': '*-C#N',                           # Carbon (not in a ring) triple bonded to a Nitrogen

    # Oxygen functional groups
    'Alcohol': '*-C-[OH]',                            # Carbon (with 4 single bonds) bonded to an OH group
    'Phenol': 'O-C1=C-C=C-C=C1',                         # An OH group bonded to an aromatic carbon ring (benzene ring) - simplified for typical representation
    'Ether': 'O(-*)(-*)',                        # Oxygen (with 2 single bonds) bonded to two Carbons
    'Epoxide': '*-C1-O-C1-*',                   # Two Carbons and one Oxygen forming a three-membered ring

    # Sulfur functional groups
    'Thiol': '*-S',                                # Sulfur (atomic number 16) bonded to one Hydrogen
    'Thioether': 'S(-*)(-*)',                  # Sulfur (atomic number 16, with 2 connections) bonded to two Carbons
    'Sulfonamide': '*-S(=O)(=O)-N-*',                  # Sulfur (not in a ring) double bonded to two Oxygens, and single bonded to a Nitrogen

    # Halogen
    'Halogen': '*-[F,Cl,Br,I]',                          # Fluorine, Chlorine, Bromine, or Iodine

    # Aromatic ring structures
    'Benzene': 'C1=C-C=C-C=C1',                             # A 6-membered aromatic ring of Carbons
    'Pyridine': 'C1=N-C=C-C=C1',                            # A 6-membered aromatic ring with one Nitrogen
    'Pyrimidine': 'C1=N-C=C-N=C1',                       # A 6-membered aromatic ring with two Nitrogens (positions 1 and 3)
    'Imidazole': 'C1=N-C=C-N1',                         # A 5-membered aromatic ring with two Nitrogens
    'Indole': 'C1=C-N-C2-C=C-C=C-C21',                    # Benzene ring fused to a pyrrole ring
    'Furan': 'C1=C-C=C-O1',                                # A 5-membered aromatic ring with one Oxygen
    'Thiophene': 'C1=C-C=C-S1',                            # A 5-membered aromatic ring with one Sulfur
    'Oxazole': 'C1=N-C=C-O1',                              # A 5-membered aromatic ring with one Nitrogen and one Oxygen

    # Special ring structures
    'Cyclopropane': 'C1-C-C1',                           # A 3-membered ring of Carbons
    'Cyclobutane': 'C1-C-C-C1',                           # A 4-membered ring of Carbons
}

# Generate molecules from all representations
# mols = []
# legends = []

def show_svg(mol, name):
    img = Draw.MolToImage(mol)
    display(img)

for name, smarts in molecules.items():
    mol = visualize_from_smarts(smarts, name)
    print(name)
    draw_svg(mol, name)
    # show_svg(mol, name)


    # if mol:
    #     mols.append(mol)
    #     legends.append(f"{name}\n{smarts}")

# Display molecules in a grid
# img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(300, 300), legends=legends)
# display(img)
# %%
