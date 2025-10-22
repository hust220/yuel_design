# %%

import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict
import seaborn as sns
from matplotlib.colors import ListedColormap
import networkx as nx

class Atom:
    """Class to represent an atom in a PDB file"""
    def __init__(self, line):
        """Initialize from a PDB ATOM/HETATM line"""
        self.line = line
        self.name = line[12:16].strip()
        self.alt_loc = line[16].strip()
        self.resname = line[17:20].strip()
        self.chain_id = line[21]
        try:
            self.res_seq = int(line[22:26])
        except ValueError:
            self.res_seq = 0
        self.x = float(line[30:38])
        self.y = float(line[38:46])
        self.z = float(line[46:54])
        try:
            self.serial = int(line[6:11])
        except ValueError:
            self.serial = 0
    
    def get_coords(self):
        """Return atom coordinates as a numpy array"""
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other_atom):
        """Calculate Euclidean distance to another atom"""
        return np.linalg.norm(self.get_coords() - other_atom.get_coords())
    
    def __str__(self):
        # return f"{self.chain_id}:{self.resname}{self.res_seq}:{self.name}({self.serial})"
        return f"{self.name}{self.serial}"


class Model:
    """Class to represent a model in a PDB file"""
    def __init__(self, model_id):
        self.id = model_id
        self.atoms = []
    
    def add_atom(self, atom):
        """Add an atom to this model"""
        self.atoms.append(atom)
    
    def get_atoms(self):
        """Return the list of atoms in this model"""
        return self.atoms


def parse_pdb_without_biopython(pdb_filepath):
    """
    Parse a PDB file and extract models and atoms without using Biopython.
    
    Args:
        pdb_filepath (str): Path to the PDB file
        
    Returns:
        list: List of Model objects
    """
    models = []
    current_model = None
    
    try:
        with open(pdb_filepath, 'r') as f:
            for line in f:
                if line.startswith("MODEL"):
                    try:
                        model_id = int(line[10:14].strip())
                    except ValueError:
                        model_id = len(models) + 1
                    current_model = Model(model_id)
                    models.append(current_model)
                
                elif line.startswith("ATOM") or line.startswith("HETATM"):
                    # If no MODEL record has been encountered yet, create a default model
                    if current_model is None:
                        current_model = Model(1)
                        models.append(current_model)
                    
                    atom = Atom(line)
                    current_model.add_atom(atom)
                
                elif line.startswith("ENDMDL"):
                    current_model = None
    
    except Exception as e:
        print(f"Error parsing PDB file: {e}")
        return []
    
    # Handle PDB files without MODEL/ENDMDL records (single model)
    if len(models) == 0:
        print("No MODEL records found. Treating file as a single model.")
        try:
            with open(pdb_filepath, 'r') as f:
                model = Model(1)
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        atom = Atom(line)
                        model.add_atom(atom)
                if model.atoms:
                    models.append(model)
        except Exception as e:
            print(f"Error parsing PDB file as single model: {e}")
    
    return models

def determine_bonds(models, bond_cutoff=1.9):
    if not models:
        print("Error: No models found in the PDB file.")
        return
    
    print(f"Number of models found: {len(models)}")
    
    # Convert all models' atoms to lists to ensure consistent indexing
    model_atoms_lists = [model.get_atoms() for model in models]
    
    # Verify all models have the same number of atoms for consistent indexing
    atom_counts = [len(atoms) for atoms in model_atoms_lists]
    if len(set(atom_counts)) > 1:
        print(f"Warning: Models have different atom counts: {atom_counts}")
        print("This may cause incorrect atom tracking by index!")
    
    # Identify bonds in the last model using indices
    last_model_atoms = model_atoms_lists[-1]
    
    if not last_model_atoms:
        print(f"Error: No atoms found in the last model (Model ID: {models[-1].id}).")
        return
    
    print(f"\nProcessing last model (Model ID: {models[-1].id}) to identify bonds using indices:")
    
    bonds_to_track = []  # List of tuples: (atom_idx1, atom_idx2)
    atom_details_map = {}  # To store atom details for plot labels
    
    for i in range(len(last_model_atoms)):
        for j in range(i + 1, len(last_model_atoms)):
            atom1_obj = last_model_atoms[i]
            atom2_obj = last_model_atoms[j]
            distance = atom1_obj.distance_to(atom2_obj)
            # print("i", i, "j", j, "distance", distance)
            
            if distance < bond_cutoff:
                # Store indices as a sorted tuple to avoid duplicates
                # print("i", i, "j", j)
                bond_pair = tuple(sorted((i, j)))
                bonds_to_track.append(bond_pair)
                
                # Store atom details for later use in plots
                if i not in atom_details_map:
                    atom_details_map[i] = str(atom1_obj)
                if j not in atom_details_map:
                    atom_details_map[j] = str(atom2_obj)
                
                print(f"  Identified bond in last model between: "
                      f"Atom (Idx {i}, Name {atom1_obj.name}, Res {atom1_obj.resname}{atom1_obj.res_seq}, Chain {atom1_obj.chain_id}) and "
                      f"Atom (Idx {j}, Name {atom2_obj.name}, Res {atom2_obj.resname}{atom2_obj.res_seq}, Chain {atom2_obj.chain_id}), "
                      f"Dist: {distance:.3f} Å")
    
    bonds_to_track = sorted(list(set(bonds_to_track)))
    
    if not bonds_to_track:
        print("No bonds identified in the last model using the specified cutoff.")
        return
    
    print(f"Total unique bonds (by atom indices) to track: {len(bonds_to_track)}")

    return bonds_to_track, atom_details_map

def track_bonds(models, bonds_to_track):
    model_atoms_lists = [model.get_atoms() for model in models]
    bond_distance_evolution = {bond_pair: [] for bond_pair in bonds_to_track}
    model_ids_for_plot = list(range(1, len(models) + 1))  # Store model IDs for x-axis
    
    for model_idx, model_atoms in enumerate(model_atoms_lists):
        for idx1, idx2 in bonds_to_track:
            # Ensure indices exist in this model's atom list
            if idx1 < len(model_atoms) and idx2 < len(model_atoms):
                atom1 = model_atoms[idx1]
                atom2 = model_atoms[idx2]
                distance = atom1.distance_to(atom2)
                bond_distance_evolution[(idx1, idx2)].append(distance)
            else:
                bond_distance_evolution[(idx1, idx2)].append(None)
                print(f"Warning: Atom indices {idx1} or {idx2} out of range in model {model_idx}")
    
    print("\n--- Bond Distance Evolution (By Atom Indices) ---")
    if not bond_distance_evolution:
        print("No bond distances were tracked.")
        return
    return bond_distance_evolution

def plot_bond_distances(bond_distance_evolution, atom_details_map, output_plot_dir="plots"):
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)
        print(f"\nPlots will be saved to '{output_plot_dir}/' directory.")
    print("len(bond_distance_evolution)", len(bond_distance_evolution))
    for (idx1, idx2), distances in bond_distance_evolution.items():
        atom1_label = atom_details_map.get(idx1, f"Index {idx1}")
        atom2_label = atom_details_map.get(idx2, f"Index {idx2}")
        print(f"\nBond between {atom1_label} <-> {atom2_label}")
        plt.figure(figsize=(4, 3))
        x = list(range(1, len(distances) + 1))
        plt.plot(x, distances, marker='o', linestyle='-', color='#8e7fb8')
        plt.title(f"Bond Distance Evolution\n{atom1_label} <-> {atom2_label}")
        plt.xlabel("Model ID")
        plt.ylabel("Distance (Å)")
        if len(x) > 10:
            plt.xticks(x[::max(1, len(x)//10)], rotation=45, ha="right")
        else:
            plt.xticks(x, rotation=45, ha="right")
        plt.grid(True)
        plt.tight_layout()
        plot_filename_base = f"bond_idx{idx1}-idx{idx2}_evolution"
        plt.savefig(os.path.join(output_plot_dir, plot_filename_base + ".png"))
        plt.savefig(os.path.join(output_plot_dir, plot_filename_base + ".svg"))
        print(f"  Plot saved as {os.path.join(output_plot_dir, plot_filename_base + '.png')} and .svg")
        plt.show()
        plt.clf()

def plot_average_bond_lengths(bond_distance_evolution, output_plot_dir="plots"):
    average_bond_lengths = {}
    for (idx1, idx2), distances in bond_distance_evolution.items():
        for i in range(len(distances)):
            average_bond_lengths.setdefault(i, 0)
            average_bond_lengths[i] += distances[i]
    for i in range(len(average_bond_lengths)):
        average_bond_lengths[i] /= len(bond_distance_evolution)
    plt.figure(figsize=(4, 3))
    print("average_bond_lengths", average_bond_lengths)
    x = list(range(len(average_bond_lengths)))
    y = [average_bond_lengths[i] for i in x]
    plt.plot(x, y, linestyle='-', color='#8e7fb8', linewidth=2)
    # plt.title("Average Bond Lengths")
    plt.xlabel("Diffusion Step")
    plt.ylabel("Average Bond Length (Å)")
    interval = 20
    ticks = [i for i in range(0, len(x)+2, interval)]
    plt.xticks(ticks, [str(i*5) for i in ticks])
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, "average_bond_lengths.png"))
    plt.savefig(os.path.join(output_plot_dir, "average_bond_lengths.svg"))
    plt.show()
    plt.clf()

def track_atom_type_changes(models):
    """
    Track changes in atom types for all atoms across models.
    Detects changes by comparing each model with the PREVIOUS model,
    not with the initial model.
    
    Args:
        models (list): List of Model objects
        
    Returns:
        dict: Dictionary with atom indices as keys and lists of binary values as values
              where 1 indicates a type change from previous model and 0 indicates no change
    """
    model_atoms_lists = [model.get_atoms() for model in models]
    
    if not model_atoms_lists:
        print("No models found to track atom type changes")
        return {}
    
    # Find the maximum number of atoms in any model
    max_atoms = max(len(atoms) for atoms in model_atoms_lists)
    print(f"Tracking type changes for up to {max_atoms} atoms across {len(models)} models")
        
    # Dictionary to store changes for each atom
    atom_type_changes = {idx: [] for idx in range(max_atoms)}
    
    # Compare atom types across consecutive models
    previous_model_atom_types = {}  # Store atom types from previous model
    
    # Initialize with first model's atom types
    for atom_idx, atom in enumerate(model_atoms_lists[0]):
        previous_model_atom_types[atom_idx] = atom.name
        # First model has no changes by definition (no previous model)
        atom_type_changes[atom_idx].append(0)
    
    # Compare consecutive models (starting from the second model)
    for model_idx in range(1, len(model_atoms_lists)):
        model_atoms = model_atoms_lists[model_idx]
        
        for atom_idx in range(max_atoms):
            change_detected = 0  # Default: no change
            
            # Ensure index exists in this model's atom list
            if atom_idx < len(model_atoms):
                atom = model_atoms[atom_idx]
                current_type = atom.name
                
                # Check if atom type has changed from PREVIOUS model (not initial)
                if atom_idx in previous_model_atom_types:
                    previous_type = previous_model_atom_types[atom_idx]
                    if current_type != previous_type:
                        change_detected = 1
                
                # Update previous model's atom type for next iteration
                previous_model_atom_types[atom_idx] = current_type
            
            # Store the change status
            atom_type_changes[atom_idx].append(change_detected)
    
    return atom_type_changes

def create_atom_type_change_matrix(atom_type_changes):
    if not atom_type_changes:
        print("No atom type changes data available")
        return None
    
    # Determine matrix dimensions
    num_atoms = len(atom_type_changes)
    num_models = len(next(iter(atom_type_changes.values())))
    
    # Create empty matrix
    change_matrix = np.zeros((num_atoms, num_models))
    
    # Fill matrix with atom type change data
    for atom_idx, changes in sorted(atom_type_changes.items()):
        for model_idx, change in enumerate(changes):
            change_matrix[atom_idx, model_idx] = change
    
    return change_matrix

def plot_atom_type_changes(atom_type_changes, atom_details_map=None, output_filename="atom_evolution_plots/atom_type_changes_matrix.png"):
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create matrix from the atom type changes data
    change_matrix = create_atom_type_change_matrix(atom_type_changes)
    print("change_matrix", change_matrix.shape)
    
    if change_matrix is None or change_matrix.size == 0:
        print("No data available to plot atom type changes")
        return
    
    # Create heatmap - limit number of y labels if too many atoms
    plt.figure(figsize=(4, 3))
    
    cmap = ListedColormap(["#a2c9ae", "#8e7fb8"])
    ax = sns.heatmap(change_matrix, cmap=cmap, cbar=False, linewidths=0.5)

    # set yticks to be the atom details map
    n_atoms = len(atom_details_map)
    ax.set_yticks(range(n_atoms))
    ax.set_yticklabels([atom_details_map[i] for i in range(n_atoms)])
    interval = 20
    ticks = [i for i in range(0, change_matrix.shape[1]+2, interval)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(i*5) for i in ticks], rotation=0)
    
    # plt.title("Atom Type Changes Across Models")
    plt.xlabel("Diffusion Step")
    # plt.ylabel("Atom")
    plt.tight_layout()
    
    # Save the plot
    base, _ = os.path.splitext(output_filename)
    plt.savefig(base + ".png")
    plt.savefig(base + ".svg")
    print(f"  Plot saved as {base + '.png'} and .svg")
    plt.show()
    plt.clf()
    
    # Calculate and plot summary statistics
    change_counts_by_model = change_matrix.sum(axis=0)
    plt.figure(figsize=(4, 3))
    plt.bar(range(1, len(change_counts_by_model) + 1), change_counts_by_model, color='#8e7fb8')
    # plt.title("Number of Atoms With Type Changes by Model")
    plt.xlabel("Diffusion Step")
    plt.ylabel("# Type Changes")
    plt.xticks(ticks, [str(i*5) for i in ticks])
    plt.tight_layout()
    
    summary_base = os.path.join(output_dir, "atom_type_changes_summary")
    plt.savefig(summary_base + ".png")
    plt.savefig(summary_base + ".svg")
    print(f"  Summary plot saved as {summary_base + '.png'} and .svg")
    plt.show()
    plt.clf()

def summarize_atom_type_changes(atom_type_changes, atom_details_map=None):
    print("\n--- Atom Type Changes Summary ---")
    
    # Count total changes per atom
    top_changing_atoms = []
    for atom_idx, changes in sorted(atom_type_changes.items()):
        total_changes = sum(changes)
        if total_changes > 0:
            total_models = len(changes)
            atom_label = atom_details_map.get(atom_idx, f"Atom {atom_idx}") if atom_details_map else f"Atom {atom_idx}"
            
            top_changing_atoms.append((atom_idx, atom_label, total_changes, total_models))
    
    # Print top changing atoms
    if top_changing_atoms:
        print(f"\nTop {min(10, len(top_changing_atoms))} atoms with most type changes:")
        for atom_idx, atom_label, total_changes, total_models in sorted(top_changing_atoms, key=lambda x: x[2], reverse=True)[:10]:
            print(f"  {atom_label}: {total_changes} changes across {total_models} models ({total_changes/total_models*100:.1f}%)")
    else:
        print("\nNo atom type changes detected across any atoms.")

    # Calculate models with most changes
    model_change_counts = defaultdict(int)
    for atom_idx, changes in atom_type_changes.items():
        for model_idx, change in enumerate(changes):
            if change == 1:
                model_change_counts[model_idx + 1] += 1  # +1 to convert 0-index to model ID
    
    # Print top models with changes
    if model_change_counts:
        print("\nModels with most atom type changes:")
        for model_id, count in sorted(model_change_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  Model {model_id}: {count} atom(s) with type changes")
    else:
        print("\nNo atom type changes detected across any models.")

def calculate_rmsd(model1_atoms, model2_atoms):
    if len(model1_atoms) != len(model2_atoms):
        print(f"Warning: Models have different number of atoms ({len(model1_atoms)} vs {len(model2_atoms)})")
        return None
    
    squared_distances = []
    for atom1, atom2 in zip(model1_atoms, model2_atoms):
        distance = atom1.distance_to(atom2)
        squared_distances.append(distance * distance)
    
    rmsd = np.sqrt(np.mean(squared_distances))
    return rmsd

def calculate_rmsd_to_last_model(models):
    if not models:
        print("Error: No models found")
        return {}
    
    last_model = models[-1]
    last_model_atoms = last_model.get_atoms()
    
    rmsd_values = []
    for model in models[:-1]:  # Exclude last model since it's the reference
        model_atoms = model.get_atoms()
        rmsd = calculate_rmsd(model_atoms, last_model_atoms)
        if rmsd is not None:
            rmsd_values.append(rmsd)
    
    return rmsd_values

def plot_rmsd_evolution(rmsd_values, output_plot_dir="plots"):
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)
    plt.figure(figsize=(3.5, 3))
    x = list(range(len(rmsd_values)))
    y = rmsd_values
    plt.plot(x, y, linestyle='-', color='#8e7fb8', linewidth=2)
    # plt.title("RMSD Evolution Relative to Last Model")
    plt.xlabel("Diffusion Step")
    plt.ylabel("RMSD (Å)")
    # Set x-ticks at intervals of 10
    if len(x) > 10:
        interval = 20
        ticks=[i for i in range(0, len(x)+2, interval)]
        plt.xticks(
            ticks=ticks,
            labels=[str(i*5) for i in ticks],
        )
    else:
        plt.xticks(x)
    plot_filename_base = os.path.join(output_plot_dir, "rmsd_evolution")
    plt.tight_layout()
    plt.savefig(plot_filename_base + ".png")
    plt.savefig(plot_filename_base + ".svg")
    print(f"RMSD evolution plot saved as {plot_filename_base + '.png'} and .svg")
    plt.show()
    plt.clf()

def determine_bonds_in_model(model_atoms, bond_cutoff=1.9):
    bonds = []
    for i in range(len(model_atoms)):
        for j in range(i + 1, len(model_atoms)):
            atom1 = model_atoms[i]
            atom2 = model_atoms[j]
            distance = atom1.distance_to(atom2)
            
            if distance < bond_cutoff:
                bond_pair = tuple(sorted((i, j)))
                bonds.append(bond_pair)
    
    return sorted(list(set(bonds)))

def track_bond_counts(models, bond_cutoff=1.9):
    bond_counts = []
    
    for model in models:
        model_atoms = model.get_atoms()
        bonds = determine_bonds_in_model(model_atoms, bond_cutoff)
        bond_counts.append(len(bonds))
    
    return bond_counts

def plot_bond_count_evolution(bond_counts, output_plot_dir="plots"):
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)
    
    plt.figure(figsize=(4, 3))
    x = list(range(1, len(bond_counts) + 1))
    plt.plot(x, bond_counts, marker='o', linestyle='-', color='#a2c9ae')
    plt.title("Evolution of Bond Counts Across Models")
    plt.xlabel("Diffusion Step")
    plt.ylabel("Number of Bonds")
    plt.grid(True)
    if len(x) > 10:
        plt.xticks(x[::max(1, len(x)//10)])
    else:
        plt.xticks(x)
    plot_filename_base = os.path.join(output_plot_dir, "bond_count_evolution")
    plt.savefig(plot_filename_base + ".png")
    plt.savefig(plot_filename_base + ".svg")
    print(f"Bond count evolution plot saved as {plot_filename_base + '.png'} and .svg")
    plt.show()
    plt.clf()

def analyze_bond_changes(models, bond_cutoff=1.9):
    bond_changes = []
    previous_bonds = set()
    
    for model_idx, model in enumerate(models):
        current_bonds = set(determine_bonds_in_model(model.get_atoms(), bond_cutoff))
        
        if model_idx > 0:  # Skip first model as we need previous bonds to compare
            bonds_gained = len(current_bonds - previous_bonds)
            bonds_lost = len(previous_bonds - current_bonds)
            bond_changes.append((model_idx + 1, bonds_gained, bonds_lost))
        
        previous_bonds = current_bonds
    
    return bond_changes

def plot_bond_changes(bond_changes, output_plot_dir="plots"):
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)
    
    model_ids = [x[0] for x in bond_changes]
    bonds_gained = [x[1] for x in bond_changes]
    bonds_lost = [x[2] for x in bond_changes]
    
    plt.figure(figsize=(7.5, 3))
    x = np.arange(len(model_ids))
    width = 0.35
    
    plt.bar(x - width/2, bonds_gained, width, label='Bonds Gained', color='#8e7fb8')
    plt.bar(x + width/2, bonds_lost, width, label='Bonds Lost', color='#a2c9ae')
    
    # plt.title("Bond Changes Between Consecutive Models")
    plt.xlabel("Diffusion Step")
    plt.ylabel("Number of Bonds")
    # Set x-ticks at intervals of 10
    interval = 10
    print("len(model_ids)", len(model_ids))
    xticks = [i for i in range(0, len(model_ids)+2, interval)]
    plt.xticks(xticks, [str(i*5) for i in xticks])
    plt.xlim(0, len(model_ids)+1)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    plot_filename_base = os.path.join(output_plot_dir, "bond_count_changes")
    plt.savefig(plot_filename_base + ".png")
    plt.savefig(plot_filename_base + ".svg")
    print(f"Bond changes plot saved as {plot_filename_base + '.png'} and .svg")
    plt.show()
    plt.clf()

def permutation_test_atom_covariation(change_matrix, n_permutations=1000, random_seed=42):
    """
    Permutation test for atom pair co-variation significance.
    Args:
        change_matrix (np.ndarray): Atom x Model binary matrix
        n_permutations (int): Number of permutations
        random_seed (int): Random seed for reproducibility
    Returns:
        observed_co (np.ndarray): Observed co-variation matrix (fraction)
        pvals (np.ndarray): p-value matrix for each atom pair
    """
    np.random.seed(random_seed)
    num_atoms, num_steps = change_matrix.shape
    observed_co = np.dot(change_matrix, change_matrix.T) / num_steps
    perm_co_distributions = np.zeros((num_atoms, num_atoms, n_permutations))
    for p in range(n_permutations):
        permuted = np.zeros_like(change_matrix)
        for i in range(num_atoms):
            permuted[i] = np.random.permutation(change_matrix[i])
        perm_co = np.dot(permuted, permuted.T) / num_steps
        perm_co_distributions[:, :, p] = perm_co
    # 计算p值：观测值大于等于随机的概率
    pvals = np.mean(perm_co_distributions >= observed_co[:, :, None], axis=2)
    return observed_co, pvals

def plot_atom_type_change_network(change_matrix, atom_details_map=None, threshold=0.5, output_filename="atom_evolution_plots/atom_type_change_network.png", pval_matrix=None, pval_cutoff=None):
    """
    Plot a network graph where nodes are atoms and edges connect atoms that co-changed above a threshold or are significant by p-value.
    Args:
        change_matrix (np.ndarray): Atom x Model binary matrix of type changes
        atom_details_map (dict): Optional, maps atom index to label
        threshold (float): Minimum co-variation (fraction) to draw an edge
        output_filename (str): Where to save the plot
        pval_matrix (np.ndarray): Optional, p-value matrix for significance
        pval_cutoff (float): If set, only plot edges with p-value < pval_cutoff
    """
    import networkx as nx
    if change_matrix is None or change_matrix.size == 0:
        print("No data available to plot atom type change network")
        return
    num_atoms = change_matrix.shape[0]
    co_occurrence = np.dot(change_matrix, change_matrix.T) / change_matrix.shape[1]
    G = nx.Graph()
    for i in range(num_atoms):
        label = atom_details_map.get(i, str(i)) if atom_details_map else str(i)
        G.add_node(i, label=label)
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            weight = co_occurrence[i, j]
            if pval_matrix is not None and pval_cutoff is not None:
                if pval_matrix[i, j] < pval_cutoff:
                    G.add_edge(i, j, weight=weight, pval=pval_matrix[i, j])
            else:
                if weight >= threshold:
                    G.add_edge(i, j, weight=weight)
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    edge_weights = [G[u][v]['weight']*5 for u, v in G.edges()]  # scale for visibility
    nx.draw_networkx_nodes(G, pos, node_color="#8e7fb8", node_size=400)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6)
    nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]['label'] for i in G.nodes}, font_size=8)
    if pval_matrix is not None and pval_cutoff is not None:
        plt.title(f"Significant Atom Type Co-variation Network (p < {pval_cutoff})")
    else:
        plt.title(f"Atom Type Change Co-variation Network (threshold ≥ {threshold})")
    plt.axis('off')
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base, _ = os.path.splitext(output_filename)
    plt.tight_layout()
    plt.savefig(base + ".png")
    plt.savefig(base + ".svg")
    print(f"  Network plot saved as {base + '.png'} and .svg")
    plt.show()
    plt.clf()

def plot_bond_and_covariation_network(last_model_atoms, significant_pairs, bonds, atom_details_map, output_filename="atom_evolution_plots/bond_vs_covariation_network.png"):
    """
    Plot a network with both last frame's bonds and significant covaried atom pairs.
    Args:
        last_model_atoms (list): List of Atom objects in the last frame
        significant_pairs (list of tuple): List of (i, j) atom index pairs with significant covariation
        bonds (list of tuple): List of (i, j) atom index pairs representing bonds
        atom_details_map (dict): Atom index to label
        output_filename (str): Where to save the plot
    """
    import networkx as nx
    G = nx.Graph()
    # Add nodes
    for i, atom in enumerate(last_model_atoms):
        label = atom_details_map.get(i, str(i)) if atom_details_map else str(i)
        G.add_node(i, label=label)
    # Add edges: bonds, covariation, overlap
    bond_set = set(tuple(sorted(b)) for b in bonds)
    covar_set = set(tuple(sorted(p)) for p in significant_pairs)
    overlap = bond_set & covar_set
    only_bond = bond_set - overlap
    only_covar = covar_set - overlap
    # Add edges with type attribute
    for i, j in only_bond:
        G.add_edge(i, j, type='bond')
    for i, j in only_covar:
        G.add_edge(i, j, type='covar')
    for i, j in overlap:
        G.add_edge(i, j, type='both')
    # Draw
    plt.figure(figsize=(7, 7))
    pos = nx.spring_layout(G, seed=42)
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="#8e7fb8", node_size=400)
    nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]['label'] for i in G.nodes}, font_size=8)
    # Draw edges by type
    edge_types = nx.get_edge_attributes(G, 'type')
    # Bonds only: solid green
    nx.draw_networkx_edges(G, pos, edgelist=[e for e, t in edge_types.items() if t=='bond'], width=2, edge_color="#4daf4a", style='solid', label='Bond')
    # Covar only: dashed orange
    nx.draw_networkx_edges(G, pos, edgelist=[e for e, t in edge_types.items() if t=='covar'], width=2, edge_color="#ff9800", style='dashed', label='Significant Covariation')
    # Both: solid red, thicker
    nx.draw_networkx_edges(G, pos, edgelist=[e for e, t in edge_types.items() if t=='both'], width=4, edge_color="#e41a1c", style='solid', label='Both')
    # Legend
    import matplotlib.lines as mlines
    bond_line = mlines.Line2D([], [], color="#4daf4a", linewidth=2, label='Bond')
    covar_line = mlines.Line2D([], [], color="#ff9800", linewidth=2, linestyle='dashed', label='Significant Covariation')
    both_line = mlines.Line2D([], [], color="#e41a1c", linewidth=4, label='Both')
    plt.legend(handles=[bond_line, covar_line, both_line], loc='best')
    plt.title("Bonds vs. Significant Atom Type Covariation")
    plt.axis('off')
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base, _ = os.path.splitext(output_filename)
    plt.tight_layout()
    plt.savefig(base + ".png")
    plt.savefig(base + ".svg")
    print(f"  Bond vs. covariation network plot saved as {base + '.png'} and .svg")
    plt.show()
    plt.clf()

#%%
current_dir = os.path.dirname(os.path.abspath(__file__))
test_pdb_file = os.path.join(current_dir, "mol_004.pdb")

models = parse_pdb_without_biopython(test_pdb_file)
bonds_to_track, atom_details_map = determine_bonds(models, bond_cutoff=1.8)

# bond_distance_evolution = track_bonds(models, bonds_to_track)
# plot_bond_distances(bond_distance_evolution, atom_details_map, output_plot_dir="bond_evolution_plots")
# plot_average_bond_lengths(bond_distance_evolution, output_plot_dir="bond_evolution_plots")

atom_type_changes = track_atom_type_changes(models)
# plot_atom_type_changes(atom_type_changes, atom_details_map, output_filename="atom_evolution_plots/atom_type_changes_matrix.png")
# print("atom_type_changes", len(atom_type_changes), len(atom_type_changes[0]))
# summarize_atom_type_changes(atom_type_changes, atom_details_map)

# rmsd_values = calculate_rmsd_to_last_model(models)
# plot_rmsd_evolution(rmsd_values, output_plot_dir="rmsd_evolution_plots")

# bond_changes = analyze_bond_changes(models, bond_cutoff=1.9)
# plot_bond_changes(bond_changes, output_plot_dir="bond_evolution_plots")

change_matrix = create_atom_type_change_matrix(atom_type_changes)

#%%
# 进行 permutation test，得到显著性矩阵
observed_co, pvals = permutation_test_atom_covariation(change_matrix, n_permutations=1000, random_seed=42)

# 只画显著共变的 network（比如 p < 0.05）
plot_atom_type_change_network(
    change_matrix,
    atom_details_map,
    output_filename="atom_evolution_plots/atom_type_change_network_significant.png",
    pval_matrix=pvals,
    pval_cutoff=0.05
)

# 也可以继续画原始的 network（如需对比）
# plot_atom_type_change_network(change_matrix, atom_details_map, threshold=0.1, output_filename="atom_evolution_plots/atom_type_change_network.png")

# %%
# After permutation test and bond determination
last_model_atoms = models[-1].get_atoms()
bonds = determine_bonds_in_model(last_model_atoms, bond_cutoff=1.8)
# Get significant covar pairs (p < 0.05, upper triangle, i != j)
significant_pairs = [(i, j) for i in range(pvals.shape[0]) for j in range(i+1, pvals.shape[1]) if pvals[i, j] < 0.05]
plot_bond_and_covariation_network(
    last_model_atoms,
    significant_pairs,
    bonds,
    atom_details_map,
    output_filename="atom_evolution_plots/bond_vs_covariation_network.png"
)

# %%
