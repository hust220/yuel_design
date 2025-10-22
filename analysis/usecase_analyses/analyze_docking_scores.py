#%%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import sys
import os
import re
sys.path.append('../../')
from db_utils import db_connection
import psycopg2

def get_all_scores(table, score_column):
    """Retrieve all medusadock_scores from the database"""
    scores = []
    with db_connection() as conn:
        with conn.cursor() as cursor:
            # Fetch scores in batches to avoid memory issues
            cursor.execute(f"SELECT {score_column} FROM {table} WHERE {score_column} IS NOT NULL")
            with tqdm(desc="Fetching scores") as pbar:
                while True:
                    batch = cursor.fetchmany(1000)  # Fetch in batches of 1000
                    if not batch:
                        break
                    scores.extend([score[0] for score in batch])
                    pbar.update(len(batch))
    return scores

def get_random_docking_scores(size=None, native=False):
    """Retrieve random docking scores from the database.
    Args:
        size (int, optional): If provided, only fetch scores for ligands of this size.
        native (bool, optional): If True, fetch native scores. If False, fetch nonnative scores.
                              If None, fetch all scores.
    """
    scores = []
    with db_connection() as conn:
        with conn.cursor() as cursor:
            # Base query
            query = "SELECT best_score FROM random_docking WHERE best_score IS NOT NULL"
            params = []
            
            # Add size filter if specified
            if size is not None:
                query += " AND ligand_size = %s"
                params.append(size)
            
            # Add native/nonnative filter if specified
            if native is not None:
                query += " AND pocket = ligand" if native else " AND pocket != ligand"
                
            # Fetch scores in batches to avoid memory issues
            cursor.execute(query, tuple(params) if params else None)
            with tqdm(desc="Fetching scores") as pbar:
                while True:
                    batch = cursor.fetchmany(1000)  # Fetch in batches of 1000
                    if not batch:
                        break
                    scores.extend([score[0] for score in batch])
                    pbar.update(len(batch))
    return scores

def get_prediction_scores(size=None):
    """Retrieve medusadock_scores from the database.
    Args:
        size (int, optional): If provided, only fetch scores for molecules of this size.
    """
    scores = []
    with db_connection() as conn:
        with conn.cursor() as cursor:
            # Base query
            query = """
                SELECT mr.medusadock_score 
                FROM medusadock_results mr
                JOIN molecules m ON mr.molecule_id = m.id
                WHERE mr.medusadock_score IS NOT NULL
            """
            
            # Add size filter if specified
            if size is not None:
                query += " AND m.size = %s"
                params = (size,)
            else:
                params = None
                
            # Fetch scores in batches to avoid memory issues
            cursor.execute(query, params)
            with tqdm(desc="Fetching scores") as pbar:
                while True:
                    batch = cursor.fetchmany(1000)  # Fetch in batches of 1000
                    if not batch:
                        break
                    scores.extend([score[0] for score in batch])
                    pbar.update(len(batch))
    return scores

def get_prediction_scores_with_equal_size():
    """Retrieve all medusadock_scores from the database where molecule size matches ligand size"""
    scores = []
    with db_connection() as conn:
        with conn.cursor() as cursor:
            # Fetch scores in batches to avoid memory issues
            cursor.execute("""
                SELECT mr.medusadock_score 
                FROM medusadock_results mr
                JOIN molecules m ON mr.molecule_id = m.id
                JOIN ligands l ON m.ligand_name = l.name
                WHERE mr.medusadock_score IS NOT NULL
                AND m.size = l.size
            """)
            with tqdm(desc="Fetching scores") as pbar:
                while True:
                    batch = cursor.fetchmany(1000)  # Fetch in batches of 1000
                    if not batch:
                        break
                    scores.extend([score[0] for score in batch])
                    pbar.update(len(batch))
    return scores

def get_native_scores():
    """Retrieve all medusadock_scores from the database"""
    scores = []
    with db_connection() as conn:
        with conn.cursor() as cursor:
            # Fetch scores in batches to avoid memory issues
            cursor.execute(f"SELECT best_score FROM random_docking WHERE best_score IS NOT NULL AND pocket=ligand")
            with tqdm(desc="Fetching scores") as pbar:
                while True:
                    batch = cursor.fetchmany(1000)  # Fetch in batches of 1000
                    if not batch:
                        break
                    scores.extend([score[0] for score in batch])
                    pbar.update(len(batch))
    return scores

def get_nonnative_scores():
    """Retrieve all medusadock_scores from the database"""
    scores = []
    with db_connection() as conn:
        with conn.cursor() as cursor:
            # Fetch scores in batches to avoid memory issues
            cursor.execute(f"SELECT best_score FROM random_docking WHERE best_score IS NOT NULL AND pocket!=ligand")
            with tqdm(desc="Fetching scores") as pbar:
                while True:
                    batch = cursor.fetchmany(1000)  # Fetch in batches of 1000
                    if not batch:
                        break
                    scores.extend([score[0] for score in batch])
                    pbar.update(len(batch))
    return scores

def get_native_score(target_name):
    """Get the docking score for the native ligand"""
    with db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT best_score, title
                FROM docking 
                WHERE title LIKE %s
                AND docking_status = 2
                LIMIT 1
            """, (f"{target_name}_native",))
            
            result = cursor.fetchone()
            if result:
                score, title = result
                print(f"\nNative ligand score for {target_name}:")
                print(f"Title: {title}")
                print(f"Score: {score:.2f}")
                return score
            else:
                print(f"No native ligand score found for {target_name}")
                return None
                
        except Exception as e:
            print(f"Error retrieving native score: {str(e)}")
            return None

def get_prediction_scores(target_name):
    """Get docking scores for all predicted ligands"""
    with db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT best_score, title
                FROM docking 
                WHERE title LIKE %s
                AND title NOT LIKE %s
                AND docking_status = 2
                ORDER BY best_score ASC
            """, (f"{target_name}%", f"{target_name}_native"))
            
            results = cursor.fetchall()
            if results:
                scores = [r[0] for r in results]
                print(f"\nFound {len(scores)} prediction scores for {target_name}")
                print(f"Best score: {min(scores):.2f}")
                print(f"Worst score: {max(scores):.2f}")
                print(f"Average score: {np.mean(scores):.2f}")
                return scores
            else:
                print(f"No prediction scores found for {target_name}")
                return []
                
        except Exception as e:
            print(f"Error retrieving prediction scores: {str(e)}")
            return []

def plot_score_distribution(scores, target_name=None, native_score=None, outfile=None):
    """Plot the distribution of docking scores
    
    Args:
        scores (list): List of scores to plot
        target_name (str, optional): Name of the target for the plot title
        native_score (float, optional): Native ligand score to mark on the plot
        outfile (str, optional): Output file path to save the plot
    """
    if not scores:
        print("No scores to plot")
        return
    
    plt.figure(figsize=(4, 3))
    
    # Plot histogram of scores
    sns.histplot(scores, bins=30, kde=True, color='#8e7fb8', edgecolor='none', alpha=0.7, label='Scores')
    
    # Add statistical annotations
    mean = np.mean(scores)
    std = np.std(scores)

    plt.axvline(mean, color='#e6b8a2', linestyle='--', linewidth=2.5, label=f'Mean: {mean:.2f}')
    plt.axvline(mean - std, color='gray', linestyle=':', linewidth=2)
    plt.axvline(mean + std, color='gray', linestyle=':', linewidth=2, label=f'±1 std: {std:.2f}')
    
    # Plot native score if available
    if native_score is not None:
        plt.axvline(native_score, color='black', linestyle='--', linewidth=2.5, 
                   label=f'Native ({native_score:.2f})')
    
    # Formatting
    plt.xlabel('Docking Score (kCal/mol)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.23), ncol=2, frameon=False)
    plt.grid(True, alpha=0.3)
    
    # Save plot if outfile is provided
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def get_best_structure(target_name):
    """Get the best pose from docking output for the best scoring structure"""
    with db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT best_pose, title, best_score
                FROM docking 
                WHERE title LIKE %s
                AND title NOT LIKE %s
                AND docking_status = 2
                ORDER BY best_score ASC
                LIMIT 1
            """, (f"{target_name}%", f"{target_name}_native"))
            
            result = cursor.fetchone()
            if result:
                best_pose, title, score = result
                return best_pose.tobytes(), title, score
            else:
                print(f"No structures found for {target_name}")
                return None, None, None
                
        except Exception as e:
            print(f"Error retrieving best structure: {str(e)}")
            return None, None, None

def analyze_scores(target_name):
    """Analyze docking scores for a target"""
    print(f"\nAnalyzing docking scores for {target_name}")
    
    # Get scores
    native_score = get_native_score(target_name)
    prediction_scores = get_prediction_scores(target_name)
    
    if prediction_scores:
        # Calculate statistics
        scores_array = np.array(prediction_scores)
        print("\nScore Statistics:")
        print(f"Mean: {np.mean(scores_array):.2f}")
        print(f"Median: {np.median(scores_array):.2f}")
        print(f"Std Dev: {np.std(scores_array):.2f}")
        print(f"Min: {np.min(scores_array):.2f}")
        print(f"Max: {np.max(scores_array):.2f}")
        
        # Calculate percentiles
        percentiles = [25, 50, 75, 90, 95]
        print("\nPercentiles:")
        for p in percentiles:
            print(f"{p}th percentile: {np.percentile(scores_array, p):.2f}")
        
        # Compare with native if available
        if native_score is not None:
            better_than_native = sum(1 for s in prediction_scores if s < native_score)
            print(f"\nPredictions better than native: {better_than_native} ({better_than_native/len(prediction_scores)*100:.1f}%)")
        
        # Plot distribution
        plot_score_distribution(prediction_scores, target_name=target_name, native_score=native_score, outfile=f'{target_name}_score_distribution.svg')
        
        # Save best structure
        best_pose, title, best_score = get_best_structure(target_name)
        if best_pose:
            output_file = f"{target_name}_best_pose.pdb"
            with open(output_file, 'wb') as f:
                f.write(best_pose)
            print(f"\nSaved best pose to {output_file}")
            print(f"Title: {title}")
            print(f"Score: {best_score:.2f}")
    else:
        print("No prediction scores available for analysis")

def plot_average_scores_by_size(interval=5):
    """Plot average MedusaDock scores and random docking scores by molecule size.
    Each bar represents the average of a range of sizes.
    
    Args:
        interval (int): The size of each range to average over (default: 5)
    """
    # Create docking_plots directory if it doesn't exist
    os.makedirs('docking_plots', exist_ok=True)
    
    # Create ranges of sizes (e.g., 10-14, 15-19, etc. for interval=5)
    size_ranges = [(i, i+interval-1) for i in range(11, 31, interval)]
    medusa_avgs = []
    random_native_avgs = []
    medusa_stds = []
    random_native_stds = []
    
    for start_size, end_size in tqdm(size_ranges, desc="Calculating averages"):
        # Get scores for all sizes in the range
        medusa_scores = []
        random_native_scores = []
        
        for size in range(start_size, end_size + 1):
            medusa_scores.extend(get_prediction_scores(size))
            random_native_scores.extend(get_random_docking_scores(size, native=True))
        
        # Calculate averages and standard deviations
        medusa_avg = np.mean(medusa_scores) if medusa_scores else None
        random_native_avg = np.mean(random_native_scores) if random_native_scores else None
        medusa_std = np.std(medusa_scores) if medusa_scores else None
        random_native_std = np.std(random_native_scores) if random_native_scores else None
        
        medusa_avgs.append(medusa_avg)
        random_native_avgs.append(random_native_avg)
        medusa_stds.append(medusa_std)
        random_native_stds.append(random_native_std)
    
    # Create the plot
    plt.figure(figsize=(3, 2.5))
    
    # Set width of bars
    barWidth = 0.35
    
    # Set positions of the bars on X axis
    r1 = np.arange(len(size_ranges))
    r2 = [x + barWidth for x in r1]
    
    # Create the bars with error bars
    plt.bar(r1, medusa_avgs, width=barWidth, label='YuelDesign', color='#8E7FB8', edgecolor='white', yerr=medusa_stds, capsize=3)
    plt.bar(r2, random_native_avgs, width=barWidth, label='Native', color='#A2C9AE', edgecolor='white', yerr=random_native_stds, capsize=3)
    
    # Customize the plot
    plt.xlabel('Molecule Size Range', fontsize=10)
    plt.ylabel('Average Score (kCal/mol)', fontsize=10)
    
    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth/2 for r in range(len(size_ranges))], 
               [f'{s1}-{s2}' for s1, s2 in size_ranges], rotation=45)
    
    # Add a grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    plt.legend(fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig('docking_plots/average_scores_by_size.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_score_distributions_comparison():
    """Plot and compare the distributions of equal-size prediction scores and native scores."""
    # Create docking_plots directory if it doesn't exist
    os.makedirs('docking_plots', exist_ok=True)
    
    # Get scores
    equal_size_scores = get_prediction_scores_with_equal_size()
    native_scores = get_random_docking_scores(native=True)
    
    # Create the plot
    plt.figure(figsize=(3, 2.5))
    
    # Plot distributions with filled areas
    sns.kdeplot(data=equal_size_scores, label='YuelDesign', color='#8E7FB8', fill=True, alpha=0.7, edgecolor='black')
    sns.kdeplot(data=native_scores, label='Native', color='#A2C9AE', fill=True, alpha=0.7, edgecolor='black')
    
    # Customize the plot
    plt.xlabel('MedusaScore', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig('docking_plots/score_distributions_comparison.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()
        
# %%

analyze_scores('7e2y')

#%%

analyze_scores('7ckz')
# %%
