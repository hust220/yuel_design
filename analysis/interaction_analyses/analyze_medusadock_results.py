#%%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import sys
import os
sys.path.append('../')
from db_utils import db_connection

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


def plot_score_distribution(scores, outfile=None):
    """Plot the distribution of MedusaDock scores"""
    if not scores:
        print("No scores to plot")
        return
    
    plt.figure(figsize=(4, 3))
    
    # Main histogram
    sns.histplot(scores, bins=50, kde=True, color='skyblue', edgecolor='none')
    
    # Add statistical annotations
    mean = np.mean(scores)
    median = np.median(scores)
    std = np.std(scores)
    
    plt.axvline(mean, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='--', linewidth=1, label=f'Median: {median:.2f}')
    plt.axvline(mean - std, color='gray', linestyle=':', linewidth=1)
    plt.axvline(mean + std, color='gray', linestyle=':', linewidth=1, label=f'±1 std: {std:.2f}')
    
    # Formatting
    # plt.title('Distribution of MedusaDock Scores', fontsize=14, pad=20)
    plt.xlabel('MedusaDock Score (kCal/mol)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()

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

#%%

if __name__ == "__main__":
    # print("Analyzing docking scores by molecule size...")
    # plot_average_scores_by_size(interval=5)
    
    print("\nComparing score distributions...")
    plot_score_distributions_comparison()
    
    # print("\nAnalyzing individual score distributions...")
    # prediction_scores = get_prediction_scores()
    # random_docking_scores = get_random_docking_scores()
    
    # if prediction_scores:
    #     print(f"\nRetrieved {len(prediction_scores)} MedusaDock scores")
    #     print(f"Prediction score range: [{min(prediction_scores):.2f}, {max(prediction_scores):.2f}]")
    #     plot_score_distribution(prediction_scores, 'MedusaDock for Generated Molecules')

    #     print(f"Retrieved {len(random_docking_scores)} random docking scores")
    #     print(f"Random docking score range: [{min(random_docking_scores):.2f}, {max(random_docking_scores):.2f}]")
    #     plot_score_distribution(random_docking_scores, 'Random Docking')
        
    #     print("Plots saved as 'medusadock_scores_distribution.png' and 'average_scores_by_size.png'")
    # else:
    #     print("No scores found in the database")
        
# %%

scores = get_native_scores()
print(f"Retrieved {len(scores)} native scores")
print(f"Native score range: [{min(scores):.2f}, {max(scores):.2f}]")
plot_score_distribution(scores)
#%%
scores = get_nonnative_scores()
print(f"Retrieved {len(scores)} nonnative scores")
print(f"Nonnative score range: [{min(scores):.2f}, {max(scores):.2f}]")
plot_score_distribution(scores, 'Nonnative Docking')
# %%
