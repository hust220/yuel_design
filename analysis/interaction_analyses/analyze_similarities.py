#%%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import sys
import os
sys.path.append('../')
from db_utils import db_connection

def get_all_similarities():
    """Retrieve all medusadock_scores from the database"""
    similarities = []
    with db_connection() as conn:
        with conn.cursor() as cursor:
            # Fetch scores in batches to avoid memory issues
            cursor.execute(f"SELECT similarity FROM molecules WHERE similarity IS NOT NULL")
            with tqdm(desc="Fetching scores") as pbar:
                while True:
                    batch = cursor.fetchmany(1000)  # Fetch in batches of 1000
                    if not batch:
                        break
                    similarities.extend([[float(i) for i in similarity[0].split()] for similarity in batch])
                    pbar.update(len(batch))
    return similarities

def plot_distribution(similarities, outfile=None):
    """Plot the distribution of similarities"""
    if not similarities:
        print("No scores to plot")
        return
    
    plt.figure(figsize=(3, 2.5))
    
    # Main histogram with KDE, ensuring area under curve equals 1
    sns.kdeplot(data=similarities, color='#8E7FB8', fill=True, alpha=0.7, edgecolor='black', common_norm=True)
    
    plt.xlabel('Similarity', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if outfile:
        plt.savefig(outfile, format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_similarity_vs_size(interval=1):
    """Plot similarity values against molecule size with three distinct regions."""
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT m.size, m.similarity
                FROM molecules m
                WHERE m.similarity IS NOT NULL
            """)
            data = cursor.fetchall()
    
    sizes = np.array([row[0] for row in data])
    similarities = np.array([float(row[1].split()[0]) for row in data])  # Get first similarity value
    
    # Bin by size with adjustable interval
    min_size = int(np.min(sizes))
    max_size = int(np.max(sizes))
    bin_edges = np.arange(min_size, max_size + interval, interval)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Prepare lists for all regions
    outliers_low = []
    q1s = []
    medians = []
    q3s = []
    outliers_high = []
    mins = []
    maxs = []
    
    for i in range(len(bin_edges) - 1):
        bin_mask = (sizes >= bin_edges[i]) & (sizes < bin_edges[i+1])
        bin_sims = similarities[bin_mask]
        if len(bin_sims) > 0:
            q1 = np.percentile(bin_sims, 25)
            q3 = np.percentile(bin_sims, 75)
            iqr = q3 - q1
            lower_whisker = q1 - 1.5 * iqr
            upper_whisker = q3 + 1.5 * iqr
            median = np.percentile(bin_sims, 50)
            min_val = np.min(bin_sims)
            max_val = np.max(bin_sims)
        else:
            q1 = q3 = lower_whisker = upper_whisker = median = min_val = max_val = np.nan
        
        outliers_low.append(lower_whisker)
        q1s.append(q1)
        medians.append(median)
        q3s.append(q3)
        outliers_high.append(upper_whisker)
        mins.append(min_val)
        maxs.append(max_val)
    
    plt.figure(figsize=(3, 2.5))
    
    # Plot outlier regions in gray
    plt.fill_between(bin_centers, mins, outliers_low, color='gray', alpha=0.4, label='Outliers')
    plt.fill_between(bin_centers, outliers_high, maxs, color='gray', alpha=0.4)
    
    # Plot 0-25% and 75-100% regions in light color
    plt.fill_between(bin_centers, outliers_low, q1s, color='#A2C9AE', alpha=0.7, label='0-25% & 75-100%')
    plt.fill_between(bin_centers, q3s, outliers_high, color='#A2C9AE', alpha=0.7)
    
    # Plot 25-75% region in darker color
    plt.fill_between(bin_centers, q1s, q3s, color='#8E7FB8', alpha=0.7, label='25-75%')
    
    # Plot the median
    plt.plot(bin_centers, medians, color='black', lw=1.5, label='Median')
    
    plt.xlabel('Molecule Size', fontsize=10)
    plt.ylabel('Similarity', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend(fontsize=8, loc='upper right')
    
    # Save and show the plot
    plt.savefig('similarity_plots/similarity_vs_size.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

# %%

# Create similarity_plots directory if it doesn't exist
os.makedirs('similarity_plots', exist_ok=True)

similarities = get_all_similarities()
# Only plot the first similarity
sims = [ls[0] for ls in similarities]
print(f"Similarity range: [{min(sims):.2f}, {max(sims):.2f}]")
plot_distribution(sims, "similarity_plots/similarity_distribution.svg")
plot_similarity_vs_size()

# %%
