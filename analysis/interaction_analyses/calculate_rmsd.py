import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_rmsd_distribution():
    """Plot the distribution of RMSD values."""
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT rmsd 
                FROM medusadock_results 
                WHERE rmsd IS NOT NULL
            """)
            rmsds = [row[0] for row in cursor.fetchall()]
    
    plt.figure(figsize=(3, 2.5))
    sns.kdeplot(data=rmsds, color='#8E7FB8', fill=True, alpha=0.7, edgecolor='black')
    plt.xlabel('RMSD', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig('docking_plots/rmsd_distribution.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_rmsd_vs_size():
    """Plot RMSD values against molecule size."""
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT m.size, mr.rmsd
                FROM medusadock_results mr
                JOIN molecules m ON mr.molecule_id = m.id
                WHERE mr.rmsd IS NOT NULL
            """)
            data = cursor.fetchall()
    
    sizes = [row[0] for row in data]
    rmsds = [row[1] for row in data]
    
    plt.figure(figsize=(3, 2.5))
    plt.scatter(sizes, rmsds, color='#8E7FB8', alpha=0.7, s=10)
    plt.xlabel('Molecule Size', fontsize=10)
    plt.ylabel('RMSD', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig('docking_plots/rmsd_vs_size.svg', format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

def analyze_rmsd():
    """Analyze RMSD values and create plots."""
    # Create docking_plots directory if it doesn't exist
    os.makedirs('docking_plots', exist_ok=True)
    
    print("Plotting RMSD distribution...")
    plot_rmsd_distribution()
    
    print("\nPlotting RMSD vs size...")
    plot_rmsd_vs_size()
    
    # Print statistics
    with db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    AVG(rmsd) as avg_rmsd,
                    MIN(rmsd) as min_rmsd,
                    MAX(rmsd) as max_rmsd,
                    COUNT(rmsd) as count
                FROM medusadock_results 
                WHERE rmsd IS NOT NULL
            """)
            avg_rmsd, min_rmsd, max_rmsd, count = cursor.fetchone()
            
            print(f"\nRMSD Statistics:")
            print(f"Number of molecules: {count}")
            print(f"Average RMSD: {avg_rmsd:.3f}")
            print(f"Min RMSD: {min_rmsd:.3f}")
            print(f"Max RMSD: {max_rmsd:.3f}")

# if __name__ == "__main__":
#     main()

# %%
analyze_rmsd() 