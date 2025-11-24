#!/usr/bin/env python3
"""
Split MOAD dataset into train/val/test sets and add split column to moad_pockets table.

Split ratios:
- val: fixed 50 samples
- test: (val + test) : train = 2:8, so test = (2/10 * total) - 50
- train: remaining samples

Usage:
    python split_moad.py
"""

import os
import sys
import random
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.db_utils import db_connection


def get_total_samples():
    """Get total number of samples in moad_pockets table."""
    with db_connection() as conn:
        with conn.cursor() as c:
            c.execute("SELECT COUNT(*) FROM moad_pockets")
            return c.fetchone()[0]


def calculate_split_sizes(total_samples):
    """
    Calculate split sizes based on the requirements:
    - val: fixed 50 samples
    - test: (val + test) : train = 2:8, so test = (2/10 * total) - 50
    - train: remaining samples
    """
    val_size = 50
    
    # Calculate test size: (val + test) : train = 2:8
    # So: (val + test) / total = 2/10
    # test = (2/10 * total) - val
    test_size = int(0.2 * total_samples) - val_size
    
    # Ensure test_size is not negative
    if test_size < 0:
        test_size = 0
    
    # Calculate train size
    train_size = total_samples - val_size - test_size
    
    # Ensure we don't exceed total samples
    if train_size < 0:
        train_size = 0
        test_size = total_samples - val_size
    
    return train_size, val_size, test_size


def get_all_pocket_ids():
    """Get all pocket IDs from moad_pockets table."""
    with db_connection() as conn:
        with conn.cursor() as c:
            c.execute("SELECT id FROM moad_pockets ORDER BY id")
            return [row[0] for row in c.fetchall()]


def assign_splits(pocket_ids, train_size, val_size, test_size):
    """
    Assign splits to pocket IDs randomly.
    
    Args:
        pocket_ids: List of all pocket IDs
        train_size: Number of samples for training
        val_size: Number of samples for validation  
        test_size: Number of samples for testing
    
    Returns:
        dict: Mapping of pocket_id -> split
    """
    # Shuffle the IDs to ensure random assignment
    shuffled_ids = pocket_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Assign splits
    split_assignments = {}
    
    # Assign validation (first val_size samples)
    for i in range(val_size):
        split_assignments[shuffled_ids[i]] = 'val'
    
    # Assign test (next test_size samples)
    for i in range(val_size, val_size + test_size):
        split_assignments[shuffled_ids[i]] = 'test'
    
    # Assign train (remaining samples)
    for i in range(val_size + test_size, len(shuffled_ids)):
        split_assignments[shuffled_ids[i]] = 'train'
    
    return split_assignments


def add_split_column():
    """Add split column to moad_pockets table if it doesn't exist."""
    with db_connection() as conn:
        with conn.cursor() as c:
            # Check if split column exists
            c.execute("""
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_name = 'moad_pockets' 
                AND column_name = 'split'
            """)
            
            if c.fetchone()[0] == 0:
                print("Adding 'split' column to moad_pockets table...")
                c.execute("ALTER TABLE moad_pockets ADD COLUMN split VARCHAR(10)")
                conn.commit()
                print("✓ Added 'split' column")
            else:
                print("✓ 'split' column already exists")


def update_split_assignments(split_assignments):
    """Update the split column with the assigned values."""
    with db_connection() as conn:
        with conn.cursor() as c:
            print("Updating split assignments...")
            
            for pocket_id, split in split_assignments.items():
                c.execute(
                    "UPDATE moad_pockets SET split = %s WHERE id = %s",
                    (split, pocket_id)
                )
            
            conn.commit()
            print("✓ Updated split assignments")


def print_split_summary(split_assignments):
    """Print summary of split assignments."""
    train_count = sum(1 for split in split_assignments.values() if split == 'train')
    val_count = sum(1 for split in split_assignments.values() if split == 'val')
    test_count = sum(1 for split in split_assignments.values() if split == 'test')
    
    total = len(split_assignments)
    
    print("\n" + "="*50)
    print("SPLIT SUMMARY")
    print("="*50)
    print(f"Total samples: {total}")
    print(f"Train: {train_count} ({train_count/total*100:.1f}%)")
    print(f"Val:   {val_count} ({val_count/total*100:.1f}%)")
    print(f"Test:  {test_count} ({test_count/total*100:.1f}%)")
    print("="*50)
    
    # Verify the 2:8 ratio for (val+test):train
    val_test_total = val_count + test_count
    if train_count > 0:
        ratio = val_test_total / train_count
        print(f"(Val+Test):Train ratio = {ratio:.2f} (target: 0.25)")
        if abs(ratio - 0.25) < 0.01:
            print("✓ Ratio is correct!")
        else:
            print("⚠ Ratio differs from target 0.25")


def main():
    """Main function to split the MOAD dataset."""
    print("MOAD Dataset Split Assignment")
    print("="*40)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Get total number of samples
    total_samples = get_total_samples()
    print(f"Total samples in moad_pockets: {total_samples}")
    
    if total_samples < 50:
        print("❌ Error: Not enough samples for validation set (need at least 50)")
        return
    
    # Calculate split sizes
    train_size, val_size, test_size = calculate_split_sizes(total_samples)
    print(f"Split sizes: train={train_size}, val={val_size}, test={test_size}")
    
    # Get all pocket IDs
    print("Fetching pocket IDs...")
    pocket_ids = get_all_pocket_ids()
    print(f"✓ Found {len(pocket_ids)} pocket IDs")
    
    # Assign splits
    print("Assigning splits...")
    split_assignments = assign_splits(pocket_ids, train_size, val_size, test_size)
    print("✓ Split assignments completed")
    
    # Add split column if needed
    add_split_column()
    
    # Update database with split assignments
    update_split_assignments(split_assignments)
    
    # Print summary
    print_split_summary(split_assignments)
    
    print("\n✓ MOAD dataset split completed successfully!")


if __name__ == "__main__":
    main()
