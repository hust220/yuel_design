# %%

import os
import glob
import pymol
from pymol import cmd

def convert_sdf_to_mol2(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)
    sdf_files = glob.glob(os.path.join(input_dir, "*.sdf"))
    print(f"Found {len(sdf_files)} .sdf files to convert")
    converted_count = 0
    failed_count = 0
    pymol.finish_launching(['pymol', '-cq'])
    for sdf_file in sdf_files:
        try:
            base_name = os.path.splitext(os.path.basename(sdf_file))[0]
            mol2_file = os.path.join(output_dir, f"{base_name}.mol2")
            obj_name = f"obj_{base_name}"
            cmd.reinitialize()
            cmd.load(sdf_file, obj_name, multiplex=0)  # Only load the first molecule
            cmd.save(mol2_file, obj_name)
            print(f"Converted: {sdf_file} -> {mol2_file}")
            converted_count += 1
        except Exception as e:
            print(f"Error converting {sdf_file}: {str(e)}")
            failed_count += 1
    print(f"\nConversion complete!")
    print(f"Successfully converted: {converted_count} files")
    print(f"Failed conversions: {failed_count} files")

if __name__ == "__main__":
    input_directory = "GPR75_analysis/cluster_7_high_qed_molecules"
    convert_sdf_to_mol2(input_directory)



# %%
