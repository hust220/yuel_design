import os
import glob
import re
import csv

output_dir = "docking-output"
output_csv = "min_energy_table.csv"

energy_pattern = re.compile(r"REMARK E_without_VDWR: ([\-0-9\.Ee]+)")

molecule_energies = {}

for pdb_file in glob.glob(os.path.join(output_dir, "*.pdb")):
    energies = []
    with open(pdb_file, "r") as f:
        for line in f:
            match = energy_pattern.search(line)
            if match:
                try:
                    energies.append(float(match.group(1)))
                except ValueError:
                    pass
    if energies:
        min_energy = min(energies)
    else:
        min_energy = None
    base = os.path.splitext(os.path.basename(pdb_file))[0]
    mol_name = base.rsplit("_run", 1)[0]
    if min_energy is not None:
        molecule_energies.setdefault(mol_name, []).append(min_energy)

results = []
for mol_name, energies in molecule_energies.items():
    results.append({"molecule": mol_name, "min_energy": min(energies)})

# 按照min_energy升序排序
results.sort(key=lambda x: x["min_energy"])

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["molecule", "min_energy"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Done! Results saved to {output_csv}") 