#%%
import os
import sys

sys.path.append('../../')

# make sure the index of residue in pdb2 is the same as pdb1 and then write to outfile
def restore_residue_index(pdb1, pdb2, outfile):
    """
    For each residue in pdb2, set its residue index and chain name to match the corresponding residue in pdb1 (by order),
    but only if the residue names match. Otherwise, leave the ATOM lines unchanged for that residue.
    Only ATOM lines are updated. HETATM and other lines are copied as is.
    The only thing assumed to be the same is the order of residues.
    """
    def get_residues(pdb_file):
        residues = []
        last_key = None
        with open(pdb_file) as f:
            for line in f:
                if line.startswith('ATOM'):
                    chain = line[21]
                    resseq = line[22:26]
                    icode = line[26]
                    resname = line[17:20]
                    key = (chain, resseq.strip(), icode, resname)
                    if last_key != key:
                        residues.append({'chain': chain, 'resseq': resseq.strip(), 'icode': icode, 'resname': resname, 'key': key, 'lines': []})
                        last_key = key
                    residues[-1]['lines'].append(line)
        return residues

    # Get residue lists for both pdb1 and pdb2 (only ATOM lines)
    residues1 = get_residues(pdb1)
    residues2 = get_residues(pdb2)

    # Build mapping: residue order -> (chain, resseq, icode, resname) from pdb1
    mapping = []
    for res in residues1:
        mapping.append((res['chain'], res['resseq'], res['icode'], res['resname']))

    # Write output
    with open(pdb2) as fin, open(outfile, 'w') as fout:
        residue_iter = iter(mapping)
        current_key = None
        for line in fin:
            if line.startswith('ATOM'):
                chain = line[21]
                resseq = line[22:26].strip()
                icode = line[26]
                resname = line[17:20]
                if resname.strip() == 'H2O':
                    fout.write(line)
                    continue
                key = (chain, resseq, icode, resname)
                # print(key, current_key)
                # If this is a new residue, advance the residue mapping
                if key != current_key:
                    current_map = next(residue_iter)
                    current_key = key
                # Only update if residue names match
                new_chain, new_resseq, new_icode, new_resname = current_map
                if new_resname.strip() == resname.strip():
                    newline = (
                        line[:21] +
                        new_chain +
                        str(new_resseq).rjust(4) +
                        (new_icode if new_icode else ' ') +
                        line[27:]
                    )
                    fout.write(newline)
                else:
                    raise ValueError(f"Residue names do not match: {new_resname} != {resname}")
            else:
                fout.write(line)

#%%
restore_residue_index('7e2y.pdb', '7e2y_best_pose.pdb', '7e2y_best_pose_restored.pdb')

#%%
restore_residue_index('7ckz.pdb', '7ckz_best_pose.pdb', '7ckz_best_pose_restored.pdb')








# %%
