import argparse
import itertools
import math
import os
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def require_binary(name):
    if shutil.which(name) is None:
        raise RuntimeError(f'{name} not found on PATH')


def dump_fasta(df, seq_col, id_col, out_path):
    with open(out_path, 'w') as handle:
        for row in df.itertuples():
            row_dict = row._asdict()
            seq = row_dict.get(seq_col)
            if not isinstance(seq, str) or not seq.strip():
                continue
            handle.write(f'>{row_dict[id_col]}\n')
            handle.write(f'{seq.strip()}\n')


def build_blast_database(fasta_path, db_prefix):
    subprocess.run(
        [
            'makeblastdb',
            '-dbtype',
            'prot',
            '-in',
            str(fasta_path),
            '-out',
            str(db_prefix),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def run_blastp_all_vs_all(fasta_path, db_prefix, threads):
    out_path = Path(fasta_path).with_suffix('.blast.tsv')
    cmd = [
        'blastp',
        '-query',
        str(fasta_path),
        '-db',
        str(db_prefix),
        '-outfmt',
        '6 qseqid sseqid pident',
        '-max_target_seqs',
        '1000000',
        '-num_threads',
        str(threads),
    ]
    with open(out_path, 'w') as out_handle:
        subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=out_handle,
            stderr=subprocess.PIPE,
        )
    return out_path


def parse_blast_hits(tsv_path, threshold):
    hits = set()
    with open(tsv_path, 'r') as handle:
        for line in handle:
            qid, sid, ident = line.strip().split('\t')
            if qid == sid:
                continue
            if float(ident) >= threshold:
                hits.add(tuple(sorted((qid, sid))))
    return hits


def ligand_fingerprints(df, smiles_col, id_col):
    fps = {}
    for row in df.itertuples():
        row_dict = row._asdict()
        smiles = row_dict.get(smiles_col)
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if not mol:
            continue
        fps[row_dict[id_col]] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return fps


def ligand_edges(df, smiles_col, id_col, threshold):
    fps = ligand_fingerprints(df, smiles_col, id_col)
    edges = set()
    ordered = list(fps.items())
    for i, (cid_i, fp_i) in enumerate(ordered):
        targets = [fp for _, fp in ordered[i + 1 :]]
        sims = DataStructs.BulkTanimotoSimilarity(fp_i, targets)
        for j, sim in enumerate(sims):
            if sim >= threshold:
                cid_j = ordered[i + 1 + j][0]
                edges.add(tuple(sorted((cid_i, cid_j))))
    return edges


def structural_edges(df, pocket_col, id_col, rmsd_threshold, max_pairs, rng, existing_pairs):
    require_binary('TM-align')
    edges = set()
    comparisons = 0
    indices = list(range(len(df)))
    rng.shuffle(indices)
    for offset_i, i in enumerate(indices):
        for j in indices[offset_i + 1 :]:
            if comparisons >= max_pairs:
                break
        a = df.iloc[i]
        b = df.iloc[j]
        pair = tuple(sorted((a[id_col], b[id_col])))
        if pair in existing_pairs:
            continue
        pocket_a = Path(a[pocket_col])
        pocket_b = Path(b[pocket_col])
        if not pocket_a.exists() or not pocket_b.exists():
            continue
        result = subprocess.run(
            ['TM-align', str(pocket_a), str(pocket_b)],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if line.strip().startswith('RMSD='):
                rmsd = float(line.split('=')[1].split()[0])
                if rmsd <= rmsd_threshold:
                    edges.add(pair)
                break
            comparisons += 1
        if comparisons >= max_pairs:
            break
    return edges


def connected_components(ids, edges):
    ids = list(ids)
    uf = UnionFind(ids)
    for a, b in edges:
        uf.union(a, b)
    groups = defaultdict(list)
    for cid in ids:
        groups[uf.find(cid)].append(cid)
    return list(groups.values())


def assign_splits(groups, total_samples, train_ratio, seed):
    rng = random.Random(seed)
    rng.shuffle(groups)
    target_train = int(round(total_samples * train_ratio))
    train, test = set(), set()
    count = 0
    for group in groups:
        if count + len(group) <= target_train:
            train.update(group)
            count += len(group)
        else:
            test.update(group)
    if count < target_train:
        deficit = target_train - count
        movable = [group for group in groups if any(cid in test for cid in group)]
        for group in movable:
            if deficit <= 0:
                break
            train.update(group)
            test.difference_update(group)
            added = len(group)
            deficit -= added
            count += added
    return train, test


def split_dataset(args):
    df = pd.read_csv(args.metadata)
    id_col = args.complex_id_column
    if id_col not in df.columns:
        raise ValueError('Complex id column missing')
    seq_col = args.protein_sequence_column
    pocket_col = args.pocket_pdb_column
    smiles_col = args.ligand_smiles_column
    for col in (seq_col, pocket_col, smiles_col):
        if col not in df.columns:
            raise ValueError(f'{col} column missing')
    tmp_ctx = None
    try:
        if args.tmp_dir:
            tmp_root = Path(args.tmp_dir)
            tmp_root.mkdir(parents=True, exist_ok=True)
        else:
            tmp_ctx = tempfile.TemporaryDirectory(prefix='moad_split_')
            tmp_root = Path(tmp_ctx.name)

        fasta_path = tmp_root / 'proteins.fasta'
        dump_fasta(df, seq_col, id_col, fasta_path)
        seq_edges = set()
        if fasta_path.stat().st_size > 0:
            require_binary('makeblastdb')
            require_binary('blastp')
            db_prefix = tmp_root / 'blast_db'
            build_blast_database(fasta_path, db_prefix)
            blast_output = run_blastp_all_vs_all(fasta_path, db_prefix, args.threads)
            seq_edges = parse_blast_hits(blast_output, args.sequence_threshold)

        ligand_edge_pairs = ligand_edges(df, smiles_col, id_col, args.tanimoto_threshold)

        rng = random.Random(args.seed)
        struct_edges = structural_edges(
            df,
            pocket_col,
            id_col,
            args.rmsd_threshold,
            args.max_structural_comparisons,
            rng,
            seq_edges.union(ligand_edge_pairs),
        )

        all_edges = seq_edges.union(ligand_edge_pairs).union(struct_edges)
        groups = connected_components(df[id_col], all_edges)
        total = len(df)
        train_ids, test_ids = assign_splits(groups, total, args.train_ratio, args.seed)
        df['split'] = df[id_col].apply(lambda cid: 'train' if cid in train_ids else 'test')
        df.to_csv(args.output, index=False)
        train_count = (df['split'] == 'train').sum()
        test_count = (df['split'] == 'test').sum()
        print(f'Saved splits to {args.output}')
        print(f'Train: {train_count}')
        print(f'Test: {test_count}')
    finally:
        if tmp_ctx:
            tmp_ctx.cleanup()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--complex-id-column', default='complex_id')
    parser.add_argument('--protein-sequence-column', default='protein_sequence')
    parser.add_argument('--pocket-pdb-column', default='pocket_pdb')
    parser.add_argument('--ligand-smiles-column', default='ligand_smiles')
    parser.add_argument('--sequence-threshold', type=float, default=30.0)
    parser.add_argument('--rmsd-threshold', type=float, default=2.0)
    parser.add_argument('--tanimoto-threshold', type=float, default=0.85)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--max-structural-comparisons', type=int, default=5000)
    parser.add_argument('--threads', type=int, default=os.cpu_count() or 4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tmp-dir')
    return parser.parse_args()


if __name__ == '__main__':
    split_dataset(parse_args())

