"""One table per natural product: LOTUS + NPAtlas structures with chemical classes, organisms, scorer keys and PubChem counts.

usage: python build_np.py <lotus_csv_gz> <npatlas_tsv> <pool_meta.parquet> <folds_structures.parquet> <pop_tmp_dir> <out_dir> [processes] [retained_knowledge.parquet]
"""
import json, os, sys, time
for _thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_thread_var] = '1'
import numpy as np
import pandas as pd


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def _chem(block):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    te = rdMolStandardize.TautomerEnumerator()
    out = []
    for s in block:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            out.append(('', '', '', np.nan, 0)); continue
        flat = Chem.MolToSmiles(m, isomericSmiles=False)
        try:
            key = Chem.MolToInchiKey(te.Canonicalize(Chem.MolFromSmiles(flat)))[:14]
        except Exception:
            key = ''
        out.append((key, flat, rdMolDescriptors.CalcMolFormula(m), Descriptors.ExactMolWt(m), m.GetNumHeavyAtoms()))
    return out


def _individual_labels(s):
    dtype = s.dtype
    return s.dropna().str.split(' $ ', regex=False).explode().astype(dtype)


def mode(s):
    s = _individual_labels(s)
    return s.value_counts().index[0] if len(s) else None


def top(s, k=3):
    s = s.dropna()
    return ';'.join(s.value_counts().index[:k]) if len(s) else None


def top_lotus(s, k=3):
    return top(_individual_labels(s), k)


def h14(a):
    arr = np.ascontiguousarray(np.asarray(a, 'S14')).view(np.uint8).reshape(-1, 14)
    hi = arr[:, :8].copy().view(np.uint64).ravel()
    lo = np.zeros((len(arr), 8), np.uint8); lo[:, :6] = arr[:, 8:]
    return hi * np.uint64(0x9E3779B97F4A7C15) + lo.view(np.uint64).ravel()


def main():
    lotus_p, npa_p, pool_p, folds_p, tmp, out = sys.argv[1:7]
    cap = min(32, int(os.environ.get('BRIDGE_JOBS', '32')))
    nw = max(1, min(cap, int(sys.argv[7]) if len(sys.argv) > 7 else cap))
    retained_p = sys.argv[8] if len(sys.argv) > 8 else None
    assert __import__('rdkit').__version__ == '2026.03.3'
    os.makedirs(out, exist_ok=True)
    L = pd.read_csv(lotus_p, low_memory=False, usecols=[
        'structure_inchikey', 'structure_smiles_2D', 'structure_taxonomy_npclassifier_01pathway',
        'structure_taxonomy_npclassifier_02superclass', 'structure_taxonomy_npclassifier_03class', 'organism_name',
        'organism_taxonomy_02kingdom', 'organism_taxonomy_06family', 'organism_taxonomy_08genus', 'reference_doi'])
    L = L.dropna(subset=['structure_inchikey'])
    L['ik14'] = L.structure_inchikey.str.slice(0, 14)
    L.columns = ['inchikey', 'smiles', 'pathway', 'superclass', 'np_class', 'organism', 'kingdom', 'family', 'genus', 'doi', 'ik14']
    log('lotus pairs', len(L), 'connectivities', L.ik14.nunique())
    g = L.groupby('ik14', sort=True)
    lot = pd.DataFrame({
        'smiles': g.smiles.first(), 'np_pathway': g.pathway.agg(mode), 'np_superclass': g.superclass.agg(mode),
        'np_class': g.np_class.agg(mode), 'n_organisms': g.organism.nunique(), 'n_references': g.doi.nunique(),
        'n_stereoisomers': g.inchikey.nunique(), 'kingdoms': g.kingdom.agg(top_lotus), 'top_families': g.family.agg(top_lotus),
        'top_genera': g.genus.agg(top_lotus)})
    log('lotus table', len(lot))
    N = pd.read_csv(npa_p, sep='\t', usecols=['npaid', 'compound_name', 'compound_inchikey', 'compound_smiles', 'origin_type',
                                               'genus', 'original_reference_year'])
    N = N.dropna(subset=['compound_inchikey']); N['ik14'] = N.compound_inchikey.str.slice(0, 14)
    gn = N.groupby('ik14', sort=True)
    npa = pd.DataFrame({'npatlas_id': gn.npaid.first(), 'npatlas_name': gn.compound_name.first(), 'npatlas_smiles': gn.compound_smiles.first(),
                        'npatlas_origin': gn.origin_type.agg(top), 'npatlas_genera': gn.genus.agg(top),
                        'npatlas_first_year': gn.original_reference_year.min()})
    log('npatlas table', len(npa))
    T = lot.join(npa, how='outer')
    T['in_lotus'] = T.index.isin(lot.index); T['in_npatlas'] = T.index.isin(npa.index)
    T['smiles'] = T.smiles.where(T.smiles.notna(), T.npatlas_smiles)
    T = T.drop(columns=['npatlas_smiles']).reset_index().rename(columns={'index': 'ik14'})
    import multiprocessing as mp
    smi = T.smiles.tolist(); blocks = [smi[i:i + 500] for i in range(0, len(smi), 500)]
    with mp.get_context('fork').Pool(nw) as pool:
        res = [r for b in pool.imap(_chem, blocks, chunksize=2) for r in b]
    T['scorer_key'] = [r[0] for r in res]; T['smiles'] = [r[1] or s for r, s in zip(res, smi)]
    T['formula'] = [r[2] for r in res]; T['mono_mass'] = [r[3] for r in res]; T['n_heavy'] = [r[4] for r in res]
    pool_keys = set(pd.read_parquet(pool_p, columns=['key']).key)
    fs = pd.read_parquet(folds_p, columns=['scorer_key', 'inchikey14'])
    train_keys = set(fs.scorer_key) | set(fs.inchikey14)
    T['in_casmi_pool'] = T.scorer_key.isin(pool_keys) | T.ik14.isin(pool_keys)
    T['in_casmi_train'] = T.scorer_key.isin(train_keys) | T.ik14.isin(train_keys)
    # PubChem counts summed over CIDs sharing the connectivity block
    ik = np.load(os.path.join(tmp, 'cid_ik14.npy'), mmap_mode='r')
    hc = h14(ik); want = h14(T.ik14.values.astype('S14'))
    sel = np.where(np.isin(hc, want))[0]
    agg = pd.DataFrame({'h': hc[sel], 'cid': sel})
    for name in ('sid', 'pmid', 'patent'):
        agg[name] = np.load(os.path.join(tmp, f'cid_{name}.npy'), mmap_mode='r')[sel]
    rep = agg.sort_values(['h', 'sid', 'cid'], ascending=[True, False, True]).drop_duplicates('h').set_index('h').cid
    sums = agg.groupby('h')[['sid', 'pmid', 'patent']].sum()
    T['pubchem_cid'] = rep.reindex(want).fillna(0).astype(np.int32).values
    for name, col in (('sid', 'pubchem_substances'), ('pmid', 'pubmed_articles'), ('patent', 'patents')):
        T[col] = sums[name].reindex(want).fillna(0).astype(np.int64).values
    cols = ['ik14', 'scorer_key', 'smiles', 'formula', 'mono_mass', 'n_heavy', 'in_lotus', 'in_npatlas', 'in_casmi_pool', 'in_casmi_train',
            'np_pathway', 'np_superclass', 'np_class', 'n_organisms', 'n_references', 'n_stereoisomers', 'kingdoms', 'top_families',
            'top_genera', 'npatlas_id', 'npatlas_name', 'npatlas_origin', 'npatlas_genera', 'npatlas_first_year',
            'pubchem_cid', 'pubchem_substances', 'pubmed_articles', 'patents']
    # Optional retained snapshot preserves its measured PubChem columns exactly.
    # Newly added raw keys retain the counts from the supplied public CID arrays.
    if retained_p:
        previous = pd.read_parquet(retained_p).set_index('ik14')
        keep = T.ik14.isin(previous.index)
        for col in ('pubchem_cid', 'pubchem_substances', 'pubmed_articles', 'patents'):
            dtype = np.int32 if col == 'pubchem_cid' else np.int64
            T.loc[keep, col] = T.loc[keep, 'ik14'].map(previous[col]).astype(dtype)
    T = T[cols].sort_values('mono_mass').reset_index(drop=True)
    for c in ('n_organisms', 'n_references', 'n_stereoisomers', 'npatlas_first_year'):
        T[c] = T[c].astype('Int32')
    T.to_parquet(os.path.join(out, 'natural_products.parquet'), index=False)
    T.to_csv(os.path.join(out, 'natural_products.csv'), index=False)
    # compact organism table: which genera produce which connectivity
    G = L.dropna(subset=['genus']).copy()
    for col in ('kingdom', 'family', 'genus'):
        dtype = G[col].dtype
        G[col] = G[col].str.split(' $ ', regex=False)
        G = G.explode(col)
        G[col] = G[col].astype(dtype)
    O = (G.groupby(['ik14', 'kingdom', 'family', 'genus'], dropna=False)
           .agg(n_species=('organism', 'nunique'), n_references=('doi', 'nunique')).reset_index())
    O.to_parquet(os.path.join(out, 'lotus_structure_genus.parquet'), index=False)
    O.to_csv(os.path.join(out, 'lotus_structure_genus.csv'), index=False)
    summ = dict(rows=int(len(T)), lotus=int(T.in_lotus.sum()), npatlas=int(T.in_npatlas.sum()), both=int((T.in_lotus & T.in_npatlas).sum()),
                in_casmi_pool=int(T.in_casmi_pool.sum()), in_casmi_train=int(T.in_casmi_train.sum()),
                new_vs_pool=int((~T.in_casmi_pool).sum()), in_pubchem=int((T.pubchem_cid > 0).sum()),
                valid_structures=int((T.scorer_key != '').sum()), genus_pairs=int(len(O)),
                pathways=T.np_pathway.value_counts().head(8).to_dict(),
                mass_150_1250=int(T.mono_mass.between(150, 1250).sum()), rdkit=__import__('rdkit').__version__)
    json.dump(summ, open(os.path.join(out, 'SUMMARY.json'), 'w'), indent=1)
    log(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
