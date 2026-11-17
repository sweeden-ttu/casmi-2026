# [Colab Companion Cell 2] Encode catalog into FP16 memory-mapped file + build FAISS & Formula indices
import faiss
import torch
from torch.cuda.amp import autocast


@torch.no_grad()
def encode_catalog_to_memmap_and_faiss(
    catalog_df: pd.DataFrame,
    mol_encoder: torch.nn.Module,
    tokenizer,
    mmap_path: Path = IndexCFG.EMBEDDINGS_MMAP,
    faiss_path: Path = IndexCFG.FAISS_INDEX_PATH,
    formula_idx_path: Path = IndexCFG.FORMULA_INDEX_PATH,
    batch_size: int = IndexCFG.SHARD_BATCH_SIZE,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    """
    Streams ChemFormer molecular embeddings directly to a float16 disk memmap,
    builds an inner-product FAISS index for global fallback search, and exports
    a CSR-style molecular formula lookup table for formula-constrained retrieval.
    """
    mol_encoder.eval().to(device)
    n_compounds = len(catalog_df)
    d_model = IndexCFG.D_MODEL

    # 1. Allocate on-disk float16 memory-mapped array: shape (N, 256)
    mmap_embs = np.memmap(
        mmap_path, dtype=np.float16, mode="w+", shape=(n_compounds, d_model)
    )

    smiles_list = catalog_df["canonical_smiles"].tolist()

    for start_idx in range(0, n_compounds, batch_size):
        end_idx = min(start_idx + batch_size, n_compounds)
        batch_smi = smiles_list[start_idx:end_idx]

        token_Batch = torch.stack(
            [tokenizer.encode(smi, add_cls=True) for smi in batch_smi]
        ).to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
            _, e_m = mol_encoder(token_Batch)  # Normalized (B, 256)

        mmap_embs[start_idx:end_idx] = e_m.cpu().numpy().astype(np.float16)

        if (start_idx // batch_size) % 50 == 0:
            mmap_embs.flush()
            print(f"Encoded {end_idx:,} / {n_compounds:,} molecules...")

    mmap_embs.flush()

    # 2. Build FAISS Inner-Product Index (Cosine similarity on L2-normalized vectors)
    quantizer = faiss.IndexFlatIP(d_model)
    nlist = min(4096, max(1, int(math.sqrt(n_compounds))))
    if n_compounds >= 10000:
        index = faiss.IndexIVFFlat(quantizer, d_model, nlist, faiss.METRIC_INNER_PRODUCT)
        # Train IVF centroids on a representative sample
        sample_size = min(n_compounds, 131072)
        sample_idx = np.random.choice(n_compounds, size=sample_size, replace=False)
        train_vecs = np.asarray(mmap_embs[sample_idx], dtype=np.float32)
        index.train(train_vecs)
    else:
        index = quantizer

    # Stream shards into FAISS index
    for start_idx in range(0, n_compounds, batch_size * 16):
        end_idx = min(start_idx + batch_size * 16, n_compounds)
        shard_fp32 = np.asarray(mmap_embs[start_idx:end_idx], dtype=np.float32)
        index.add(shard_fp32)

    faiss.write_index(index, str(faiss_path))
    print(f"Saved FAISS index ({index.ntotal:,} vectors) to {faiss_path}")

    # 3. Build CSR-style inverted index for exact Molecular Formula retrieval (GLMR Formula Library)
    formula_groups = catalog_df.groupby("molecular_formula")["catalog_id"].apply(
        lambda s: np.asarray(s.values, dtype=np.int32)
    )
    formulas = np.array(list(formula_groups.index), dtype=str)
    lengths = np.array([len(arr) for arr in formula_groups.values], dtype=np.int32)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat_indices = (
        np.concatenate(list(formula_groups.values))
        if len(lengths) > 0
        else np.empty(0, dtype=np.int32)
    )

    np.savez_compressed(
        formula_idx_path,
        formulas=formulas,
        offsets=offsets,
        flat_indices=flat_indices,
    )
    print(f"Saved {len(formulas):,} unique molecular formula buckets to {formula_idx_path}")