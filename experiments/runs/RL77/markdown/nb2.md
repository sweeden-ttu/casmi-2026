<!-- intro -->
# CASMI 2026 · RL77 Notebook 2 — Offline Candidate Store (Catalog, FP16 Memmap, FAISS)

*Assembled from **CASMI 2026 Colab Rubric.pdf** (pages 21–27). Python cells are the reviewed `experiments/runs/RL77/nb2/cell_NN.py` files; text cells are taken from the PDF.*

> **Assistant (end of the GLMR template):** Would you like the companion script to pre-compute and shard the PubChem/COCONUT ChemFormer embeddings into a memory-mapped FAISS index for your offline Kaggle dataset upload?
>
> **User prompt:** yes

This companion pipeline curates and deduplicates compounds from `train.parquet`, **COCONUT**, and **PubChem** by `inchikey14`, sorts the catalog by monoisotopic mass for $O(\log N)$ binary-search slicing, encodes molecules into `float16` memory-mapped **ChemFormer** embeddings alongside a **FAISS** index, and packages the artifacts for offline Kaggle execution.

<!-- before:cell_01 -->
## Cell 1: Candidate Corpus Curation, `inchikey14` Deduplication & Mass Sorting

Sorting the deduplicated catalog by `exact_mass` enables contiguous memory-mapped slicing (`np.searchsorted`) during inference, avoiding full-database scans while supporting both the **Weight-based** and **Formula-based** retrieval libraries evaluated in GLMR.

> **Review fix:** Google Drive paths become the competition mount, the `casmi26-rl77-candidates` COCONUT export (495k structures), and `/kaggle/working`. Tautomer canonicalization of ~770k structures runs on a process pool under a time budget (tier 1 always completes). `exact_mass` is stored as float64 (float32 rounds ~0.06 ppm at 1000 Da), and an empty catalog raises instead of failing later in `sort_values`.

<!-- before:cell_02 -->
## Cell 2: FP16 Memory-Mapped ChemFormer Encoding & FAISS Index Builder

Encoding into an on-disk `np.memmap` array in `float16` reduces storage by 50% (1M compounds × 256 dimensions × 2 bytes ≈ 512 MB) and allows Kaggle notebooks to map the embeddings directly from `/kaggle/input` without copying them into system RAM.

> **Review fix:** `torch.cuda.amp.autocast` is deprecated; `torch.autocast(device_type=...)` is used. Batches are length-sorted and trimmed to their longest SMILES, which leaves the masked `[CLS]` embedding unchanged and cuts CPU encoding cost ~6× (median SMILES is 39 of 256 tokens). This notebook runs on a CPU session, so it uses no GPU quota.

<!-- before:cell_03 -->
## Cell 3: Offline Wheel Bundler & Kaggle Dataset Uploader

Kaggle submissions run with internet disabled. This cell packages the sharded index, checkpoints, and pinned `rdkit==2026.03.3` wheels into a single versioned Kaggle Dataset.

> **Review fix:** `matchms` was bundled but never imported (and pulls in numba), and `pyarrow` ships with the Kaggle image. Wheels are downloaded with the running interpreter (`sys.executable -m pip`) so they match the image, and a failed download stops the run instead of producing an unusable offline dataset. Publishing happens from the repository with `tools/kaggle_pipeline.py publish-output nb2`.

<!-- before:cell_04 -->
## Cell 4: Zero-Copy Offline Retriever for Your Kaggle Submission Notebook

Drop this class into your Kaggle submission notebook (replacing `retrieve_top_k_candidates` in Cell 7 and Cell 8 of the main template). It reads embeddings directly from disk via `mode="r"` memory mapping and slices mass windows with `np.searchsorted` in under 5 milliseconds per molecule.

> **Review fix:** With fewer than K candidates in the window, the original returned only the FAISS hits and dropped the mass-matched candidates; FAISS also pads missing results with `-1`, which silently indexed the last memmap row. Constrained candidates are now kept first and FAISS fills the rest. Neutral mass uses `neutral_mass_from_precursor()` (formate/acetate spellings, `[2M+X]`), and `top_k`, `ppm_tol`, `in_library_boost`, and `nprobe` come from the Evidently weights.

<!-- before:cell_05 -->
## Cell 5: Build the Store on the Competition Data and Measure Retrieval

The PDF stops at the retriever class. This cell runs Cells 1–4 for real: the notebook-1 molecule encoder embeds every candidate, and retrieval is scored on notebook 1's held-out timsTOF molecules (MRR@25, recall@K, and whether the true structure is in the mass window at all). Outputs in `/kaggle/working` become the `casmi26-rl77-nb2-store-outputs` dataset that notebooks 3–5 and the offline submission attach.

`in_library_boost` is **not** fitted here: every validation molecule comes from `train.parquet` (tier 1), so a validation sweep would always push the boost up while the hidden test set also contains out-of-library structures.

<!-- end -->
## Memory & Latency Profile on Kaggle

| Resource Metric | Unsharded In-Memory Tensor | Sharded FP16 Memmap + Binary Search |
| --- | --- | --- |
| **1.5M Compound Disk Size** | 1.54 GB (`float32`) | 768 MB (`float16 .mmap`) |
| **Kaggle Peak System RAM** | Over 6.5 GB (risks OOM crash) | Under 450 MB (OS page-cache backed) |
| **Mass-Window Slice Latency** | 45 ms per molecule (boolean mask) | Under 1.2 ms per molecule (`searchsorted`) |
| **Total 400-Molecule Retrieval** | ~18 seconds | Under 1 second (plus ~7 min Beam Search) |

The table above shows how sorting by `exact_mass` and mapping `float16` vectors keeps total Kaggle submission runtime under 8 minutes while leaving over 12 GB of system RAM free for PyTorch beam search and ensemble blending. *(Figures are the PDF's estimates; `metrics.json` records the measured run.)*
