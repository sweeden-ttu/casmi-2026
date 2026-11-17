Diagnostic sugar and acyl neutral-loss annotations for the natural-product knowledge table, the public candidate pool and the competition training spectra.

Files and alignment:

- `knowledge_units.parquet`: 153,964 rows, aligned to the refreshed `natural_products.parquet` from the 2026-10-20 LOTUS update; `source_row` is zero-based and `ik14` is retained for verification.
- `pool_units.parquet`: 710,701 rows, aligned to public `pool_meta.parquet`; `source_row` is zero-based and `key` is retained.
- `training_observed_losses.parquet`: 2,539,608 rows in competition `train.parquet` order; `spectrum_row` is zero-based. No peak arrays are included.
- `unit_definitions.json`, `sugarloss_units.py`: formulas, RDKit exact isotope masses, SMARTS, and the ring-based O-glycoside algorithm.
- `structure_validation.json`: named PubChem examples and positive/negative controls.
- `selected_rules.json`: per-unit, per-polarity rules fitted on identity folds 1–4 only. Ranking eligibility also requires training true positives >=20, precision >=0.5 and lift >=2.

Structure unit columns are nonnegative counts. `valid_structure` is a parsing flag. Sugars are O-glycosidic saturated five- or six-membered rings, with carbon and attached-oxygen counts distinguishing hexose, deoxyhexose (including 2-deoxy sugars), pentose and uronic acid. Stereochemistry is ignored. `hex_deoxy_disaccharide` counts linked deoxyhexosyl-to-hexoside pairs; `rutinoside_like` additionally requires a terminal-methyl donor and a 1-to-6 topology and does not establish stereochemical identity. Free reducing sugars, C-glycosides, N-glycosides and anomeric esters are excluded. `sugar_ring_other`, `sugar_pyranose` and `sugar_furanose` preserve additional topology information. Aromatic acyl groups require the stated substituents and ester connectivity. Modified or fused sugar rings and acyl groups with substituted phenolic oxygens can be missed.

Spectrum measurements use relative-to-base-peak intensities, at least 1%. For each unit, `precursor`, `carrier_shift`, and `fragment_pair` channels have a maximum `strength` and match counts `n01`, `n05`, `n10` at 1%, 5%, and 10% intensity. A fragment-pair match requires both peaks to pass the threshold; its strength is their minimum intensity. Duplicate input peaks can contribute multiple matches. Tolerance is max(0.001 Da, 10 ppm of the neutral-loss mass). The 1 mDa floor accommodates subtraction of two measured/rounded masses.

Precursor losses account for declared charge. A separate carrier-shift hypothesis accounts for conversion from common ammonium, sodium, potassium, chloride, formate and acetate adducts to the protonated/deprotonated ion. Fragment-pair losses assume singly charged products. The charge field describes the scan assumption; unparsed adducts use one charge. `eligible_single_charge` identifies polarity-consistent monomer precursors with a parsed single charge. Multiply charged, unknown-charge, dimer and polarity-conflicting precursors have descriptive scan measurements but are excluded from rule fitting, primary agreement and selected observations.

`<unit>_observed` applies the frozen rule to an eligible spectrum. `<unit>_ranking_evidence` also requires the training reliability gate. A missing observed loss is not evidence that the structure lacks that unit. Match multiplicity is not a chemical group count.

These are derived annotations on competition/public source tables. Retain the upstream competition and source-specific data terms and attribution (Enveda; the public candidate pipeline by Ahmed Berat Özer; LOTUS and NPAtlas for the knowledge table). PubChem supplied the five named reference structures. The empirical validation uses existing training spectra, not hidden-test spectra. Column order and positional alignment are part of the interface; join by the retained row identifiers when using a different table version.
