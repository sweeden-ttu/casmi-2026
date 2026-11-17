This table brings together 153,964 known natural products from LOTUS and NPAtlas with their chemical class, the organisms they were reported from and the number of papers describing them, keyed by the competition's scorer key; most are also in PubChem, but PubChem links only 38.4% of them to a PubMed article, while LOTUS records the source literature for 98.6%. Use it to favour known and well-documented natural products when several candidates fit a spectrum equally well, as the [data notebook](https://www.kaggle.com/code/dmitriigluzdov/casmi-26-the-data-behind-spectra-to-structures) does. It also shows which compound classes have few spectra in the competition training file: only 8.0% of these natural products have one, from 4.6% of terpenoids, the largest class, to 22.1% of carbohydrates.

![Known natural products grouped by chemical pathway, with the share having competition training spectra shown in each bar.](https://raw.githubusercontent.com/Gluzdov-Dmitrii/Enveda-CASMI-2026/main/kaggle-datasets/casmi26-natural-product-knowledge-table/pathway_coverage.png)

## How to use it

- **Rank candidates.** Give known natural products and those with more references a small bonus. The data notebook adds +1.0 for COCONUT, LOTUS or NPAtlas membership and 0.5·ln(1 + LOTUS references) to its PubChem-channel score; the [popularity dataset](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior) holds both as arrays aligned to its PubChem tier (`pc_np.npy`, `pc_lotus_refs.npy`).
- **Look up a mass.** List the known natural products within a few ppm of a neutral mass, with class, organisms and references (example below).
- **Find gaps in the training data.** Group by `np_pathway` or `np_class` and compare `in_casmi_train`: the first figure shows which classes a model has hardly seen.
- **Check sugar losses.** `sugar_and_acyl_units/` tells which structures carry sugars and which training spectra show their losses: a hexose loss is a fairly reliable sign of a hexoside (precision 0.59 in positive and 0.71 in negative mode), acyl losses are not, and a simple ranking bonus built on them did not improve the panel.
- **Check chemistry assumptions.** `spectra_chemistry/` measures which elements the training, pool and PubChem structures contain, how much of each spectrum the precursor's subformulas explain, which neutral losses and fragment ions are common, how often fragment substructures are catalogued, and how often physical rules hold for the true structures. These numbers are inputs for candidate filters and ranker features.
- **Extend a spectral library.** `external_reference_spectra/` adds 1,384 public spectra that are not near-duplicates of training spectra.

## Quick start

The example reads the dataset from either supported Kaggle mount location.

```python
from pathlib import Path
import pandas as pd
slug = 'casmi26-natural-product-knowledge-table'
D = next(p for p in (Path('/kaggle/input/datasets/dmitriigluzdov') / slug, Path('/kaggle/input') / slug) if p.is_dir())
natural_products = pd.read_parquet(D / 'natural_products.parquet')
print(natural_products[['np_pathway', 'kingdoms', 'in_casmi_train']].head())
```

For example, a neutral mass of 192.064476 Da from the public panel can be searched within 10 ppm [parts per million]. The three matching records include one reported from Vaccinium, Actinidia and Lotus, supported by 59 references. The rows show chemical classes, reported genera [groups of related species] and reference counts; an empty result means no match in this table, rather than proof that a candidate is not natural.

```python
neutral_mass = 192.064476
matches = natural_products[(natural_products.mono_mass - neutral_mass).abs() <= neutral_mass * 10e-6]
print(matches[['ik14', 'np_class', 'top_genera', 'n_references']].head())
```

![Known natural-product matches and other PubChem structures within 10 ppm of three measured public panel masses.](https://raw.githubusercontent.com/Gluzdov-Dmitrii/Enveda-CASMI-2026/main/kaggle-datasets/casmi26-natural-product-knowledge-table/mass_window_use.png)

The figure counts distinct source connectivity keys in each PubChem mass window; “other” means absent from this LOTUS/NPAtlas table, and does not establish biological origin.

## What is inside

| Main file | What it contains |
|---|---|
| `natural_products.parquet` | This table describes 153,964 structures, their classes, organisms, documentation and training-data coverage. |
| `lotus_structure_genus.parquet` | This table links structures to reported genera [groups of related species] and supporting references. |
| `natural_products.csv`, `lotus_structure_genus.csv` | These CSV copies contain the same rows for Kaggle preview and spreadsheets. |
| `candidate_joins.parquet` | This table identifies matching rows in the public structure collections. |
| `build_np.py`, `SUMMARY.json`, `JOIN_NOTES.json` | These files provide the build code, summary counts and explanations of changed matches. |
| `descriptors.parquet`, `DESCRIPTOR_METHOD.json` | These files add molecular properties, a model score for natural-product likeness, ring-system summaries and counts of structures sharing a formula. |
| `external_reference_spectra/` | 1,384 public MassBank, MoNA and GNPS reference spectra of 782 structures that are not near-duplicates of training spectra (110 of the structures have no training spectrum at all), with each record's source and licence. |
| `sugar_and_acyl_units/` | Counts of O-glycosidic sugars (hexose, deoxyhexose, pentose, uronic acid), hexose–deoxyhexose disaccharides, acyl esters (galloyl, caffeoyl, feruloyl, sinapoyl, malonyl), sulfate and phosphate for every structure of this table and of the public candidate pool, and the sugar and acyl losses observed in each of the 2,539,608 training spectra, with the rules and checks. |
| `spectra_chemistry/` | Element census of training, pool and PubChem structures; formula annotation of 2,123 training spectra; catalogue coverage of fragment formulas and H-capped fragment substructures; frequent neutral losses and fragment ions; electron-parity statistics; one fragmentation graph; 17 physical checks with their measured behaviour. `FILE_DESCRIPTIONS.json` describes every file. |

Each main-table row represents an InChIKey connectivity block [the first 14 InChIKey characters, identifying atom connections without their spatial arrangement]. The separate scorer key follows the competition’s structure-standardisation rule. Only 8.0% of the records have a competition training spectrum, so the table is useful for studying molecules without that reference evidence.

The candidate [a possible molecular answer] joins point into the pool [the public training/natural-product collection], the PubChem tier [the mass-sorted PubChem collection] and PubChemLite. The original membership flags are retained; the new explicit joins differ for one record, which is listed rather than silently changed.

Column definitions, build steps and validation are in `TECHNICAL_NOTES.md`.

## How it was made

LOTUS and NPAtlas provide broad open natural-product coverage with organism and literature links; COCONUT structures are already in the candidate pool. The build merges the LOTUS frozen release of 2026-04-13 with NPAtlas v2024_09 (retrieved 2026-09-30 and checked again on 2026-10-20). RDKit [software for working with molecular structures] recalculates the structures, formulas, masses and competition scorer keys. PubChem supplies documentation counts, and the public candidate collection and validation assignments supply coverage flags. The candidate joins make matches to each collection explicit while preserving the original table.

## Updates

- **2026-10-23:** Added measured example spectra, a fragmentation tree, element and neutral-loss tables and the physical-rule measurements used by the data notebook to `spectra_chemistry/`.
- **2026-10-22:** Added `spectra_chemistry/`: element census, spectrum formula annotation, neutral losses and fragment ions, fragment catalogue coverage, a fragmentation graph and physical checks.
- **2026-10-20, second update:** Moved to the LOTUS frozen release of 2026-04-13: 153,964 structures (2,569 added, 1,322 removed), refreshed organisms, references and classes, and 25 corrected scorer keys; added sugar and acyl unit counts and observed losses (`sugar_and_acyl_units/`).
- **2026-10-20:** Added public reference spectra not present in the competition training file; the training file already contains 99.4% of the eligible public MassBank, MoNA, GNPS and MSnLib records.
- **2026-10-19, second update:** Added molecular descriptors, natural-product likeness, ring-system summaries and formula ambiguity counts; restored the two CSV views and the build script, which were missing from the previous version.
- **2026-10-19:** Added explicit candidate matches and source checks.

## Licence and credits

The LOTUS frozen export of 2026-04-13 on [Zenodo](https://doi.org/10.5281/zenodo.19360665) is CC BY 4.0; credit the [LOTUS initiative](https://lotus.naturalproducts.net/) and [Rutz et al., eLife 2022](https://doi.org/10.7554/eLife.70780). NPAtlas is [CC BY-NC 4.0](https://www.npatlas.org/download); cite [van Santen et al., Nucleic Acids Research 2022](https://doi.org/10.1093/nar/gkab941) wherever the `npatlas_*` columns are used. The compilation follows the most restrictive component terms: **non-commercial use with attribution**. Retain snapshot-specific source and competition-derived restrictions.

The `spectra_chemistry/` tables are derived from the competition training file and the catalogues above and follow their terms; method references (MetFrag, MAGMa, SIRIUS, the Seven Golden Rules, ICEBERG) are listed in `spectra_chemistry/METHOD_CREDITS.json`.

Chemical pathway, superclass and class labels come from [NPClassifier, Kim et al., 2021](https://doi.org/10.1021/acs.jnatprod.1c00399), distributed with LOTUS. Documentation comes from [PubChem](https://pubchem.ncbi.nlm.nih.gov/), U.S. National Library of Medicine; cite [Kim et al., PubChem 2025 update](https://doi.org/10.1093/nar/gkae1059) and retain source-specific rights. Pool membership refers to [Ahmed Berat Özer’s training/COCONUT collection](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v2-pool); training membership refers to Enveda’s competition data. No competition spectra or test data are included.

Descriptor calculations use RDKit (BSD 3-Clause) and the [Ertl et al. natural-product likeness model](https://doi.org/10.1021/ci700286x).

The external reference spectra come from MassBank (MassBank-data release 2026.03; MassBank contributors), MoNA (Fiehn Lab, UC Davis, and original submitters) and GNPS (contributing laboratories and submitters); record-level licences (CC BY 4.0, CC0 1.0) and original attributions are kept with every spectrum and summarised in `external_reference_spectra/LICENCES.md`.
