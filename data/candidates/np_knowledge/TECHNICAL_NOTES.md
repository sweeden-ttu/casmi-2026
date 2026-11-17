# CASMI26 Natural Product Knowledge Table — Technical notes

These notes retain the complete methods, file and column definitions, measurements, caveats, updates and source credits. The companion Kaggle card provides a short introduction.

## Terms used in these notes

**Candidate** [a possible molecular answer].

**Pool** [a collection of candidate structures].

**Tier** [the mass-sorted PubChem structure collection used here].

**Prior** [extra ranking evidence, such as how often a compound is documented].

**Fold** [one group reserved for checking a model while the other groups supply training data].

**InChIKey connectivity block** [the first 14 InChIKey characters, identifying connectivity without stereochemistry]. The scorer key is calculated from the exact exported stereo-free SMILES after RDKit chooses a standard tautomer [a form differing in proton positions and bond arrangement].

**MRR@25** [the mean of 1 divided by the correct answer’s rank among the first 25 unique scored identities; a missing answer or a rank beyond 25 contributes zero].

## Dataset overview

One row per known natural product: **153,964 connectivities** from [LOTUS](https://lotus.naturalproducts.net/) and [NPAtlas](https://www.npatlas.org/), each with its chemical class, the organisms that make it, the key the [CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra) scorer would compute, and how well PubChem documents it. The notebook [CASMI 26: From Spectra to Structures](https://www.kaggle.com/code/dmitriigluzdov/casmi-26-from-spectra-to-structures) mounts this table for its analysis of natural-product classes, their training coverage and their mass ranges.

The hidden test of the competition comes from biological extracts. Only **8.0%** of these natural products have a training spectrum (12,291 of 153,964), while 98% are present in the public candidate pool, so most of them must be identified without a reference spectrum. This table describes that chemistry: what classes it contains, which organisms produce it, and how much is known about each compound.

## 2026-10-19: explicit candidate joins

`candidate_joins.parquet` adds 153,964 rows aligned one-for-one with `natural_products.parquet`. It records `pool_row` (scorer_key to public pool.key), `tier_row` and `tier_match_count` (raw ik14 to the mass-sorted tier), and `pubchemlite_row` / `pubchemlite_match_count` (raw ik14 to the PubChemLite add-on). Indices are zero-based; -1 denotes no match. The smallest source row is the representative when several rows share the raw connectivity.

The joins find 151,114 current scorer-key pool matches, 144,153 tier matches and 7,107 PubChemLite matches. Values and membership flags are refreshed, with the original raw-key/scorer-key definitions and mass ordering retained. One record differs between the membership flag (scorer key OR raw key) and the explicit scorer-key-only pool join; `JOIN_NOTES.json` lists it.

Notebook integration: the knowledge table composition groups these tier/PubChemLite joins by NP pathway. Its LOTUS references also supply the precomputed tier prior in the companion popularity dataset. The current LOTUS frozen release is 2026-04-13; NPAtlas retains v2024_09 table-relevant content, originally retrieved 2026-09-30 and checked against a public TSV redownload on 2026-10-20.

Reproduction: join `candidate_joins.parquet` by `ik14` with validation `one_to_one`, or assert its ik14 column equals the knowledge table before positional use. Stored connectivity and scorer-standardised keys are separately labelled.

Source credits: PubChem / U.S. National Library of Medicine (https://doi.org/10.1093/nar/gkae1059), LOTUS / Rutz et al. (https://doi.org/10.7554/eLife.70780), Natural Products Atlas and contributors (https://doi.org/10.1093/nar/gkab941). LOTUS frozen 2026-04-13 is CC BY 4.0 (https://doi.org/10.5281/zenodo.19360665). The current NPAtlas download page specifies CC BY-NC 4.0; retain snapshot-specific terms and attribution. Existing PubChem/source-specific and competition-derived restrictions are retained. See `SOURCE_CHECK_20261019.md`.

No new spectra or hidden-test labels are added. Public panel labels are used only for evaluation. A local documented-natural-product panel does not establish a hidden-test improvement. Earlier MANIFEST files, including `MANIFEST_20261019.json`, describe earlier snapshots; `CHANGED_FILES.json` and `LOTUS_REFRESH.json` describe the current refresh.

## 2026-10-20, second update: LOTUS 2026-04-13 refresh

The table now uses the MD5-verified LOTUS frozen release of 2026-04-13, [Zenodo record 19360665](https://zenodo.org/records/19360665), DOI [10.5281/zenodo.19360665](https://doi.org/10.5281/zenodo.19360665), retrieved 2026-10-20. The release is CC BY 4.0. Raw connectivity grouping, source preference, membership definitions, column types and mass ordering are retained; current values and aligned joins are refreshed.

**Historical source note.** The table used LOTUS 2023-01-06 until 2026-10-20. The older [Zenodo record 7534071](https://zenodo.org/records/7534071) currently also reports CC BY 4.0; current metadata does not establish which licence was displayed at its original publication.

| Measure | Before refresh | After refresh |
|---|---:|---:|
| Raw connectivity rows | 152,717 | 153,964 |
| LOTUS members | 147,983 | 151,885 |
| NPAtlas members | 33,497 | 33,497 |
| Both sources | 28,763 | 31,418 |
| Structure–genus rows | 291,660 | 293,503 |
| Training membership flags | 12,547 | 12,291 |
| Pool membership flags | 150,002 | 151,115 |
| Absent under pool membership flag | 2,715 | 2,849 |
| Explicit scorer-key pool joins | 149,994 | 151,114 |
| Raw-key tier joins | 143,473 | 144,153 |
| Raw-key PubChemLite joins | 7,506 | 7,107 |
| Pool flag / explicit join differences | 8 | 1 |

`LOTUS_REFRESH.json` records the source receipts and validation; `CHANGED_FILES.json` identifies the refreshed files. Earlier dated manifests remain historical. These metadata changes do not establish an accuracy gain: the public comparison panel consists entirely of documented natural products.


## Contents at a glance

Each row of `natural_products` is one natural product, identified by its InChIKey connectivity block (the first 14 characters of the InChIKey, which identify a structure without stereochemistry). The tables are stored as Parquet for fast, typed loading (`pd.read_parquet`); `natural_products.csv` and `lotus_structure_genus.csv` hold exactly the same rows as CSV so that they can be previewed on Kaggle and opened in any spreadsheet.

Three rows of `natural_products` (selected columns):

| ik14 | formula | mono_mass | np_pathway / np_superclass / np_class | n_organisms | top_genera | npatlas_name | in_casmi_train | pubmed_articles |
|---|---|---:|---|---:|---|---|---|---:|
| `RYYVLZVUVIJVGH` | C8H10N4O2 | 194.0804 | Alkaloids / Pseudoalkaloids / Purine alkaloids | 72 | Camellia; Coffea; Ilex | Caffeine | True | 33,846 |
| `MXXWOMGUGJBKIW` | C17H19NO3 | 285.1365 | Alkaloids / Lysine alkaloids / Piperidine alkaloids | 26 | Piper; Macropiper; Aglaia | Piperine | True | 1,201 |
| `REFJWTPEDVJJIY` | C15H10O7 | 302.0427 | Shikimates and Phenylpropanoids / Flavonoids / Flavonols | 1,745 | Hypericum; Rhododendron; Ilex | (not in NPAtlas) | True | 18,205 |

Caffeine (the tea, coffee and holly alkaloid), piperine (black pepper) and quercetin (a flavonol found in over a thousand organisms) all have training spectra; 92% of the table does not. `lotus_structure_genus` lists every producing genus of a structure with its kingdom, family and the number of supporting references.

A minimal loading example:

```python
import os
import pandas as pd
D = next(p for p in ('/kaggle/input/datasets/dmitriigluzdov/casmi26-natural-product-knowledge-table', '/kaggle/input/casmi26-natural-product-knowledge-table')
         if os.path.isdir(p))   # newer and older mount paths
np_table = pd.read_parquet(f'{D}/natural_products.parquet')      # sorted by neutral monoisotopic mass

# candidates within 10 ppm of a neutral mass, with their chemical class and documentation
m = 354.1467
hits = np_table[(np_table.mono_mass - m).abs() < m * 10e-6]
print(hits[['smiles', 'np_superclass', 'kingdoms', 'n_organisms', 'pubmed_articles', 'in_casmi_train']])

# which plant genera make a given structure
genus = pd.read_parquet(f'{D}/lotus_structure_genus.parquet')
print(genus[genus.ik14 == hits.ik14.iloc[0]].sort_values('n_references', ascending=False).head())
```

## Files

| File | Rows | Content |
|---|---:|---|
| `natural_products.parquet` | 153,964 | One row per InChIKey connectivity block; columns described below. |
| `lotus_structure_genus.parquet` | 293,503 | LOTUS structure–genus pairs with kingdom, family, species count and reference count. |
| `natural_products.csv`, `lotus_structure_genus.csv` | 153,964 / 293,503 | The same two tables as CSV, for preview and spreadsheets. |
| `SUMMARY.json` | | Row counts, overlaps and the RDKit version. |
| `build_np.py` | | The complete build script. |

### Columns of `natural_products.parquet`

| Column | Meaning |
|---|---|
| `ik14` | InChIKey connectivity block (first 14 characters) from the source database. |
| `scorer_key` | Key computed the way the competition scorer does it: InChIKey14 after RDKit 2026.03.3 TautomerEnumerator canonicalizes the exact exported stereo-free SMILES. |
| `smiles`, `formula`, `mono_mass`, `n_heavy` | Stereo-free structure, formula, neutral monoisotopic mass (Da), heavy atoms. |
| `in_lotus`, `in_npatlas` | Source membership (31,418 compounds are in both). |
| `in_casmi_pool`, `in_casmi_train` | Present in the public 710,701-structure candidate pool / has a training spectrum. |
| `np_pathway`, `np_superclass`, `np_class` | [NPClassifier](https://doi.org/10.1021/acs.jnatprod.1c00399) biosynthetic pathway (7), superclass (76) and class (666), from LOTUS. |
| `n_organisms`, `n_references`, `n_stereoisomers` | Distinct organisms, literature references and stereoisomers recorded in LOTUS. |
| `kingdoms`, `top_families`, `top_genera` | Most frequent producing taxa (up to three, separated by semicolons). |
| `npatlas_id`, `npatlas_name`, `npatlas_origin`, `npatlas_genera`, `npatlas_first_year` | NPAtlas record: microbial origin (bacterium or fungus), genus and year of first report. |
| `pubchem_cid`, `pubchem_substances`, `pubmed_articles`, `patents` | Most-deposited PubChem CID of the connectivity and counts summed over its CIDs. September fields are retained exactly for existing raw keys; newly added raw keys alone use separately verified current public files, retrieved 2026-10-20. |

## What the table shows

* **Classes.** Terpenoids 59,335 · shikimates and phenylpropanoids 28,198 · alkaloids 20,831 · polyketides 20,177 · amino acids and peptides 8,783 · fatty acids 8,054 · carbohydrates 1,485.
* **Organisms.** Plants dominate LOTUS (94,989 structures have Archaeplastida as their only stored kingdom label); NPAtlas adds bacterial and fungal metabolites. Under the retained distinct-organism-name definition, 67.7% of LOTUS structures are reported from a single organism.
* **Documentation.** 38% of the compounds have at least one PubMed article linked in PubChem and 35% appear in a patent, far above a typical PubChem structure (3.5% with a PubMed article).
* **Coverage.** 147,855 structures fall in the 150–1,250 Da range of the public PubChem tier; 2,849 are absent from the public candidate pool.

## Possible uses

* Class-aware validation: MRR@25 reported by `np_pathway` shows which chemistry a model handles.
* Priors: producing taxa, class frequencies and documentation counts as ranking features for same-formula isomers.
* Candidate expansion: the 2,849 structures missing from the public pool.
* Analogue reasoning: candidates grouped by `np_superclass` reveal close relatives of a library hit.

## How it was built

`build_np.py` merges the MD5-verified LOTUS frozen metadata of 2026-04-13 ([Zenodo 19360665](https://zenodo.org/records/19360665), DOI [10.5281/zenodo.19360665](https://doi.org/10.5281/zenodo.19360665)) and NPAtlas v2024_09 by InChIKey connectivity block, recomputes structures, formulas, masses and scorer keys with RDKit 2026.03.3, and joins PubChem counts from Compound Extras (`CID-SID`, `CID-PMID`, `CID-Patent`, `CID-InChI-Key`). September fields are retained exactly for existing raw keys; newly added raw keys alone use separately verified current public files, retrieved 2026-10-20. Component file dates and checksums are recorded in `LOTUS_REFRESH.json`. Pool and training membership come from the public [candidate pool](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets) and [identity folds](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-identity-disjoint-validation-folds).

The old and new LOTUS metadata headers are identical. The 2026 export packs some classifier values with literal ` $ ` separators; these values are decoded before applying the original modal single-label rule. Taxonomy retains the original top-three frequency rule and source-order tie behavior. The decoder reproduces all 2023 modal/top-taxonomy fields and structure–genus rows exactly. The delivered labels comprise 7 pathways, 76 superclasses and 666 classes; no packed classifier strings remain. Organism names, DOI strings and structure strings are not split.

NPAtlas retains v2024_09 table-relevant content: the original snapshot was retrieved 2026-09-30, with public TSV redownload and equivalence checks on 2026-10-20. In the redownload equivalence check, all five aggregated NPAtlas fields and all 4,734 old NPAtlas-only selected structures match the original table. Refreshed selected structures still follow the original LOTUS-first source preference. Original raw-file byte identity cannot be established because the original payload and checksum were not retained.

Scorer keys apply RDKit 2026.03.3 TautomerEnumerator to the exact exported stereo-free SMILES. This corrects 25 representation-sensitive legacy-builder keys; all 153,964 delivered keys were checked against the public scorer. This compatibility correction is separate from LOTUS source curation.

## Terms and credit

LOTUS ([Rutz et al., eLife 2022](https://doi.org/10.7554/eLife.70780)) frozen 2026-04-13 is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), DOI [10.5281/zenodo.19360665](https://doi.org/10.5281/zenodo.19360665). NPAtlas ([van Santen et al., Nucleic Acids Research 2022](https://doi.org/10.1093/nar/gkab941)) is released under [CC BY-NC 4.0](https://www.npatlas.org/download) (non-commercial use) and is to be cited wherever the `npatlas_*` columns are used. PubChem data are distributed by NCBI for free reuse. This compilation therefore follows the most restrictive component terms: non-commercial use with attribution. No competition spectra or test data are included.

## Acknowledgements

The natural-product records are the work of the [LOTUS](https://lotus.naturalproducts.net/) initiative ([Rutz et al., eLife 2022](https://doi.org/10.7554/eLife.70780)) and the [Natural Products Atlas](https://www.npatlas.org/) ([van Santen et al., Nucleic Acids Research 2022](https://doi.org/10.1093/nar/gkab941)). The biosynthetic pathway, superclass and class labels are [NPClassifier](https://doi.org/10.1021/acs.jnatprod.1c00399) annotations (Kim et al., J. Nat. Prod. 2021) distributed with LOTUS. Documentation counts come from [PubChem](https://pubchem.ncbi.nlm.nih.gov/) (U.S. National Library of Medicine). The `in_casmi_pool` flag refers to the public candidate pool of training and COCONUT structures by [Ahmed Berat Özer](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v2-pool), and `in_casmi_train` to the competition training file provided by Enveda.

## Reading and alignment checks

The CSV views and `build_np.py` are included alongside the Parquet files. The figure counts the refreshed `in_casmi_train` flags and includes 7,101 records without a pathway label, of which 372 have training spectra. A monoisotopic mass is the neutral molecule's mass calculated with each element's most abundant isotope, in daltons (Da). NPClassifier organises chemical labels from broad pathways to more specific superclasses and classes.

The refreshed membership flag marks 2,849 structures absent from the pool. The explicit current join finds 151,114 matches, leaving 2,850 unmatched; `JOIN_NOTES.json` explains the one differing record. The membership flag uses scorer key OR raw key, while the explicit pool join uses scorer key alone. Both measurements retain their original definitions. These catalogue counts do not estimate competition-test coverage.

The mass-window example uses a tolerance of 10 parts per million (ppm). A genus is a biological grouping of related species; Archaeplastida is the plant/algal lineage.

## 2026-10-19, second update: molecular descriptors and formula ambiguity

`descriptors.parquet` has 153,964 rows in exactly the `ik14` order of `natural_products.parquet`. The restored `natural_products.csv`, `lotus_structure_genus.csv` and `build_np.py` are present. The CSV views contain the same rows and columns as their respective Parquet tables; their row counts are 153,964 and 293,503. The build script remains available alongside the descriptor method record.

### Files and column definitions

| File | Rows / format | Bytes | Purpose |
|---|---|---:|---|
| `DESCRIPTOR_METHOD.json` | JSON | 3,694 | Descriptor definitions, RDKit and NP-score model provenance, source hashes, missing-value rules and alignment checks. |
| `descriptors.parquet` | 153,964; Parquet | 5,679,231 | 153,964 aligned natural-product rows with RDKit descriptors, NP-likeness, scaffolds, validity and formula-ambiguity counts. |

#### Columns of `descriptors.parquet`

| Column | Type | Meaning |
|---|---|---|
| `ik14` | large_string | Stored 14-character InChIKey connectivity block; order matches natural_products.parquet. |
| `mw` | float | Average molecular weight calculated with RDKit MolWt (g/mol); not monoisotopic mass. |
| `clogp` | float | RDKit Wildman-Crippen calculated logP, unitless. |
| `tpsa` | float | RDKit topological polar surface area in square angstroms. |
| `hbd` | uint16 | Number of hydrogen-bond donors, RDKit Lipinski definition. |
| `hba` | uint16 | Number of hydrogen-bond acceptors, RDKit Lipinski definition. |
| `rotatable_bonds` | uint16 | RDKit default strict rotatable-bond count. |
| `rings` | uint16 | RDKit ring count. |
| `aromatic_rings` | uint16 | RDKit aromatic-ring count. |
| `fsp3` | float | Fraction of carbon atoms with sp3 hybridization. |
| `heavy_atoms` | uint16 | Number of non-hydrogen atoms. |
| `formal_charge` | int16 | Total formal charge of the source molecular structure. |
| `np_likeness` | float | RDKit Contrib NP_Score value using its distributed open-data model; null if undefined for an atypical structure. |
| `murcko_scaffold` | large_string | Stereo-free Bemis-Murcko scaffold SMILES of the source structure; empty for acyclic molecules. |
| `generic_scaffold` | large_string | RDKit generic Murcko scaffold SMILES, with atoms carbon and bonds single; empty for acyclic molecules and null if the transformation is undefined for an otherwise valid structure. |
| `rdkit_valid` | bool | Whether the stored knowledge-table SMILES parses as a sanitized RDKit molecule. Invalid rows retain their position with null chemistry descriptors and scaffolds; formula counts still join the original table formula. |
| `knowledge_formula_isomers` | uint16 | Exact number of knowledge-table rows with identical RDKit molecular formula, including this row. |
| `tier_formula_isomers` | uint16 | Exact uint16 number of successfully parsed public tier rows with the original table formula; zero if absent. Invalid knowledge SMILES still use the stored table formula for this join. |

### Method and validation

RDKit 2026.03.3 computes the molecular descriptors from the selected knowledge-table SMILES. The refresh reuses 147,841 unchanged selected structures and computes 6,123 new or changed rows. Floating descriptors use float32, counts use uint16 and formal charge uses signed int16. NP-likeness uses the distributed RDKit Contrib NP_Score model, trained on openly available natural products and ZINC background structures. The model SHA-256 is `34db5b02d762e05a0243e640aee24202a5303b7ec810ced7ff929a233e082ab6`; the scoring code SHA-256 is `fad19fda1fbf6688c54a15f8dd6ff4fb8d4c0139a3c11537e9b81d1d0332a56c`.

A Bemis–Murcko scaffold is the ring system and connecting chains of a molecule. Both its stereo-free SMILES and its generic form are calculated from the source structure, without another tautomer conversion. The generic form replaces atoms with carbon and bonds with single bonds. Valid acyclic molecules have empty scaffold strings; invalid molecules have null scaffolds. A failed generic-scaffold transformation on an otherwise valid molecule leaves only that scaffold value null; valid numeric descriptors and `rdkit_valid=True` are retained. Repeated scaffold strings use dictionary encoding in Parquet.

`knowledge_formula_isomers` counts rows of this knowledge table with the same formula. `tier_formula_isomers` counts successfully parsed rows of the 105,875,489-row public PubChem tier with that formula. Matching uses exact RDKit formula text, including charge, with default isotope handling (`separateIsotopes=False`). These are row counts including aliases, rather than counts of unique scorer identities or stereoisomers. Knowledge-table formula counts are rebuilt for all rows. Exact tier counts are reused for previously known formulas; 749 newly observed formulas were independently resolved against 204,862 exact tier-window rows.

All 153,964 identifiers match the refreshed table order. Recomputed formulas and heavy-atom counts agree for every valid structure; knowledge-formula counts also agree for every row. There are 0 invalid source SMILES and 0 null NP scores. Four sanitized metal-containing structures have undefined RDKit generic scaffolds; only those scaffold values are null, while `rdkit_valid=True` and valid numeric descriptors are retained. `DESCRIPTOR_METHOD.json` lists the identifiers and exceptions. There are 8,493 valid acyclic/empty scaffolds. Tier formula counts are nonzero for 146,797 rows and all fit uint16 exactly. The data table and method record total 5,682,925 bytes. `DESCRIPTOR_METHOD.json` records source hashes, missing values and alignment checks.

### Limits and source terms

The source table includes unusual ions, atoms and small fragments; it is not uniformly curated as neutral natural products. NP-likeness is a model descriptor, not a biological label. Formula ambiguity counts source rows and cannot distinguish candidates already restricted to one formula. The companion [popularity resource](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior) lists 14,410 tier formula failures in `formula_failures.parquet`; these rows have zero counts and are excluded from valid-formula totals.

Credit RDKit contributors (BSD 3-Clause) and Ertl, Roggo and Schuffenhauer, “Natural Product-likeness Score and Its Application for Prioritization of Compound Libraries”, J. Chem. Inf. Model. 48, 68–74 (2008), https://doi.org/10.1021/ci700286x. LOTUS frozen 2026-04-13 is CC BY 4.0 (DOI https://doi.org/10.5281/zenodo.19360665; Rutz et al., https://doi.org/10.7554/eLife.70780); the retained NPAtlas snapshot is conservatively treated as CC BY-NC 4.0 (https://www.npatlas.org/download; https://doi.org/10.1093/nar/gkab941). PubChem / U.S. National Library of Medicine is credited at https://doi.org/10.1093/nar/gkae1059. Existing source and competition-derived terms remain applicable.

### Companion build records

`CARD_ADDITIONS.md` explains the additions, `ADDED_FILE_DESCRIPTIONS.json` records build-time file and column metadata, and `ADDITIONS_MANIFEST.json` records build-time file sizes, hashes and provenance. The current card and these notes describe the delivery layout. A distribution inventory must describe the actual distributed filenames and dataset locations.

## Practical mass-window example and comparison

For example, a neutral mass of 192.064476 Da from the public panel can be searched within 10 ppm [parts per million]. The three matching records include one reported from 63 organism names, including Vaccinium, Actinidia and Lotus, supported by 59 references. The rows show chemical classes, reported genera [groups of related species] and reference counts; an empty result means no match in this table, rather than proof that a candidate is not natural.

```python
neutral_mass = 192.064476
matches = natural_products[(natural_products.mono_mass - neutral_mass).abs() <= neutral_mass * 10e-6]
print(matches[['ik14', 'np_class', 'top_genera', 'n_references']].head())
```

![Known natural-product matches and other PubChem structures within 10 ppm of three measured public panel masses.](https://raw.githubusercontent.com/Gluzdov-Dmitrii/Enveda-CASMI-2026/main/kaggle-datasets/casmi26-natural-product-knowledge-table/mass_window_use.png)

The figure counts distinct source connectivity keys in each PubChem mass window; “other” means absent from this LOTUS/NPAtlas table, and does not establish biological origin.


The three panel identifiers are the first three in lexical order, selected before measuring table coverage. Windows use measured neutral mass ±10 ppm; array bounds include the lower endpoint and exclude the upper endpoint. Counts collapse duplicate PubChem rows by their source connectivity block; they do not recalculate competition scorer keys. A match means present in this dated LOTUS/NPAtlas compilation, not a comprehensive natural-product label.

| Panel identifier | Measured neutral mass (Da) | Tier rows | Distinct source keys | Known-table keys | Other keys |
|---|---:|---:|---:|---:|---:|
| AAWZDTNXLSGCEK | 192.064476452 | 3,481 | 2,930 | 3 | 2,927 |
| ACNHBCIZLNNLRS | 435.240846873 | 5,564 | 4,663 | 9 | 4,654 |
| ACWBQPMHZXGDFX | 435.227659299 | 7,648 | 6,230 | 1 | 6,229 |

LOTUS and NPAtlas were used because they combine broad open structure coverage with organism and literature links. A comparative claim that these are the largest databases was not established by the supplied files. COCONUT coverage is already available through the public candidate pool.

## External public reference spectra (2026-10-20)

The `external_reference_spectra/` folder contains 1,384 screened public MS/MS reference spectra for 782 scorer keys, per-record source attribution and licences, structure coverage, candidate preparation, record compatibility diagnostics and precomputed links to existing training-library SIDs. The LOTUS refresh updates the knowledge table and the external structure-coverage annotations; spectra, attribution, SID links and runtime integration files retain their existing values.

Under the declared exact/near-duplicate screen, the training file already contains 99.4% of the eligible public MassBank/MoNA/GNPS/MSnLib records. Specifically, 263,995 of 265,582 eligible record occurrences pass the near-training similarity rule. This is a spectrum-overlap result, not a claim of independent acquisitions or hidden-test improvement.

The notebook extension enables 751 spectra for 469 already trained structures: 265 MassBank, 258 MoNA and 228 GNPS spectra (521 positive, 230 negative). It uses only existing, pool-reachable SIDs with verified scorer identities and exact masses, recognized adducts, passing precursor/mass diagnostics and consistent polarity. Of 1,224 trained-structure delta spectra, 457 fail the selected quality/link checks and 16 further rows have inconsistent adduct polarity. The 160 spectra for 110 new-to-training structures remain available as data; they are not added to the runtime library or candidate pool. No feature-bank rows are appended. The candidate-additions table is a preparation artifact, not an enabled runtime pool extension. Quarantined records are absent from the retained delta.

Scorer identities use RDKit 2026.03.3: parse, remove stereochemistry, canonicalize the tautomer, then retain the first InChIKey block. No neutralization or largest-fragment selection is performed. SID links refer to the distributed structure table and pool. Multiple existing SIDs are resolved by verified scorer key/mass and pool reachability, preferring an identical library key, then lowest SID. No full-table molecular standardization runs in the notebook.

The optional extension builds separate CSR arrays with bounded memmap copies and streams spectrum metadata. Source cleaning keeps at most 512 peaks, applies a 0.001 relative-intensity floor and removes peaks above precursor + 2 Da. Missing files, checksum/schema mismatches, invalid indices or a failed extended Library/Engine construction select the original training cache with one diagnostic line. A successful load prints `external reference spectra: 751 added for 469 structures`.

The source snapshot selection is MassBank-data release 2026.03; MoNA LC-MS/MS export 2026-10-20 00:56:39.961 UTC; GNPS community plus 13 direct natural-product collections; and NIH/TargetMol MSnLib exports from Zenodo record 21105617. MSnLib contributes zero retained delta spectra. DOWNLOADS.csv and SOURCE_MANIFESTS.json record all 24 public source files, versions, URLs, dates, MD5 and SHA-256 hashes. Record-level licences and original authors, accession, copyright, publication and source URLs are retained in external_delta.parquet. Retained terms are CC BY 4.0 (510 spectra), CC0 1.0 (438), CC BY (403) and CC-BY (33); version-unspecified strings remain literal. Restricted or unclear records were excluded. See LICENCES.md for component terms and official links.

Source credits: MassBank contributors and MassBank Europe/MassBank-data; MoNA at the Fiehn Lab, UC Davis, and original submitters; GNPS and the contributing PIs, data collectors and submitters; MSnLib creators Corinna Brungs, Robin Schmid and Tomas Pluskal. Harmonization comprises parsing and field/unit normalization, scorer-standardized structure identifiers, sorted/base-peak-normalized peaks and eligibility/duplicate screens. These modifications do not imply contributor endorsement.

Cluster validation executes the patched cells 35–36, the full base-engine feature wrapper and ranker, with Kaggle-like mounts, on 12 visible molecules and 20 public panel molecules. For panel queries, all enveda-np-examples references are excluded from direct search and analog representatives. Runtime and output comparisons are recorded in INTEGRATION.json. CPU cluster emulation does not establish the exact runtime or peak RAM of a Kaggle T4 session, and downstream engines, spectrum simulators and final fusion are not part of this validation.

Measured CPU-cluster results: 1/12 visible and 1/20 panel top-60 lists change; top-25 changes are 1/12 and 1/20. The first full cache build plus extension takes 213.08 s and peaks at 5.391 GiB RSS. The final revision reuses the original cache after full hash verification; its extension rebuild/check takes 24.17 s. Engine initialization plus the 32 queries takes 107.49 s without the extension and 104.83 s with it, with peaks 5.974 and 5.939 GiB RSS. Forced load-failure fallback reproduces the unmodified notebook's 12 visible candidate lists and feature bytes. These public checks demonstrate integration compatibility, not a hidden-test gain.

## Changelog

* **2026-10-20, second update:** refreshed LOTUS to frozen 2026-04-13, corrected source attribution and licence, refreshed membership and candidate joins, decoded packed classifier labels, checked scorer keys against exported SMILES, and updated descriptors, formula counts and organism examples. NPAtlas table-relevant content and retained PubChem fields are preserved; the dated section above records the count comparison.

## 2026-10-20, second update: sugar and acyl units (`sugar_and_acyl_units/`)

Diagnostic sugar and acyl neutral-loss annotations, in the folder `sugar_and_acyl_units/`.

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

On the public panel a bonus or penalty built from these rules did not improve ranking (PubChem search with the documentation prior MRR@25 0.969 unchanged at weight 0.25; main list without the answer in the pool 0.3605 to 0.3597), so the annotations are provided as data, not as a recommended ranking rule.
