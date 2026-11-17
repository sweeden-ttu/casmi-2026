# Source metadata check — 2026-10-19

Metadata was read from the primary source websites. No new database payload was downloaded.

- LOTUS initiative: https://lotus.nprod.net/ explicitly identifies LOTUS data as CC0 and says the older Natural Products Online interface is unmaintained. The retained table uses the frozen 2023-01-06 metadata (Zenodo 7534071). A mutable Wikidata view is not a pinned replacement snapshot.
- NPAtlas downloads: https://www.npatlas.org/download labels the latest downloadable release v2024_09, DB version 2024_09, website version 3.0.2. The page explicitly links Creative Commons Attribution-Noncommercial 4.0 International. This conflicts with the earlier dataset/notebook description of NPAtlas as CC BY 4.0. The snapshot already retrieved on 2026-09-30 is retained; no claim of a new release is made. NPAtlas-derived fields should conservatively carry CC BY-NC 4.0 unless snapshot-specific permission establishes broader reuse. Credit the Natural Products Atlas and its contributors (https://doi.org/10.1093/nar/gkab941).
- PubChem data: retain the pinned September 2026 inputs and NCBI data-use/source-specific notices already shipped in the popularity-prior dataset.

This update improves alignment, reproducibility and documentation of existing public resources. It contains no new spectra or hidden-test labels.
