# Evidently Remote Workspace Browser & Table Join Studio (VS Code Extension)

A Visual Studio Code extension for browsing all projects, datasets, and reports on a local or remote **Evidently 0.7.x** workspace (`http://192.168.0.15:8000`) and joining tables to create and publish Evidently evaluation reports.

## Features

1. **Activity Bar Workspace Explorer (`Evidently Workspace`)**:
   - Lists all projects on the Evidently remote workspace (`http://192.168.0.15:8000`).
   - Groups multi-version datasets (such as `casmi-rl77-weights`) into expandable version groups while exposing one-click actions on the latest version.
   - Browses all project reports/snapshots with tags, timestamps, metadata, and linked datasets (and gracefully handles legacy projects where the server returns HTTP 500 on `/reports`).
   - Instant tree filtering (`Evidently: Filter Projects, Datasets & Reports`).

2. **Interactive Dataset Browser**:
   - Server-side paginated browsing and column-click sorting (`/api/datasets/{id}`).
   - Full `DataDefinition` / column type inspection (`numerical`, `string`, `datetime`), including automatic fallback for dynamic OpenTelemetry `tracing` datasets.
   - One-click **Use as Left Table in Join**, **Use as Right Table in Join**, **Create Report**, and **Export CSV**.

3. **Interactive Report Viewer**:
   - Searchable **Metrics Table** (`/download?report_format=json`), **Widgets Summary** (`/data`), and embedded **Interactive Evidently HTML Report** (`/download?report_format=html`).
   - Clickable **Linked Datasets** badges that jump directly to the input/output/joined datasets in the Dataset Browser.

4. **Table Join & Report Builder Studio (`Evidently: Join Tables & Create Report`)**:
   - Select any two tables across the same project or different projects on the remote workspace.
   - Automatic join key suggestion (`exact_id`, `exact`, `case_insensitive`, `semantic_id`) and multi-column join key support (`Left Column = Right Column`).
   - Supports `inner`, `left`, `right`, `outer` (full outer), `union` (row concatenation), and `cross` joins with normalized or exact key matching.
   - Live **Join Preview** showing matched/unmatched row diagnostics, key overlap percentage, column profiles (missing %, unique count, min/max/mean/std), and preview rows with CSV export.
   - Generates and publishes **Evidently 0.7.x** reports (`DataSummaryPreset`, `DataDriftPreset`, `DatasetCorrelations`, `RegressionPreset`, `ClassificationPreset`, `ValueStats`) directly to the remote workspace and optionally saves the joined dataset linked to the new report snapshot via `SnapshotLink`.
