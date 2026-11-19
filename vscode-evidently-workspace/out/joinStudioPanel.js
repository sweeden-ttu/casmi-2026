"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.JoinStudioPanel = void 0;
const fs = __importStar(require("fs"));
const vscode = __importStar(require("vscode"));
const bridgeRunner_1 = require("./bridgeRunner");
const config_1 = require("./config");
const joinEngine_1 = require("./joinEngine");
function escapeHtml(text) {
    return String(text ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}
class JoinStudioPanel {
    static instance;
    panel;
    client;
    extensionPath;
    callbacks;
    projects = [];
    datasetsByProject = new Map();
    schemasByDataset = new Map();
    leftProjectId = "";
    leftDatasetId = "";
    rightProjectId = "";
    rightDatasetId = "";
    targetProjectId = "";
    lastPreviewResult = null;
    disposables = [];
    static async show(client, extensionPath, callbacks, initialSelection) {
        if (JoinStudioPanel.instance) {
            JoinStudioPanel.instance.client = client;
            JoinStudioPanel.instance.callbacks = callbacks;
            JoinStudioPanel.instance.panel.reveal(vscode.ViewColumn.One);
            if (initialSelection) {
                await JoinStudioPanel.instance.applySelection(initialSelection.side, initialSelection.project, initialSelection.dataset);
            }
            return JoinStudioPanel.instance;
        }
        const panel = vscode.window.createWebviewPanel("evidentlyJoinStudio", "Evidently: Table Join & Report Studio", vscode.ViewColumn.One, {
            enableScripts: true,
            retainContextWhenHidden: true,
        });
        const studio = new JoinStudioPanel(panel, client, extensionPath, callbacks);
        JoinStudioPanel.instance = studio;
        await studio.initialize(initialSelection);
        return studio;
    }
    constructor(panel, client, extensionPath, callbacks) {
        this.panel = panel;
        this.client = client;
        this.extensionPath = extensionPath;
        this.callbacks = callbacks;
        this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
        this.panel.webview.onDidReceiveMessage(async (msg) => {
            await this.handleMessage(msg);
        }, null, this.disposables);
    }
    dispose() {
        JoinStudioPanel.instance = undefined;
        this.panel.dispose();
        while (this.disposables.length) {
            const d = this.disposables.pop();
            if (d) {
                d.dispose();
            }
        }
    }
    async ensureProjectDatasets(projectId) {
        if (!projectId) {
            return [];
        }
        if (!this.datasetsByProject.has(projectId)) {
            const list = await this.client.listDatasets(projectId);
            this.datasetsByProject.set(projectId, list);
        }
        return this.datasetsByProject.get(projectId) || [];
    }
    async ensureDatasetSchema(datasetId) {
        if (!datasetId) {
            return null;
        }
        if (!this.schemasByDataset.has(datasetId)) {
            const schema = await this.client.getDatasetSchema(datasetId);
            this.schemasByDataset.set(datasetId, schema);
        }
        return this.schemasByDataset.get(datasetId) || null;
    }
    async initialize(initialSelection) {
        this.panel.webview.html = this.buildShellHtml();
        try {
            this.projects = await this.client.listProjects();
            const defaultProj = initialSelection?.project ||
                this.projects.find((p) => p.name.includes("casmi")) ||
                this.projects[0];
            if (defaultProj) {
                this.leftProjectId = defaultProj.id;
                this.rightProjectId = defaultProj.id;
                this.targetProjectId = defaultProj.id;
                const datasets = await this.ensureProjectDatasets(defaultProj.id);
                if (datasets.length > 0) {
                    this.leftDatasetId = datasets[0].id;
                    // Pick a distinct dataset for right table if available
                    const second = datasets.find((d) => d.name !== datasets[0].name) ||
                        datasets[1] ||
                        datasets[0];
                    this.rightDatasetId = second.id;
                }
            }
            if (initialSelection) {
                if (initialSelection.side === "left") {
                    this.leftProjectId = initialSelection.project.id;
                    this.leftDatasetId = initialSelection.dataset.id;
                    this.targetProjectId = initialSelection.project.id;
                }
                else {
                    this.rightProjectId = initialSelection.project.id;
                    this.rightDatasetId = initialSelection.dataset.id;
                }
            }
            await this.pushStateToWebview();
        }
        catch (err) {
            await this.panel.webview.postMessage({
                type: "error",
                message: `Failed to initialize Join Studio: ${err?.message || err}`,
            });
        }
    }
    async applySelection(side, project, dataset) {
        await this.ensureProjectDatasets(project.id);
        if (side === "left") {
            this.leftProjectId = project.id;
            this.leftDatasetId = dataset.id;
            this.targetProjectId = project.id;
        }
        else {
            this.rightProjectId = project.id;
            this.rightDatasetId = dataset.id;
        }
        await this.pushStateToWebview();
    }
    async pushStateToWebview() {
        const [leftDatasets, rightDatasets, leftSchema, rightSchema] = await Promise.all([
            this.ensureProjectDatasets(this.leftProjectId),
            this.ensureProjectDatasets(this.rightProjectId),
            this.ensureDatasetSchema(this.leftDatasetId),
            this.rightDatasetId
                ? this.ensureDatasetSchema(this.rightDatasetId)
                : Promise.resolve(null),
        ]);
        const suggestions = leftSchema && rightSchema
            ? (0, joinEngine_1.suggestJoinKeys)(leftSchema.columnNames, rightSchema.columnNames)
            : [];
        await this.panel.webview.postMessage({
            type: "stateUpdate",
            projects: this.projects,
            leftProjectId: this.leftProjectId,
            leftDatasetId: this.leftDatasetId,
            leftDatasets,
            leftSchema,
            rightProjectId: this.rightProjectId,
            rightDatasetId: this.rightDatasetId,
            rightDatasets,
            rightSchema,
            targetProjectId: this.targetProjectId || this.leftProjectId,
            suggestions,
        });
    }
    async handleMessage(msg) {
        switch (msg?.type) {
            case "ready": {
                await this.pushStateToWebview();
                break;
            }
            case "changeLeftProject": {
                this.leftProjectId = String(msg.projectId || "");
                const ds = await this.ensureProjectDatasets(this.leftProjectId);
                this.leftDatasetId = ds[0]?.id || "";
                if (!this.targetProjectId) {
                    this.targetProjectId = this.leftProjectId;
                }
                await this.pushStateToWebview();
                break;
            }
            case "changeLeftDataset": {
                this.leftDatasetId = String(msg.datasetId || "");
                await this.pushStateToWebview();
                break;
            }
            case "changeRightProject": {
                this.rightProjectId = String(msg.projectId || "");
                const ds = await this.ensureProjectDatasets(this.rightProjectId);
                this.rightDatasetId = ds[0]?.id || "";
                await this.pushStateToWebview();
                break;
            }
            case "changeRightDataset": {
                this.rightDatasetId = String(msg.datasetId || "");
                await this.pushStateToWebview();
                break;
            }
            case "previewJoin": {
                await this.handlePreviewJoin(msg.payload);
                break;
            }
            case "exportJoinedCsv": {
                await this.handleExportJoinedCsv();
                break;
            }
            case "createReport": {
                await this.handleCreateReport(msg.payload);
                break;
            }
            case "openPublishedReport": {
                if (msg.projectId && msg.snapshotId) {
                    this.callbacks.onReportPublished(String(msg.projectId), String(msg.snapshotId), msg.joinedDatasetId ? String(msg.joinedDatasetId) : null, msg.htmlPath ? String(msg.htmlPath) : undefined);
                }
                break;
            }
            case "openDataset": {
                if (msg.projectId && msg.datasetId) {
                    this.callbacks.onOpenDatasetById(String(msg.projectId), String(msg.datasetId));
                }
                break;
            }
            case "openExternalUrl": {
                if (msg.url) {
                    await vscode.env.openExternal(vscode.Uri.parse(String(msg.url)));
                }
                break;
            }
        }
    }
    async handlePreviewJoin(payload) {
        try {
            const leftId = String(payload?.leftDatasetId || this.leftDatasetId);
            const rightId = String(payload?.rightDatasetId ?? this.rightDatasetId);
            if (!leftId) {
                throw new Error("Please select a Left Table.");
            }
            const joinType = (payload?.joinType || "inner");
            const matchMode = (payload?.matchMode || "normalized");
            const leftOn = Array.isArray(payload?.leftOn) ? payload.leftOn : [];
            const rightOn = Array.isArray(payload?.rightOn) ? payload.rightOn : [];
            const suffixes = [
                String(payload?.leftSuffix ?? "_left"),
                String(payload?.rightSuffix ?? "_right"),
            ];
            const selectedLeftColumns = Array.isArray(payload?.selectedLeftColumns) && payload.selectedLeftColumns.length > 0
                ? payload.selectedLeftColumns
                : undefined;
            const selectedRightColumns = Array.isArray(payload?.selectedRightColumns) && payload.selectedRightColumns.length > 0
                ? payload.selectedRightColumns
                : undefined;
            const leftCsv = await this.client.downloadDatasetCsv(leftId);
            const leftTable = (0, joinEngine_1.parseCsv)(leftCsv);
            let result;
            if (!rightId) {
                // Single table preview
                result = (0, joinEngine_1.joinTables)(leftTable, { columns: [], rows: [] }, {
                    joinType: "union",
                    leftOn: [],
                    rightOn: [],
                    selectedLeftColumns,
                    maxOutputRows: 50000,
                });
            }
            else {
                const rightCsv = await this.client.downloadDatasetCsv(rightId);
                const rightTable = (0, joinEngine_1.parseCsv)(rightCsv);
                result = (0, joinEngine_1.joinTables)(leftTable, rightTable, {
                    joinType,
                    leftOn,
                    rightOn,
                    matchMode,
                    suffixes,
                    selectedLeftColumns,
                    selectedRightColumns,
                    dropNullKeys: true,
                    maxOutputRows: 50000,
                });
            }
            this.lastPreviewResult = result;
            await this.panel.webview.postMessage({
                type: "joinPreviewResult",
                columns: result.columns,
                diagnostics: result.diagnostics,
                columnProfiles: result.columnProfiles,
                rows: result.rows.slice(0, 150),
            });
        }
        catch (err) {
            await this.panel.webview.postMessage({
                type: "error",
                message: err?.message || String(err),
            });
        }
    }
    async handleExportJoinedCsv() {
        if (!this.lastPreviewResult) {
            vscode.window.showWarningMessage("Run 'Preview Joined Table' first before exporting CSV.");
            return;
        }
        try {
            const targetUri = await vscode.window.showSaveDialog({
                defaultUri: vscode.Uri.file("joined_dataset.csv"),
                filters: { "CSV Files": ["csv"] },
                saveLabel: "Export Joined Table CSV",
            });
            if (!targetUri) {
                return;
            }
            const csv = (0, joinEngine_1.serializeCsv)(this.lastPreviewResult.columns, this.lastPreviewResult.rows);
            fs.writeFileSync(targetUri.fsPath, csv, "utf8");
            vscode.window.showInformationMessage(`Exported joined table (${this.lastPreviewResult.rows.length} rows) to ${targetUri.fsPath}`);
        }
        catch (err) {
            vscode.window.showErrorMessage(`Failed to export joined CSV: ${err?.message || err}`);
        }
    }
    async handleCreateReport(payload) {
        try {
            const cfg = (0, config_1.getWorkspaceConfig)();
            const spec = {
                mode: "create_report",
                server_url: this.client.getBaseUrl(),
                api_key: cfg.apiKey,
                left_dataset_id: String(payload?.leftDatasetId || this.leftDatasetId),
                right_dataset_id: payload?.rightDatasetId
                    ? String(payload.rightDatasetId)
                    : undefined,
                join_type: (payload?.joinType || "inner"),
                left_on: Array.isArray(payload?.leftOn) ? payload.leftOn : [],
                right_on: Array.isArray(payload?.rightOn) ? payload.rightOn : [],
                match_mode: (payload?.matchMode || "normalized"),
                suffixes: [
                    String(payload?.leftSuffix ?? "_left"),
                    String(payload?.rightSuffix ?? "_right"),
                ],
                selected_left_columns: Array.isArray(payload?.selectedLeftColumns) &&
                    payload.selectedLeftColumns.length > 0
                    ? payload.selectedLeftColumns
                    : undefined,
                selected_right_columns: Array.isArray(payload?.selectedRightColumns) &&
                    payload.selectedRightColumns.length > 0
                    ? payload.selectedRightColumns
                    : undefined,
                target_project_id: String(payload?.targetProjectId || this.targetProjectId || this.leftProjectId),
                report_name: String(payload?.reportName || "").trim(),
                tags: String(payload?.tags || "joined-report")
                    .split(",")
                    .map((t) => t.trim())
                    .filter(Boolean),
                presets: Array.isArray(payload?.presets)
                    ? payload.presets
                    : ["DataSummaryPreset"],
                target_column: String(payload?.targetColumn || "").trim(),
                prediction_column: String(payload?.predictionColumn || "").trim(),
                metric_columns: Array.isArray(payload?.metricColumns)
                    ? payload.metricColumns
                    : [],
                include_correlations: Boolean(payload?.includeCorrelations ?? true),
                include_tests: Boolean(payload?.includeTests ?? false),
                reference_source: payload?.referenceSource || "",
                save_joined_dataset: Boolean(payload?.saveJoinedDataset ?? true),
                joined_dataset_name: String(payload?.joinedDatasetName || "").trim(),
                publish_to_workspace: Boolean(payload?.publishToWorkspace ?? true),
            };
            const res = await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Evidently: Joining tables & generating report...",
                cancellable: false,
            }, async () => {
                return (0, bridgeRunner_1.runEvidentlyBridge)(cfg.pythonPath, spec, this.extensionPath, 120000);
            });
            let htmlContent = "";
            if (res.htmlPath && fs.existsSync(res.htmlPath)) {
                htmlContent = fs.readFileSync(res.htmlPath, "utf8");
            }
            await this.panel.webview.postMessage({
                type: "reportCreated",
                result: res,
                htmlContent,
            });
            this.datasetsByProject.delete(spec.target_project_id || "");
            this.callbacks.onReportPublished(spec.target_project_id || this.leftProjectId, null, // Do not auto-pop another window on top of Join Studio unless user clicks "Open in Report Viewer"
            res.joinedDatasetId, res.htmlPath);
            vscode.window.showInformationMessage(`Evidently Report "${res.reportName}" generated${res.snapshotId ? ` & published (Snapshot ${res.snapshotId.slice(0, 8)})` : ""}!`);
        }
        catch (err) {
            await this.panel.webview.postMessage({
                type: "error",
                message: err?.message || String(err),
            });
            vscode.window.showErrorMessage(`Failed to create Evidently report: ${err?.message || err}`);
        }
    }
    buildShellHtml() {
        return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>
  :root {
    --border: var(--vscode-panel-border, rgba(128,128,128,0.28));
    --card-bg: var(--vscode-editorWidget-background, rgba(128,128,128,0.08));
    --accent: var(--vscode-button-background, #0e639c);
    --accent-fg: var(--vscode-button-foreground, #ffffff);
  }
  body {
    font-family: var(--vscode-font-family, system-ui, sans-serif);
    font-size: 13px;
    color: var(--vscode-foreground);
    background: var(--vscode-editor-background);
    margin: 0;
    padding: 16px 20px;
  }
  h2 { margin: 0 0 4px 0; font-size: 18px; }
  h3 { margin: 0 0 10px 0; font-size: 14px; }
  .subtitle { font-size: 12px; opacity: 0.78; margin-bottom: 16px; }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 14px;
  }
  @media (max-width: 860px) {
    .grid-2 { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    margin-bottom: 14px;
  }
  .field-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 10px;
  }
  .field-row label {
    font-size: 11.5px;
    font-weight: 600;
    opacity: 0.85;
  }
  select, input[type="text"] {
    background: var(--vscode-input-background);
    color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, var(--border));
    padding: 6px 8px;
    border-radius: 4px;
    font-size: 12px;
  }
  button {
    background: var(--accent);
    color: var(--accent-fg);
    border: 1px solid transparent;
    padding: 6px 14px;
    border-radius: 4px;
    font-size: 12.5px;
    cursor: pointer;
    font-weight: 500;
  }
  button:hover { opacity: 0.92; }
  button.secondary {
    background: var(--vscode-button-secondaryBackground, rgba(128,128,128,0.2));
    color: var(--vscode-button-secondaryForeground, var(--vscode-foreground));
    border-color: var(--border);
  }
  button.sm {
    padding: 3px 8px;
    font-size: 11.5px;
  }
  .schema-list {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    max-height: 110px;
    overflow-y: auto;
    padding: 6px;
    background: var(--vscode-editor-background);
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  .col-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 7px;
    border-radius: 3px;
    border: 1px solid var(--border);
    font-size: 11px;
    background: var(--card-bg);
  }
  .key-pair-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }
  .key-pair-row select {
    flex: 1;
  }
  .suggestion-chip {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 12px;
    background: rgba(56, 139, 253, 0.18);
    border: 1px solid rgba(56, 139, 253, 0.4);
    color: var(--vscode-foreground);
    font-size: 11.5px;
    cursor: pointer;
    margin-right: 6px;
    margin-bottom: 5px;
  }
  .suggestion-chip:hover {
    background: rgba(56, 139, 253, 0.32);
  }
  .stats-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(125px, 1fr));
    gap: 10px;
    margin: 12px 0;
  }
  .stat-card {
    background: var(--vscode-editor-background);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 8px 12px;
  }
  .stat-val {
    font-size: 17px;
    font-weight: 700;
  }
  .stat-lbl {
    font-size: 11px;
    opacity: 0.72;
  }
  .table-wrap {
    overflow-x: auto;
    max-height: 320px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 5px;
    margin-top: 8px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  th, td {
    padding: 6px 9px;
    border: 1px solid var(--border);
    text-align: left;
    white-space: nowrap;
    max-width: 320px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  th {
    position: sticky;
    top: 0;
    background: var(--vscode-editorGroupHeader-tabsBackground, #252526);
    z-index: 2;
  }
  .preset-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 8px;
    margin: 8px 0 12px 0;
  }
  .preset-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: 5px;
    background: var(--vscode-editor-background);
  }
  .banner-error {
    border: 1px solid #f14c4c;
    background: rgba(241, 76, 76, 0.12);
    padding: 10px 14px;
    border-radius: 5px;
    margin-bottom: 12px;
    display: none;
  }
  .banner-success {
    border: 1px solid #3fb950;
    background: rgba(63, 185, 80, 0.12);
    padding: 12px 14px;
    border-radius: 5px;
    margin-bottom: 12px;
    display: none;
  }
  #reportHtmlFrame {
    width: 100%;
    height: 680px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: #ffffff;
    margin-top: 10px;
  }
  .muted { opacity: 0.65; }
</style>
</head>
<body>
  <h2>🔀 Evidently Table Join & Report Builder Studio</h2>
  <div class="subtitle">
    Select tables from any project on the local Evidently remote workspace, join them on one or more keys, preview the joined dataset, and publish an Evidently evaluation report.
  </div>

  <div id="errorBanner" class="banner-error"></div>
  <div id="successBanner" class="banner-success"></div>

  <!-- STEP 1: SELECT LEFT & RIGHT TABLES -->
  <div class="grid-2">
    <div class="card">
      <h3>1A. Left Table (Primary Dataset)</h3>
      <div class="field-row">
        <label>Project</label>
        <select id="leftProjectSelect" onchange="onLeftProjectChange()"></select>
      </div>
      <div class="field-row">
        <label>Dataset</label>
        <select id="leftDatasetSelect" onchange="onLeftDatasetChange()"></select>
      </div>
      <div class="field-row">
        <label id="leftSchemaLabel">Columns (0)</label>
        <div id="leftSchemaBox" class="schema-list"></div>
      </div>
    </div>

    <div class="card">
      <h3>1B. Right Table (Table to Join)</h3>
      <div class="field-row">
        <label>Project</label>
        <select id="rightProjectSelect" onchange="onRightProjectChange()"></select>
      </div>
      <div class="field-row">
        <label>Dataset (or select None for single-table report)</label>
        <select id="rightDatasetSelect" onchange="onRightDatasetChange()"></select>
      </div>
      <div class="field-row">
        <label id="rightSchemaLabel">Columns (0)</label>
        <div id="rightSchemaBox" class="schema-list"></div>
      </div>
    </div>
  </div>

  <!-- STEP 2: JOIN CONFIGURATION -->
  <div class="card">
    <h3>2. Join Keys & Relational Options</h3>
    <div style="display:flex;flex-wrap:wrap;gap:14px;margin-bottom:10px;">
      <div class="field-row" style="min-width:170px;">
        <label>Join Type</label>
        <select id="joinTypeSelect">
          <option value="inner">Inner Join (matching rows only)</option>
          <option value="left">Left Join (all left rows + matching right)</option>
          <option value="right">Right Join (all right rows + matching left)</option>
          <option value="outer">Full Outer Join (all rows from both)</option>
          <option value="union">Union / Concatenate Rows</option>
          <option value="cross">Cross Join (Cartesian product)</option>
        </select>
      </div>
      <div class="field-row" style="min-width:210px;">
        <label>Key Matching Mode</label>
        <select id="matchModeSelect">
          <option value="normalized">Normalized (trim, case-insensitive, numeric 40 == 40.0)</option>
          <option value="exact">Exact string/value match</option>
        </select>
      </div>
      <div class="field-row" style="width:120px;">
        <label>Left Collision Suffix</label>
        <input type="text" id="leftSuffixInput" value="_left" />
      </div>
      <div class="field-row" style="width:120px;">
        <label>Right Collision Suffix</label>
        <input type="text" id="rightSuffixInput" value="_right" />
      </div>
    </div>

    <div style="margin-bottom:8px;">
      <label style="font-size:11.5px;font-weight:600;opacity:0.85;">Suggested Join Keys (click to use):</label>
      <div id="suggestionsBox" style="margin-top:5px;"></div>
    </div>

    <div style="margin-bottom:10px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <label style="font-size:11.5px;font-weight:600;opacity:0.85;">Active Join Key Pairs (Left Column = Right Column)</label>
        <button class="secondary sm" onclick="addKeyPairRow()">+ Add Key Pair</button>
      </div>
      <div id="keyPairsContainer"></div>
    </div>

    <div style="display:flex;gap:10px;align-items:center;">
      <button onclick="runPreviewJoin()">🔍 Preview Joined Table</button>
      <button class="secondary" id="exportCsvBtn" onclick="exportJoinedCsv()" style="display:none;">⬇ Export Joined CSV</button>
      <span id="previewStatus" class="muted" style="font-size:12px;"></span>
    </div>
  </div>

  <!-- STEP 2B: JOIN PREVIEW RESULTS -->
  <div id="previewCard" class="card" style="display:none;">
    <h3>Joined Table Diagnostics & Data Preview</h3>
    <div id="diagnosticsCards" class="stats-cards"></div>

    <details style="margin-top:10px;">
      <summary style="cursor:pointer;font-weight:600;font-size:12.5px;">Column Profiles & Statistics (<span id="profileColCount">0</span> columns)</summary>
      <div class="table-wrap">
        <table id="profilesTable">
          <thead>
            <tr>
              <th>Column</th>
              <th>Source</th>
              <th>Type</th>
              <th>Non-Null</th>
              <th>Missing %</th>
              <th>Unique</th>
              <th>Min</th>
              <th>Max</th>
              <th>Mean</th>
              <th>Std</th>
              <th>Sample Values</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </details>

    <div style="margin-top:12px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <strong style="font-size:12.5px;">Joined Rows Preview</strong>
        <input type="text" id="previewFilterInput" placeholder="Filter preview rows..." oninput="filterPreviewTable()" style="width:240px;" />
      </div>
      <div class="table-wrap">
        <table id="previewDataTable">
          <thead></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- STEP 3: CREATE & PUBLISH EVIDENTLY REPORT -->
  <div class="card">
    <h3>3. Create & Publish Evidently Report from Joined Table</h3>
    <div class="grid-2">
      <div>
        <div class="field-row">
          <label>Target Project (where report & joined dataset will be saved)</label>
          <select id="targetProjectSelect"></select>
        </div>
        <div class="field-row">
          <label>Report Name</label>
          <input type="text" id="reportNameInput" placeholder="e.g. rl77-weights-vs-baseline-join" />
        </div>
        <div class="field-row">
          <label>Report Tags (comma-separated)</label>
          <input type="text" id="reportTagsInput" value="joined-report, vscode" />
        </div>
      </div>
      <div>
        <div class="field-row">
          <label>Reference Dataset for Drift Comparison (when DataDriftPreset is enabled)</label>
          <select id="referenceSourceSelect">
            <option value="left">Compare against Left Table (shared columns)</option>
            <option value="right">Compare against Right Table (shared columns)</option>
            <option value="split_half">Compare 1st half vs 2nd half of Joined Table</option>
          </select>
        </div>
        <div style="display:flex;gap:10px;">
          <div class="field-row" style="flex:1;">
            <label>Target Column (for Regression/Classification)</label>
            <select id="targetColSelect"><option value="">(None)</option></select>
          </div>
          <div class="field-row" style="flex:1;">
            <label>Prediction Column (for Regression/Classification)</label>
            <select id="predictionColSelect"><option value="">(None)</option></select>
          </div>
        </div>
        <div class="field-row">
          <label>Joined Dataset Name (if saving joined table to workspace)</label>
          <input type="text" id="joinedDatasetNameInput" placeholder="Auto-named from Report Name if left blank" />
        </div>
      </div>
    </div>

    <label style="font-size:11.5px;font-weight:600;opacity:0.85;">Evidently Presets & Metrics to Include:</label>
    <div class="preset-grid">
      <label class="preset-item">
        <input type="checkbox" class="preset-cb" value="DataSummaryPreset" checked />
        <div>
          <strong>DataSummaryPreset</strong>
          <div class="muted" style="font-size:11px;">Column stats, missing values, distributions for all joined columns</div>
        </div>
      </label>
      <label class="preset-item">
        <input type="checkbox" class="preset-cb" value="DataDriftPreset" />
        <div>
          <strong>DataDriftPreset</strong>
          <div class="muted" style="font-size:11px;">Distribution shift & drift analysis against Reference dataset</div>
        </div>
      </label>
      <label class="preset-item">
        <input type="checkbox" id="correlationsCb" checked />
        <div>
          <strong>DatasetCorrelations</strong>
          <div class="muted" style="font-size:11px;">Correlation matrix across numerical columns</div>
        </div>
      </label>
      <label class="preset-item">
        <input type="checkbox" class="preset-cb" value="RegressionPreset" />
        <div>
          <strong>RegressionPreset</strong>
          <div class="muted" style="font-size:11px;">MAE, RMSE, R², error distribution (select Target & Prediction)</div>
        </div>
      </label>
      <label class="preset-item">
        <input type="checkbox" class="preset-cb" value="ClassificationPreset" />
        <div>
          <strong>ClassificationPreset</strong>
          <div class="muted" style="font-size:11px;">Accuracy, Precision, Recall, F1 (select Target & Prediction)</div>
        </div>
      </label>
      <label class="preset-item">
        <input type="checkbox" id="includeTestsCb" />
        <div>
          <strong>Include Automatic Tests</strong>
          <div class="muted" style="font-size:11px;">Run Evidently auto-generated test conditions on metrics</div>
        </div>
      </label>
    </div>

    <div style="display:flex;flex-wrap:wrap;gap:18px;margin:10px 0 14px 0;">
      <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
        <input type="checkbox" id="publishWorkspaceCb" checked />
        <strong>Publish Report Snapshot to Evidently Remote Workspace</strong>
      </label>
      <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
        <input type="checkbox" id="saveJoinedDatasetCb" checked />
        <strong>Save Joined Table as an Evidently Dataset (linked to report)</strong>
      </label>
    </div>

    <div style="display:flex;gap:10px;align-items:center;">
      <button onclick="runCreateReport()">🚀 Generate & Publish Evidently Report</button>
      <span id="createReportStatus" class="muted" style="font-size:12px;"></span>
    </div>
  </div>

  <!-- STEP 4: GENERATED REPORT VIEW -->
  <div id="generatedReportCard" class="card" style="display:none;">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <h3 id="generatedReportTitle">Generated Evidently Report</h3>
      <div id="generatedReportActions" style="display:flex;gap:8px;"></div>
    </div>
    <iframe id="reportHtmlFrame" sandbox="allow-scripts allow-same-origin"></iframe>
  </div>

  <script>
    const vscode = acquireVsCodeApi();
    let appState = {
      projects: [],
      leftProjectId: '',
      leftDatasetId: '',
      leftDatasets: [],
      leftSchema: null,
      rightProjectId: '',
      rightDatasetId: '',
      rightDatasets: [],
      rightSchema: null,
      targetProjectId: '',
      suggestions: [],
    };
    let activeKeyPairs = [];
    let previewColumns = [];

    function esc(text) {
      return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function send(type, extra = {}) {
      vscode.postMessage({ type, ...extra });
    }

    function showError(msg) {
      const b = document.getElementById('errorBanner');
      if (!msg) {
        b.style.display = 'none';
        b.textContent = '';
      } else {
        b.style.display = 'block';
        b.textContent = '⚠️ ' + msg;
      }
    }

    function onLeftProjectChange() {
      const pid = document.getElementById('leftProjectSelect').value;
      send('changeLeftProject', { projectId: pid });
    }

    function onLeftDatasetChange() {
      const did = document.getElementById('leftDatasetSelect').value;
      send('changeLeftDataset', { datasetId: did });
    }

    function onRightProjectChange() {
      const pid = document.getElementById('rightProjectSelect').value;
      send('changeRightProject', { projectId: pid });
    }

    function onRightDatasetChange() {
      const did = document.getElementById('rightDatasetSelect').value;
      send('changeRightDataset', { datasetId: did });
    }

    function renderSelectOptions(selectEl, items, selectedValue, includeNone = false) {
      selectEl.innerHTML = '';
      if (includeNone) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = '(None — single-dataset report)';
        if (!selectedValue) opt.selected = true;
        selectEl.appendChild(opt);
      }
      for (const item of items) {
        const opt = document.createElement('option');
        opt.value = item.value;
        opt.textContent = item.label;
        if (item.value === selectedValue) {
          opt.selected = true;
        }
        selectEl.appendChild(opt);
      }
    }

    function renderSchemaBox(boxEl, labelEl, schema) {
      boxEl.innerHTML = '';
      const cols = schema && Array.isArray(schema.columns) ? schema.columns : [];
      labelEl.textContent = 'Columns (' + cols.length + ')' + (schema && schema.rowCount ? ' · ' + schema.rowCount + ' rows' : '');
      for (const c of cols) {
        const span = document.createElement('span');
        span.className = 'col-chip';
        span.textContent = c.name + ' (' + c.type + ')';
        boxEl.appendChild(span);
      }
    }

    function renderSuggestions() {
      const box = document.getElementById('suggestionsBox');
      box.innerHTML = '';
      const sugs = appState.suggestions || [];
      if (sugs.length === 0) {
        const sp = document.createElement('span');
        sp.className = 'muted';
        sp.textContent = 'No identically named columns detected — choose any Left and Right columns below.';
        box.appendChild(sp);
        return;
      }
      for (const s of sugs.slice(0, 10)) {
        const chip = document.createElement('span');
        chip.className = 'suggestion-chip';
        chip.textContent = s.leftColumn + ' = ' + s.rightColumn;
        chip.title = 'Click to add or select this join key pair';
        chip.onclick = () => {
          const exists = activeKeyPairs.some(
            (p) => p.left === s.leftColumn && p.right === s.rightColumn
          );
          if (!exists) {
            if (activeKeyPairs.length === 1 && !activeKeyPairs[0].left && !activeKeyPairs[0].right) {
              activeKeyPairs[0] = { left: s.leftColumn, right: s.rightColumn };
            } else {
              activeKeyPairs.push({ left: s.leftColumn, right: s.rightColumn });
            }
            renderKeyPairs();
          }
        };
        box.appendChild(chip);
      }
    }

    function renderKeyPairs() {
      const container = document.getElementById('keyPairsContainer');
      container.innerHTML = '';
      const leftCols = (appState.leftSchema && appState.leftSchema.columnNames) || [];
      const rightCols = (appState.rightSchema && appState.rightSchema.columnNames) || [];

      if (activeKeyPairs.length === 0) {
        if (appState.suggestions && appState.suggestions.length > 0) {
          activeKeyPairs.push({
            left: appState.suggestions[0].leftColumn,
            right: appState.suggestions[0].rightColumn,
          });
        } else {
          activeKeyPairs.push({
            left: leftCols[0] || '',
            right: rightCols[0] || '',
          });
        }
      }

      activeKeyPairs.forEach((pair, idx) => {
        const row = document.createElement('div');
        row.className = 'key-pair-row';

        const lSel = document.createElement('select');
        renderSelectOptions(
          lSel,
          leftCols.map((c) => ({ value: c, label: 'Left: ' + c })),
          pair.left
        );
        lSel.onchange = () => {
          activeKeyPairs[idx].left = lSel.value;
        };

        const eqSpan = document.createElement('span');
        eqSpan.textContent = '=';
        eqSpan.style.fontWeight = 'bold';

        const rSel = document.createElement('select');
        renderSelectOptions(
          rSel,
          rightCols.map((c) => ({ value: c, label: 'Right: ' + c })),
          pair.right
        );
        rSel.onchange = () => {
          activeKeyPairs[idx].right = rSel.value;
        };

        const delBtn = document.createElement('button');
        delBtn.className = 'secondary sm';
        delBtn.textContent = '✕';
        delBtn.title = 'Remove key pair';
        delBtn.onclick = () => {
          activeKeyPairs.splice(idx, 1);
          renderKeyPairs();
        };

        row.appendChild(lSel);
        row.appendChild(eqSpan);
        row.appendChild(rSel);
        if (activeKeyPairs.length > 1) {
          row.appendChild(delBtn);
        }
        container.appendChild(row);
      });
    }

    function addKeyPairRow() {
      const leftCols = (appState.leftSchema && appState.leftSchema.columnNames) || [];
      const rightCols = (appState.rightSchema && appState.rightSchema.columnNames) || [];
      activeKeyPairs.push({
        left: leftCols[0] || '',
        right: rightCols[0] || '',
      });
      renderKeyPairs();
    }

    function updateCandidateColumnsForReport(cols) {
      previewColumns = cols || [];
      const tSel = document.getElementById('targetColSelect');
      const pSel = document.getElementById('predictionColSelect');
      const curT = tSel.value;
      const curP = pSel.value;
      const items = previewColumns.map((c) => ({ value: c, label: c }));
      renderSelectOptions(tSel, items, curT, true);
      renderSelectOptions(pSel, items, curP, true);
    }

    function updateDefaultReportName() {
      const nameInput = document.getElementById('reportNameInput');
      if (nameInput.value && !nameInput.dataset.autoGenerated) {
        return;
      }
      const lDs = (appState.leftDatasets || []).find((d) => d.id === appState.leftDatasetId);
      const rDs = (appState.rightDatasets || []).find((d) => d.id === appState.rightDatasetId);
      const jt = document.getElementById('joinTypeSelect').value || 'inner';
      const lName = lDs ? lDs.name : 'left';
      const rName = rDs ? rDs.name : '';
      nameInput.value = rName ? (lName + ' [' + jt + ' join] ' + rName) : (lName + ' report');
      nameInput.dataset.autoGenerated = 'true';
    }

    document.getElementById('reportNameInput').addEventListener('input', (e) => {
      e.target.dataset.autoGenerated = '';
    });

    function collectJoinPayload() {
      const leftOn = activeKeyPairs.map((p) => p.left).filter(Boolean);
      const rightOn = activeKeyPairs.map((p) => p.right).filter(Boolean);
      const presets = Array.from(document.querySelectorAll('.preset-cb:checked')).map((el) => el.value);
      return {
        leftProjectId: document.getElementById('leftProjectSelect').value,
        leftDatasetId: document.getElementById('leftDatasetSelect').value,
        rightProjectId: document.getElementById('rightProjectSelect').value,
        rightDatasetId: document.getElementById('rightDatasetSelect').value,
        joinType: document.getElementById('joinTypeSelect').value,
        matchMode: document.getElementById('matchModeSelect').value,
        leftSuffix: document.getElementById('leftSuffixInput').value || '_left',
        rightSuffix: document.getElementById('rightSuffixInput').value || '_right',
        leftOn,
        rightOn,
        targetProjectId: document.getElementById('targetProjectSelect').value,
        reportName: document.getElementById('reportNameInput').value,
        tags: document.getElementById('reportTagsInput').value,
        referenceSource: document.getElementById('referenceSourceSelect').value,
        targetColumn: document.getElementById('targetColSelect').value,
        predictionColumn: document.getElementById('predictionColSelect').value,
        joinedDatasetName: document.getElementById('joinedDatasetNameInput').value,
        presets,
        includeCorrelations: document.getElementById('correlationsCb').checked,
        includeTests: document.getElementById('includeTestsCb').checked,
        publishToWorkspace: document.getElementById('publishWorkspaceCb').checked,
        saveJoinedDataset: document.getElementById('saveJoinedDatasetCb').checked,
      };
    }

    function runPreviewJoin() {
      showError('');
      document.getElementById('previewStatus').textContent = 'Running join preview...';
      send('previewJoin', { payload: collectJoinPayload() });
    }

    function exportJoinedCsv() {
      send('exportJoinedCsv');
    }

    function runCreateReport() {
      showError('');
      document.getElementById('createReportStatus').textContent = 'Generating & publishing Evidently report...';
      send('createReport', { payload: collectJoinPayload() });
    }

    function renderJoinPreview(msg) {
      document.getElementById('previewStatus').textContent =
        'Preview ready: ' + msg.diagnostics.joinedRowCount + ' joined rows × ' + msg.columns.length + ' columns.';
      document.getElementById('exportCsvBtn').style.display = 'inline-block';
      document.getElementById('previewCard').style.display = 'block';

      const d = msg.diagnostics;
      const cardsBox = document.getElementById('diagnosticsCards');
      const cardItems = [
        { lbl: 'Left Rows', val: d.leftRowCount },
        { lbl: 'Right Rows', val: d.rightRowCount },
        { lbl: 'Joined Rows', val: d.joinedRowCount },
        { lbl: 'Matched Left', val: d.matchedLeftRows + ' (' + d.unmatchedLeftRows + ' unmatched)' },
        { lbl: 'Matched Right', val: d.matchedRightRows + ' (' + d.unmatchedRightRows + ' unmatched)' },
        { lbl: 'Key Overlap', val: d.keyOverlapPct + '% (' + d.overlappingKeys + ' keys)' },
        { lbl: 'Total Columns', val: msg.columns.length },
      ];
      cardsBox.innerHTML = cardItems
        .map(
          (c) =>
            '<div class="stat-card"><div class="stat-val">' +
            c.val +
            '</div><div class="stat-lbl">' +
            c.lbl +
            '</div></div>'
        )
        .join('');

      // Render column profiles
      const profiles = msg.columnProfiles || [];
      document.getElementById('profileColCount').textContent = String(profiles.length);
      const profTbody = document.querySelector('#profilesTable tbody');
      profTbody.innerHTML = profiles
        .map(
          (p) =>
            '<tr>' +
            '<td><strong>' + esc(p.name) + '</strong></td>' +
            '<td>' + esc(p.source) + '</td>' +
            '<td>' + esc(p.inferredType) + '</td>' +
            '<td>' + esc(p.nonNullCount) + '</td>' +
            '<td>' + esc(p.missingPct) + '%</td>' +
            '<td>' + esc(p.uniqueCount) + '</td>' +
            '<td>' + esc(p.min ?? '-') + '</td>' +
            '<td>' + esc(p.max ?? '-') + '</td>' +
            '<td>' + esc(p.mean ?? '-') + '</td>' +
            '<td>' + esc(p.std ?? '-') + '</td>' +
            '<td>' + (p.sampleValues || []).map((sv) => esc(sv)).join(', ') + '</td>' +
            '</tr>'
        )
        .join('');

      // Render preview rows table
      const thead = document.querySelector('#previewDataTable thead');
      const tbody = document.querySelector('#previewDataTable tbody');
      thead.innerHTML =
        '<tr><th>#</th>' +
        msg.columns.map((c) => '<th>' + esc(c) + '</th>').join('') +
        '</tr>';
      const rows = msg.rows || [];
      if (rows.length === 0) {
        tbody.innerHTML =
          '<tr><td colspan="' +
          (msg.columns.length + 1) +
          '" style="text-align:center;padding:18px;">0 rows matched the join criteria. Try switching Join Type to "Left Join" or "Full Outer Join", or check your join key columns.</td></tr>';
      } else {
        tbody.innerHTML = rows
          .map(
            (r, idx) =>
              '<tr class="preview-row"><td>' +
              (idx + 1) +
              '</td>' +
              msg.columns
                .map((c) => '<td>' + (r[c] === null || r[c] === undefined ? '<span class="muted">null</span>' : esc(r[c])) + '</td>')
                .join('') +
              '</tr>'
          )
          .join('');
      }

      updateCandidateColumnsForReport(msg.columns);
    }

    function filterPreviewTable() {
      const q = (document.getElementById('previewFilterInput').value || '').toLowerCase();
      for (const r of document.querySelectorAll('#previewDataTable tbody tr.preview-row')) {
        r.style.display = !q || r.textContent.toLowerCase().includes(q) ? '' : 'none';
      }
    }

    window.addEventListener('message', (event) => {
      const msg = event.data;
      if (msg.type === 'stateUpdate') {
        appState = msg;
        const projItems = (msg.projects || []).map((p) => ({
          value: p.id,
          label: p.name,
        }));
        renderSelectOptions(document.getElementById('leftProjectSelect'), projItems, msg.leftProjectId);
        renderSelectOptions(document.getElementById('rightProjectSelect'), projItems, msg.rightProjectId);
        renderSelectOptions(document.getElementById('targetProjectSelect'), projItems, msg.targetProjectId);

        const formatDsLabel = (d) =>
          d.name + ' (' + d.row_count + '×' + d.column_count + ' · ' + String(d.created_at || '').slice(0, 16) + ' · ' + d.id.slice(0, 8) + ')';

        renderSelectOptions(
          document.getElementById('leftDatasetSelect'),
          (msg.leftDatasets || []).map((d) => ({ value: d.id, label: formatDsLabel(d) })),
          msg.leftDatasetId
        );
        renderSelectOptions(
          document.getElementById('rightDatasetSelect'),
          (msg.rightDatasets || []).map((d) => ({ value: d.id, label: formatDsLabel(d) })),
          msg.rightDatasetId,
          true
        );

        renderSchemaBox(
          document.getElementById('leftSchemaBox'),
          document.getElementById('leftSchemaLabel'),
          msg.leftSchema
        );
        renderSchemaBox(
          document.getElementById('rightSchemaBox'),
          document.getElementById('rightSchemaLabel'),
          msg.rightSchema
        );

        // Reset activeKeyPairs to suggestions when datasets change
        activeKeyPairs = [];
        renderSuggestions();
        renderKeyPairs();
        updateDefaultReportName();

        const combinedCols = Array.from(
          new Set([
            ...((msg.leftSchema && msg.leftSchema.columnNames) || []),
            ...((msg.rightSchema && msg.rightSchema.columnNames) || []),
          ])
        );
        updateCandidateColumnsForReport(combinedCols);
      } else if (msg.type === 'joinPreviewResult') {
        renderJoinPreview(msg);
      } else if (msg.type === 'reportCreated') {
        document.getElementById('createReportStatus').textContent = 'Report generated successfully!';
        const res = msg.result;
        if (res.columns && res.diagnostics) {
          renderJoinPreview({
            columns: res.columns,
            diagnostics: res.diagnostics,
            columnProfiles: res.columnProfiles || [],
            rows: res.rows || [],
          });
        }
        const sb = document.getElementById('successBanner');
        sb.style.display = 'block';
        sb.innerHTML =
          '<strong>✅ Evidently Report Created:</strong> <code>' +
          esc(res.reportName) +
          '</code>' +
          (res.snapshotId ? ' · Snapshot ID: <code>' + esc(res.snapshotId) + '</code>' : '') +
          (res.joinedDatasetId ? ' · Joined Dataset ID: <code>' + esc(res.joinedDatasetId) + '</code>' : '');

        const card = document.getElementById('generatedReportCard');
        card.style.display = 'block';
        document.getElementById('generatedReportTitle').textContent =
          '📈 Generated Evidently Report: ' + res.reportName;

        const actBox = document.getElementById('generatedReportActions');
        actBox.innerHTML = '';
        if (res.projectId && res.snapshotId) {
          const openRepBtn = document.createElement('button');
          openRepBtn.textContent = 'Open in Report Viewer Panel';
          openRepBtn.onclick = () =>
            send('openPublishedReport', {
              projectId: res.projectId,
              snapshotId: res.snapshotId,
              joinedDatasetId: res.joinedDatasetId,
              htmlPath: res.htmlPath,
            });
          actBox.appendChild(openRepBtn);
        }
        if (res.projectId && res.joinedDatasetId) {
          const openDsBtn = document.createElement('button');
          openDsBtn.className = 'secondary';
          openDsBtn.textContent = '📊 Browse Saved Joined Dataset';
          openDsBtn.onclick = () =>
            send('openDataset', {
              projectId: res.projectId,
              datasetId: res.joinedDatasetId,
            });
          actBox.appendChild(openDsBtn);
        }
        if (res.reportUrl) {
          const extBtn = document.createElement('button');
          extBtn.className = 'secondary';
          extBtn.textContent = '↗ Open in Evidently Web UI';
          extBtn.onclick = () => send('openExternalUrl', { url: res.reportUrl });
          actBox.appendChild(extBtn);
        }

        if (msg.htmlContent) {
          document.getElementById('reportHtmlFrame').srcdoc = msg.htmlContent;
        }
      } else if (msg.type === 'error') {
        document.getElementById('previewStatus').textContent = '';
        document.getElementById('createReportStatus').textContent = '';
        showError(msg.message);
      }
    });

    send('ready');
  </script>
</body>
</html>`;
    }
}
exports.JoinStudioPanel = JoinStudioPanel;
//# sourceMappingURL=joinStudioPanel.js.map