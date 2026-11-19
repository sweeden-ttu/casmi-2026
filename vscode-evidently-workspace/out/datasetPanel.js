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
exports.DatasetBrowserPanel = void 0;
const fs = __importStar(require("fs"));
const vscode = __importStar(require("vscode"));
function escapeHtml(text) {
    return String(text ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}
class DatasetBrowserPanel {
    static panels = new Map();
    panel;
    client;
    project;
    dataset;
    callbacks;
    currentPage = 1;
    pageSize;
    sortByColumn;
    sortAscending = true;
    disposables = [];
    static async show(client, project, dataset, defaultPageSize, callbacks) {
        const existing = DatasetBrowserPanel.panels.get(dataset.id);
        if (existing) {
            existing.client = client;
            existing.project = project;
            existing.dataset = dataset;
            existing.callbacks = callbacks;
            existing.panel.reveal(vscode.ViewColumn.One);
            await existing.loadAndRender();
            return existing;
        }
        const panel = vscode.window.createWebviewPanel("evidentlyDatasetBrowser", `Dataset: ${dataset.name}`, vscode.ViewColumn.One, {
            enableScripts: true,
            retainContextWhenHidden: true,
        });
        const instance = new DatasetBrowserPanel(panel, client, project, dataset, defaultPageSize, callbacks);
        DatasetBrowserPanel.panels.set(dataset.id, instance);
        await instance.loadAndRender();
        return instance;
    }
    constructor(panel, client, project, dataset, pageSize, callbacks) {
        this.panel = panel;
        this.client = client;
        this.project = project;
        this.dataset = dataset;
        this.pageSize = pageSize;
        this.callbacks = callbacks;
        this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
        this.panel.webview.onDidReceiveMessage(async (msg) => {
            await this.handleMessage(msg);
        }, null, this.disposables);
    }
    dispose() {
        DatasetBrowserPanel.panels.delete(this.dataset.id);
        this.panel.dispose();
        while (this.disposables.length) {
            const d = this.disposables.pop();
            if (d) {
                d.dispose();
            }
        }
    }
    async handleMessage(msg) {
        switch (msg?.type) {
            case "page": {
                this.currentPage = Math.max(1, Number(msg.page) || 1);
                await this.loadAndRender();
                break;
            }
            case "pageSize": {
                this.pageSize = Math.max(10, Math.min(500, Number(msg.pageSize) || 50));
                this.currentPage = 1;
                await this.loadAndRender();
                break;
            }
            case "sort": {
                const col = String(msg.column || "");
                if (col) {
                    if (this.sortByColumn === col) {
                        this.sortAscending = !this.sortAscending;
                    }
                    else {
                        this.sortByColumn = col;
                        this.sortAscending = true;
                    }
                    this.currentPage = 1;
                    await this.loadAndRender();
                }
                break;
            }
            case "refresh": {
                await this.loadAndRender();
                break;
            }
            case "selectLeft": {
                this.callbacks.onSelectForJoin("left", this.project, this.dataset);
                break;
            }
            case "selectRight": {
                this.callbacks.onSelectForJoin("right", this.project, this.dataset);
                break;
            }
            case "createReport": {
                this.callbacks.onCreateReportFromDataset(this.project, this.dataset);
                break;
            }
            case "exportCsv": {
                await this.exportCsv();
                break;
            }
            case "openExternal": {
                const url = `${this.client.getBaseUrl()}/projects/${this.project.id}/datasets/${this.dataset.id}`;
                await vscode.env.openExternal(vscode.Uri.parse(url));
                break;
            }
        }
    }
    async exportCsv() {
        try {
            const safeName = this.dataset.name.replace(/[^a-zA-Z0-9._-]+/g, "_");
            const targetUri = await vscode.window.showSaveDialog({
                defaultUri: vscode.Uri.file(`${safeName}_${this.dataset.id.slice(0, 8)}.csv`),
                filters: { "CSV Files": ["csv"] },
                saveLabel: "Export Dataset CSV",
            });
            if (!targetUri) {
                return;
            }
            const csv = await this.client.downloadDatasetCsv(this.dataset.id);
            fs.writeFileSync(targetUri.fsPath, csv, "utf8");
            vscode.window.showInformationMessage(`Exported dataset "${this.dataset.name}" to ${targetUri.fsPath}`);
        }
        catch (err) {
            vscode.window.showErrorMessage(`Failed to export dataset CSV: ${err?.message || err}`);
        }
    }
    async loadAndRender() {
        try {
            const page = await this.client.getDatasetPage(this.dataset.id, {
                pageSize: this.pageSize,
                currentPage: this.currentPage,
                sortByColumn: this.sortByColumn,
                sortAscending: this.sortAscending,
            });
            this.panel.webview.html = this.buildHtml(page);
        }
        catch (err) {
            this.panel.webview.html = this.buildErrorHtml(err?.message || String(err));
        }
    }
    buildErrorHtml(errorMsg) {
        return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<style>
  body { font-family: var(--vscode-font-family, sans-serif); padding: 24px; color: var(--vscode-foreground); background: var(--vscode-editor-background); }
  .err-box { border: 1px solid var(--vscode-inputValidation-errorBorder, #f14c4c); background: rgba(241, 76, 76, 0.1); padding: 16px; border-radius: 6px; }
  button { background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; margin-top: 12px; }
</style>
</head>
<body>
  <div class="err-box">
    <h3>Failed to load dataset "${escapeHtml(this.dataset.name)}" (${escapeHtml(this.dataset.id)})</h3>
    <pre>${escapeHtml(errorMsg)}</pre>
    <button onclick="acquireVsCodeApi().postMessage({ type: 'refresh' })">Retry</button>
  </div>
</body>
</html>`;
    }
    buildHtml(page) {
        const columns = page.metadata?.columns || [];
        const items = page.items || [];
        const totalRows = page.metadata?.row_count ?? this.dataset.row_count ?? items.length;
        const totalPages = Math.max(1, page.total_pages || Math.ceil(totalRows / this.pageSize));
        const currentPage = page.current_page || this.currentPage;
        const schemaBadges = columns
            .map((col) => {
            const typeClass = col.type === "numerical"
                ? "badge-num"
                : col.type === "datetime"
                    ? "badge-dt"
                    : "badge-str";
            return `<span class="col-pill" title="Click to sort by ${escapeHtml(col.name)}" onclick="sortCol('${escapeHtml(col.name.replace(/'/g, "\\'"))}')">
          <span class="col-name">${escapeHtml(col.name)}</span>
          <span class="type-badge ${typeClass}">${escapeHtml(col.type)}</span>
        </span>`;
        })
            .join("");
        const headerCells = columns
            .map((col) => {
            const isSorted = this.sortByColumn === col.name;
            const arrow = isSorted ? (this.sortAscending ? " ▲" : " ▼") : "";
            return `<th onclick="sortCol('${escapeHtml(col.name.replace(/'/g, "\\'"))}')" title="Click to sort by ${escapeHtml(col.name)} (${escapeHtml(col.type)})">
          <div class="th-inner">
            <span>${escapeHtml(col.name)}${arrow}</span>
            <span class="th-type">${escapeHtml(col.type)}</span>
          </div>
        </th>`;
        })
            .join("");
        const startRowIndex = (currentPage - 1) * this.pageSize + 1;
        const bodyRows = items.length === 0
            ? `<tr><td colspan="${Math.max(1, columns.length + 1)}" class="empty-cell">No rows in this page</td></tr>`
            : items
                .map((row, idx) => {
                const cells = columns
                    .map((col, cIdx) => {
                    const val = cIdx < row.length ? row[cIdx] : null;
                    if (val === null || val === undefined || val === "") {
                        return `<td class="null-cell">null</td>`;
                    }
                    const isNum = col.type === "numerical" && typeof val === "number";
                    return `<td class="${isNum ? "num-cell" : ""}">${escapeHtml(val)}</td>`;
                })
                    .join("");
                return `<tr class="data-row"><td class="row-num">${startRowIndex + idx}</td>${cells}</tr>`;
            })
                .join("");
        const tagsHtml = this.dataset.tags && this.dataset.tags.length > 0
            ? this.dataset.tags
                .map((t) => `<span class="tag-pill">${escapeHtml(t)}</span>`)
                .join("")
            : `<span class="muted">none</span>`;
        return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>
  :root {
    --border: var(--vscode-panel-border, rgba(128,128,128,0.25));
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
  .header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
  }
  .title-block h2 {
    margin: 0 0 6px 0;
    font-size: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .meta-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    font-size: 12px;
    opacity: 0.9;
  }
  .meta-item strong {
    opacity: 0.75;
  }
  .desc-box {
    margin-top: 8px;
    font-size: 12px;
    opacity: 0.85;
    max-width: 900px;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  button {
    background: var(--accent);
    color: var(--accent-fg);
    border: 1px solid transparent;
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
    font-weight: 500;
  }
  button:hover {
    opacity: 0.92;
  }
  button.secondary {
    background: var(--vscode-button-secondaryBackground, rgba(128,128,128,0.2));
    color: var(--vscode-button-secondaryForeground, var(--vscode-foreground));
    border-color: var(--border);
  }
  button:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .schema-bar {
    margin: 12px 0;
    padding: 10px 12px;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  .schema-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.75;
    margin-bottom: 8px;
    font-weight: 600;
  }
  .pills-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .col-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    border-radius: 4px;
    background: var(--vscode-editor-background);
    border: 1px solid var(--border);
    font-size: 11.5px;
    cursor: pointer;
  }
  .col-pill:hover {
    border-color: var(--accent);
  }
  .type-badge {
    font-size: 10px;
    padding: 1px 5px;
    border-radius: 3px;
    font-weight: 600;
  }
  .badge-num { background: rgba(56, 139, 253, 0.2); color: #58a6ff; }
  .badge-str { background: rgba(163, 113, 247, 0.2); color: #d2a8ff; }
  .badge-dt { background: rgba(63, 185, 80, 0.2); color: #7ee787; }
  .tag-pill {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 3px;
    background: rgba(56, 139, 253, 0.18);
    font-size: 11px;
    margin-right: 4px;
  }
  .controls-bar {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin: 12px 0;
  }
  .search-input {
    background: var(--vscode-input-background);
    color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, var(--border));
    padding: 5px 10px;
    border-radius: 4px;
    width: 280px;
    font-size: 12px;
  }
  .pagination {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
  }
  select {
    background: var(--vscode-dropdown-background, var(--vscode-input-background));
    color: var(--vscode-dropdown-foreground, var(--vscode-foreground));
    border: 1px solid var(--vscode-dropdown-border, var(--border));
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
  }
  .table-container {
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
    max-height: calc(100vh - 280px);
    overflow-y: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  th, td {
    padding: 7px 10px;
    border-bottom: 1px solid var(--border);
    border-right: 1px solid var(--border);
    text-align: left;
    max-width: 380px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  th {
    position: sticky;
    top: 0;
    background: var(--vscode-editorGroupHeader-tabsBackground, #252526);
    cursor: pointer;
    user-select: none;
    z-index: 2;
  }
  th:hover {
    background: var(--vscode-list-hoverBackground);
  }
  .th-inner {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .th-type {
    font-size: 10px;
    opacity: 0.65;
    font-weight: normal;
  }
  tr.data-row:hover {
    background: var(--vscode-list-hoverBackground);
  }
  .row-num {
    opacity: 0.55;
    font-family: var(--vscode-editor-font-family, monospace);
    width: 42px;
    text-align: right;
  }
  .num-cell {
    font-family: var(--vscode-editor-font-family, monospace);
    text-align: right;
  }
  .null-cell {
    opacity: 0.4;
    font-style: italic;
  }
  .empty-cell {
    text-align: center;
    padding: 24px;
    opacity: 0.7;
  }
  .muted { opacity: 0.6; }
</style>
</head>
<body>
  <div class="header">
    <div class="title-block">
      <h2>📊 ${escapeHtml(page.metadata?.name || this.dataset.name)}</h2>
      <div class="meta-grid">
        <span class="meta-item"><strong>Project:</strong> ${escapeHtml(this.project.name)}</span>
        <span class="meta-item"><strong>Dataset ID:</strong> <code>${escapeHtml(this.dataset.id)}</code></span>
        <span class="meta-item"><strong>Rows:</strong> ${escapeHtml(totalRows)}</span>
        <span class="meta-item"><strong>Columns:</strong> ${escapeHtml(columns.length)}</span>
        <span class="meta-item"><strong>Origin:</strong> ${escapeHtml(this.dataset.origin)}</span>
        <span class="meta-item"><strong>Created:</strong> ${escapeHtml(this.dataset.created_at)}</span>
        <span class="meta-item"><strong>Tags:</strong> ${tagsHtml}</span>
      </div>
      ${page.metadata?.description || this.dataset.description
            ? `<div class="desc-box">${escapeHtml(page.metadata?.description || this.dataset.description)}</div>`
            : ""}
    </div>
    <div class="actions">
      <button onclick="send('selectLeft')" title="Send this dataset to the Join & Report Studio as the Left Table">⬅ Use as Left Table in Join</button>
      <button onclick="send('selectRight')" title="Send this dataset to the Join & Report Studio as the Right Table">➡ Use as Right Table in Join</button>
      <button class="secondary" onclick="send('createReport')" title="Open Join & Report Studio with this dataset">📈 Create Report</button>
      <button class="secondary" onclick="send('exportCsv')">⬇ Export CSV</button>
      <button class="secondary" onclick="send('openExternal')">↗ Open in Evidently UI</button>
      <button class="secondary" onclick="send('refresh')">↻ Refresh</button>
    </div>
  </div>

  <div class="schema-bar">
    <div class="schema-title">Columns & Data Definition (${columns.length} columns — click any column to sort)</div>
    <div class="pills-wrap">${schemaBadges || '<span class="muted">No columns</span>'}</div>
  </div>

  <div class="controls-bar">
    <div>
      <input
        type="text"
        id="rowSearch"
        class="search-input"
        placeholder="Filter rows on this page..."
        oninput="filterRows()"
      />
    </div>
    <div class="pagination">
      <span>Rows per page:</span>
      <select onchange="changePageSize(this.value)">
        ${[25, 50, 100, 250]
            .map((sz) => `<option value="${sz}" ${sz === this.pageSize ? "selected" : ""}>${sz}</option>`)
            .join("")}
      </select>
      <button class="secondary" ${currentPage <= 1 ? "disabled" : ""} onclick="goPage(1)">« First</button>
      <button class="secondary" ${currentPage <= 1 ? "disabled" : ""} onclick="goPage(${currentPage - 1})">‹ Prev</button>
      <span>Page <strong>${currentPage}</strong> of <strong>${totalPages}</strong></span>
      <button class="secondary" ${currentPage >= totalPages ? "disabled" : ""} onclick="goPage(${currentPage + 1})">Next ›</button>
      <button class="secondary" ${currentPage >= totalPages ? "disabled" : ""} onclick="goPage(${totalPages})">Last »</button>
    </div>
  </div>

  <div class="table-container">
    <table id="dataTable">
      <thead>
        <tr>
          <th class="row-num">#</th>
          ${headerCells}
        </tr>
      </thead>
      <tbody>
        ${bodyRows}
      </tbody>
    </table>
  </div>

  <script>
    const vscode = acquireVsCodeApi();
    function send(type, extra = {}) {
      vscode.postMessage({ type, ...extra });
    }
    function goPage(page) {
      send('page', { page });
    }
    function changePageSize(pageSize) {
      send('pageSize', { pageSize: Number(pageSize) });
    }
    function sortCol(column) {
      send('sort', { column });
    }
    function filterRows() {
      const q = (document.getElementById('rowSearch').value || '').toLowerCase();
      const rows = document.querySelectorAll('#dataTable tbody tr.data-row');
      for (const r of rows) {
        const txt = r.textContent.toLowerCase();
        r.style.display = !q || txt.includes(q) ? '' : 'none';
      }
    }
  </script>
</body>
</html>`;
    }
}
exports.DatasetBrowserPanel = DatasetBrowserPanel;
//# sourceMappingURL=datasetPanel.js.map