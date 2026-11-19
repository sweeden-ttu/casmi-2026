import * as fs from "fs";
import * as vscode from "vscode";
import {
  EvidentlyClient,
  EvidentlyProject,
  EvidentlyReportInfo,
} from "./evidentlyClient";
import { getReportDisplayName } from "./treeProvider";

function escapeHtml(text: any): string {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatMetricValue(val: any): string {
  if (val === null || val === undefined) {
    return "null";
  }
  if (typeof val === "number") {
    return Number.isInteger(val) ? String(val) : val.toFixed(6).replace(/\.?0+$/, "");
  }
  if (typeof val === "object") {
    const entries = Object.entries(val);
    if (entries.length <= 6) {
      return entries
        .map(([k, v]) => `${k}: ${typeof v === "number" ? Number(v.toFixed(4)) : JSON.stringify(v)}`)
        .join(", ");
    }
    return JSON.stringify(val).slice(0, 200) + "...";
  }
  return String(val);
}

export class ReportViewerPanel {
  private static panels = new Map<string, ReportViewerPanel>();

  private readonly panel: vscode.WebviewPanel;
  private client: EvidentlyClient;
  private project: EvidentlyProject;
  private report: EvidentlyReportInfo;
  private localHtmlPath?: string;
  private onOpenDatasetById: (projectId: string, datasetId: string) => void;
  private disposables: vscode.Disposable[] = [];

  public static async show(
    client: EvidentlyClient,
    project: EvidentlyProject,
    report: EvidentlyReportInfo,
    onOpenDatasetById: (projectId: string, datasetId: string) => void,
    localHtmlPath?: string
  ): Promise<ReportViewerPanel> {
    const key = `${project.id}:${report.id}`;
    const existing = ReportViewerPanel.panels.get(key);
    if (existing) {
      existing.client = client;
      existing.project = project;
      existing.report = report;
      existing.localHtmlPath = localHtmlPath;
      existing.panel.reveal(vscode.ViewColumn.One);
      await existing.loadAndRender();
      return existing;
    }

    const title = getReportDisplayName(report);
    const panel = vscode.window.createWebviewPanel(
      "evidentlyReportViewer",
      `Report: ${title}`,
      vscode.ViewColumn.One,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
      }
    );

    const instance = new ReportViewerPanel(
      panel,
      client,
      project,
      report,
      onOpenDatasetById,
      localHtmlPath
    );
    ReportViewerPanel.panels.set(key, instance);
    await instance.loadAndRender();
    return instance;
  }

  private constructor(
    panel: vscode.WebviewPanel,
    client: EvidentlyClient,
    project: EvidentlyProject,
    report: EvidentlyReportInfo,
    onOpenDatasetById: (projectId: string, datasetId: string) => void,
    localHtmlPath?: string
  ) {
    this.panel = panel;
    this.client = client;
    this.project = project;
    this.report = report;
    this.onOpenDatasetById = onOpenDatasetById;
    this.localHtmlPath = localHtmlPath;

    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
    this.panel.webview.onDidReceiveMessage(
      async (msg) => {
        await this.handleMessage(msg);
      },
      null,
      this.disposables
    );
  }

  public dispose(): void {
    const key = `${this.project.id}:${this.report.id}`;
    ReportViewerPanel.panels.delete(key);
    this.panel.dispose();
    while (this.disposables.length) {
      const d = this.disposables.pop();
      if (d) {
        d.dispose();
      }
    }
  }

  private async handleMessage(msg: any): Promise<void> {
    switch (msg?.type) {
      case "refresh": {
        await this.loadAndRender();
        break;
      }
      case "loadFullHtml": {
        await this.sendFullHtmlToWebview();
        break;
      }
      case "openDataset": {
        if (msg.datasetId) {
          this.onOpenDatasetById(this.project.id, String(msg.datasetId));
        }
        break;
      }
      case "exportHtml": {
        await this.exportHtml();
        break;
      }
      case "exportJson": {
        await this.exportJson();
        break;
      }
      case "openExternal": {
        const url = `${this.client.getBaseUrl()}/projects/${this.project.id}/reports/${this.report.id}`;
        await vscode.env.openExternal(vscode.Uri.parse(url));
        break;
      }
    }
  }

  private async sendFullHtmlToWebview(): Promise<void> {
    try {
      let htmlContent = "";
      if (this.localHtmlPath && fs.existsSync(this.localHtmlPath)) {
        htmlContent = fs.readFileSync(this.localHtmlPath, "utf8");
      } else {
        htmlContent = await this.client.downloadReportHtml(
          this.project.id,
          this.report.id
        );
      }
      await this.panel.webview.postMessage({
        type: "fullHtmlLoaded",
        html: htmlContent,
      });
    } catch (err: any) {
      await this.panel.webview.postMessage({
        type: "fullHtmlError",
        error: err?.message || String(err),
      });
    }
  }

  private async exportHtml(): Promise<void> {
    try {
      const title = getReportDisplayName(this.report).replace(/[^a-zA-Z0-9._-]+/g, "_");
      const targetUri = await vscode.window.showSaveDialog({
        defaultUri: vscode.Uri.file(`${title}_${this.report.id.slice(0, 8)}.html`),
        filters: { "HTML Report": ["html"] },
        saveLabel: "Export Evidently HTML Report",
      });
      if (!targetUri) {
        return;
      }
      const html =
        this.localHtmlPath && fs.existsSync(this.localHtmlPath)
          ? fs.readFileSync(this.localHtmlPath, "utf8")
          : await this.client.downloadReportHtml(this.project.id, this.report.id);
      fs.writeFileSync(targetUri.fsPath, html, "utf8");
      vscode.window.showInformationMessage(`Saved HTML report to ${targetUri.fsPath}`);
    } catch (err: any) {
      vscode.window.showErrorMessage(`Failed to export HTML report: ${err?.message || err}`);
    }
  }

  private async exportJson(): Promise<void> {
    try {
      const title = getReportDisplayName(this.report).replace(/[^a-zA-Z0-9._-]+/g, "_");
      const targetUri = await vscode.window.showSaveDialog({
        defaultUri: vscode.Uri.file(`${title}_${this.report.id.slice(0, 8)}.json`),
        filters: { "JSON Report": ["json"] },
        saveLabel: "Export Evidently JSON Report",
      });
      if (!targetUri) {
        return;
      }
      const jsonObj = await this.client.downloadReportJson(
        this.project.id,
        this.report.id
      );
      fs.writeFileSync(targetUri.fsPath, JSON.stringify(jsonObj, null, 2), "utf8");
      vscode.window.showInformationMessage(`Saved JSON report to ${targetUri.fsPath}`);
    } catch (err: any) {
      vscode.window.showErrorMessage(`Failed to export JSON report: ${err?.message || err}`);
    }
  }

  public async loadAndRender(): Promise<void> {
    try {
      const [meta, reportJson, reportData] = await Promise.all([
        this.client
          .getReportMetadata(this.project.id, this.report.id)
          .catch(() => this.report),
        this.client
          .downloadReportJson(this.project.id, this.report.id)
          .catch(() => ({ metrics: [], tests: [] })),
        this.client
          .getReportData(this.project.id, this.report.id)
          .catch(() => ({ widgets: [] })),
      ]);
      this.report = { ...this.report, ...meta };
      this.panel.webview.html = this.buildHtml(
        reportJson.metrics || [],
        reportJson.tests || [],
        reportData.widgets || []
      );
    } catch (err: any) {
      this.panel.webview.html = `<body style="padding:20px;font-family:sans-serif;">
        <h3>Failed to load report ${escapeHtml(this.report.id)}</h3>
        <pre>${escapeHtml(err?.message || err)}</pre>
      </body>`;
    }
  }

  private collectLinkedDatasets(): { label: string; datasetId: string }[] {
    const links: { label: string; datasetId: string }[] = [];
    const seen = new Set<string>();
    const add = (label: string, id?: string | null) => {
      if (id && typeof id === "string" && id.trim() && !seen.has(id.trim())) {
        seen.add(id.trim());
        links.push({ label, datasetId: id.trim() });
      }
    };

    const dsLinks = this.report.links?.datasets;
    if (dsLinks) {
      add("Current Dataset", dsLinks.output?.current);
      add("Reference Dataset", dsLinks.output?.reference);
      add("Input Current", dsLinks.input?.current);
      add("Input Reference", dsLinks.input?.reference);
      for (const [k, v] of Object.entries(dsLinks.output?.additional || {})) {
        add(`Additional (${k})`, v);
      }
      for (const [k, v] of Object.entries(dsLinks.input?.additional || {})) {
        add(`Input (${k})`, v);
      }
    }

    const meta = this.report.metadata || {};
    add("Left Joined Table", meta.left_dataset_id);
    add("Right Joined Table", meta.right_dataset_id);

    return links;
  }

  private buildHtml(metrics: any[], tests: any[], widgets: any[]): string {
    const title = getReportDisplayName(this.report);
    const linkedDatasets = this.collectLinkedDatasets();

    const linkedButtons =
      linkedDatasets.length > 0
        ? linkedDatasets
            .map(
              (l) =>
                `<button class="secondary sm-btn" onclick="openDataset('${escapeHtml(
                  l.datasetId
                )}')" title="Open dataset ${escapeHtml(l.datasetId)}">📊 ${escapeHtml(
                  l.label
                )} (<code>${escapeHtml(l.datasetId.slice(0, 8))}</code>)</button>`
            )
            .join("")
        : `<span class="muted">None recorded</span>`;

    const tagsHtml =
      this.report.tags && this.report.tags.length > 0
        ? this.report.tags
            .map((t) => `<span class="tag-pill">${escapeHtml(t)}</span>`)
            .join("")
        : `<span class="muted">none</span>`;

    const metadataEntries = Object.entries(this.report.metadata || {});
    const metaBadges =
      metadataEntries.length > 0
        ? metadataEntries
            .map(
              ([k, v]) =>
                `<span class="meta-pill"><strong>${escapeHtml(k)}:</strong> ${escapeHtml(
                  typeof v === "object" ? JSON.stringify(v) : v
                )}</span>`
            )
            .join("")
        : `<span class="muted">none</span>`;

    const metricsRows =
      metrics.length === 0
        ? `<tr><td colspan="4" class="empty-cell">No metrics found in JSON export</td></tr>`
        : metrics
            .map((m, idx) => {
              const mName = m.metric_name || m.metric || m.id || `Metric ${idx + 1}`;
              const mCol = m.config?.column || m.column || "dataset";
              const mType = (m.config?.type || "").replace("evidently:metric_v2:", "");
              const mVal = formatMetricValue(m.value ?? m.result);
              return `<tr class="metric-row">
                <td><code>${escapeHtml(mName)}</code></td>
                <td><span class="type-badge">${escapeHtml(mCol)}</span></td>
                <td>${escapeHtml(mType)}</td>
                <td class="val-cell">${escapeHtml(mVal)}</td>
              </tr>`;
            })
            .join("");

    const widgetsHtml =
      widgets.length === 0
        ? `<div class="muted" style="padding:16px;">No widget summaries available. Switch to the "Interactive Evidently Report (HTML)" tab to view full charts.</div>`
        : widgets
            .map((w) => {
              const wTitle = w.title || w.type || "Widget";
              const wType = w.type || "widget";
              let details = "";
              if (Array.isArray(w.params?.counters)) {
                details = `<div class="counters-row">${w.params.counters
                  .map(
                    (c: any) =>
                      `<div class="counter-box"><div class="counter-val">${escapeHtml(
                        c.value
                      )}</div><div class="counter-lbl">${escapeHtml(c.label)}</div></div>`
                  )
                  .join("")}</div>`;
              } else if (Array.isArray(w.params?.header) && Array.isArray(w.params?.data)) {
                const headers = w.params.header
                  .map((h: any) => `<th>${escapeHtml(h)}</th>`)
                  .join("");
                const rows = w.params.data
                  .slice(0, 12)
                  .map(
                    (r: any[]) =>
                      `<tr>${r.map((cell: any) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`
                  )
                  .join("");
                details = `<div style="overflow-x:auto;margin-top:8px;"><table><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table></div>`;
              } else if (w.params?.data && w.params?.layout) {
                const traceCount = Array.isArray(w.params.data) ? w.params.data.length : 1;
                details = `<div class="muted" style="margin-top:4px;">Plotly chart (${traceCount} trace${
                  traceCount === 1 ? "" : "s"
                }) — view interactive plot in the HTML Report tab.</div>`;
              }
              return `<div class="widget-card">
                <div class="widget-header">
                  <strong>${escapeHtml(wTitle)}</strong>
                  <span class="type-badge">${escapeHtml(wType)}</span>
                </div>
                ${details}
              </div>`;
            })
            .join("");

    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
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
  }
  .meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    font-size: 12px;
    margin-bottom: 6px;
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
  button.secondary {
    background: var(--vscode-button-secondaryBackground, rgba(128,128,128,0.2));
    color: var(--vscode-button-secondaryForeground, var(--vscode-foreground));
    border-color: var(--border);
  }
  button.sm-btn {
    padding: 3px 8px;
    font-size: 11.5px;
    margin-right: 6px;
  }
  .tag-pill, .meta-pill {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    background: var(--card-bg);
    border: 1px solid var(--border);
    font-size: 11.5px;
    margin-right: 5px;
    margin-bottom: 4px;
  }
  .type-badge {
    font-size: 10.5px;
    padding: 2px 6px;
    border-radius: 3px;
    background: rgba(56, 139, 253, 0.2);
    color: #58a6ff;
  }
  .tabs {
    display: flex;
    gap: 4px;
    margin: 14px 0;
    border-bottom: 1px solid var(--border);
  }
  .tab-btn {
    background: transparent;
    color: var(--vscode-foreground);
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    padding: 8px 16px;
    font-size: 13px;
    cursor: pointer;
    opacity: 0.75;
  }
  .tab-btn.active {
    border-bottom-color: var(--accent);
    opacity: 1;
    font-weight: 600;
  }
  .tab-pane {
    display: none;
  }
  .tab-pane.active {
    display: block;
  }
  .search-input {
    background: var(--vscode-input-background);
    color: var(--vscode-input-foreground);
    border: 1px solid var(--vscode-input-border, var(--border));
    padding: 6px 10px;
    border-radius: 4px;
    width: 300px;
    font-size: 12px;
    margin-bottom: 10px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  th, td {
    padding: 7px 10px;
    border: 1px solid var(--border);
    text-align: left;
  }
  th {
    background: var(--vscode-editorGroupHeader-tabsBackground, #252526);
  }
  .val-cell {
    font-family: var(--vscode-editor-font-family, monospace);
  }
  .widget-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 12px;
    margin-top: 10px;
  }
  .widget-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
  }
  .widget-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .counters-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 10px;
  }
  .counter-box {
    background: var(--vscode-editor-background);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 12px;
    min-width: 90px;
  }
  .counter-val {
    font-size: 16px;
    font-weight: 700;
  }
  .counter-lbl {
    font-size: 11px;
    opacity: 0.75;
  }
  #htmlIframe {
    width: 100%;
    height: calc(100vh - 220px);
    border: 1px solid var(--border);
    border-radius: 6px;
    background: #ffffff;
  }
  .muted { opacity: 0.6; }
</style>
</head>
<body>
  <div class="header">
    <div class="title-block">
      <h2>📈 ${escapeHtml(title)}</h2>
      <div class="meta-row">
        <span><strong>Project:</strong> ${escapeHtml(this.project.name)}</span>
        <span><strong>Snapshot ID:</strong> <code>${escapeHtml(this.report.id)}</code></span>
        <span><strong>Timestamp:</strong> ${escapeHtml(this.report.timestamp)}</span>
        <span><strong>Metrics:</strong> ${metrics.length}</span>
        <span><strong>Widgets:</strong> ${widgets.length}</span>
        <span><strong>Tests:</strong> ${tests.length}</span>
      </div>
      <div class="meta-row">
        <span><strong>Tags:</strong> ${tagsHtml}</span>
      </div>
      <div class="meta-row">
        <span><strong>Metadata:</strong> ${metaBadges}</span>
      </div>
      <div class="meta-row" style="align-items:center;">
        <span><strong>Linked Datasets:</strong> ${linkedButtons}</span>
      </div>
    </div>
    <div class="actions">
      <button onclick="switchTab('htmlTab')">🖥 Interactive HTML Report</button>
      <button class="secondary" onclick="send('exportHtml')">⬇ Export HTML</button>
      <button class="secondary" onclick="send('exportJson')">⬇ Export JSON</button>
      <button class="secondary" onclick="send('openExternal')">↗ Open in Evidently UI</button>
      <button class="secondary" onclick="send('refresh')">↻ Refresh</button>
    </div>
  </div>

  <div class="tabs">
    <button class="tab-btn active" id="btn-metricsTab" onclick="switchTab('metricsTab')">
      Metrics Table (${metrics.length})
    </button>
    <button class="tab-btn" id="btn-widgetsTab" onclick="switchTab('widgetsTab')">
      Widgets Summary (${widgets.length})
    </button>
    <button class="tab-btn" id="btn-htmlTab" onclick="switchTab('htmlTab')">
      Interactive Evidently Report (HTML)
    </button>
  </div>

  <div id="metricsTab" class="tab-pane active">
    <input
      type="text"
      id="metricSearch"
      class="search-input"
      placeholder="Filter metrics by name, column, or value..."
      oninput="filterMetrics()"
    />
    <table id="metricsTable">
      <thead>
        <tr>
          <th>Metric</th>
          <th>Column</th>
          <th>Type</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        ${metricsRows}
      </tbody>
    </table>
  </div>

  <div id="widgetsTab" class="tab-pane">
    <div class="widget-grid">
      ${widgetsHtml}
    </div>
  </div>

  <div id="htmlTab" class="tab-pane">
    <div id="htmlStatus" style="margin-bottom:8px;font-size:12px;">
      Click below or wait for the standalone Evidently HTML report to load...
      <button class="secondary sm-btn" onclick="requestHtml()">Reload HTML Report</button>
    </div>
    <iframe id="htmlIframe" sandbox="allow-scripts allow-same-origin"></iframe>
  </div>

  <script>
    const vscode = acquireVsCodeApi();
    let htmlLoaded = false;

    function send(type, extra = {}) {
      vscode.postMessage({ type, ...extra });
    }

    function openDataset(datasetId) {
      send('openDataset', { datasetId });
    }

    function switchTab(tabId) {
      for (const pane of document.querySelectorAll('.tab-pane')) {
        pane.classList.remove('active');
      }
      for (const btn of document.querySelectorAll('.tab-btn')) {
        btn.classList.remove('active');
      }
      document.getElementById(tabId).classList.add('active');
      document.getElementById('btn-' + tabId).classList.add('active');
      if (tabId === 'htmlTab' && !htmlLoaded) {
        requestHtml();
      }
    }

    function requestHtml() {
      document.getElementById('htmlStatus').textContent = 'Downloading interactive Evidently HTML report...';
      send('loadFullHtml');
    }

    function filterMetrics() {
      const q = (document.getElementById('metricSearch').value || '').toLowerCase();
      for (const r of document.querySelectorAll('#metricsTable tbody tr.metric-row')) {
        r.style.display = !q || r.textContent.toLowerCase().includes(q) ? '' : 'none';
      }
    }

    window.addEventListener('message', (event) => {
      const msg = event.data;
      if (msg.type === 'fullHtmlLoaded') {
        htmlLoaded = true;
        document.getElementById('htmlStatus').textContent = 'Interactive Evidently HTML report loaded.';
        const iframe = document.getElementById('htmlIframe');
        iframe.srcdoc = msg.html;
      } else if (msg.type === 'fullHtmlError') {
        document.getElementById('htmlStatus').textContent = 'Error loading HTML report: ' + msg.error;
      }
    });
  </script>
</body>
</html>`;
  }
}
