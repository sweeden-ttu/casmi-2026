import * as vscode from "vscode";
import {
  EvidentlyClient,
  EvidentlyDatasetInfo,
  EvidentlyProject,
  EvidentlyReportInfo,
} from "./evidentlyClient";

export type TreeNodeKind =
  | "project"
  | "datasetsFolder"
  | "reportsFolder"
  | "datasetVersionGroup"
  | "dataset"
  | "report"
  | "info";

export interface EvidentlyTreeNodeData {
  kind: TreeNodeKind;
  project?: EvidentlyProject;
  dataset?: EvidentlyDatasetInfo;
  datasetVersions?: EvidentlyDatasetInfo[];
  report?: EvidentlyReportInfo;
  datasetsCache?: EvidentlyDatasetInfo[];
  reportsCache?: EvidentlyReportInfo[];
  reportsError?: string;
  message?: string;
}

function formatShortTimestamp(iso?: string | null): string {
  if (!iso) {
    return "";
  }
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) {
    return String(iso).slice(0, 16);
  }
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const min = String(d.getMinutes()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd} ${hh}:${min}`;
}

function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) {
    return "0 B";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function getReportDisplayName(report: EvidentlyReportInfo): string {
  if (report.name && report.name.trim() && report.name !== "None") {
    return report.name.trim();
  }
  if (report.metadata && typeof report.metadata.kind === "string" && report.metadata.kind.trim()) {
    return report.metadata.kind.trim();
  }
  if (Array.isArray(report.tags) && report.tags.length > 0) {
    return report.tags.join(" · ");
  }
  return `Report ${report.id.slice(0, 8)}`;
}

export class EvidentlyTreeItem extends vscode.TreeItem {
  public readonly data: EvidentlyTreeNodeData;

  constructor(
    label: string,
    collapsibleState: vscode.TreeItemCollapsibleState,
    data: EvidentlyTreeNodeData
  ) {
    super(label, collapsibleState);
    this.data = data;
    this.contextValue = data.kind;
  }
}

export class EvidentlyTreeDataProvider
  implements vscode.TreeDataProvider<EvidentlyTreeItem>
{
  private _onDidChangeTreeData = new vscode.EventEmitter<
    EvidentlyTreeItem | undefined | null | void
  >();
  public readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private client: EvidentlyClient;
  private groupVersions: boolean;
  private filterText: string = "";

  private projectsCache: EvidentlyProject[] | null = null;
  private projectDatasetsMap = new Map<string, EvidentlyDatasetInfo[]>();
  private projectReportsMap = new Map<
    string,
    { reports: EvidentlyReportInfo[]; error?: string }
  >();

  constructor(client: EvidentlyClient, groupVersions: boolean = true) {
    this.client = client;
    this.groupVersions = groupVersions;
  }

  public setClient(client: EvidentlyClient, groupVersions: boolean): void {
    this.client = client;
    this.groupVersions = groupVersions;
    this.refresh();
  }

  public setFilter(text: string): void {
    this.filterText = (text || "").trim().toLowerCase();
    this._onDidChangeTreeData.fire();
  }

  public getFilter(): string {
    return this.filterText;
  }

  public refresh(): void {
    this.projectsCache = null;
    this.projectDatasetsMap.clear();
    this.projectReportsMap.clear();
    this._onDidChangeTreeData.fire();
  }

  public async getCachedProjects(): Promise<EvidentlyProject[]> {
    if (!this.projectsCache) {
      this.projectsCache = await this.client.listProjects();
    }
    return this.projectsCache;
  }

  public async getCachedDatasets(projectId: string): Promise<EvidentlyDatasetInfo[]> {
    if (!this.projectDatasetsMap.has(projectId)) {
      const ds = await this.client.listDatasets(projectId);
      this.projectDatasetsMap.set(projectId, ds);
    }
    return this.projectDatasetsMap.get(projectId) || [];
  }

  public async getCachedReports(
    projectId: string
  ): Promise<{ reports: EvidentlyReportInfo[]; error?: string }> {
    if (!this.projectReportsMap.has(projectId)) {
      const res = await this.client.listReports(projectId);
      this.projectReportsMap.set(projectId, res);
    }
    return this.projectReportsMap.get(projectId) || { reports: [] };
  }

  public getTreeItem(element: EvidentlyTreeItem): vscode.TreeItem {
    return element;
  }

  private matchesFilter(textParts: (string | undefined | null)[]): boolean {
    if (!this.filterText) {
      return true;
    }
    const combined = textParts
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return combined.includes(this.filterText);
  }

  public async getChildren(
    element?: EvidentlyTreeItem
  ): Promise<EvidentlyTreeItem[]> {
    if (!element) {
      return this.getRootProjectNodes();
    }

    const { data } = element;

    if (data.kind === "project" && data.project) {
      return this.getProjectChildFolders(data.project);
    }

    if (data.kind === "datasetsFolder" && data.project) {
      return this.getDatasetNodes(
        data.project,
        data.datasetsCache || (await this.getCachedDatasets(data.project.id))
      );
    }

    if (data.kind === "datasetVersionGroup" && data.project && data.datasetVersions) {
      return data.datasetVersions.map((ds, idx) =>
        this.createDatasetItem(data.project!, ds, true, idx === 0)
      );
    }

    if (data.kind === "reportsFolder" && data.project) {
      const cached =
        data.reportsCache !== undefined
          ? { reports: data.reportsCache, error: data.reportsError }
          : await this.getCachedReports(data.project.id);
      return this.getReportNodes(data.project, cached.reports, cached.error);
    }

    return [];
  }

  private async getRootProjectNodes(): Promise<EvidentlyTreeItem[]> {
    try {
      const projects = await this.getCachedProjects();
      if (projects.length === 0) {
        const emptyItem = new EvidentlyTreeItem(
          `No projects found on ${this.client.getBaseUrl()}`,
          vscode.TreeItemCollapsibleState.None,
          { kind: "info" }
        );
        emptyItem.iconPath = new vscode.ThemeIcon("info");
        return [emptyItem];
      }

      const items: EvidentlyTreeItem[] = [];
      for (const proj of projects) {
        // When filter is active, check if project name matches OR any dataset/report inside matches
        let datasets: EvidentlyDatasetInfo[] | undefined;
        let reportsRes: { reports: EvidentlyReportInfo[]; error?: string } | undefined;

        if (this.filterText) {
          const projectMatches = this.matchesFilter([
            proj.name,
            proj.description,
            proj.id,
          ]);
          if (!projectMatches) {
            datasets = await this.getCachedDatasets(proj.id);
            reportsRes = await this.getCachedReports(proj.id);
            const hasMatchingDataset = datasets.some((d) =>
              this.matchesFilter([d.name, d.description, d.id, ...(d.tags || [])])
            );
            const hasMatchingReport = reportsRes.reports.some((r) =>
              this.matchesFilter([
                getReportDisplayName(r),
                r.id,
                ...(r.tags || []),
              ])
            );
            if (!hasMatchingDataset && !hasMatchingReport) {
              continue;
            }
          }
        }

        const collapsible = this.filterText
          ? vscode.TreeItemCollapsibleState.Expanded
          : proj.name.includes("casmi")
          ? vscode.TreeItemCollapsibleState.Expanded
          : vscode.TreeItemCollapsibleState.Collapsed;

        const item = new EvidentlyTreeItem(proj.name, collapsible, {
          kind: "project",
          project: proj,
        });
        item.iconPath = new vscode.ThemeIcon("project");
        item.description = proj.id.slice(0, 8);
        item.tooltip = new vscode.MarkdownString(
          `**Project:** \`${proj.name}\`\n\n**ID:** \`${proj.id}\`\n\n${
            proj.description ? proj.description : ""
          }`
        );
        items.push(item);
      }

      if (items.length === 0 && this.filterText) {
        const noMatch = new EvidentlyTreeItem(
          `No items match filter "${this.filterText}"`,
          vscode.TreeItemCollapsibleState.None,
          { kind: "info" }
        );
        noMatch.iconPath = new vscode.ThemeIcon("search-stop");
        noMatch.command = {
          command: "evidentlyWorkspace.clearFilter",
          title: "Clear Filter",
        };
        return [noMatch];
      }

      return items;
    } catch (err: any) {
      const errItem = new EvidentlyTreeItem(
        `Cannot connect to ${this.client.getBaseUrl()}: ${err?.message || err}`,
        vscode.TreeItemCollapsibleState.None,
        { kind: "info" }
      );
      errItem.iconPath = new vscode.ThemeIcon("error");
      errItem.command = {
        command: "evidentlyWorkspace.configureServer",
        title: "Configure Server",
      };
      return [errItem];
    }
  }

  private async getProjectChildFolders(
    project: EvidentlyProject
  ): Promise<EvidentlyTreeItem[]> {
    const [datasets, reportsRes] = await Promise.all([
      this.getCachedDatasets(project.id),
      this.getCachedReports(project.id),
    ]);

    const filteredDatasets = this.filterText
      ? datasets.filter(
          (d) =>
            this.matchesFilter([project.name]) ||
            this.matchesFilter([d.name, d.description, d.id, ...(d.tags || [])])
        )
      : datasets;

    const filteredReports = this.filterText
      ? reportsRes.reports.filter(
          (r) =>
            this.matchesFilter([project.name]) ||
            this.matchesFilter([
              getReportDisplayName(r),
              r.id,
              ...(r.tags || []),
            ])
        )
      : reportsRes.reports;

    const dsFolder = new EvidentlyTreeItem(
      `Datasets (${filteredDatasets.length})`,
      filteredDatasets.length > 0
        ? vscode.TreeItemCollapsibleState.Expanded
        : vscode.TreeItemCollapsibleState.Collapsed,
      {
        kind: "datasetsFolder",
        project,
        datasetsCache: filteredDatasets,
      }
    );
    dsFolder.iconPath = new vscode.ThemeIcon("database");

    const repLabel = reportsRes.error
      ? "Reports (!)"
      : `Reports (${filteredReports.length})`;
    const repFolder = new EvidentlyTreeItem(
      repLabel,
      filteredReports.length > 0 || reportsRes.error
        ? vscode.TreeItemCollapsibleState.Expanded
        : vscode.TreeItemCollapsibleState.Collapsed,
      {
        kind: "reportsFolder",
        project,
        reportsCache: filteredReports,
        reportsError: reportsRes.error,
      }
    );
    repFolder.iconPath = new vscode.ThemeIcon("graph");

    return [dsFolder, repFolder];
  }

  private getDatasetNodes(
    project: EvidentlyProject,
    datasets: EvidentlyDatasetInfo[]
  ): EvidentlyTreeItem[] {
    if (datasets.length === 0) {
      const empty = new EvidentlyTreeItem(
        "No datasets in this project",
        vscode.TreeItemCollapsibleState.None,
        { kind: "info", project }
      );
      empty.iconPath = new vscode.ThemeIcon("circle-slash");
      return [empty];
    }

    if (!this.groupVersions) {
      return datasets.map((d) => this.createDatasetItem(project, d, false, false));
    }

    // Group datasets by exact name while preserving newest-first order
    const groups = new Map<string, EvidentlyDatasetInfo[]>();
    for (const d of datasets) {
      const list = groups.get(d.name) || [];
      list.push(d);
      groups.set(d.name, list);
    }

    const items: EvidentlyTreeItem[] = [];
    for (const [name, list] of groups.entries()) {
      if (list.length === 1) {
        items.push(this.createDatasetItem(project, list[0], false, true));
      } else {
        const latest = list[0];
        const groupItem = new EvidentlyTreeItem(
          name,
          vscode.TreeItemCollapsibleState.Collapsed,
          {
            kind: "datasetVersionGroup",
            project,
            dataset: latest,
            datasetVersions: list,
          }
        );
        groupItem.iconPath = new vscode.ThemeIcon("layers");
        groupItem.description = `${list.length} versions · latest: ${latest.row_count}×${latest.column_count}`;
        groupItem.tooltip = new vscode.MarkdownString(
          `**Dataset Group:** \`${name}\` (${list.length} versions)\n\n` +
            `**Latest ID:** \`${latest.id}\`\n\n` +
            `**Dimensions:** ${latest.row_count} rows × ${latest.column_count} columns\n\n` +
            `**Created:** ${formatShortTimestamp(latest.created_at)}\n\n` +
            `*(Expand to view all versions, or use inline buttons to browse/join the latest version)*`
        );
        items.push(groupItem);
      }
    }
    return items;
  }

  private createDatasetItem(
    project: EvidentlyProject,
    ds: EvidentlyDatasetInfo,
    isVersionChild: boolean,
    isLatest: boolean
  ): EvidentlyTreeItem {
    const label = isVersionChild
      ? `${formatShortTimestamp(ds.created_at) || ds.id.slice(0, 8)}${
          isLatest ? " (latest)" : ""
        }`
      : ds.name;

    const item = new EvidentlyTreeItem(
      label,
      vscode.TreeItemCollapsibleState.None,
      {
        kind: "dataset",
        project,
        dataset: ds,
      }
    );
    item.iconPath = new vscode.ThemeIcon(
      ds.origin === "tracing" ? "pulse" : "table"
    );
    const dims =
      ds.row_count === 0 && ds.origin === "tracing"
        ? "tracing stream"
        : `${ds.row_count} rows × ${ds.column_count} cols`;
    item.description = isVersionChild
      ? `${dims} · ${ds.id.slice(0, 8)}`
      : `${dims} · ${formatShortTimestamp(ds.created_at)}`;

    item.tooltip = new vscode.MarkdownString(
      `**Dataset:** \`${ds.name}\`\n\n` +
        `**ID:** \`${ds.id}\`\n\n` +
        `**Project:** \`${project.name}\`\n\n` +
        `**Dimensions:** ${ds.row_count} rows × ${ds.column_count} columns (${formatBytes(
          ds.size_bytes
        )})\n\n` +
        `**Origin:** \`${ds.origin}\`\n\n` +
        `**Created:** ${ds.created_at}\n\n` +
        (ds.tags && ds.tags.length > 0
          ? `**Tags:** ${ds.tags.map((t) => `\`${t}\``).join(", ")}\n\n`
          : "") +
        (ds.description ? `${ds.description}` : "")
    );

    item.command = {
      command: "evidentlyWorkspace.openDataset",
      title: "Browse Dataset",
      arguments: [item],
    };
    return item;
  }

  private getReportNodes(
    project: EvidentlyProject,
    reports: EvidentlyReportInfo[],
    error?: string
  ): EvidentlyTreeItem[] {
    if (error) {
      const errItem = new EvidentlyTreeItem(
        `Reports unavailable (${error})`,
        vscode.TreeItemCollapsibleState.None,
        { kind: "info", project }
      );
      errItem.iconPath = new vscode.ThemeIcon("warning");
      errItem.tooltip = `The Evidently server returned an error when listing reports for project ${project.name} (${project.id}):\n${error}\n\nDatasets in this project can still be browsed and joined.`;
      return [errItem];
    }

    if (reports.length === 0) {
      const empty = new EvidentlyTreeItem(
        "No reports in this project",
        vscode.TreeItemCollapsibleState.None,
        { kind: "info", project }
      );
      empty.iconPath = new vscode.ThemeIcon("circle-slash");
      return [empty];
    }

    return reports.map((rep) => {
      const title = getReportDisplayName(rep);
      const item = new EvidentlyTreeItem(
        title,
        vscode.TreeItemCollapsibleState.None,
        {
          kind: "report",
          project,
          report: rep,
        }
      );
      item.iconPath = new vscode.ThemeIcon("graph-line");
      const tagStr =
        rep.tags && rep.tags.length > 0 ? ` [${rep.tags.slice(0, 3).join(", ")}]` : "";
      item.description = `${formatShortTimestamp(rep.timestamp)}${tagStr}`;

      const metaLines = Object.entries(rep.metadata || {})
        .slice(0, 8)
        .map(([k, v]) => `- \`${k}\`: \`${String(v)}\``)
        .join("\n");

      item.tooltip = new vscode.MarkdownString(
        `**Report:** \`${title}\`\n\n` +
          `**Snapshot ID:** \`${rep.id}\`\n\n` +
          `**Project:** \`${project.name}\`\n\n` +
          `**Timestamp:** ${rep.timestamp}\n\n` +
          (rep.tags && rep.tags.length > 0
            ? `**Tags:** ${rep.tags.map((t) => `\`${t}\``).join(", ")}\n\n`
            : "") +
          (metaLines ? `**Metadata:**\n${metaLines}` : "")
      );

      item.command = {
        command: "evidentlyWorkspace.openReport",
        title: "View Report",
        arguments: [item],
      };
      return item;
    });
  }
}
