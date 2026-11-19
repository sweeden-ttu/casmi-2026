import * as fs from "fs";
import * as vscode from "vscode";
import { getWorkspaceConfig } from "./config";
import { DatasetBrowserPanel } from "./datasetPanel";
import {
  EvidentlyClient,
  EvidentlyDatasetInfo,
  EvidentlyProject,
  EvidentlyReportInfo,
} from "./evidentlyClient";
import { JoinStudioPanel } from "./joinStudioPanel";
import { ReportViewerPanel } from "./reportPanel";
import {
  EvidentlyTreeDataProvider,
  EvidentlyTreeItem,
  getReportDisplayName,
} from "./treeProvider";

export function activate(context: vscode.ExtensionContext): void {
  let cfg = getWorkspaceConfig();
  const client = new EvidentlyClient(cfg.serverUrl, cfg.apiKey);
  const treeProvider = new EvidentlyTreeDataProvider(
    client,
    cfg.groupDatasetVersions
  );

  const treeView = vscode.window.createTreeView("evidentlyProjectsTree", {
    treeDataProvider: treeProvider,
    showCollapseAll: true,
  });
  context.subscriptions.push(treeView);

  // Status bar item
  const statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    90
  );
  statusBarItem.command = "evidentlyWorkspace.openJoinBuilder";
  const updateStatusBar = () => {
    const host = cfg.serverUrl.replace(/^https?:\/\//, "");
    const filter = treeProvider.getFilter();
    statusBarItem.text = `$(database) Evidently: ${host}${
      filter ? ` [filter: ${filter}]` : ""
    }`;
    statusBarItem.tooltip = `Connected to Evidently Remote Workspace (${cfg.serverUrl}). Click to open Table Join & Report Studio.`;
    statusBarItem.show();
  };
  updateStatusBar();
  context.subscriptions.push(statusBarItem);

  // React to VS Code settings changes
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("evidentlyWorkspace")) {
        cfg = getWorkspaceConfig();
        client.setConfig(cfg.serverUrl, cfg.apiKey);
        treeProvider.setClient(client, cfg.groupDatasetVersions);
        updateStatusBar();
      }
    })
  );

  const openDatasetById = async (projectId: string, datasetId: string) => {
    try {
      const projects = await treeProvider.getCachedProjects();
      let targetProject = projects.find((p) => p.id === projectId);
      let datasets = targetProject
        ? await treeProvider.getCachedDatasets(targetProject.id)
        : [];
      let targetDataset = datasets.find((d) => d.id === datasetId);

      if (!targetDataset) {
        for (const p of projects) {
          const pDs = await treeProvider.getCachedDatasets(p.id);
          const found = pDs.find((d) => d.id === datasetId);
          if (found) {
            targetProject = p;
            targetDataset = found;
            break;
          }
        }
      }

      if (!targetProject || !targetDataset) {
        vscode.window.showWarningMessage(
          `Dataset ${datasetId} was not found on ${cfg.serverUrl}.`
        );
        return;
      }

      await openDatasetPanel(targetProject, targetDataset);
    } catch (err: any) {
      vscode.window.showErrorMessage(
        `Failed to open dataset ${datasetId}: ${err?.message || err}`
      );
    }
  };

  const openJoinStudioWithSelection = async (
    side?: "left" | "right",
    project?: EvidentlyProject,
    dataset?: EvidentlyDatasetInfo
  ) => {
    await JoinStudioPanel.show(
      client,
      context.extensionPath,
      {
        onReportPublished: async (
          projectId,
          snapshotId,
          _joinedDatasetId,
          htmlPath
        ) => {
          treeProvider.refresh();
          if (snapshotId) {
            const projects = await treeProvider.getCachedProjects();
            const proj =
              projects.find((p) => p.id === projectId) || {
                id: projectId,
                name: projectId,
              };
            const reportsRes = await treeProvider.getCachedReports(projectId);
            const rep = reportsRes.reports.find((r) => r.id === snapshotId) || {
              id: snapshotId,
              name: "Joined Report",
              timestamp: new Date().toISOString(),
              metadata: {},
              tags: ["joined-report"],
            };
            await ReportViewerPanel.show(
              client,
              proj,
              rep,
              openDatasetById,
              htmlPath
            );
          }
        },
        onOpenDatasetById: openDatasetById,
      },
      side && project && dataset ? { side, project, dataset } : undefined
    );
  };

  const openDatasetPanel = async (
    project: EvidentlyProject,
    dataset: EvidentlyDatasetInfo
  ) => {
    await DatasetBrowserPanel.show(
      client,
      project,
      dataset,
      cfg.defaultPageSize,
      {
        onSelectForJoin: (side, proj, ds) => {
          void openJoinStudioWithSelection(side, proj, ds);
        },
        onCreateReportFromDataset: (proj, ds) => {
          void openJoinStudioWithSelection("left", proj, ds);
        },
      }
    );
  };

  const resolveDatasetArg = async (
    item?: EvidentlyTreeItem
  ): Promise<{ project: EvidentlyProject; dataset: EvidentlyDatasetInfo } | null> => {
    if (item?.data?.project && item?.data?.dataset) {
      return { project: item.data.project, dataset: item.data.dataset };
    }
    const projects = await treeProvider.getCachedProjects();
    if (projects.length === 0) {
      vscode.window.showWarningMessage("No Evidently projects found.");
      return null;
    }
    const pickedProj = await vscode.window.showQuickPick(
      projects.map((p) => ({
        label: p.name,
        description: p.id.slice(0, 8),
        detail: p.description || "",
        project: p,
      })),
      { placeHolder: "Select an Evidently Project" }
    );
    if (!pickedProj) {
      return null;
    }
    const datasets = await treeProvider.getCachedDatasets(pickedProj.project.id);
    if (datasets.length === 0) {
      vscode.window.showWarningMessage(
        `No datasets in project "${pickedProj.project.name}".`
      );
      return null;
    }
    const pickedDs = await vscode.window.showQuickPick(
      datasets.map((d) => ({
        label: d.name,
        description: `${d.row_count} rows × ${d.column_count} cols · ${d.created_at.slice(
          0,
          16
        )}`,
        detail: d.id,
        dataset: d,
      })),
      { placeHolder: "Select a Dataset" }
    );
    if (!pickedDs) {
      return null;
    }
    return { project: pickedProj.project, dataset: pickedDs.dataset };
  };

  const resolveReportArg = async (
    item?: EvidentlyTreeItem
  ): Promise<{ project: EvidentlyProject; report: EvidentlyReportInfo } | null> => {
    if (item?.data?.project && item?.data?.report) {
      return { project: item.data.project, report: item.data.report };
    }
    const projects = await treeProvider.getCachedProjects();
    if (projects.length === 0) {
      vscode.window.showWarningMessage("No Evidently projects found.");
      return null;
    }
    const pickedProj = await vscode.window.showQuickPick(
      projects.map((p) => ({
        label: p.name,
        description: p.id.slice(0, 8),
        project: p,
      })),
      { placeHolder: "Select an Evidently Project" }
    );
    if (!pickedProj) {
      return null;
    }
    const reportsRes = await treeProvider.getCachedReports(pickedProj.project.id);
    if (reportsRes.error) {
      vscode.window.showErrorMessage(
        `Reports unavailable for "${pickedProj.project.name}": ${reportsRes.error}`
      );
      return null;
    }
    if (reportsRes.reports.length === 0) {
      vscode.window.showWarningMessage(
        `No reports in project "${pickedProj.project.name}".`
      );
      return null;
    }
    const pickedRep = await vscode.window.showQuickPick(
      reportsRes.reports.map((r) => ({
        label: getReportDisplayName(r),
        description: `${r.timestamp.slice(0, 16)} · ${(r.tags || []).join(", ")}`,
        detail: r.id,
        report: r,
      })),
      { placeHolder: "Select an Evidently Report" }
    );
    if (!pickedRep) {
      return null;
    }
    return { project: pickedProj.project, report: pickedRep.report };
  };

  // Register Commands
  context.subscriptions.push(
    vscode.commands.registerCommand("evidentlyWorkspace.refresh", () => {
      cfg = getWorkspaceConfig();
      client.setConfig(cfg.serverUrl, cfg.apiKey);
      treeProvider.setClient(client, cfg.groupDatasetVersions);
      updateStatusBar();
    }),

    vscode.commands.registerCommand("evidentlyWorkspace.filterTree", async () => {
      const current = treeProvider.getFilter();
      const input = await vscode.window.showInputBox({
        prompt: "Filter Evidently projects, datasets, and reports by name, tag, or ID",
        value: current,
        placeHolder: "e.g. casmi, rl77, weights, bench, leaderboard...",
      });
      if (input !== undefined) {
        treeProvider.setFilter(input);
        treeView.description = input ? `Filter: "${input}"` : undefined;
        updateStatusBar();
      }
    }),

    vscode.commands.registerCommand("evidentlyWorkspace.clearFilter", () => {
      treeProvider.setFilter("");
      treeView.description = undefined;
      updateStatusBar();
    }),

    vscode.commands.registerCommand(
      "evidentlyWorkspace.configureServer",
      async () => {
        const newUrl = await vscode.window.showInputBox({
          prompt: "Evidently Remote Workspace Server URL",
          value: cfg.serverUrl,
          placeHolder: "http://192.168.0.15:8000",
        });
        if (!newUrl) {
          return;
        }
        const vsCfg = vscode.workspace.getConfiguration("evidentlyWorkspace");
        await vsCfg.update(
          "serverUrl",
          newUrl.trim(),
          vscode.ConfigurationTarget.Global
        );
        cfg = getWorkspaceConfig();
        client.setConfig(cfg.serverUrl, cfg.apiKey);
        treeProvider.refresh();
        updateStatusBar();
        vscode.window.showInformationMessage(
          `Evidently server URL set to ${cfg.serverUrl}`
        );
      }
    ),

    vscode.commands.registerCommand(
      "evidentlyWorkspace.openDataset",
      async (item?: EvidentlyTreeItem) => {
        const target = await resolveDatasetArg(item);
        if (!target) {
          return;
        }
        await openDatasetPanel(target.project, target.dataset);
      }
    ),

    vscode.commands.registerCommand(
      "evidentlyWorkspace.openReport",
      async (item?: EvidentlyTreeItem) => {
        const target = await resolveReportArg(item);
        if (!target) {
          return;
        }
        await ReportViewerPanel.show(
          client,
          target.project,
          target.report,
          openDatasetById
        );
      }
    ),

    vscode.commands.registerCommand(
      "evidentlyWorkspace.openJoinBuilder",
      async (item?: EvidentlyTreeItem) => {
        if (item?.data?.project && item?.data?.dataset) {
          await openJoinStudioWithSelection(
            "left",
            item.data.project,
            item.data.dataset
          );
          return;
        }
        if (item?.data?.project) {
          const datasets = await treeProvider.getCachedDatasets(
            item.data.project.id
          );
          if (datasets.length > 0) {
            await openJoinStudioWithSelection(
              "left",
              item.data.project,
              datasets[0]
            );
            return;
          }
        }
        await openJoinStudioWithSelection();
      }
    ),

    vscode.commands.registerCommand(
      "evidentlyWorkspace.selectLeftTable",
      async (item?: EvidentlyTreeItem) => {
        const target = await resolveDatasetArg(item);
        if (!target) {
          return;
        }
        await openJoinStudioWithSelection(
          "left",
          target.project,
          target.dataset
        );
      }
    ),

    vscode.commands.registerCommand(
      "evidentlyWorkspace.selectRightTable",
      async (item?: EvidentlyTreeItem) => {
        const target = await resolveDatasetArg(item);
        if (!target) {
          return;
        }
        await openJoinStudioWithSelection(
          "right",
          target.project,
          target.dataset
        );
      }
    ),

    vscode.commands.registerCommand(
      "evidentlyWorkspace.createReportFromDataset",
      async (item?: EvidentlyTreeItem) => {
        const target = await resolveDatasetArg(item);
        if (!target) {
          return;
        }
        await openJoinStudioWithSelection(
          "left",
          target.project,
          target.dataset
        );
      }
    ),

    vscode.commands.registerCommand(
      "evidentlyWorkspace.exportDatasetCsv",
      async (item?: EvidentlyTreeItem) => {
        const target = await resolveDatasetArg(item);
        if (!target) {
          return;
        }
        try {
          const safeName = target.dataset.name.replace(
            /[^a-zA-Z0-9._-]+/g,
            "_"
          );
          const uri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file(
              `${safeName}_${target.dataset.id.slice(0, 8)}.csv`
            ),
            filters: { "CSV Files": ["csv"] },
          });
          if (!uri) {
            return;
          }
          const csv = await client.downloadDatasetCsv(target.dataset.id);
          fs.writeFileSync(uri.fsPath, csv, "utf8");
          vscode.window.showInformationMessage(
            `Exported dataset "${target.dataset.name}" to ${uri.fsPath}`
          );
        } catch (err: any) {
          vscode.window.showErrorMessage(
            `Failed to export dataset CSV: ${err?.message || err}`
          );
        }
      }
    ),

    vscode.commands.registerCommand(
      "evidentlyWorkspace.exportReportHtml",
      async (item?: EvidentlyTreeItem) => {
        const target = await resolveReportArg(item);
        if (!target) {
          return;
        }
        try {
          const safeName = getReportDisplayName(target.report).replace(
            /[^a-zA-Z0-9._-]+/g,
            "_"
          );
          const uri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file(
              `${safeName}_${target.report.id.slice(0, 8)}.html`
            ),
            filters: { "HTML Files": ["html"] },
          });
          if (!uri) {
            return;
          }
          const html = await client.downloadReportHtml(
            target.project.id,
            target.report.id
          );
          fs.writeFileSync(uri.fsPath, html, "utf8");
          vscode.window.showInformationMessage(
            `Exported report HTML to ${uri.fsPath}`
          );
        } catch (err: any) {
          vscode.window.showErrorMessage(
            `Failed to export report HTML: ${err?.message || err}`
          );
        }
      }
    ),

    vscode.commands.registerCommand(
      "evidentlyWorkspace.openInBrowser",
      async (item?: EvidentlyTreeItem) => {
        const base = client.getBaseUrl();
        let url = base;
        if (item?.data?.project && item?.data?.report) {
          url = `${base}/projects/${item.data.project.id}/reports/${item.data.report.id}`;
        } else if (item?.data?.project && item?.data?.dataset) {
          url = `${base}/projects/${item.data.project.id}/datasets/${item.data.dataset.id}`;
        } else if (item?.data?.project) {
          url = `${base}/projects/${item.data.project.id}`;
        }
        await vscode.env.openExternal(vscode.Uri.parse(url));
      }
    )
  );
}

export function deactivate(): void {
  // No-op
}
