import { spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { ColumnProfile, JoinDiagnostics, JoinType, KeyMatchMode } from "./joinEngine";

export interface AdditionalJoinSpec {
  dataset_id?: string;
  csv_path?: string;
  csv_text?: string;
  join_type?: JoinType;
  left_on: string[];
  right_on: string[];
  match_mode?: KeyMatchMode;
  suffixes?: [string, string];
}

export interface BridgeReportSpec {
  mode?: "preview_join" | "create_report" | "delete_artifacts";
  server_url: string;
  api_key?: string;
  left_dataset_id?: string;
  right_dataset_id?: string;
  left_csv_path?: string;
  right_csv_path?: string;
  left_csv_text?: string;
  right_csv_text?: string;
  join_type?: JoinType;
  left_on?: string[];
  right_on?: string[];
  match_mode?: KeyMatchMode;
  suffixes?: [string, string];
  selected_left_columns?: string[];
  selected_right_columns?: string[];
  drop_null_keys?: boolean;
  additional_joins?: AdditionalJoinSpec[];
  target_project_id?: string;
  report_name?: string;
  tags?: string[];
  metadata?: Record<string, string>;
  presets?: string[];
  target_column?: string;
  prediction_column?: string;
  metric_columns?: string[];
  include_correlations?: boolean;
  include_tests?: boolean;
  reference_source?: "" | "left" | "right" | "split_half" | "dataset";
  reference_dataset_id?: string;
  save_joined_dataset?: boolean;
  joined_dataset_name?: string;
  joined_dataset_description?: string;
  publish_to_workspace?: boolean;
  html_output_path?: string;
  max_rows?: number;
  preview_limit?: number;
  // For delete_artifacts
  project_id?: string;
  snapshot_ids?: string[];
  dataset_ids?: string[];
}

export interface BridgeReportResult {
  status: "ok" | "error";
  mode?: string;
  error?: string;
  traceback?: string;
  reportName?: string;
  projectId?: string | null;
  snapshotId?: string | null;
  joinedDatasetId?: string | null;
  reportUrl?: string | null;
  htmlPath?: string;
  columns?: string[];
  diagnostics?: JoinDiagnostics;
  columnProfiles?: ColumnProfile[];
  rows?: Record<string, any>[];
  metricsSummary?: {
    metrics: any[];
    tests: any[];
  };
  deletedSnapshots?: string[];
  deletedDatasets?: string[];
}

export function resolveBridgeScriptPath(extensionPath?: string): string {
  const candidates: string[] = [];
  if (extensionPath) {
    candidates.push(path.join(extensionPath, "python", "evidently_bridge.py"));
  }
  candidates.push(path.resolve(__dirname, "..", "python", "evidently_bridge.py"));
  candidates.push(
    "/mnt/LMSTUDIO/CASMI/vscode-evidently-workspace/python/evidently_bridge.py"
  );
  for (const c of candidates) {
    if (fs.existsSync(c)) {
      return c;
    }
  }
  throw new Error(
    `Could not locate python/evidently_bridge.py (checked: ${candidates.join(", ")})`
  );
}

export function runEvidentlyBridge(
  pythonPath: string,
  spec: BridgeReportSpec,
  extensionPath?: string,
  timeoutMs: number = 120000
): Promise<BridgeReportResult> {
  return new Promise((resolve, reject) => {
    let scriptPath: string;
    try {
      scriptPath = resolveBridgeScriptPath(extensionPath);
    } catch (err) {
      return reject(err);
    }

    const child = spawn(pythonPath, [scriptPath, "-"], {
      stdio: ["pipe", "pipe", "pipe"],
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
      },
    });

    const stdoutChunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];

    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`Evidently Python bridge timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    child.stdout.on("data", (chunk: Buffer) => stdoutChunks.push(Buffer.from(chunk)));
    child.stderr.on("data", (chunk: Buffer) => stderrChunks.push(Buffer.from(chunk)));

    child.on("error", (err) => {
      clearTimeout(timer);
      reject(new Error(`Failed to start Python (${pythonPath}): ${err.message}`));
    });

    child.on("close", (code) => {
      clearTimeout(timer);
      const stdout = Buffer.concat(stdoutChunks).toString("utf8").trim();
      const stderr = Buffer.concat(stderrChunks).toString("utf8").trim();

      // Find the last JSON object line in stdout
      const lines = stdout.split(/\r?\n/).filter((l) => l.trim().startsWith("{"));
      const jsonLine = lines.length > 0 ? lines[lines.length - 1] : stdout;

      try {
        const parsed = JSON.parse(jsonLine) as BridgeReportResult;
        if (parsed.status === "error" || code !== 0) {
          reject(
            new Error(
              parsed.error ||
                `Evidently bridge exited with code ${code}: ${stderr || stdout}`
            )
          );
          return;
        }
        resolve(parsed);
      } catch {
        reject(
          new Error(
            `Failed to parse Evidently bridge output (exit code ${code}): ${
              stderr || stdout
            }`
          )
        );
      }
    });

    child.stdin.write(JSON.stringify(spec));
    child.stdin.end();
  });
}
