import * as fs from "fs";
import * as os from "os";
import * as path from "path";

export interface EvidentlyWorkspaceConfig {
  serverUrl: string;
  apiKey: string;
  apiKeyPath: string;
  pythonPath: string;
  defaultPageSize: number;
  groupDatasetVersions: boolean;
}

function expandHome(filepath: string): string {
  if (!filepath) {
    return filepath;
  }
  if (filepath === "~") {
    return os.homedir();
  }
  if (filepath.startsWith("~/") || filepath.startsWith("~\\")) {
    return path.join(os.homedir(), filepath.slice(2));
  }
  return filepath;
}

function readEnvTokenFromDotEnv(workspaceRoots: string[]): string {
  for (const root of workspaceRoots) {
    const envFile = path.join(root, ".env");
    try {
      if (fs.existsSync(envFile)) {
        const text = fs.readFileSync(envFile, "utf8");
        for (const rawLine of text.split(/\r?\n/)) {
          const line = rawLine.trim();
          if (!line || line.startsWith("#")) {
            continue;
          }
          const match = line.match(/^EVIDENTLY_TOKEN\s*=\s*(.+)$/);
          if (match) {
            const val = match[1].trim().replace(/^["']|["']$/g, "");
            if (val) {
              return val;
            }
          }
        }
      }
    } catch {
      // ignore read errors
    }
  }
  return "";
}

export function resolveApiKey(
  explicitKey: string,
  configuredKeyPath: string,
  workspaceRoots: string[] = []
): string {
  if (explicitKey && explicitKey.trim()) {
    return explicitKey.trim();
  }
  if (process.env.EVIDENTLY_TOKEN && process.env.EVIDENTLY_TOKEN.trim()) {
    return process.env.EVIDENTLY_TOKEN.trim();
  }
  const candidatePaths = [
    expandHome(configuredKeyPath || "~/.config/evidently/api_key"),
    path.join(os.homedir(), ".config", "evidently", "api_key"),
    path.join(os.homedir(), ".config", "evidently", "token"),
  ];
  for (const p of candidatePaths) {
    try {
      if (p && fs.existsSync(p)) {
        const content = fs.readFileSync(p, "utf8").trim();
        if (content) {
          return content;
        }
      }
    } catch {
      // ignore
    }
  }
  const dotEnvToken = readEnvTokenFromDotEnv(workspaceRoots);
  if (dotEnvToken) {
    return dotEnvToken;
  }
  return "";
}

export function resolvePythonPath(
  configuredPythonPath: string,
  workspaceRoots: string[] = []
): string {
  const candidates: string[] = [];
  if (configuredPythonPath && configuredPythonPath.trim()) {
    candidates.push(expandHome(configuredPythonPath.trim()));
  }
  for (const root of workspaceRoots) {
    candidates.push(path.join(root, ".venv", "bin", "python"));
  }
  candidates.push("/mnt/LMSTUDIO/CASMI/.venv/bin/python");
  for (const candidate of candidates) {
    try {
      if (candidate && fs.existsSync(candidate)) {
        return candidate;
      }
    } catch {
      // ignore
    }
  }
  return configuredPythonPath || "python3";
}

export function getWorkspaceConfig(
  overrides: Partial<EvidentlyWorkspaceConfig> = {}
): EvidentlyWorkspaceConfig {
  let vsConfig: any = null;
  const workspaceRoots: string[] = [];
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const vscode = require("vscode");
    vsConfig = vscode.workspace.getConfiguration("evidentlyWorkspace");
    const folders = vscode.workspace.workspaceFolders || [];
    for (const f of folders) {
      if (f?.uri?.fsPath) {
        workspaceRoots.push(f.uri.fsPath);
      }
    }
  } catch {
    // Running outside VS Code extension host (e.g. unit/integration tests)
  }
  if (workspaceRoots.length === 0) {
    workspaceRoots.push("/mnt/LMSTUDIO/CASMI", process.cwd());
  }

  const rawServerUrl =
    overrides.serverUrl ??
    process.env.EVIDENTLY_URL ??
    (vsConfig ? vsConfig.get("serverUrl", "http://192.168.0.15:8000") : "http://192.168.0.15:8000");

  const serverUrl = String(rawServerUrl || "http://192.168.0.15:8000").replace(/\/+$/, "");

  const explicitApiKey =
    overrides.apiKey ?? (vsConfig ? vsConfig.get("apiKey", "") : "");

  const apiKeyPath =
    overrides.apiKeyPath ??
    (vsConfig ? vsConfig.get("apiKeyPath", "~/.config/evidently/api_key") : "~/.config/evidently/api_key");

  const configuredPython =
    overrides.pythonPath ??
    (vsConfig
      ? vsConfig.get("pythonPath", "/mnt/LMSTUDIO/CASMI/.venv/bin/python")
      : "/mnt/LMSTUDIO/CASMI/.venv/bin/python");

  const defaultPageSize =
    overrides.defaultPageSize ??
    (vsConfig ? Number(vsConfig.get("defaultPageSize", 50)) || 50 : 50);

  const groupDatasetVersions =
    overrides.groupDatasetVersions ??
    (vsConfig ? Boolean(vsConfig.get("groupDatasetVersions", true)) : true);

  return {
    serverUrl,
    apiKey: resolveApiKey(explicitApiKey, apiKeyPath, workspaceRoots),
    apiKeyPath,
    pythonPath: resolvePythonPath(configuredPython, workspaceRoots),
    defaultPageSize: Math.max(5, Math.min(500, defaultPageSize)),
    groupDatasetVersions,
  };
}
