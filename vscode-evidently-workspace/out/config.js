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
exports.resolveApiKey = resolveApiKey;
exports.resolvePythonPath = resolvePythonPath;
exports.getWorkspaceConfig = getWorkspaceConfig;
const fs = __importStar(require("fs"));
const os = __importStar(require("os"));
const path = __importStar(require("path"));
function expandHome(filepath) {
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
function readEnvTokenFromDotEnv(workspaceRoots) {
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
        }
        catch {
            // ignore read errors
        }
    }
    return "";
}
function resolveApiKey(explicitKey, configuredKeyPath, workspaceRoots = []) {
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
        }
        catch {
            // ignore
        }
    }
    const dotEnvToken = readEnvTokenFromDotEnv(workspaceRoots);
    if (dotEnvToken) {
        return dotEnvToken;
    }
    return "";
}
function resolvePythonPath(configuredPythonPath, workspaceRoots = []) {
    const candidates = [];
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
        }
        catch {
            // ignore
        }
    }
    return configuredPythonPath || "python3";
}
function getWorkspaceConfig(overrides = {}) {
    let vsConfig = null;
    const workspaceRoots = [];
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
    }
    catch {
        // Running outside VS Code extension host (e.g. unit/integration tests)
    }
    if (workspaceRoots.length === 0) {
        workspaceRoots.push("/mnt/LMSTUDIO/CASMI", process.cwd());
    }
    const rawServerUrl = overrides.serverUrl ??
        process.env.EVIDENTLY_URL ??
        (vsConfig ? vsConfig.get("serverUrl", "http://192.168.0.15:8000") : "http://192.168.0.15:8000");
    const serverUrl = String(rawServerUrl || "http://192.168.0.15:8000").replace(/\/+$/, "");
    const explicitApiKey = overrides.apiKey ?? (vsConfig ? vsConfig.get("apiKey", "") : "");
    const apiKeyPath = overrides.apiKeyPath ??
        (vsConfig ? vsConfig.get("apiKeyPath", "~/.config/evidently/api_key") : "~/.config/evidently/api_key");
    const configuredPython = overrides.pythonPath ??
        (vsConfig
            ? vsConfig.get("pythonPath", "/mnt/LMSTUDIO/CASMI/.venv/bin/python")
            : "/mnt/LMSTUDIO/CASMI/.venv/bin/python");
    const defaultPageSize = overrides.defaultPageSize ??
        (vsConfig ? Number(vsConfig.get("defaultPageSize", 50)) || 50 : 50);
    const groupDatasetVersions = overrides.groupDatasetVersions ??
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
//# sourceMappingURL=config.js.map