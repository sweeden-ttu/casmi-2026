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
exports.resolveBridgeScriptPath = resolveBridgeScriptPath;
exports.runEvidentlyBridge = runEvidentlyBridge;
const child_process_1 = require("child_process");
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
function resolveBridgeScriptPath(extensionPath) {
    const candidates = [];
    if (extensionPath) {
        candidates.push(path.join(extensionPath, "python", "evidently_bridge.py"));
    }
    candidates.push(path.resolve(__dirname, "..", "python", "evidently_bridge.py"));
    candidates.push("/mnt/LMSTUDIO/CASMI/vscode-evidently-workspace/python/evidently_bridge.py");
    for (const c of candidates) {
        if (fs.existsSync(c)) {
            return c;
        }
    }
    throw new Error(`Could not locate python/evidently_bridge.py (checked: ${candidates.join(", ")})`);
}
function runEvidentlyBridge(pythonPath, spec, extensionPath, timeoutMs = 120000) {
    return new Promise((resolve, reject) => {
        let scriptPath;
        try {
            scriptPath = resolveBridgeScriptPath(extensionPath);
        }
        catch (err) {
            return reject(err);
        }
        const child = (0, child_process_1.spawn)(pythonPath, [scriptPath, "-"], {
            stdio: ["pipe", "pipe", "pipe"],
            env: {
                ...process.env,
                PYTHONUNBUFFERED: "1",
            },
        });
        const stdoutChunks = [];
        const stderrChunks = [];
        const timer = setTimeout(() => {
            child.kill("SIGKILL");
            reject(new Error(`Evidently Python bridge timed out after ${timeoutMs}ms`));
        }, timeoutMs);
        child.stdout.on("data", (chunk) => stdoutChunks.push(Buffer.from(chunk)));
        child.stderr.on("data", (chunk) => stderrChunks.push(Buffer.from(chunk)));
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
                const parsed = JSON.parse(jsonLine);
                if (parsed.status === "error" || code !== 0) {
                    reject(new Error(parsed.error ||
                        `Evidently bridge exited with code ${code}: ${stderr || stdout}`));
                    return;
                }
                resolve(parsed);
            }
            catch {
                reject(new Error(`Failed to parse Evidently bridge output (exit code ${code}): ${stderr || stdout}`));
            }
        });
        child.stdin.write(JSON.stringify(spec));
        child.stdin.end();
    });
}
//# sourceMappingURL=bridgeRunner.js.map