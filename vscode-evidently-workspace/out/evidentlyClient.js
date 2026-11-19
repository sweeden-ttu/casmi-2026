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
exports.EvidentlyClient = void 0;
const http = __importStar(require("http"));
const https = __importStar(require("https"));
const url_1 = require("url");
class EvidentlyClient {
    baseUrl;
    apiKey;
    bearerToken = null;
    timeoutMs;
    constructor(baseUrl, apiKey = "", timeoutMs = 30000) {
        this.baseUrl = baseUrl.replace(/\/+$/, "");
        this.apiKey = apiKey.trim();
        this.timeoutMs = timeoutMs;
    }
    getBaseUrl() {
        return this.baseUrl;
    }
    setConfig(baseUrl, apiKey) {
        const cleanUrl = baseUrl.replace(/\/+$/, "");
        if (cleanUrl !== this.baseUrl || apiKey.trim() !== this.apiKey) {
            this.baseUrl = cleanUrl;
            this.apiKey = apiKey.trim();
            this.bearerToken = null;
        }
    }
    rawRequest(method, pathWithQuery, headers = {}, body) {
        return new Promise((resolve, reject) => {
            const targetUrl = new url_1.URL(pathWithQuery.startsWith("http")
                ? pathWithQuery
                : `${this.baseUrl}${pathWithQuery.startsWith("/") ? "" : "/"}${pathWithQuery}`);
            const isHttps = targetUrl.protocol === "https:";
            const transport = isHttps ? https : http;
            const reqHeaders = {
                Accept: "application/json, text/plain, */*",
                ...headers,
            };
            if (body !== undefined && !reqHeaders["Content-Length"]) {
                reqHeaders["Content-Length"] = String(Buffer.byteLength(body));
            }
            const req = transport.request(targetUrl, {
                method,
                headers: reqHeaders,
                timeout: this.timeoutMs,
            }, (res) => {
                const chunks = [];
                res.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
                res.on("end", () => {
                    const fullBody = Buffer.concat(chunks).toString("utf8");
                    resolve({
                        statusCode: res.statusCode || 0,
                        headers: res.headers,
                        body: fullBody,
                    });
                });
            });
            req.on("timeout", () => {
                req.destroy(new Error(`Request to ${targetUrl.toString()} timed out after ${this.timeoutMs}ms`));
            });
            req.on("error", (err) => reject(err));
            if (body !== undefined) {
                req.write(body);
            }
            req.end();
        });
    }
    async login(force = false) {
        if (this.bearerToken && !force) {
            return this.bearerToken;
        }
        if (!this.apiKey) {
            return null;
        }
        try {
            const resp = await this.rawRequest("GET", "/api/users/login", {
                "X-Evidently-Token": this.apiKey,
            });
            if (resp.statusCode >= 200 && resp.statusCode < 300 && resp.body.trim()) {
                this.bearerToken = resp.body.trim().replace(/^"|"$/g, "");
                return this.bearerToken;
            }
        }
        catch {
            // Server may not require login or may be unreachable; caller will see error on actual endpoint
        }
        return null;
    }
    async request(method, pathWithQuery, extraHeaders = {}, body, retried = false) {
        const token = await this.login(false);
        const headers = { ...extraHeaders };
        if (token) {
            headers["Authorization"] = `Bearer ${token}`;
        }
        else if (this.apiKey) {
            headers["X-Evidently-Token"] = this.apiKey;
            headers["evidently-secret"] = this.apiKey;
        }
        const resp = await this.rawRequest(method, pathWithQuery, headers, body);
        if ((resp.statusCode === 401 || resp.statusCode === 403) && !retried && this.apiKey) {
            await this.login(true);
            return this.request(method, pathWithQuery, extraHeaders, body, true);
        }
        return resp;
    }
    async requestJson(method, pathWithQuery, extraHeaders = {}, body) {
        const payload = body !== undefined
            ? typeof body === "string"
                ? body
                : JSON.stringify(body)
            : undefined;
        const headers = { ...extraHeaders };
        if (payload !== undefined && !headers["Content-Type"]) {
            headers["Content-Type"] = "application/json";
        }
        const resp = await this.request(method, pathWithQuery, headers, payload);
        if (resp.statusCode < 200 || resp.statusCode >= 300) {
            throw new Error(`HTTP ${resp.statusCode} on ${method} ${pathWithQuery}: ${resp.body.slice(0, 300)}`);
        }
        if (!resp.body || !resp.body.trim()) {
            throw new Error(`Empty JSON response from ${method} ${pathWithQuery}`);
        }
        return JSON.parse(resp.body);
    }
    async getVersion() {
        return this.requestJson("GET", "/api/version");
    }
    async listProjects() {
        try {
            const projects = await this.requestJson("GET", "/api/v2/projects");
            return Array.isArray(projects) ? projects : [];
        }
        catch {
            const projects = await this.requestJson("GET", "/api/projects");
            return Array.isArray(projects) ? projects : [];
        }
    }
    async listDatasets(projectId) {
        const params = new URLSearchParams({ project_id: projectId });
        const data = await this.requestJson("GET", `/api/datasets?${params.toString()}`);
        const list = Array.isArray(data?.datasets) ? data.datasets : [];
        return list.sort((a, b) => {
            const ta = a.created_at ? Date.parse(a.created_at) : 0;
            const tb = b.created_at ? Date.parse(b.created_at) : 0;
            return tb - ta;
        });
    }
    async getDatasetPage(datasetId, options = {}) {
        const params = new URLSearchParams();
        params.set("page_size", String(options.pageSize ?? 50));
        params.set("current_page", String(options.currentPage ?? 1));
        if (options.sortByColumn) {
            params.set("sort_by_column", options.sortByColumn);
            params.set("sort_ascending", String(options.sortAscending ?? true));
        }
        if (options.filters && options.filters.length > 0) {
            for (const f of options.filters) {
                params.append("filters", f);
            }
        }
        return this.requestJson("GET", `/api/datasets/${encodeURIComponent(datasetId)}?${params.toString()}`);
    }
    async getDatasetSchema(datasetId) {
        let dataDef = {};
        let allColumns = [];
        try {
            const ddResp = await this.requestJson("GET", `/api/datasets/${encodeURIComponent(datasetId)}/data_definition`);
            dataDef = ddResp.data_definition || {};
            allColumns = Array.isArray(ddResp.all_columns) ? ddResp.all_columns : [];
        }
        catch {
            // Fallback to getDatasetPage below
        }
        // Also fetch a 1-row page when allColumns is empty (e.g. tracing dataset) or to get exact column types & row count
        try {
            const page = await this.getDatasetPage(datasetId, { pageSize: 1, currentPage: 1 });
            const cols = Array.isArray(page?.metadata?.columns) ? page.metadata.columns : [];
            if (cols.length > 0) {
                return {
                    datasetId,
                    columns: cols,
                    columnNames: cols.map((c) => c.name),
                    rowCount: page.metadata?.row_count ?? 0,
                    dataDefinition: page.metadata?.data_definition || dataDef,
                };
            }
        }
        catch {
            // Ignore page fallback error if data_definition already succeeded
        }
        const numCols = new Set(dataDef.numerical_columns || []);
        const dtCols = new Set(dataDef.datetime_columns || []);
        const columns = allColumns.map((name) => ({
            name,
            type: numCols.has(name) ? "numerical" : dtCols.has(name) ? "datetime" : "string",
            is_index: dataDef.id_column === name,
        }));
        return {
            datasetId,
            columns,
            columnNames: allColumns,
            rowCount: 0,
            dataDefinition: dataDef,
        };
    }
    async downloadDatasetCsv(datasetId) {
        const resp = await this.request("GET", `/api/datasets/${encodeURIComponent(datasetId)}/download?format=csv`);
        if (resp.statusCode < 200 || resp.statusCode >= 300) {
            throw new Error(`Failed to download CSV for dataset ${datasetId}: HTTP ${resp.statusCode}`);
        }
        return resp.body;
    }
    async listReports(projectId) {
        // Try /snapshots first so we get dataset links, then fall back to /reports
        for (const suffix of ["snapshots", "reports"]) {
            try {
                const resp = await this.request("GET", `/api/projects/${encodeURIComponent(projectId)}/${suffix}`);
                if (resp.statusCode >= 200 && resp.statusCode < 300) {
                    const parsed = JSON.parse(resp.body);
                    if (Array.isArray(parsed)) {
                        const reports = parsed.map((item) => ({
                            id: String(item.id),
                            name: item.name ?? null,
                            timestamp: String(item.timestamp || ""),
                            metadata: item.metadata || {},
                            tags: Array.isArray(item.tags) ? item.tags : [],
                            links: item.links || undefined,
                        }));
                        reports.sort((a, b) => {
                            const ta = a.timestamp ? Date.parse(a.timestamp) : 0;
                            const tb = b.timestamp ? Date.parse(b.timestamp) : 0;
                            return tb - ta;
                        });
                        return { reports };
                    }
                }
                else if (resp.statusCode >= 500) {
                    // Do not crash on projects with broken snapshot storage on the server
                    return {
                        reports: [],
                        error: `Server returned HTTP ${resp.statusCode} when listing reports for this project.`,
                    };
                }
            }
            catch (err) {
                if (suffix === "reports") {
                    return {
                        reports: [],
                        error: err?.message || String(err),
                    };
                }
            }
        }
        return { reports: [] };
    }
    async getReportMetadata(projectId, snapshotId) {
        return this.requestJson("GET", `/api/projects/${encodeURIComponent(projectId)}/${encodeURIComponent(snapshotId)}/metadata`);
    }
    async getReportData(projectId, snapshotId) {
        const raw = await this.requestJson("GET", `/api/projects/${encodeURIComponent(projectId)}/${encodeURIComponent(snapshotId)}/data`);
        if (typeof raw === "string") {
            return JSON.parse(raw);
        }
        return raw;
    }
    async downloadReportJson(projectId, snapshotId) {
        const raw = await this.requestJson("GET", `/api/projects/${encodeURIComponent(projectId)}/${encodeURIComponent(snapshotId)}/download?report_format=json`);
        if (typeof raw === "string") {
            return JSON.parse(raw);
        }
        return {
            metrics: Array.isArray(raw?.metrics) ? raw.metrics : [],
            tests: Array.isArray(raw?.tests) ? raw.tests : [],
        };
    }
    async downloadReportHtml(projectId, snapshotId) {
        const resp = await this.request("GET", `/api/projects/${encodeURIComponent(projectId)}/${encodeURIComponent(snapshotId)}/download?report_format=html`);
        if (resp.statusCode < 200 || resp.statusCode >= 300) {
            throw new Error(`Failed to download report HTML (${projectId}/${snapshotId}): HTTP ${resp.statusCode}`);
        }
        return resp.body;
    }
}
exports.EvidentlyClient = EvidentlyClient;
//# sourceMappingURL=evidentlyClient.js.map