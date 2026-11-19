import * as http from "http";
import * as https from "https";
import { URL } from "url";

export interface EvidentlyProject {
  id: string;
  name: string;
  description?: string | null;
  created_at?: string | null;
  version?: string;
}

export interface EvidentlyDatasetInfo {
  id: string;
  project_id: string;
  name: string;
  description: string;
  size_bytes: number;
  created_at: string;
  author_name?: string;
  row_count: number;
  column_count: number;
  origin: string;
  metadata: Record<string, any>;
  tags: string[];
}

export interface EvidentlyDatasetColumn {
  name: string;
  type: "numerical" | "string" | "datetime" | string;
  is_index?: boolean;
}

export interface EvidentlyDatasetPage {
  items: any[][];
  metadata: {
    name: string;
    description: string;
    row_count: number;
    data_definition?: Record<string, any>;
    columns: EvidentlyDatasetColumn[];
  };
  page_size: number;
  current_page: number;
  total_pages: number;
}

export interface EvidentlyDatasetSchema {
  datasetId: string;
  columns: EvidentlyDatasetColumn[];
  columnNames: string[];
  rowCount: number;
  dataDefinition: Record<string, any>;
}

export interface EvidentlySnapshotLinks {
  datasets?: {
    input?: {
      reference?: string | null;
      current?: string | null;
      additional?: Record<string, string>;
    };
    output?: {
      reference?: string | null;
      current?: string | null;
      additional?: Record<string, string>;
    };
  };
}

export interface EvidentlyReportInfo {
  id: string;
  name?: string | null;
  timestamp: string;
  metadata: Record<string, any>;
  tags: string[];
  links?: EvidentlySnapshotLinks;
}

export interface ListReportsResult {
  reports: EvidentlyReportInfo[];
  error?: string;
}

export interface HttpResponseData {
  statusCode: number;
  headers: http.IncomingHttpHeaders;
  body: string;
}

export class EvidentlyClient {
  private baseUrl: string;
  private apiKey: string;
  private bearerToken: string | null = null;
  private timeoutMs: number;

  constructor(baseUrl: string, apiKey: string = "", timeoutMs: number = 30000) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.apiKey = apiKey.trim();
    this.timeoutMs = timeoutMs;
  }

  public getBaseUrl(): string {
    return this.baseUrl;
  }

  public setConfig(baseUrl: string, apiKey: string): void {
    const cleanUrl = baseUrl.replace(/\/+$/, "");
    if (cleanUrl !== this.baseUrl || apiKey.trim() !== this.apiKey) {
      this.baseUrl = cleanUrl;
      this.apiKey = apiKey.trim();
      this.bearerToken = null;
    }
  }

  private rawRequest(
    method: string,
    pathWithQuery: string,
    headers: Record<string, string> = {},
    body?: string | Buffer
  ): Promise<HttpResponseData> {
    return new Promise((resolve, reject) => {
      const targetUrl = new URL(
        pathWithQuery.startsWith("http")
          ? pathWithQuery
          : `${this.baseUrl}${pathWithQuery.startsWith("/") ? "" : "/"}${pathWithQuery}`
      );
      const isHttps = targetUrl.protocol === "https:";
      const transport = isHttps ? https : http;

      const reqHeaders: Record<string, string> = {
        Accept: "application/json, text/plain, */*",
        ...headers,
      };
      if (body !== undefined && !reqHeaders["Content-Length"]) {
        reqHeaders["Content-Length"] = String(Buffer.byteLength(body));
      }

      const req = transport.request(
        targetUrl,
        {
          method,
          headers: reqHeaders,
          timeout: this.timeoutMs,
        },
        (res) => {
          const chunks: Buffer[] = [];
          res.on("data", (chunk: Buffer) => chunks.push(Buffer.from(chunk)));
          res.on("end", () => {
            const fullBody = Buffer.concat(chunks).toString("utf8");
            resolve({
              statusCode: res.statusCode || 0,
              headers: res.headers,
              body: fullBody,
            });
          });
        }
      );

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

  public async login(force: boolean = false): Promise<string | null> {
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
    } catch {
      // Server may not require login or may be unreachable; caller will see error on actual endpoint
    }
    return null;
  }

  private async request(
    method: string,
    pathWithQuery: string,
    extraHeaders: Record<string, string> = {},
    body?: string | Buffer,
    retried: boolean = false
  ): Promise<HttpResponseData> {
    const token = await this.login(false);
    const headers: Record<string, string> = { ...extraHeaders };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    } else if (this.apiKey) {
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

  private async requestJson<T>(
    method: string,
    pathWithQuery: string,
    extraHeaders: Record<string, string> = {},
    body?: any
  ): Promise<T> {
    const payload =
      body !== undefined
        ? typeof body === "string"
          ? body
          : JSON.stringify(body)
        : undefined;
    const headers: Record<string, string> = { ...extraHeaders };
    if (payload !== undefined && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const resp = await this.request(method, pathWithQuery, headers, payload);
    if (resp.statusCode < 200 || resp.statusCode >= 300) {
      throw new Error(
        `HTTP ${resp.statusCode} on ${method} ${pathWithQuery}: ${resp.body.slice(0, 300)}`
      );
    }
    if (!resp.body || !resp.body.trim()) {
      throw new Error(`Empty JSON response from ${method} ${pathWithQuery}`);
    }
    return JSON.parse(resp.body) as T;
  }

  public async getVersion(): Promise<{ application: string; version: string; commit?: string }> {
    return this.requestJson("GET", "/api/version");
  }

  public async listProjects(): Promise<EvidentlyProject[]> {
    try {
      const projects = await this.requestJson<EvidentlyProject[]>("GET", "/api/v2/projects");
      return Array.isArray(projects) ? projects : [];
    } catch {
      const projects = await this.requestJson<EvidentlyProject[]>("GET", "/api/projects");
      return Array.isArray(projects) ? projects : [];
    }
  }

  public async listDatasets(projectId: string): Promise<EvidentlyDatasetInfo[]> {
    const params = new URLSearchParams({ project_id: projectId });
    const data = await this.requestJson<{ datasets?: EvidentlyDatasetInfo[] }>(
      "GET",
      `/api/datasets?${params.toString()}`
    );
    const list = Array.isArray(data?.datasets) ? data.datasets : [];
    return list.sort((a, b) => {
      const ta = a.created_at ? Date.parse(a.created_at) : 0;
      const tb = b.created_at ? Date.parse(b.created_at) : 0;
      return tb - ta;
    });
  }

  public async getDatasetPage(
    datasetId: string,
    options: {
      pageSize?: number;
      currentPage?: number;
      sortByColumn?: string;
      sortAscending?: boolean;
      filters?: string[];
    } = {}
  ): Promise<EvidentlyDatasetPage> {
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
    return this.requestJson<EvidentlyDatasetPage>(
      "GET",
      `/api/datasets/${encodeURIComponent(datasetId)}?${params.toString()}`
    );
  }

  public async getDatasetSchema(datasetId: string): Promise<EvidentlyDatasetSchema> {
    let dataDef: Record<string, any> = {};
    let allColumns: string[] = [];
    try {
      const ddResp = await this.requestJson<{
        data_definition?: Record<string, any>;
        all_columns?: string[];
      }>("GET", `/api/datasets/${encodeURIComponent(datasetId)}/data_definition`);
      dataDef = ddResp.data_definition || {};
      allColumns = Array.isArray(ddResp.all_columns) ? ddResp.all_columns : [];
    } catch {
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
    } catch {
      // Ignore page fallback error if data_definition already succeeded
    }

    const numCols = new Set<string>(dataDef.numerical_columns || []);
    const dtCols = new Set<string>(dataDef.datetime_columns || []);
    const columns: EvidentlyDatasetColumn[] = allColumns.map((name) => ({
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

  public async downloadDatasetCsv(datasetId: string): Promise<string> {
    const resp = await this.request(
      "GET",
      `/api/datasets/${encodeURIComponent(datasetId)}/download?format=csv`
    );
    if (resp.statusCode < 200 || resp.statusCode >= 300) {
      throw new Error(
        `Failed to download CSV for dataset ${datasetId}: HTTP ${resp.statusCode}`
      );
    }
    return resp.body;
  }

  public async listReports(projectId: string): Promise<ListReportsResult> {
    // Try /snapshots first so we get dataset links, then fall back to /reports
    for (const suffix of ["snapshots", "reports"]) {
      try {
        const resp = await this.request(
          "GET",
          `/api/projects/${encodeURIComponent(projectId)}/${suffix}`
        );
        if (resp.statusCode >= 200 && resp.statusCode < 300) {
          const parsed = JSON.parse(resp.body);
          if (Array.isArray(parsed)) {
            const reports: EvidentlyReportInfo[] = parsed.map((item: any) => ({
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
        } else if (resp.statusCode >= 500) {
          // Do not crash on projects with broken snapshot storage on the server
          return {
            reports: [],
            error: `Server returned HTTP ${resp.statusCode} when listing reports for this project.`,
          };
        }
      } catch (err: any) {
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

  public async getReportMetadata(
    projectId: string,
    snapshotId: string
  ): Promise<EvidentlyReportInfo> {
    return this.requestJson<EvidentlyReportInfo>(
      "GET",
      `/api/projects/${encodeURIComponent(projectId)}/${encodeURIComponent(snapshotId)}/metadata`
    );
  }

  public async getReportData(
    projectId: string,
    snapshotId: string
  ): Promise<{ name?: string; widgets: any[] }> {
    const raw = await this.requestJson<any>(
      "GET",
      `/api/projects/${encodeURIComponent(projectId)}/${encodeURIComponent(snapshotId)}/data`
    );
    if (typeof raw === "string") {
      return JSON.parse(raw);
    }
    return raw;
  }

  public async downloadReportJson(
    projectId: string,
    snapshotId: string
  ): Promise<{ metrics: any[]; tests: any[] }> {
    const raw = await this.requestJson<any>(
      "GET",
      `/api/projects/${encodeURIComponent(projectId)}/${encodeURIComponent(
        snapshotId
      )}/download?report_format=json`
    );
    if (typeof raw === "string") {
      return JSON.parse(raw);
    }
    return {
      metrics: Array.isArray(raw?.metrics) ? raw.metrics : [],
      tests: Array.isArray(raw?.tests) ? raw.tests : [],
    };
  }

  public async downloadReportHtml(
    projectId: string,
    snapshotId: string
  ): Promise<string> {
    const resp = await this.request(
      "GET",
      `/api/projects/${encodeURIComponent(projectId)}/${encodeURIComponent(
        snapshotId
      )}/download?report_format=html`
    );
    if (resp.statusCode < 200 || resp.statusCode >= 300) {
      throw new Error(
        `Failed to download report HTML (${projectId}/${snapshotId}): HTTP ${resp.statusCode}`
      );
    }
    return resp.body;
  }
}
