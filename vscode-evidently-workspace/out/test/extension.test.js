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
const assert = __importStar(require("assert"));
const fs = __importStar(require("fs"));
const node_test_1 = require("node:test");
const bridgeRunner_1 = require("../bridgeRunner");
const config_1 = require("../config");
const evidentlyClient_1 = require("../evidentlyClient");
const joinEngine_1 = require("../joinEngine");
(0, node_test_1.describe)("JoinEngine Unit Tests (CSV Parser, Key Suggestions, Relational Joins)", () => {
    (0, node_test_1.it)("parses RFC-4180 CSV with quoted commas, newlines, escaped quotes, and numeric coercion", () => {
        const csv = 'id,name,score,note\n1,"Alpha, Inc.",42.5,"Line 1\n""Quoted"" Line 2"\n2,Beta,40.0,\n';
        const table = (0, joinEngine_1.parseCsv)(csv);
        assert.deepStrictEqual(table.columns, ["id", "name", "score", "note"]);
        assert.strictEqual(table.rows.length, 2);
        assert.strictEqual(table.rows[0].id, 1);
        assert.strictEqual(table.rows[0].name, "Alpha, Inc.");
        assert.strictEqual(table.rows[0].score, 42.5);
        assert.strictEqual(table.rows[0].note, 'Line 1\n"Quoted" Line 2');
        assert.strictEqual(table.rows[1].note, null);
        const roundTrip = (0, joinEngine_1.parseCsv)((0, joinEngine_1.serializeCsv)(table.columns, table.rows));
        assert.deepStrictEqual(roundTrip.columns, table.columns);
        assert.strictEqual(roundTrip.rows.length, 2);
        assert.strictEqual(roundTrip.rows[0].note, 'Line 1\n"Quoted" Line 2');
    });
    (0, node_test_1.it)("handles empty CSV inputs gracefully", () => {
        assert.deepStrictEqual((0, joinEngine_1.parseCsv)(""), { columns: [], rows: [] });
        assert.deepStrictEqual((0, joinEngine_1.parseCsv)("   \n  "), { columns: [], rows: [] });
    });
    (0, node_test_1.it)("suggests high-value exact, case-insensitive, and semantic join keys", () => {
        const leftCols = ["molecule_id", "precursor_mz", "note", "Slug"];
        const rightCols = ["molecule", "slug", "note", "formula"];
        const suggestions = (0, joinEngine_1.suggestJoinKeys)(leftCols, rightCols);
        assert.ok(suggestions.length >= 3);
        // Slug <-> slug or molecule_id <-> molecule should outrank generic 'note'
        const topCols = suggestions.slice(0, 2).map((s) => `${s.leftColumn}=${s.rightColumn}`);
        assert.ok(topCols.includes("Slug=slug") || topCols.includes("molecule_id=molecule"), `Unexpected top suggestions: ${JSON.stringify(topCols)}`);
    });
    (0, node_test_1.it)("executes inner, left, right, outer, cross, and union joins with normalized keys", () => {
        const left = (0, joinEngine_1.parseCsv)("key,val,shared\nA,10,L1\n40,20,L2\nC,30,L3\n");
        const right = (0, joinEngine_1.parseCsv)("key,score,shared\na ,100,R1\n40.0,200,R2\nD,400,R4\n");
        // Inner join with normalized keys ('A' == 'a ', '40' == '40.0')
        const inner = (0, joinEngine_1.joinTables)(left, right, {
            joinType: "inner",
            leftOn: ["key"],
            rightOn: ["key"],
            matchMode: "normalized",
            suffixes: ["_l", "_r"],
        });
        assert.strictEqual(inner.rows.length, 2);
        assert.strictEqual(inner.diagnostics.matchedLeftRows, 2);
        assert.strictEqual(inner.diagnostics.unmatchedLeftRows, 1);
        assert.ok(inner.columns.includes("key"));
        assert.ok(inner.columns.includes("shared_l"));
        assert.ok(inner.columns.includes("shared_r"));
        // Left join
        const leftJoin = (0, joinEngine_1.joinTables)(left, right, {
            joinType: "left",
            leftOn: ["key"],
            rightOn: ["key"],
        });
        assert.strictEqual(leftJoin.rows.length, 3);
        // Right join
        const rightJoin = (0, joinEngine_1.joinTables)(left, right, {
            joinType: "right",
            leftOn: ["key"],
            rightOn: ["key"],
        });
        assert.strictEqual(rightJoin.rows.length, 3);
        const dRow = rightJoin.rows.find((r) => r.key === "D");
        assert.ok(dRow, "Right-only key D must have coalesced key column");
        assert.strictEqual(dRow.score, 400);
        // Outer join
        const outerJoin = (0, joinEngine_1.joinTables)(left, right, {
            joinType: "outer",
            leftOn: ["key"],
            rightOn: ["key"],
        });
        assert.strictEqual(outerJoin.rows.length, 4);
        // Cross join
        const crossJoin = (0, joinEngine_1.joinTables)(left, right, {
            joinType: "cross",
            leftOn: [],
            rightOn: [],
        });
        assert.strictEqual(crossJoin.rows.length, 9);
        // Union join
        const unionJoin = (0, joinEngine_1.joinTables)(left, right, {
            joinType: "union",
            leftOn: [],
            rightOn: [],
        });
        assert.strictEqual(unionJoin.rows.length, 6);
    });
    (0, node_test_1.it)("supports multi-column composite join keys and differently named key columns", () => {
        const left = (0, joinEngine_1.parseCsv)("mol_id,adduct,intensity\nM1,[M+H]+,0.9\nM1,[M+Na]+,0.4\n");
        const right = (0, joinEngine_1.parseCsv)("molecule,ion_mode,rank\nM1,[M+H]+,1\nM1,[M-H]-,2\n");
        const res = (0, joinEngine_1.joinTables)(left, right, {
            joinType: "inner",
            leftOn: ["mol_id", "adduct"],
            rightOn: ["molecule", "ion_mode"],
        });
        assert.strictEqual(res.rows.length, 1);
        assert.strictEqual(res.rows[0].mol_id, "M1");
        assert.strictEqual(res.rows[0].rank, 1);
    });
    (0, node_test_1.it)("handles pre-existing _left and _right columns without duplicate column collisions", () => {
        const left = (0, joinEngine_1.parseCsv)("id,val,val_left\n1,10,100\n2,20,200\n");
        const right = (0, joinEngine_1.parseCsv)("id,val,val_right\n1,30,300\n2,40,400\n");
        const res = (0, joinEngine_1.joinTables)(left, right, {
            joinType: "inner",
            leftOn: ["id"],
            rightOn: ["id"],
            suffixes: ["_left", "_right"],
        });
        assert.strictEqual(res.rows.length, 2);
        assert.strictEqual(new Set(res.columns).size, res.columns.length);
        assert.strictEqual(res.columns.length, 5);
    });
});
(0, node_test_1.describe)("Live Integration Tests against Local Evidently Remote Workspace", () => {
    const cfg = (0, config_1.getWorkspaceConfig)();
    const client = new evidentlyClient_1.EvidentlyClient(cfg.serverUrl, cfg.apiKey);
    const casmiProjectId = "01a1d2dd-5ca0-7d05-a07a-135ba489166d";
    (0, node_test_1.it)("connects, authenticates, and lists all projects on the local Evidently remote workspace", async () => {
        const version = await client.getVersion();
        assert.ok(version.version, "Expected Evidently version string");
        const projects = await client.listProjects();
        assert.ok(projects.length >= 15, `Expected >=15 projects, got ${projects.length}`);
        const casmi = projects.find((p) => p.id === casmiProjectId);
        assert.ok(casmi, "Expected enveda-casmi26-molecule-id-mass-spectra project");
    });
    (0, node_test_1.it)("lists datasets, fetches paginated sorted rows, and resolves schemas (including tracing datasets)", async () => {
        const datasets = await client.listDatasets(casmiProjectId);
        assert.ok(datasets.length >= 20, `Expected >=20 datasets in CASMI project, got ${datasets.length}`);
        const rl77 = datasets.find((d) => d.name === "casmi-rl77-weights");
        assert.ok(rl77, "Expected casmi-rl77-weights dataset");
        const page = await client.getDatasetPage(rl77.id, {
            pageSize: 10,
            currentPage: 1,
            sortByColumn: "name",
            sortAscending: true,
        });
        assert.strictEqual(page.items.length, 10);
        assert.ok(page.metadata.columns.length >= 8);
        const schema = await client.getDatasetSchema(rl77.id);
        assert.ok(schema.columnNames.includes("name"));
        assert.ok(schema.columnNames.includes("value"));
        // Verify 0-row / tracing dataset fallback in gemma-4-developer-agent
        const tracingDatasetId = "01a171ef-d5fb-7e86-8c30-6527ca5bc19d";
        const tracingSchema = await client.getDatasetSchema(tracingDatasetId);
        assert.ok(tracingSchema.columnNames.length >= 10, `Expected tracing dataset fallback schema to have columns, got ${tracingSchema.columnNames.length}`);
    });
    (0, node_test_1.it)("lists reports on healthy projects and handles HTTP 500 report errors gracefully without throwing", async () => {
        const casmiReports = await client.listReports(casmiProjectId);
        assert.strictEqual(casmiReports.error, undefined);
        assert.ok(casmiReports.reports.length >= 20);
        const sampleReport = casmiReports.reports[0];
        const jsonMetrics = await client.downloadReportJson(casmiProjectId, sampleReport.id);
        assert.ok(Array.isArray(jsonMetrics.metrics));
        const reportData = await client.getReportData(casmiProjectId, sampleReport.id);
        assert.ok(Array.isArray(reportData.widgets));
        // Corrupt snapshot project on live server returns HTTP 500 on /reports
        const brokenProjectId = "01a14491-3616-7d3f-9fd5-765e5183dd28";
        const brokenReports = await client.listReports(brokenProjectId);
        assert.deepStrictEqual(brokenReports.reports, []);
        assert.ok(brokenReports.error && brokenReports.error.includes("500"));
    });
    (0, node_test_1.it)("joins two live Evidently datasets and generates + publishes + cleans up an Evidently report via Python bridge", async () => {
        const datasets = await client.listDatasets(casmiProjectId);
        const topicsDs = datasets.find((d) => d.name === "kaggle discussions: topics");
        const messagesDs = datasets.find((d) => d.name === "kaggle discussions: messages");
        assert.ok(topicsDs && messagesDs, "Expected topics and messages datasets in CASMI project");
        // 1. Test TypeScript live join on downloaded CSVs
        const [topicsCsv, messagesCsv] = await Promise.all([
            client.downloadDatasetCsv(topicsDs.id),
            client.downloadDatasetCsv(messagesDs.id),
        ]);
        const joinedTs = (0, joinEngine_1.joinTables)((0, joinEngine_1.parseCsv)(topicsCsv), (0, joinEngine_1.parseCsv)(messagesCsv), {
            joinType: "inner",
            leftOn: ["topic_id"],
            rightOn: ["topic_id"],
        });
        assert.ok(joinedTs.rows.length > 100, `Expected >100 joined rows, got ${joinedTs.rows.length}`);
        assert.strictEqual(joinedTs.diagnostics.keyOverlapPct, 100);
        // 2. Test Python bridge report generation + round-trip publish to workspace + cleanup
        const rl77Ds = datasets.find((d) => d.name === "casmi-rl77-weights");
        const casmiWeightsDs = datasets.find((d) => d.name === "casmi-weights");
        const bridgeResult = await (0, bridgeRunner_1.runEvidentlyBridge)(cfg.pythonPath, {
            mode: "create_report",
            server_url: cfg.serverUrl,
            api_key: cfg.apiKey,
            left_dataset_id: rl77Ds.id,
            right_dataset_id: casmiWeightsDs.id,
            join_type: "outer",
            left_on: ["name"],
            right_on: ["name"],
            target_project_id: casmiProjectId,
            report_name: "vscode-ext-automated-test-join-report",
            tags: ["automated-test", "joined-report"],
            presets: ["DataSummaryPreset", "DataDriftPreset"],
            include_correlations: true,
            reference_source: "left",
            save_joined_dataset: true,
            joined_dataset_name: "vscode-ext-automated-test-joined-table",
            publish_to_workspace: true,
        });
        assert.strictEqual(bridgeResult.status, "ok");
        assert.ok(bridgeResult.snapshotId, "Expected published snapshotId");
        assert.ok(bridgeResult.joinedDatasetId, "Expected published joinedDatasetId");
        assert.ok(bridgeResult.htmlPath && fs.existsSync(bridgeResult.htmlPath));
        assert.ok((bridgeResult.metricsSummary?.metrics?.length || 0) > 5);
        try {
            // Verify report & joined dataset are accessible via EvidentlyClient
            const meta = await client.getReportMetadata(casmiProjectId, bridgeResult.snapshotId);
            assert.strictEqual(meta.id, bridgeResult.snapshotId);
            assert.strictEqual(meta.links?.datasets?.output?.current, bridgeResult.joinedDatasetId);
            const joinedPage = await client.getDatasetPage(bridgeResult.joinedDatasetId, {
                pageSize: 5,
                currentPage: 1,
            });
            assert.ok(joinedPage.items.length > 0);
        }
        finally {
            // Clean up temporary test artifacts from the server and local tmp
            if (bridgeResult.htmlPath && fs.existsSync(bridgeResult.htmlPath)) {
                fs.unlinkSync(bridgeResult.htmlPath);
            }
            const cleanup = await (0, bridgeRunner_1.runEvidentlyBridge)(cfg.pythonPath, {
                mode: "delete_artifacts",
                server_url: cfg.serverUrl,
                api_key: cfg.apiKey,
                project_id: casmiProjectId,
                snapshot_ids: bridgeResult.snapshotId ? [bridgeResult.snapshotId] : [],
                dataset_ids: bridgeResult.joinedDatasetId ? [bridgeResult.joinedDatasetId] : [],
            });
            assert.strictEqual(cleanup.status, "ok");
        }
    });
});
//# sourceMappingURL=extension.test.js.map