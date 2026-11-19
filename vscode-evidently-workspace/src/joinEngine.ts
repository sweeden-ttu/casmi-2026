export type JoinType = "inner" | "left" | "right" | "outer" | "cross" | "union";
export type KeyMatchMode = "normalized" | "exact";

export interface ParsedTable {
  columns: string[];
  rows: Record<string, any>[];
}

export interface JoinOptions {
  joinType: JoinType;
  leftOn: string[];
  rightOn: string[];
  matchMode?: KeyMatchMode;
  suffixes?: [string, string];
  selectedLeftColumns?: string[];
  selectedRightColumns?: string[];
  dropNullKeys?: boolean;
  maxOutputRows?: number;
}

export interface ColumnProfile {
  name: string;
  source: "key" | "left" | "right" | "union";
  inferredType: "numerical" | "string" | "datetime" | "boolean";
  nonNullCount: number;
  missingCount: number;
  missingPct: number;
  uniqueCount: number;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  std?: number | null;
  sampleValues: string[];
}

export interface JoinDiagnostics {
  joinType: JoinType;
  leftOn: string[];
  rightOn: string[];
  leftRowCount: number;
  rightRowCount: number;
  joinedRowCount: number;
  truncated: boolean;
  matchedLeftRows: number;
  unmatchedLeftRows: number;
  matchedRightRows: number;
  unmatchedRightRows: number;
  distinctLeftKeys: number;
  distinctRightKeys: number;
  overlappingKeys: number;
  keyOverlapPct: number;
}

export interface JoinResult {
  columns: string[];
  rows: Record<string, any>[];
  diagnostics: JoinDiagnostics;
  columnProfiles: ColumnProfile[];
}

export interface SuggestedJoinKeyPair {
  leftColumn: string;
  rightColumn: string;
  reason: "exact_id" | "exact" | "case_insensitive" | "semantic_id";
  score: number;
}

export function parseCsv(csvText: string): ParsedTable {
  if (!csvText || !csvText.trim()) {
    return { columns: [], rows: [] };
  }

  const rawRows: string[][] = [];
  let currentRow: string[] = [];
  let currentField = "";
  let inQuotes = false;
  const len = csvText.length;

  for (let i = 0; i < len; i++) {
    const ch = csvText[i];
    if (inQuotes) {
      if (ch === '"') {
        if (i + 1 < len && csvText[i + 1] === '"') {
          currentField += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        currentField += ch;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ",") {
        currentRow.push(currentField);
        currentField = "";
      } else if (ch === "\r") {
        if (i + 1 < len && csvText[i + 1] === "\n") {
          i++;
        }
        currentRow.push(currentField);
        rawRows.push(currentRow);
        currentRow = [];
        currentField = "";
      } else if (ch === "\n") {
        currentRow.push(currentField);
        rawRows.push(currentRow);
        currentRow = [];
        currentField = "";
      } else {
        currentField += ch;
      }
    }
  }

  if (inQuotes || currentField.length > 0 || currentRow.length > 0) {
    currentRow.push(currentField);
    rawRows.push(currentRow);
  }

  if (rawRows.length === 0) {
    return { columns: [], rows: [] };
  }

  const header = rawRows[0].map((c, idx) => {
    const cleaned = idx === 0 ? c.replace(/^\uFEFF/, "").trim() : c.trim();
    return cleaned || `col_${idx + 1}`;
  });

  // Deduplicate header names if any
  const seenHeader = new Map<string, number>();
  const columns = header.map((col) => {
    const count = seenHeader.get(col) || 0;
    seenHeader.set(col, count + 1);
    return count === 0 ? col : `${col}_${count + 1}`;
  });

  const rows: Record<string, any>[] = [];
  for (let r = 1; r < rawRows.length; r++) {
    const arr = rawRows[r];
    if (arr.length === 1 && arr[0] === "") {
      continue;
    }
    const obj: Record<string, any> = {};
    for (let c = 0; c < columns.length; c++) {
      const val = c < arr.length ? arr[c] : "";
      obj[columns[c]] = coerceScalar(val);
    }
    rows.push(obj);
  }

  return { columns, rows };
}

export function coerceScalar(raw: any): any {
  if (raw === null || raw === undefined) {
    return null;
  }
  if (typeof raw === "number" || typeof raw === "boolean") {
    return raw;
  }
  const str = String(raw);
  if (str === "" || str.toLowerCase() === "null" || str.toLowerCase() === "nan" || str.toLowerCase() === "none") {
    return null;
  }
  // Check strict finite number (avoid converting leading-zero strings like "01a1..." or zip codes)
  if (/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(str.trim())) {
    const num = Number(str.trim());
    if (Number.isFinite(num)) {
      return num;
    }
  }
  return str;
}

export function serializeCsv(columns: string[], rows: Record<string, any>[]): string {
  const escapeCell = (val: any): string => {
    if (val === null || val === undefined) {
      return "";
    }
    const str = String(val);
    if (/[",\r\n]/.test(str)) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };

  const lines: string[] = [];
  lines.push(columns.map(escapeCell).join(","));
  for (const row of rows) {
    lines.push(columns.map((c) => escapeCell(row[c])).join(","));
  }
  return lines.join("\n") + "\n";
}

function normalizeKeyComponent(val: any, mode: KeyMatchMode): string | null {
  if (val === null || val === undefined || val === "") {
    return null;
  }
  if (typeof val === "number") {
    if (!Number.isFinite(val)) {
      return null;
    }
    return mode === "normalized" ? String(Number(val)) : String(val);
  }
  const str = String(val);
  if (mode === "exact") {
    return str;
  }
  const trimmed = str.trim();
  if (!trimmed) {
    return null;
  }
  // Canonicalize numeric strings so "40" and "40.0" match
  if (/^-?(?:0|[1-9]\d*)(?:\.\d+)?$/.test(trimmed)) {
    const n = Number(trimmed);
    if (Number.isFinite(n)) {
      return String(n);
    }
  }
  return trimmed.toLowerCase();
}

function buildCompositeKey(
  row: Record<string, any>,
  keyCols: string[],
  mode: KeyMatchMode
): { key: string; hasNull: boolean } {
  const parts: string[] = [];
  let hasNull = false;
  for (const col of keyCols) {
    const norm = normalizeKeyComponent(row[col], mode);
    if (norm === null) {
      hasNull = true;
      parts.push("\u0000NULL\u0000");
    } else {
      parts.push(norm);
    }
  }
  return {
    key: parts.join("\u001F"),
    hasNull,
  };
}

function uniqueColumnName(desired: string, existing: Set<string>): string {
  if (!existing.has(desired)) {
    existing.add(desired);
    return desired;
  }
  let idx = 2;
  while (existing.has(`${desired}_${idx}`)) {
    idx++;
  }
  const finalName = `${desired}_${idx}`;
  existing.add(finalName);
  return finalName;
}

export function suggestJoinKeys(
  leftColumns: string[],
  rightColumns: string[]
): SuggestedJoinKeyPair[] {
  const suggestions: SuggestedJoinKeyPair[] = [];
  const rightSet = new Set(rightColumns);
  const rightLowerMap = new Map<string, string>();
  for (const rc of rightColumns) {
    if (!rightLowerMap.has(rc.toLowerCase())) {
      rightLowerMap.set(rc.toLowerCase(), rc);
    }
  }

  const isHighValueKey = (col: string): boolean =>
    /(?:^|_)(?:id|key|name|slug|molecule|topic|team|exp|version|code|index|inchikey)(?:$|_)/i.test(
      col
    );
  const isLowValueKey = (col: string): boolean =>
    /^(?:note|notes|updated_utc|timestamp|created_at|description|text|body|url)$/i.test(
      col
    );

  const addedPairs = new Set<string>();
  const addPair = (
    leftColumn: string,
    rightColumn: string,
    reason: SuggestedJoinKeyPair["reason"],
    score: number
  ) => {
    const pairKey = `${leftColumn}:::${rightColumn}`;
    if (!addedPairs.has(pairKey)) {
      addedPairs.add(pairKey);
      suggestions.push({ leftColumn, rightColumn, reason, score });
    }
  };

  for (const lc of leftColumns) {
    if (rightSet.has(lc)) {
      const score = isHighValueKey(lc) ? 100 : isLowValueKey(lc) ? 35 : 75;
      addPair(lc, lc, isHighValueKey(lc) ? "exact_id" : "exact", score);
    } else if (rightLowerMap.has(lc.toLowerCase())) {
      const rc = rightLowerMap.get(lc.toLowerCase())!;
      const score = isHighValueKey(lc) ? 90 : 65;
      addPair(lc, rc, "case_insensitive", score);
    }
  }

  // Semantic ID matching (e.g. molecule_id <-> molecule, exp_id <-> id)
  const normalizeStem = (c: string) =>
    c
      .toLowerCase()
      .replace(/_?(?:id|key|code|num)$/i, "")
      .replace(/[^a-z0-9]/g, "");

  for (const lc of leftColumns) {
    const lStem = normalizeStem(lc);
    if (!lStem || lStem.length < 3) {
      continue;
    }
    for (const rc of rightColumns) {
      if (lc === rc) {
        continue;
      }
      const rStem = normalizeStem(rc);
      if (lStem === rStem && (isHighValueKey(lc) || isHighValueKey(rc))) {
        addPair(lc, rc, "semantic_id", 60);
      }
    }
  }

  return suggestions.sort((a, b) => b.score - a.score);
}

export function profileColumns(
  columns: string[],
  rows: Record<string, any>[],
  columnSources: Map<string, ColumnProfile["source"]>
): ColumnProfile[] {
  const totalRows = rows.length;
  return columns.map((col) => {
    let nonNullCount = 0;
    let numericCount = 0;
    let boolCount = 0;
    let datetimeCount = 0;
    const nums: number[] = [];
    const uniqueSet = new Set<string>();
    const sampleValues: string[] = [];

    for (const row of rows) {
      const val = row[col];
      if (val === null || val === undefined || val === "") {
        continue;
      }
      nonNullCount++;
      const strVal = String(val);
      if (uniqueSet.size < 50000) {
        uniqueSet.add(strVal);
      }
      if (sampleValues.length < 4 && !sampleValues.includes(strVal)) {
        sampleValues.push(strVal.length > 48 ? strVal.slice(0, 45) + "..." : strVal);
      }

      if (typeof val === "number" && Number.isFinite(val)) {
        numericCount++;
        nums.push(val);
      } else if (typeof val === "boolean") {
        boolCount++;
      } else if (
        typeof val === "string" &&
        /^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?/.test(val)
      ) {
        datetimeCount++;
      }
    }

    let inferredType: ColumnProfile["inferredType"] = "string";
    if (nonNullCount > 0) {
      if (numericCount === nonNullCount) {
        inferredType = "numerical";
      } else if (boolCount === nonNullCount) {
        inferredType = "boolean";
      } else if (datetimeCount >= Math.ceil(nonNullCount * 0.8)) {
        inferredType = "datetime";
      }
    }

    let min: number | null = null;
    let max: number | null = null;
    let mean: number | null = null;
    let std: number | null = null;

    if (nums.length > 0) {
      min = nums[0];
      max = nums[0];
      let sum = 0;
      for (const n of nums) {
        if (n < min) {
          min = n;
        }
        if (n > max) {
          max = n;
        }
        sum += n;
      }
      mean = sum / nums.length;
      if (nums.length > 1) {
        let sqDiff = 0;
        for (const n of nums) {
          const d = n - mean;
          sqDiff += d * d;
        }
        std = Math.sqrt(sqDiff / (nums.length - 1));
      } else {
        std = 0;
      }
    }

    const missingCount = Math.max(0, totalRows - nonNullCount);
    const missingPct = totalRows > 0 ? Number(((missingCount / totalRows) * 100).toFixed(2)) : 0;

    return {
      name: col,
      source: columnSources.get(col) || "left",
      inferredType,
      nonNullCount,
      missingCount,
      missingPct,
      uniqueCount: uniqueSet.size,
      min,
      max,
      mean: mean !== null ? Number(mean.toFixed(6)) : null,
      std: std !== null ? Number(std.toFixed(6)) : null,
      sampleValues,
    };
  });
}

export function joinTables(
  leftTable: ParsedTable,
  rightTable: ParsedTable,
  options: JoinOptions
): JoinResult {
  const joinType: JoinType = options.joinType || "inner";
  const matchMode: KeyMatchMode = options.matchMode || "normalized";
  const suffixes: [string, string] = options.suffixes || ["_left", "_right"];
  const dropNullKeys = options.dropNullKeys ?? true;
  const maxOutputRows = options.maxOutputRows && options.maxOutputRows > 0 ? options.maxOutputRows : 100000;

  const leftKeep =
    options.selectedLeftColumns && options.selectedLeftColumns.length > 0
      ? leftTable.columns.filter((c) => options.selectedLeftColumns!.includes(c))
      : [...leftTable.columns];
  const rightKeep =
    options.selectedRightColumns && options.selectedRightColumns.length > 0
      ? rightTable.columns.filter((c) => options.selectedRightColumns!.includes(c))
      : [...rightTable.columns];

  // Handle UNION (vertical concatenation)
  if (joinType === "union") {
    const unionCols: string[] = [];
    const seen = new Set<string>();
    for (const c of [...leftKeep, ...rightKeep]) {
      if (!seen.has(c)) {
        seen.add(c);
        unionCols.push(c);
      }
    }
    const outRows: Record<string, any>[] = [];
    let truncated = false;
    for (const r of leftTable.rows) {
      if (outRows.length >= maxOutputRows) {
        truncated = true;
        break;
      }
      const rowObj: Record<string, any> = {};
      for (const c of unionCols) {
        rowObj[c] = r[c] ?? null;
      }
      outRows.push(rowObj);
    }
    for (const r of rightTable.rows) {
      if (outRows.length >= maxOutputRows) {
        truncated = true;
        break;
      }
      const rowObj: Record<string, any> = {};
      for (const c of unionCols) {
        rowObj[c] = r[c] ?? null;
      }
      outRows.push(rowObj);
    }

    const colSources = new Map<string, ColumnProfile["source"]>();
    for (const c of unionCols) {
      colSources.set(c, "union");
    }

    return {
      columns: unionCols,
      rows: outRows,
      diagnostics: {
        joinType,
        leftOn: [],
        rightOn: [],
        leftRowCount: leftTable.rows.length,
        rightRowCount: rightTable.rows.length,
        joinedRowCount: outRows.length,
        truncated,
        matchedLeftRows: leftTable.rows.length,
        unmatchedLeftRows: 0,
        matchedRightRows: rightTable.rows.length,
        unmatchedRightRows: 0,
        distinctLeftKeys: 0,
        distinctRightKeys: 0,
        overlappingKeys: 0,
        keyOverlapPct: 100,
      },
      columnProfiles: profileColumns(unionCols, outRows, colSources),
    };
  }

  const leftOn = options.leftOn || [];
  const rightOn = options.rightOn || [];

  if (joinType !== "cross") {
    if (leftOn.length === 0 || rightOn.length === 0) {
      throw new Error("At least one join key column must be specified for left and right tables.");
    }
    if (leftOn.length !== rightOn.length) {
      throw new Error(
        `Mismatched join key count: leftOn has ${leftOn.length} column(s), rightOn has ${rightOn.length} column(s).`
      );
    }
    for (const lk of leftOn) {
      if (!leftTable.columns.includes(lk)) {
        throw new Error(`Left join key column "${lk}" does not exist in left table.`);
      }
    }
    for (const rk of rightOn) {
      if (!rightTable.columns.includes(rk)) {
        throw new Error(`Right join key column "${rk}" does not exist in right table.`);
      }
    }
  }

  // Ensure key columns are included
  for (const lk of leftOn) {
    if (!leftKeep.includes(lk)) {
      leftKeep.unshift(lk);
    }
  }
  for (const rk of rightOn) {
    if (!rightKeep.includes(rk)) {
      rightKeep.unshift(rk);
    }
  }

  // Determine output column names and coalesced shared key columns
  const sharedKeyMap = new Map<string, string>(); // leftKeyCol -> rightKeyCol when name matches
  if (joinType !== "cross") {
    for (let i = 0; i < leftOn.length; i++) {
      if (leftOn[i] === rightOn[i]) {
        sharedKeyMap.set(leftOn[i], rightOn[i]);
      }
    }
  }

  const rightNonSharedCols = rightKeep.filter((rc) => !Array.from(sharedKeyMap.values()).includes(rc));
  const rightCollisionSet = new Set(rightNonSharedCols);
  const leftCollisionSet = new Set(leftKeep.filter((lc) => !sharedKeyMap.has(lc)));

  const existingOutputCols = new Set<string>();
  const leftColMapping: { src: string; out: string; isSharedKey: boolean }[] = [];
  const rightColMapping: { src: string; out: string }[] = [];
  const columnSources = new Map<string, ColumnProfile["source"]>();

  for (const lc of leftKeep) {
    if (sharedKeyMap.has(lc)) {
      const outName = uniqueColumnName(lc, existingOutputCols);
      leftColMapping.push({ src: lc, out: outName, isSharedKey: true });
      columnSources.set(outName, "key");
    } else if (rightCollisionSet.has(lc)) {
      const outName = uniqueColumnName(`${lc}${suffixes[0]}`, existingOutputCols);
      leftColMapping.push({ src: lc, out: outName, isSharedKey: false });
      columnSources.set(outName, "left");
    } else {
      const outName = uniqueColumnName(lc, existingOutputCols);
      leftColMapping.push({ src: lc, out: outName, isSharedKey: false });
      columnSources.set(outName, leftOn.includes(lc) ? "key" : "left");
    }
  }

  for (const rc of rightNonSharedCols) {
    if (leftCollisionSet.has(rc) || sharedKeyMap.has(rc)) {
      const outName = uniqueColumnName(`${rc}${suffixes[1]}`, existingOutputCols);
      rightColMapping.push({ src: rc, out: outName });
      columnSources.set(outName, rightOn.includes(rc) ? "key" : "right");
    } else {
      const outName = uniqueColumnName(rc, existingOutputCols);
      rightColMapping.push({ src: rc, out: outName });
      columnSources.set(outName, rightOn.includes(rc) ? "key" : "right");
    }
  }

  const outputColumns = [
    ...leftColMapping.map((m) => m.out),
    ...rightColMapping.map((m) => m.out),
  ];

  const mergeRowPair = (
    lRow: Record<string, any> | null,
    rRow: Record<string, any> | null
  ): Record<string, any> => {
    const merged: Record<string, any> = {};
    for (const lm of leftColMapping) {
      if (lm.isSharedKey) {
        const lVal = lRow ? lRow[lm.src] : null;
        const rVal = rRow ? rRow[sharedKeyMap.get(lm.src)!] : null;
        merged[lm.out] = lVal !== null && lVal !== undefined && lVal !== "" ? lVal : rVal ?? null;
      } else {
        merged[lm.out] = lRow ? lRow[lm.src] ?? null : null;
      }
    }
    for (const rm of rightColMapping) {
      merged[rm.out] = rRow ? rRow[rm.src] ?? null : null;
    }
    return merged;
  };

  const outRows: Record<string, any>[] = [];
  let truncated = false;

  if (joinType === "cross") {
    for (const lRow of leftTable.rows) {
      for (const rRow of rightTable.rows) {
        if (outRows.length >= maxOutputRows) {
          truncated = true;
          break;
        }
        outRows.push(mergeRowPair(lRow, rRow));
      }
      if (truncated) {
        break;
      }
    }

    return {
      columns: outputColumns,
      rows: outRows,
      diagnostics: {
        joinType,
        leftOn: [],
        rightOn: [],
        leftRowCount: leftTable.rows.length,
        rightRowCount: rightTable.rows.length,
        joinedRowCount: outRows.length,
        truncated,
        matchedLeftRows: leftTable.rows.length > 0 && rightTable.rows.length > 0 ? leftTable.rows.length : 0,
        unmatchedLeftRows: rightTable.rows.length === 0 ? leftTable.rows.length : 0,
        matchedRightRows: leftTable.rows.length > 0 && rightTable.rows.length > 0 ? rightTable.rows.length : 0,
        unmatchedRightRows: leftTable.rows.length === 0 ? rightTable.rows.length : 0,
        distinctLeftKeys: 0,
        distinctRightKeys: 0,
        overlappingKeys: 0,
        keyOverlapPct: 100,
      },
      columnProfiles: profileColumns(outputColumns, outRows, columnSources),
    };
  }

  // Index right table by composite key
  const rightIndex = new Map<string, { row: Record<string, any>; idx: number }[]>();
  const distinctRightKeySet = new Set<string>();
  const nullKeyRightRows: { row: Record<string, any>; idx: number }[] = [];

  for (let rIdx = 0; rIdx < rightTable.rows.length; rIdx++) {
    const rRow = rightTable.rows[rIdx];
    const { key, hasNull } = buildCompositeKey(rRow, rightOn, matchMode);
    if (hasNull && dropNullKeys) {
      nullKeyRightRows.push({ row: rRow, idx: rIdx });
      continue;
    }
    distinctRightKeySet.add(key);
    let bucket = rightIndex.get(key);
    if (!bucket) {
      bucket = [];
      rightIndex.set(key, bucket);
    }
    bucket.push({ row: rRow, idx: rIdx });
  }

  const distinctLeftKeySet = new Set<string>();
  const matchedRightIndices = new Set<number>();
  let matchedLeftRows = 0;
  let unmatchedLeftRows = 0;

  for (let lIdx = 0; lIdx < leftTable.rows.length; lIdx++) {
    const lRow = leftTable.rows[lIdx];
    const { key, hasNull } = buildCompositeKey(lRow, leftOn, matchMode);
    if (!hasNull || !dropNullKeys) {
      distinctLeftKeySet.add(key);
    }
    const matches = !hasNull || !dropNullKeys ? rightIndex.get(key) : undefined;
    if (matches && matches.length > 0) {
      matchedLeftRows++;
      for (const m of matches) {
        matchedRightIndices.add(m.idx);
        if (outRows.length < maxOutputRows) {
          outRows.push(mergeRowPair(lRow, m.row));
        } else {
          truncated = true;
        }
      }
    } else {
      unmatchedLeftRows++;
      if (joinType === "left" || joinType === "outer") {
        if (outRows.length < maxOutputRows) {
          outRows.push(mergeRowPair(lRow, null));
        } else {
          truncated = true;
        }
      }
    }
  }

  if (joinType === "right" || joinType === "outer") {
    for (let rIdx = 0; rIdx < rightTable.rows.length; rIdx++) {
      if (!matchedRightIndices.has(rIdx)) {
        if (outRows.length < maxOutputRows) {
          outRows.push(mergeRowPair(null, rightTable.rows[rIdx]));
        } else {
          truncated = true;
        }
      }
    }
  }

  let overlappingKeys = 0;
  for (const lk of distinctLeftKeySet) {
    if (distinctRightKeySet.has(lk)) {
      overlappingKeys++;
    }
  }
  const denomKeys = Math.max(1, Math.min(distinctLeftKeySet.size, distinctRightKeySet.size));
  const keyOverlapPct =
    distinctLeftKeySet.size === 0 || distinctRightKeySet.size === 0
      ? 0
      : Number(((overlappingKeys / denomKeys) * 100).toFixed(2));

  const matchedRightRows = matchedRightIndices.size;
  const unmatchedRightRows = Math.max(0, rightTable.rows.length - matchedRightRows);

  return {
    columns: outputColumns,
    rows: outRows,
    diagnostics: {
      joinType,
      leftOn,
      rightOn,
      leftRowCount: leftTable.rows.length,
      rightRowCount: rightTable.rows.length,
      joinedRowCount: outRows.length,
      truncated,
      matchedLeftRows,
      unmatchedLeftRows,
      matchedRightRows,
      unmatchedRightRows,
      distinctLeftKeys: distinctLeftKeySet.size,
      distinctRightKeys: distinctRightKeySet.size,
      overlappingKeys,
      keyOverlapPct,
    },
    columnProfiles: profileColumns(outputColumns, outRows, columnSources),
  };
}
