from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

INPUT_OUTPUT_PAIRS = [
    (
        Path("/data/disk4/workspace/projects/glassdoor/outputs/firm_year_glassdoor_major_customer.parquet"),
        Path("/data/disk4/workspace/projects/glassdoor/outputs/firm_year_glassdoor_major_customer.dta"),
    ),
    (
        Path("/data/disk4/workspace/projects/glassdoor/outputs/firm_year_glassdoor_union.parquet"),
        Path("/data/disk4/workspace/projects/glassdoor/outputs/firm_year_glassdoor_union.dta"),
    ),
]


def _sanitize_stata_name(name: str, used: set[str], max_len: int = 32) -> str:
    # Stata variable names must be <= 32 chars; keep names readable and deterministic.
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not cleaned:
        cleaned = "v"
    if cleaned[0].isdigit():
        cleaned = f"v_{cleaned}"

    base = cleaned[:max_len]
    if base not in used:
        used.add(base)
        return base

    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
    reserve = 1 + len(digest)
    candidate = f"{base[: max_len - reserve]}_{digest}"

    suffix = 1
    while candidate in used:
        s = str(suffix)
        reserve2 = 1 + len(digest) + 1 + len(s)
        candidate = f"{base[: max_len - reserve2]}_{digest}_{s}"
        suffix += 1

    used.add(candidate)
    return candidate


def _rename_columns_for_stata(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
    used: set[str] = set()
    col_map: Dict[str, str] = {}
    for col in df.columns:
        col_map[col] = _sanitize_stata_name(str(col), used=used, max_len=32)
    renamed = df.rename(columns=col_map)
    return renamed, col_map


def _prepare_dtypes_for_stata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        s = out[col]

        if pd.api.types.is_categorical_dtype(s):
            out[col] = s.astype("string").astype(object)
            continue

        if pd.api.types.is_string_dtype(s):
            out[col] = s.astype(object)
            continue

        # Stata writer does not support nullable integer/boolean extension dtypes with NA well.
        if pd.api.types.is_integer_dtype(s) and s.isna().any():
            out[col] = s.astype("float64")
            continue

        if pd.api.types.is_bool_dtype(s) and s.isna().any():
            out[col] = s.astype("float64")
            continue

    return out


def convert_one(parquet_path: Path, dta_path: Path) -> None:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input not found: {parquet_path}")

    print("=" * 100)
    print(f"Reading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Rows: {len(df):,}, Columns: {len(df.columns):,}")

    df = _prepare_dtypes_for_stata(df)
    df, col_map = _rename_columns_for_stata(df)

    dta_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing Stata: {dta_path}")
    df.to_stata(dta_path, write_index=False, version=118)

    map_path = dta_path.with_suffix(".name_map.json")
    with map_path.open("w", encoding="utf-8") as f:
        json.dump(col_map, f, indent=2, ensure_ascii=False)

    print(f"Wrote name map: {map_path}")


def main() -> None:
    for parquet_path, dta_path in INPUT_OUTPUT_PAIRS:
        convert_one(parquet_path, dta_path)

    print("\nDone. Converted all firm-year parquet files to Stata .dta.")


if __name__ == "__main__":
    main()
