from __future__ import annotations

import gzip
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

CHUNKSIZE = int(os.getenv("REVELIO_REVIEW_CHUNKSIZE", "3000000"))
COMMON_CHUNKSIZE = int(os.getenv("REVELIO_COMMON_CHUNKSIZE", "3000000"))

REVELIO_COMMON_PATH = Path("/data/disk4/database/Revelio Labs/revelio_common") / "revelio_common.zip"
REVIEWS_ZIP_PATH = Path("/data/disk4/database/Revelio Labs/revelio_sentiment") / "sentiment_individual_reviews.zip"

OUTPUT_PARQUET_PATH = (
    Path("/data/disk4/workspace/projects/glassdoor/outputs")
    / "sentiment_individual_reviews_with_gvkey.parquet"
)
OUTPUT_DIAG_PATH = (
    Path("/data/disk4/workspace/projects/glassdoor/outputs")
    / "sentiment_individual_reviews_with_gvkey_diagnostics.json"
)
MAPPING_CACHE_PATH = (
    Path("/data/disk4/workspace/projects/glassdoor/outputs")
    / "revelio_common_rcid_gvkey_mapping.parquet"
)
MAPPING_CACHE_META_PATH = (
    Path("/data/disk4/workspace/projects/glassdoor/outputs")
    / "revelio_common_rcid_gvkey_mapping_meta.json"
)

SUPPORTED_SUFFIXES = (".csv", ".csv.gz", ".parquet")
ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "cp1252", "latin1")
USE_MAPPING_CACHE = os.getenv("REVELIO_USE_MAPPING_CACHE", "1") == "1"
FORCE_REBUILD_MAPPING_CACHE = os.getenv("REVELIO_REBUILD_MAPPING_CACHE", "0") == "1"


def _normalize_string(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "NULL": pd.NA,
            "null": pd.NA,
            "<NA>": pd.NA,
        }
    )


def _file_signature(path: Path) -> Dict[str, int]:
    st = path.stat()
    return {
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _load_mapping_from_cache(cache_path: Path) -> pd.Series:
    df = pd.read_parquet(cache_path, columns=["rcid", "gvkey"])
    df["rcid"] = _normalize_string(df["rcid"])
    df["gvkey"] = _normalize_string(df["gvkey"])
    df = df[df["rcid"].notna() & df["gvkey"].notna()].copy()
    mapping = df.set_index("rcid")["gvkey"].astype("string")
    mapping.index.name = "rcid"
    mapping.name = "gvkey"
    return mapping


def _write_mapping_cache(
    mapping: pd.Series,
    cache_path: Path,
    meta_path: Path,
    source_path: Path,
    rcid_total_duplicates: int,
    rcid_conflict_count: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df = mapping.rename("gvkey").reset_index()
    mapping_df.to_parquet(cache_path, index=False)

    meta = {
        "source_path": str(source_path),
        "source_signature": _file_signature(source_path),
        "rows_in_mapping": int(len(mapping)),
        "rcid_total_duplicates": int(rcid_total_duplicates),
        "rcid_conflict_count": int(rcid_conflict_count),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _can_use_mapping_cache(
    source_path: Path,
    cache_path: Path,
    meta_path: Path,
) -> bool:
    if not cache_path.exists() or not meta_path.exists():
        return False

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False

    source_sig = _file_signature(source_path)
    cached_sig = meta.get("source_signature", {})

    return (
        int(cached_sig.get("size", -1)) == source_sig["size"]
        and int(cached_sig.get("mtime_ns", -1)) == source_sig["mtime_ns"]
    )


def _supported_member_names(member_names: List[str]) -> List[str]:
    supported = [name for name in member_names if name.lower().endswith(SUPPORTED_SUFFIXES)]
    return sorted(supported)


def _select_member(member_names: List[str], zip_path: Path) -> str:
    supported = _supported_member_names(member_names)
    if not supported:
        raise ValueError(
            f"No supported member found in {zip_path}. Supported formats: {SUPPORTED_SUFFIXES}"
        )

    # Deterministic selection: lexicographically first supported member.
    selected = supported[0]
    print(f"Selected member for processing: {selected}")
    return selected


def _extract_member_to_tempfile(zip_path: Path, member_name: str) -> str:
    suffix = ".parquet" if member_name.lower().endswith(".parquet") else ".tmp"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    with ZipFile(zip_path, "r") as zf, zf.open(member_name, "r") as src, open(tmp_path, "wb") as dst:
        while True:
            buf = src.read(8 * 1024 * 1024)
            if not buf:
                break
            dst.write(buf)

    return tmp_path


def _guess_csv_encoding_from_zip_member(zip_path: Path, member_name: str, is_gzip: bool) -> str:
    sample_size = 2 * 1024 * 1024
    with ZipFile(zip_path, "r") as zf, zf.open(member_name, "r") as raw:
        if is_gzip:
            with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
                sample = gz.read(sample_size)
        else:
            sample = raw.read(sample_size)

    for enc in ENCODING_CANDIDATES:
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue

    return "latin1"


def inspect_zip_members(zip_path: Path) -> List[str]:
    print("=" * 100)
    print(f"Inspecting ZIP: {zip_path}")
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file does not exist: {zip_path}")

    with ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()

    if not infos:
        raise ValueError(f"ZIP file is empty: {zip_path}")

    for info in infos:
        print(f"- {info.filename} ({info.file_size:,} bytes)")

    member_names = [info.filename for info in infos]
    supported = _supported_member_names(member_names)
    print("Supported members:")
    for name in supported:
        print(f"  * {name}")

    return supported


def _read_common_member(zip_path: Path, member_name: str) -> pd.DataFrame:
    usecols = ["rcid", "gvkey"]
    dtype = {"rcid": "string", "gvkey": "string"}

    member_lower = member_name.lower()
    if member_lower.endswith(".csv"):
        encoding = _guess_csv_encoding_from_zip_member(zip_path, member_name, is_gzip=False)
        print(f"Reading common CSV member with encoding: {encoding}")
        with ZipFile(zip_path, "r") as zf, zf.open(member_name, "r") as f:
            return pd.read_csv(
                f,
                usecols=usecols,
                dtype=dtype,
                low_memory=False,
                encoding=encoding,
                encoding_errors="replace",
            )

    if member_lower.endswith(".csv.gz"):
        encoding = _guess_csv_encoding_from_zip_member(zip_path, member_name, is_gzip=True)
        print(f"Reading common CSV.GZ member with encoding: {encoding}")
        with ZipFile(zip_path, "r") as zf, zf.open(member_name, "r") as f:
            with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                return pd.read_csv(
                    gz,
                    usecols=usecols,
                    dtype=dtype,
                    low_memory=False,
                    encoding=encoding,
                    encoding_errors="replace",
                )

    if member_lower.endswith(".parquet"):
        tmp_path = _extract_member_to_tempfile(zip_path, member_name)
        try:
            return pd.read_parquet(tmp_path, columns=usecols)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    raise ValueError(f"Unsupported member format: {member_name}")


def _iter_common_chunks(zip_path: Path, member_name: str, chunksize: int) -> Iterator[pd.DataFrame]:
    usecols = ["rcid", "gvkey"]
    dtype = {"rcid": "string", "gvkey": "string"}

    member_lower = member_name.lower()
    if member_lower.endswith(".csv"):
        encoding = _guess_csv_encoding_from_zip_member(zip_path, member_name, is_gzip=False)
        print(f"Reading common CSV member with encoding: {encoding}, chunksize={chunksize:,}")
        with ZipFile(zip_path, "r") as zf, zf.open(member_name, "r") as f:
            reader = pd.read_csv(
                f,
                usecols=usecols,
                dtype=dtype,
                low_memory=False,
                encoding=encoding,
                encoding_errors="replace",
                chunksize=chunksize,
            )
            for chunk in reader:
                yield chunk
        return

    if member_lower.endswith(".csv.gz"):
        encoding = _guess_csv_encoding_from_zip_member(zip_path, member_name, is_gzip=True)
        print(f"Reading common CSV.GZ member with encoding: {encoding}, chunksize={chunksize:,}")
        with ZipFile(zip_path, "r") as zf, zf.open(member_name, "r") as f:
            with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                reader = pd.read_csv(
                    gz,
                    usecols=usecols,
                    dtype=dtype,
                    low_memory=False,
                    encoding=encoding,
                    encoding_errors="replace",
                    chunksize=chunksize,
                )
                for chunk in reader:
                    yield chunk
        return

    if member_lower.endswith(".parquet"):
        tmp_path = _extract_member_to_tempfile(zip_path, member_name)
        try:
            parquet_file = pq.ParquetFile(tmp_path)
            for batch in parquet_file.iter_batches(batch_size=chunksize, columns=usecols):
                chunk = batch.to_pandas()
                chunk["rcid"] = chunk["rcid"].astype("string")
                chunk["gvkey"] = chunk["gvkey"].astype("string")
                yield chunk
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return

    raise ValueError(f"Unsupported member format: {member_name}")


def read_revelio_common_mapping(revelio_common_path: Path) -> Tuple[pd.Series, bool]:
    print("\nReading Revelio common mapping...")

    if USE_MAPPING_CACHE and not FORCE_REBUILD_MAPPING_CACHE:
        if _can_use_mapping_cache(
            source_path=revelio_common_path,
            cache_path=MAPPING_CACHE_PATH,
            meta_path=MAPPING_CACHE_META_PATH,
        ):
            print(f"Using cached mapping: {MAPPING_CACHE_PATH}")
            mapping = _load_mapping_from_cache(MAPPING_CACHE_PATH)
            print(f"Loaded cached rcid->gvkey mapping size: {len(mapping):,}")
            return mapping, True

    if FORCE_REBUILD_MAPPING_CACHE:
        print("REVELIO_REBUILD_MAPPING_CACHE=1, forcing mapping rebuild.")

    supported = inspect_zip_members(revelio_common_path)
    member_name = _select_member(supported, revelio_common_path)

    total_rows = 0
    rcid_seen: set[str] = set()
    duplicate_rcid_values: set[str] = set()

    # Keep lexicographically smallest non-missing gvkey per rcid.
    # This is equivalent to "sort then take first non-missing gvkey" without global sort.
    mapping_dict: Dict[str, str] = {}
    gvkey_sets_by_rcid: Dict[str, set[str]] = {}

    for idx, chunk in enumerate(
        _iter_common_chunks(revelio_common_path, member_name, COMMON_CHUNKSIZE),
        start=1,
    ):
        total_rows += len(chunk)
        chunk["rcid"] = _normalize_string(chunk["rcid"])
        chunk["gvkey"] = _normalize_string(chunk["gvkey"])

        chunk = chunk[chunk["rcid"].notna()]

        for rcid, gvkey in zip(chunk["rcid"], chunk["gvkey"]):
            rcid_str = str(rcid)
            if rcid_str in rcid_seen:
                duplicate_rcid_values.add(rcid_str)
            else:
                rcid_seen.add(rcid_str)

            if pd.isna(gvkey):
                continue

            gvkey_str = str(gvkey)

            existing = mapping_dict.get(rcid_str)
            if existing is None or gvkey_str < existing:
                mapping_dict[rcid_str] = gvkey_str

            s = gvkey_sets_by_rcid.get(rcid_str)
            if s is None:
                s = set()
                gvkey_sets_by_rcid[rcid_str] = s
            s.add(gvkey_str)

        print(
            f"Common chunk {idx}: rows={len(chunk):,}, total_rows={total_rows:,}, "
            f"mapped_rcids={len(mapping_dict):,}"
        )

    rcid_total_duplicates = len(duplicate_rcid_values)
    rcid_conflict_count = sum(1 for vals in gvkey_sets_by_rcid.values() if len(vals) > 1)

    if rcid_total_duplicates > 0:
        print(f"[WARN] Duplicate rcid values detected in Revelio common: {rcid_total_duplicates:,}")
    if rcid_conflict_count > 0:
        print(
            "[WARN] rcid values mapping to multiple non-missing gvkeys: "
            f"{rcid_conflict_count:,}. Keeping lexicographically first non-missing gvkey."
        )

    mapping = pd.Series(mapping_dict, dtype="string")
    mapping.index.name = "rcid"
    mapping.name = "gvkey"

    print(f"Loaded Revelio common rows: {total_rows:,}")
    print(f"Final rcid->gvkey mapping size: {len(mapping):,}")

    if USE_MAPPING_CACHE:
        _write_mapping_cache(
            mapping=mapping,
            cache_path=MAPPING_CACHE_PATH,
            meta_path=MAPPING_CACHE_META_PATH,
            source_path=revelio_common_path,
            rcid_total_duplicates=rcid_total_duplicates,
            rcid_conflict_count=rcid_conflict_count,
        )
        print(f"Wrote mapping cache: {MAPPING_CACHE_PATH}")
        print(f"Wrote mapping cache meta: {MAPPING_CACHE_META_PATH}")

    return mapping, False


def iter_review_chunks(zip_reviews: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    print("\nPreparing review chunk iterator...")
    supported = inspect_zip_members(zip_reviews)
    member_name = _select_member(supported, zip_reviews)

    dtype_ids = {
        "rcid": "string",
        "ultimate_parent_rcid": "string",
    }

    member_lower = member_name.lower()
    if member_lower.endswith(".csv"):
        encoding = _guess_csv_encoding_from_zip_member(zip_reviews, member_name, is_gzip=False)
        print(f"Reading review CSV member with encoding: {encoding}")
        with ZipFile(zip_reviews, "r") as zf, zf.open(member_name, "r") as f:
            reader = pd.read_csv(
                f,
                chunksize=chunksize,
                dtype=dtype_ids,
                low_memory=False,
                encoding=encoding,
                encoding_errors="replace",
            )
            for chunk in reader:
                yield chunk
        return

    if member_lower.endswith(".csv.gz"):
        encoding = _guess_csv_encoding_from_zip_member(zip_reviews, member_name, is_gzip=True)
        print(f"Reading review CSV.GZ member with encoding: {encoding}")
        with ZipFile(zip_reviews, "r") as zf, zf.open(member_name, "r") as f:
            with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                reader = pd.read_csv(
                    gz,
                    chunksize=chunksize,
                    dtype=dtype_ids,
                    low_memory=False,
                    encoding=encoding,
                    encoding_errors="replace",
                )
                for chunk in reader:
                    yield chunk
        return

    if member_lower.endswith(".parquet"):
        tmp_path = _extract_member_to_tempfile(zip_reviews, member_name)
        try:
            parquet_file = pq.ParquetFile(tmp_path)
            for batch in parquet_file.iter_batches(batch_size=chunksize):
                chunk = batch.to_pandas()
                for col in ("rcid", "ultimate_parent_rcid"):
                    if col in chunk.columns:
                        chunk[col] = chunk[col].astype("string")
                yield chunk
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return

    raise ValueError(f"Unsupported member format for reviews: {member_name}")


def merge_chunk_with_gvkey(chunk: pd.DataFrame, mapping: pd.Series) -> pd.DataFrame:
    out = chunk.copy()

    for col in ("rcid", "ultimate_parent_rcid"):
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out), dtype="string")
        out[col] = _normalize_string(out[col])

    gvkey_by_rcid = out["rcid"].map(mapping)
    gvkey_by_parent = out["ultimate_parent_rcid"].map(mapping)

    out["gvkey"] = gvkey_by_rcid.combine_first(gvkey_by_parent)

    match_source = pd.Series(pd.NA, index=out.index, dtype="string")
    match_source.loc[gvkey_by_rcid.notna()] = "rcid"
    match_source.loc[gvkey_by_rcid.isna() & gvkey_by_parent.notna()] = "ultimate_parent_rcid"
    out["gvkey_match_source"] = match_source

    out["gvkey"] = _normalize_string(out["gvkey"])
    return out


def _align_table_to_schema(table: pa.Table, target_schema: pa.Schema) -> pa.Table:
    current = table

    # Ensure all target columns exist.
    missing_cols = [name for name in target_schema.names if name not in current.column_names]
    for name in missing_cols:
        n = current.num_rows
        current = current.append_column(name, pa.nulls(n))

    # Select and order columns to match target schema.
    current = current.select(target_schema.names)

    # Cast each column to target type.
    casted_arrays = []
    for field in target_schema:
        arr = current[field.name]
        if arr.type == field.type:
            casted_arrays.append(arr)
        else:
            casted_arrays.append(pc.cast(arr, target_type=field.type, safe=False))

    return pa.Table.from_arrays(casted_arrays, schema=target_schema)


def main() -> None:
    print("Starting merge_revelio_reviews_with_gvkey job...")
    print(f"Review chunk size: {CHUNKSIZE:,}")
    print(f"Common chunk size: {COMMON_CHUNKSIZE:,}")
    print(f"Use mapping cache: {USE_MAPPING_CACHE}")
    print(f"Force rebuild mapping cache: {FORCE_REBUILD_MAPPING_CACHE}")
    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    mapping, mapping_cache_used = read_revelio_common_mapping(REVELIO_COMMON_PATH)

    total_rows_read = 0
    rows_matched_by_rcid = 0
    rows_matched_by_ultimate_parent_rcid = 0
    rows_kept = 0
    rows_dropped_unmatched = 0

    unique_gvkeys: set[str] = set()
    unique_rcids: set[str] = set()

    writer: Optional[pq.ParquetWriter] = None
    writer_schema: Optional[pa.Schema] = None
    review_columns: Optional[List[str]] = None

    try:
        for chunk_idx, chunk in enumerate(iter_review_chunks(REVIEWS_ZIP_PATH, CHUNKSIZE), start=1):
            if review_columns is None:
                review_columns = list(chunk.columns)

            merged = merge_chunk_with_gvkey(chunk, mapping)

            chunk_total = len(merged)
            chunk_matched_by_rcid = int((merged["gvkey_match_source"] == "rcid").sum())
            chunk_matched_by_parent = int(
                (merged["gvkey_match_source"] == "ultimate_parent_rcid").sum()
            )

            kept = merged[merged["gvkey"].notna()].copy()
            chunk_kept = len(kept)
            chunk_dropped = chunk_total - chunk_kept

            total_rows_read += chunk_total
            rows_matched_by_rcid += chunk_matched_by_rcid
            rows_matched_by_ultimate_parent_rcid += chunk_matched_by_parent
            rows_kept += chunk_kept
            rows_dropped_unmatched += chunk_dropped

            if chunk_kept > 0:
                unique_gvkeys.update(kept["gvkey"].dropna().astype(str).unique().tolist())
                unique_rcids.update(kept["rcid"].dropna().astype(str).unique().tolist())

                table = pa.Table.from_pandas(kept, preserve_index=False)
                if writer is None:
                    writer_schema = table.schema
                    writer = pq.ParquetWriter(
                        OUTPUT_PARQUET_PATH,
                        writer_schema,
                        compression="snappy",
                    )
                else:
                    table = _align_table_to_schema(table, writer_schema)
                writer.write_table(table)

            print(
                f"Chunk {chunk_idx}: read={chunk_total:,}, matched_rcid={chunk_matched_by_rcid:,}, "
                f"matched_ultimate_parent_rcid={chunk_matched_by_parent:,}, kept={chunk_kept:,}, "
                f"dropped={chunk_dropped:,}"
            )
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        print("No matched rows found. Writing an empty parquet file with expected columns.")
        empty_cols = (review_columns or []) + ["gvkey", "gvkey_match_source"]
        empty_df = pd.DataFrame({col: pd.Series(dtype="string") for col in empty_cols})
        empty_table = pa.Table.from_pandas(empty_df, preserve_index=False)
        pq.write_table(empty_table, OUTPUT_PARQUET_PATH, compression="snappy")

    diagnostics: Dict[str, object] = {
        "rows_read": int(total_rows_read),
        "rows_matched_by_rcid": int(rows_matched_by_rcid),
        "rows_matched_by_ultimate_parent_rcid": int(rows_matched_by_ultimate_parent_rcid),
        "rows_with_gvkey_kept": int(rows_kept),
        "rows_unmatched_dropped": int(rows_dropped_unmatched),
        "unique_gvkeys": int(len(unique_gvkeys)),
        "unique_rcids_in_output": int(len(unique_rcids)),
        "output_file_path": str(OUTPUT_PARQUET_PATH),
        "mapping_cache_used": bool(mapping_cache_used),
        "mapping_cache_path": str(MAPPING_CACHE_PATH),
        "script_run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "chunksize": CHUNKSIZE,
    }

    with OUTPUT_DIAG_PATH.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print("\nDiagnostics summary:")
    print(json.dumps(diagnostics, indent=2))
    print(f"\nWrote parquet to: {OUTPUT_PARQUET_PATH}")
    print(f"Wrote diagnostics to: {OUTPUT_DIAG_PATH}")


if __name__ == "__main__":
    main()
