from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CHUNKSIZE = int(os.getenv("GLASSDOOR_FIRM_YEAR_CHUNKSIZE", "1000000"))

INPUT_PATH = (
    Path("/data/disk4/workspace/projects/glassdoor/outputs")
    / "glassdoor_review_level_clean.parquet"
)
OUTPUT_MAJOR_PATH = (
    Path("/data/disk4/workspace/projects/glassdoor/outputs")
    / "firm_year_glassdoor_major_customer.parquet"
)
OUTPUT_UNION_PATH = (
    Path("/data/disk4/workspace/projects/glassdoor/outputs")
    / "firm_year_glassdoor_union.parquet"
)
OUTPUT_DIAG_PATH = (
    Path("/data/disk4/workspace/projects/glassdoor/outputs")
    / "firm_year_glassdoor_aggregate_diagnostics.json"
)

RATING_MAP = {
    "gd_rating": "GD_rating",
    "gd_outlook": "GD_outlook",
    "gd_career_opp": "GD_career_opp",
    "gd_ceo": "GD_ceo",
    "gd_comp_benefit": "GD_comp_benefit",
    "gd_culture": "GD_culture",
    "gd_diversity": "GD_diversity",
    "gd_recommend": "GD_recommend",
    "gd_senior_mgmt": "GD_senior_mgmt",
    "gd_wlb": "GD_wlb",
}

TEXT_MEAN_VARS = [
    "summary_len",
    "advice_len",
    "pros_len",
    "cons_len",
    "total_text_len",
    "pros_word_count",
    "cons_word_count",
]

JOB_GROUP_DUMMY_MAP = {
    "sales": "job_sales",
    "rd": "job_rd",
    "operations": "job_operations",
    "admin": "job_admin",
    "management": "job_management",
    "senior": "job_senior",
    "entry_level": "job_entry_level",
    "client_facing": "job_client_facing",
    "high_hc": "job_high_hc",
    "specialized": "job_specialized",
    "core_business": "job_core_business",
    "performance_linked": "job_performance_linked",
    "compliance": "job_compliance",
    "low_level_core": "job_low_level_core",
    "high_level_core": "job_high_level_core",
    "non_core_staff": "job_non_core_staff",
}

UNION_ROLE_DUMMY_COLUMNS = [
    "role_management_supervisory",
    "role_legal",
    "role_hr_labor_relations",
    "role_strategy_corporate",
    "role_sales_commission",
    "role_high_level_professional",
    "role_owner_nonemployee",
    "role_rank_and_file_likely",
]

UNION_ROLE_SUMMARY_FLAG_COLUMNS = [
    "role_likely_excluded_from_union",
    "role_likely_unionizable",
    "role_ambiguous_union_status",
]

UNION_ROLE_CLASS_VALUES = [
    "likely_excluded",
    "likely_unionizable",
    "ambiguous",
]

SUBGROUP_RATING_VARS = [
    "GD_rating",
    "GD_career_opp",
    "GD_comp_benefit",
    "GD_senior_mgmt",
    "GD_wlb",
    "GD_culture",
    "GD_diversity",
]

BASE_COLUMNS = [
    "gvkey",
    "review_year",
    "rcid",
    "company",
    "ultimate_parent_company_name",
    "is_current_employee",
    "is_former_employee",
]

ALL_REQUIRED_COLUMNS = (
    BASE_COLUMNS
    + list(RATING_MAP.keys())
    + TEXT_MEAN_VARS
    + list(JOB_GROUP_DUMMY_MAP.values())
    + UNION_ROLE_DUMMY_COLUMNS
    + UNION_ROLE_SUMMARY_FLAG_COLUMNS
    + ["role_union_classification"]
)


def iter_review_chunks(input_path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    pf = pq.ParquetFile(input_path)
    schema_names = set(pf.schema.names)
    selected = [c for c in ALL_REQUIRED_COLUMNS if c in schema_names]

    print(f"Reading columns ({len(selected)}): {selected}")
    for batch in pf.iter_batches(batch_size=chunksize, columns=selected):
        yield batch.to_pandas()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def mode_string(counter: Counter) -> str | None:
    if not counter:
        return None
    max_cnt = max(counter.values())
    candidates = sorted([k for k, v in counter.items() if v == max_cnt])
    return candidates[0] if candidates else None


def _new_firm_year_state() -> Dict:
    return {
        "n_reviews": 0,
        "n_current_emp": 0,
        "n_former_emp": 0,
        "rcid_set": set(),
        "company_set": set(),
        "company_counter": Counter(),
        "ultimate_counter": Counter(),
        "rating_sum": defaultdict(float),
        "rating_count": defaultdict(int),
        "text_sum": defaultdict(float),
        "text_count": defaultdict(int),
        "job_dummy_sum": defaultdict(float),
        "subgroup_review_count": defaultdict(int),
        "subgroup_rating_sum": defaultdict(float),
        "subgroup_rating_count": defaultdict(int),
        "union_role_dummy_sum": defaultdict(float),
        "union_role_flag_sum": defaultdict(float),
        "union_role_class_counter": Counter(),
    }


def aggregate_chunk_to_firm_year(chunk: pd.DataFrame) -> Tuple[DefaultDict[Tuple[str, int], Dict], int, int]:
    partial: DefaultDict[Tuple[str, int], Dict] = defaultdict(_new_firm_year_state)

    # Standardize keys and keep usable rows only.
    chunk["gvkey"] = chunk["gvkey"].astype("string").str.strip()
    chunk["review_year"] = safe_numeric(chunk["review_year"]).astype("Int64")

    before = len(chunk)
    chunk = chunk[chunk["gvkey"].notna() & chunk["review_year"].notna()].copy()
    used = len(chunk)

    if used == 0:
        return partial, before, used

    # Normalize numeric aggregands.
    for src in RATING_MAP:
        if src in chunk.columns:
            chunk[src] = safe_numeric(chunk[src])

    for var in TEXT_MEAN_VARS:
        if var in chunk.columns:
            chunk[var] = safe_numeric(chunk[var])

    for dummy in JOB_GROUP_DUMMY_MAP.values():
        if dummy in chunk.columns:
            chunk[dummy] = safe_numeric(chunk[dummy]).fillna(0)
            chunk[dummy] = np.where(chunk[dummy] > 0, 1.0, 0.0)

    for dummy in UNION_ROLE_DUMMY_COLUMNS:
        if dummy in chunk.columns:
            chunk[dummy] = safe_numeric(chunk[dummy]).fillna(0)
            chunk[dummy] = np.where(chunk[dummy] > 0, 1.0, 0.0)

    for flag in UNION_ROLE_SUMMARY_FLAG_COLUMNS:
        if flag in chunk.columns:
            chunk[flag] = safe_numeric(chunk[flag]).fillna(0)
            chunk[flag] = np.where(chunk[flag] > 0, 1.0, 0.0)

    if "is_current_employee" in chunk.columns:
        chunk["is_current_employee"] = safe_numeric(chunk["is_current_employee"])
    else:
        chunk["is_current_employee"] = np.nan

    if "is_former_employee" in chunk.columns:
        chunk["is_former_employee"] = safe_numeric(chunk["is_former_employee"])
    else:
        chunk["is_former_employee"] = np.nan

    for row in chunk.itertuples(index=False):
        gvkey = getattr(row, "gvkey")
        review_year = int(getattr(row, "review_year"))
        key = (str(gvkey), review_year)
        st = partial[key]

        st["n_reviews"] += 1

        if getattr(row, "is_current_employee", np.nan) == 1:
            st["n_current_emp"] += 1
        if getattr(row, "is_former_employee", np.nan) == 1:
            st["n_former_emp"] += 1

        rcid = getattr(row, "rcid", None)
        if pd.notna(rcid):
            st["rcid_set"].add(str(rcid).strip())

        company = getattr(row, "company", None)
        if pd.notna(company):
            company_s = str(company).strip()
            if company_s:
                st["company_set"].add(company_s)
                st["company_counter"][company_s] += 1

        up_company = getattr(row, "ultimate_parent_company_name", None)
        if pd.notna(up_company):
            up_s = str(up_company).strip()
            if up_s:
                st["ultimate_counter"][up_s] += 1

        # Base rating means and counts.
        for src, dst in RATING_MAP.items():
            val = getattr(row, src, np.nan)
            if pd.notna(val):
                st["rating_sum"][dst] += float(val)
                st["rating_count"][dst] += 1

        # Text means.
        for var in TEXT_MEAN_VARS:
            val = getattr(row, var, np.nan)
            if pd.notna(val):
                st["text_sum"][var] += float(val)
                st["text_count"][var] += 1

        # Job composition shares and subgroup rating means.
        for grp, dummy_col in JOB_GROUP_DUMMY_MAP.items():
            is_in_group = getattr(row, dummy_col, 0.0)
            if pd.notna(is_in_group):
                st["job_dummy_sum"][grp] += float(is_in_group)

            if is_in_group == 1:
                st["subgroup_review_count"][grp] += 1
                for rating_var in SUBGROUP_RATING_VARS:
                    src_col = next(k for k, v in RATING_MAP.items() if v == rating_var)
                    r_val = getattr(row, src_col, np.nan)
                    if pd.notna(r_val):
                        key2 = f"{grp}::{rating_var}"
                        st["subgroup_rating_sum"][key2] += float(r_val)
                        st["subgroup_rating_count"][key2] += 1

        # Union exclusion role shares and counts.
        for dummy_col in UNION_ROLE_DUMMY_COLUMNS:
            val = getattr(row, dummy_col, 0.0)
            if pd.notna(val):
                st["union_role_dummy_sum"][dummy_col] += float(val)

        for flag_col in UNION_ROLE_SUMMARY_FLAG_COLUMNS:
            val = getattr(row, flag_col, 0.0)
            if pd.notna(val):
                st["union_role_flag_sum"][flag_col] += float(val)

        cls = getattr(row, "role_union_classification", None)
        if pd.notna(cls):
            cls_s = str(cls).strip()
            if cls_s in UNION_ROLE_CLASS_VALUES:
                st["union_role_class_counter"][cls_s] += 1

    return partial, before, used


def combine_partial_aggregates(
    combined: DefaultDict[Tuple[str, int], Dict],
    partial: DefaultDict[Tuple[str, int], Dict],
) -> None:
    for key, st2 in partial.items():
        st1 = combined[key]
        st1["n_reviews"] += st2["n_reviews"]
        st1["n_current_emp"] += st2["n_current_emp"]
        st1["n_former_emp"] += st2["n_former_emp"]

        st1["rcid_set"].update(st2["rcid_set"])
        st1["company_set"].update(st2["company_set"])
        st1["company_counter"].update(st2["company_counter"])
        st1["ultimate_counter"].update(st2["ultimate_counter"])

        for k2, v2 in st2["rating_sum"].items():
            st1["rating_sum"][k2] += v2
        for k2, v2 in st2["rating_count"].items():
            st1["rating_count"][k2] += v2

        for k2, v2 in st2["text_sum"].items():
            st1["text_sum"][k2] += v2
        for k2, v2 in st2["text_count"].items():
            st1["text_count"][k2] += v2

        for k2, v2 in st2["job_dummy_sum"].items():
            st1["job_dummy_sum"][k2] += v2

        for k2, v2 in st2["subgroup_review_count"].items():
            st1["subgroup_review_count"][k2] += v2
        for k2, v2 in st2["subgroup_rating_sum"].items():
            st1["subgroup_rating_sum"][k2] += v2
        for k2, v2 in st2["subgroup_rating_count"].items():
            st1["subgroup_rating_count"][k2] += v2

        for k2, v2 in st2["union_role_dummy_sum"].items():
            st1["union_role_dummy_sum"][k2] += v2
        for k2, v2 in st2["union_role_flag_sum"].items():
            st1["union_role_flag_sum"][k2] += v2
        st1["union_role_class_counter"].update(st2["union_role_class_counter"])


def finalize_firm_year_panel(combined: DefaultDict[Tuple[str, int], Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for (gvkey, year), st in combined.items():
        row: Dict = {
            "gvkey": gvkey,
            "review_year": int(year),
            "n_reviews": int(st["n_reviews"]),
            "n_current_emp": int(st["n_current_emp"]),
            "n_former_emp": int(st["n_former_emp"]),
            "n_unique_rcid": int(len(st["rcid_set"])),
            "n_unique_company_names": int(len(st["company_set"])),
            "company_name_mode": mode_string(st["company_counter"]),
            "ultimate_parent_company_name_mode": mode_string(st["ultimate_counter"]),
        }

        denom = row["n_current_emp"] + row["n_former_emp"]
        row["pct_current"] = (row["n_current_emp"] / denom) if denom > 0 else np.nan

        # Ratings and counts
        for dst in RATING_MAP.values():
            cnt = int(st["rating_count"].get(dst, 0))
            sm = float(st["rating_sum"].get(dst, 0.0))
            row[dst] = (sm / cnt) if cnt > 0 else np.nan
            row[f"n_{dst}"] = cnt

        # Text means
        for var in TEXT_MEAN_VARS:
            cnt = int(st["text_count"].get(var, 0))
            sm = float(st["text_sum"].get(var, 0.0))
            row[f"avg_{var}"] = (sm / cnt) if cnt > 0 else np.nan

        # Job shares
        for grp in JOB_GROUP_DUMMY_MAP:
            sm = float(st["job_dummy_sum"].get(grp, 0.0))
            row[f"pct_{grp}"] = (sm / row["n_reviews"]) if row["n_reviews"] > 0 else np.nan

        # Subgroup counts and subgroup rating means
        for grp in JOB_GROUP_DUMMY_MAP:
            n_grp = int(st["subgroup_review_count"].get(grp, 0))
            row[f"n_{grp}_reviews"] = n_grp

            for rating_var in SUBGROUP_RATING_VARS:
                key2 = f"{grp}::{rating_var}"
                cnt = int(st["subgroup_rating_count"].get(key2, 0))
                sm = float(st["subgroup_rating_sum"].get(key2, 0.0))
                row[f"{grp}_{rating_var}"] = (sm / cnt) if cnt > 0 else np.nan

        row["has_10_reviews"] = int(row["n_reviews"] >= 10)
        row["has_25_reviews"] = int(row["n_reviews"] >= 25)
        row["has_50_reviews"] = int(row["n_reviews"] >= 50)

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["gvkey", "review_year"]).reset_index(drop=True)
    return out


def add_union_role_aggregates(
    union_panel: pd.DataFrame,
    combined: DefaultDict[Tuple[str, int], Dict],
) -> pd.DataFrame:
    if union_panel.empty:
        return union_panel

    extra_rows: List[Dict] = []
    for (gvkey, year), st in combined.items():
        n_reviews = int(st["n_reviews"])
        row: Dict = {
            "gvkey": str(gvkey),
            "review_year": int(year),
        }

        for dummy_col in UNION_ROLE_DUMMY_COLUMNS:
            sm = float(st["union_role_dummy_sum"].get(dummy_col, 0.0))
            row[f"share_{dummy_col}"] = (sm / n_reviews) if n_reviews > 0 else np.nan

        for flag_col in UNION_ROLE_SUMMARY_FLAG_COLUMNS:
            sm = float(st["union_role_flag_sum"].get(flag_col, 0.0))
            row[f"n_{flag_col}"] = int(round(sm))
            row[f"share_{flag_col}"] = (sm / n_reviews) if n_reviews > 0 else np.nan

        for cls in UNION_ROLE_CLASS_VALUES:
            cnt = int(st["union_role_class_counter"].get(cls, 0))
            row[f"n_role_class_{cls}"] = cnt
            row[f"share_role_class_{cls}"] = (cnt / n_reviews) if n_reviews > 0 else np.nan

        row["role_union_classification_mode"] = mode_string(st["union_role_class_counter"])
        extra_rows.append(row)

    extra_df = pd.DataFrame(extra_rows)
    out = union_panel.merge(extra_df, on=["gvkey", "review_year"], how="left")
    return out


def write_outputs(major_df: pd.DataFrame, union_df: pd.DataFrame, major_path: Path, union_path: Path) -> None:
    major_path.parent.mkdir(parents=True, exist_ok=True)
    major_df.to_parquet(major_path, index=False, compression="snappy")
    union_df.to_parquet(union_path, index=False, compression="snappy")


def main() -> None:
    print("=" * 100)
    print("build_firm_year_aggregates.py")
    print(f"Input: {INPUT_PATH}")
    print(f"Output major: {OUTPUT_MAJOR_PATH}")
    print(f"Output union: {OUTPUT_UNION_PATH}")
    print(f"Chunksize: {CHUNKSIZE:,}")
    print("=" * 100)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    combined: DefaultDict[Tuple[str, int], Dict] = defaultdict(_new_firm_year_state)

    total_rows_read = 0
    total_rows_used = 0

    for chunk_idx, chunk in enumerate(iter_review_chunks(INPUT_PATH, CHUNKSIZE), start=1):
        partial, before, used = aggregate_chunk_to_firm_year(chunk)
        combine_partial_aggregates(combined, partial)

        total_rows_read += before
        total_rows_used += used

        print(
            f"Chunk {chunk_idx}: read={before:,}, used={used:,}, "
            f"firm_year_keys_so_far={len(combined):,}"
        )

    major_panel = finalize_firm_year_panel(combined)
    union_panel = add_union_role_aggregates(major_panel.copy(), combined)

    # Validate one row per gvkey x review_year
    if not major_panel.empty:
        dup = major_panel.duplicated(["gvkey", "review_year"]).sum()
        if dup > 0:
            raise ValueError(f"Found duplicate gvkey-review_year rows after aggregation: {dup}")

    # Keep major-customer output schema unchanged; union output contains extra role aggregates.
    write_outputs(major_panel, union_panel, OUTPUT_MAJOR_PATH, OUTPUT_UNION_PATH)

    n_fy = len(major_panel)
    n_gvkey = int(major_panel["gvkey"].nunique()) if n_fy > 0 else 0
    year_min = int(major_panel["review_year"].min()) if n_fy > 0 else None
    year_max = int(major_panel["review_year"].max()) if n_fy > 0 else None

    if n_fy > 0:
        n_reviews_desc = major_panel["n_reviews"].describe().to_dict()
        n_reviews_dist = {k: (float(v) if pd.notna(v) else None) for k, v in n_reviews_desc.items()}
    else:
        n_reviews_dist = {}

    n_has_10 = int(major_panel["has_10_reviews"].sum()) if n_fy > 0 else 0
    n_has_25 = int(major_panel["has_25_reviews"].sum()) if n_fy > 0 else 0
    n_has_50 = int(major_panel["has_50_reviews"].sum()) if n_fy > 0 else 0

    major_missingness = {}
    for col in RATING_MAP.values():
        major_missingness[col] = float(major_panel[col].isna().mean()) if n_fy > 0 else None

    union_added_columns = sorted([c for c in union_panel.columns if c not in major_panel.columns])

    diagnostics = {
        "input_path": str(INPUT_PATH),
        "output_major_path": str(OUTPUT_MAJOR_PATH),
        "output_union_path": str(OUTPUT_UNION_PATH),
        "total_review_rows_read": int(total_rows_read),
        "total_review_rows_used": int(total_rows_used),
        "firm_year_observations": int(n_fy),
        "unique_gvkeys": int(n_gvkey),
        "review_year_min": year_min,
        "review_year_max": year_max,
        "n_reviews_distribution": n_reviews_dist,
        "n_firm_year_has_10_reviews": n_has_10,
        "n_firm_year_has_25_reviews": n_has_25,
        "n_firm_year_has_50_reviews": n_has_50,
        "share_firm_year_has_10_reviews": (n_has_10 / n_fy) if n_fy > 0 else None,
        "share_firm_year_has_25_reviews": (n_has_25 / n_fy) if n_fy > 0 else None,
        "share_firm_year_has_50_reviews": (n_has_50 / n_fy) if n_fy > 0 else None,
        "missingness_rates_major_ratings": major_missingness,
        "major_output_column_count": int(len(major_panel.columns)),
        "union_output_column_count": int(len(union_panel.columns)),
        "union_added_columns": union_added_columns,
        "script_run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    OUTPUT_DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_DIAG_PATH.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print("\nDiagnostics:")
    print(json.dumps(diagnostics, indent=2))
    print(f"\nWrote major-customer file: {OUTPUT_MAJOR_PATH}")
    print(f"Wrote union file: {OUTPUT_UNION_PATH}")
    print(f"Wrote diagnostics: {OUTPUT_DIAG_PATH}")


if __name__ == "__main__":
    main()
