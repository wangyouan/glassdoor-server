from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNKSIZE = int(os.getenv("GLASSDOOR_REVIEW_CLEAN_CHUNKSIZE", "1000000"))

INPUT_PARQUET_PATH = (
    Path("/data/disk4/workspace/projects/glassdoor/outputs")
    / "sentiment_individual_reviews_with_gvkey.parquet"
)
OUTPUT_PARQUET_PATH = (
    Path("/data/disk4/workspace/projects/glassdoor/outputs")
    / "glassdoor_review_level_clean.parquet"
)
OUTPUT_DIAG_PATH = (
    Path("/data/disk4/workspace/projects/glassdoor/outputs")
    / "glassdoor_review_level_clean_diagnostics.json"
)

# ---------------------------------------------------------------------------
# Rating column renaming map
# ---------------------------------------------------------------------------

RATING_RENAME: Dict[str, str] = {
    "rating_overall": "gd_rating",
    "rating_business_outlook": "gd_outlook",
    "rating_career_opportunities": "gd_career_opp",
    "rating_ceo": "gd_ceo",
    "rating_compensation_and_benefits": "gd_comp_benefit",
    "rating_culture_and_values": "gd_culture",
    "rating_diversity_and_inclusion": "gd_diversity",
    "rating_recommend_to_friend": "gd_recommend",
    "rating_senior_leadership": "gd_senior_mgmt",
    "rating_work_life_balance": "gd_wlb",
}

# ---------------------------------------------------------------------------
# Job-category keyword rules (edit these lists freely)
# ---------------------------------------------------------------------------

JOB_KEYWORD_RULES: Dict[str, List[str]] = {
    "job_sales": [
        "sale", "sales", "account exec", "account manager", "business development",
        "biz dev", "bd ", "channel", "revenue", "quota", "client success",
        "customer success", "pre-sales", "presales",
    ],
    "job_rd": [
        "research", "scientist", "data scientist", "machine learning", "ml ",
        "engineer", "developer", "dev ", "software", "r&d", "r & d",
        "analytics", "analyst", "quantitative", "quant ", "phd", "science",
        "product manager", "pm ", "product owner",
    ],
    "job_operations": [
        "operation", "ops", "supply chain", "logistics", "warehouse",
        "fulfillment", "procurement", "sourcing", "manufacturing", "production",
        "quality assurance", "qa ", "quality control", "facilities",
    ],
    "job_admin": [
        "admin", "administrative", "receptionist", "office manager",
        "coordinator", "assistant", "secretary", "clerical", "support staff",
    ],
    "job_management": [
        "manager", "director", "vp ", "vice president", "president",
        "chief", "ceo", "coo", "cfo", "cto", "head of", "lead", "principal",
        "supervisor", "team lead", "managing",
    ],
    "job_senior": [
        "senior", "sr.", "sr ", "staff ", "principal", "distinguished",
        "fellow ", "architect", "lead ", "expert", "specialist ii", "specialist iii",
    ],
    "job_entry_level": [
        "intern", "associate", "junior", "jr.", "jr ", "entry", "trainee",
        "apprentice", "new grad", "fresh",
    ],
    "job_client_facing": [
        "customer service", "customer support", "client service", "client facing",
        "account", "support representative", "helpdesk", "help desk",
        "service representative", "call center", "contact center",
    ],
    "job_high_hc": [
        "human capital", "human resources", "hr ", "h.r.", "talent acquisition",
        "recruiter", "recruiting", "people operations", "people team",
        "people partner", "hrbp",
    ],
    "job_specialized": [
        "legal", "counsel", "attorney", "compliance", "regulatory",
        "finance", "financial", "accounting", "accountant", "tax ",
        "treasury", "audit", "actuarial", "actuary",
    ],
    "job_core_business": [
        "product", "marketing", "growth", "strategy", "corporate development",
        "investor relation", "communications", "public relations", "pr ",
    ],
    "job_performance_linked": [
        "trader", "portfolio", "investment", "broker", "underwriter",
        "agent", "representative", "field rep", "territory",
    ],
    "job_compliance": [
        "compliance", "risk", "internal audit", "sox", "regulatory affairs",
        "policy", "ethics", "governance",
    ],
    "job_low_level_core": [
        "technician", "tech ", "associate engineer", "junior engineer",
        "process associate", "analyst i", "analyst ii",
    ],
    "job_high_level_core": [
        "senior engineer", "senior developer", "senior analyst",
        "senior manager", "senior director", "vp of engineering",
        "principal engineer", "staff engineer",
    ],
    "job_non_core_staff": [
        "cafeteria", "security guard", "janitor", "cleaning", "maintenance",
        "driver", "delivery", "retail associate", "cashier", "store associate",
    ],
}

# ---------------------------------------------------------------------------
# Union exclusion role classification (heuristic based on job titles)
# These categories identify roles typically excluded from unionization under
# U.S. NLRA principles or roles less likely to be bargaining-unit workers.
# This is empirical classification for analysis, NOT legal determination.
# ---------------------------------------------------------------------------

UNION_EXCLUSION_KEYWORD_RULES: Dict[str, List[str]] = {
    "role_management_supervisory": [
        "manager", "director", "executive", "vp ", "vice president", "president",
        "chief", "ceo", "cfo", "coo", "cto", "cio", "chro", "head of",
        "managing director", "general manager", "store manager", "operations manager",
        "supervisor", "team lead", "shift lead", "lead supervisor", "foreman",
        "forewoman", "principal", "senior leadership",
    ],
    "role_legal": [
        "attorney", "lawyer", "legal", "counsel", "general counsel",
        "corporate counsel", "associate counsel", "litigation", "paralegal",
        "law clerk", "legal assistant",
    ],
    "role_hr_labor_relations": [
        "human resources", "hr ", "h.r.", "people operations", "people partner",
        "hrbp", "recruiter", "recruiting", "talent acquisition", "employee relations",
        "labor relations", "industrial relations", "compensation", "benefits manager",
        "payroll manager",
    ],
    "role_strategy_corporate": [
        "strategy", "corporate development", "corp dev", "business strategy",
        "management consultant", "consultant", "internal consultant", "chief of staff",
        "transformation", "strategic initiatives", "corporate planning",
    ],
    "role_sales_commission": [
        "sales", "account executive", "account manager", "business development",
        "bd ", "sales executive", "sales manager", "territory manager", "quota",
        "commission", "revenue", "channel sales", "enterprise sales",
        "customer success manager",
    ],
    "role_high_level_professional": [
        "software engineer", "senior software engineer", "staff engineer",
        "principal engineer", "architect", "data scientist", "senior data scientist",
        "machine learning engineer", "ml engineer", "quant", "quantitative",
        "investment banker", "investment banking", "financial analyst",
        "finance manager", "portfolio manager", "trader", "research scientist",
        "product manager", "product owner",
    ],
    "role_owner_nonemployee": [
        "founder", "co-founder", "owner", "partner", "managing partner",
        "independent contractor", "contractor", "freelancer", "self-employed",
        "consultant contractor",
    ],
    "role_rank_and_file_likely": [
        "operator", "technician", "mechanic", "assembler", "production",
        "manufacturing", "warehouse", "fulfillment", "logistics", "driver",
        "delivery", "maintenance", "janitor", "cleaner", "security guard",
        "retail associate", "cashier", "store associate", "customer service",
        "customer support", "call center", "contact center", "service representative",
        "field technician", "installer", "clerk", "hourly", "line worker",
        "associate", "assistant", "front desk", "receptionist", "server",
        "cook", "nurse", "caregiver",
    ],
}

# ---------------------------------------------------------------------------
# Final output columns
# ---------------------------------------------------------------------------

FINAL_COLUMNS: List[str] = [
    # Identifiers
    "gvkey", "rcid", "ultimate_parent_rcid", "company_id", "company",
    "ultimate_parent_company_name", "review_id",
    # Dates
    "review_date_clean", "review_year", "review_month", "review_ym",
    # Location
    "country", "state", "metro_area", "location_raw",
    # Job / reviewer
    "job_title_raw", "job_title_clean", "role_k1500_clean", "seniority_clean",
    "employment_status_clean", "is_current_employee", "is_former_employee",
    # Job dummies (general classification)
    *list(JOB_KEYWORD_RULES.keys()),
    # Union exclusion role dummies (heuristic NLRA-based classification)
    *list(UNION_EXCLUSION_KEYWORD_RULES.keys()),
    # Union role classification summary
    "role_likely_excluded_from_union", "role_likely_unionizable",
    "role_ambiguous_union_status", "role_exclusion_reason",
    "role_union_classification",
    # Ratings
    *list(RATING_RENAME.values()),
    # Text lengths
    "summary_len", "advice_len", "pros_len", "cons_len",
    "total_text_len", "pros_word_count", "cons_word_count",
    # Other review fields
    "review_count_helpful", "review_count_not_helpful",
    "review_language_id", "review_iscovid19", "gvkey_match_source",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def normalize_string(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.replace(
        {
            "": pd.NA, "nan": pd.NA, "NaN": pd.NA,
            "None": pd.NA, "NULL": pd.NA, "null": pd.NA, "<NA>": pd.NA,
        }
    )


def safe_to_numeric(
    series: pd.Series,
    col_name: str,
    invalid_counts: Dict[str, int],
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    n_invalid = int(series.notna().sum()) - int(numeric.notna().sum())
    if n_invalid > 0:
        invalid_counts[col_name] = invalid_counts.get(col_name, 0) + n_invalid
    return numeric


def parse_review_dates(df: pd.DataFrame) -> pd.DataFrame:
    date_series: Optional[pd.Series] = None

    if "review_date" in df.columns:
        date_series = pd.to_datetime(df["review_date"], errors="coerce", utc=False)

    if "review_time" in df.columns:
        fallback = pd.to_datetime(df["review_time"], errors="coerce", utc=False)
        if date_series is None:
            date_series = fallback
        else:
            date_series = date_series.fillna(fallback)

    if date_series is None:
        date_series = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    df["review_date_clean"] = date_series.dt.date.astype("object").where(
        date_series.notna(), other=None
    )
    df["review_year"] = date_series.dt.year.astype("Int64")
    df["review_month"] = date_series.dt.month.astype("Int64")
    df["review_ym"] = date_series.dt.to_period("M").astype("string").replace(
        {"NaT": pd.NA}
    )

    return df


def standardize_ratings(
    df: pd.DataFrame,
    invalid_counts: Dict[str, int],
) -> pd.DataFrame:
    for src, dst in RATING_RENAME.items():
        if src in df.columns:
            df[dst] = safe_to_numeric(df[src], dst, invalid_counts)
        else:
            df[dst] = pd.Series(np.nan, index=df.index, dtype="float64")
    return df


def standardize_employment_status(df: pd.DataFrame) -> pd.DataFrame:
    is_current = pd.array([pd.NA] * len(df), dtype="Int8")
    is_former = pd.array([pd.NA] * len(df), dtype="Int8")
    status_clean = pd.array([pd.NA] * len(df), dtype="string")

    # Use reviewer_current_job boolean if available
    if "reviewer_current_job" in df.columns:
        rcj = df["reviewer_current_job"]
        current_mask = rcj == True  # noqa: E712
        former_mask = rcj == False  # noqa: E712
        is_current = pd.array(
            [1 if c else (0 if f else pd.NA) for c, f in zip(current_mask, former_mask)],
            dtype="Int8",
        )
        is_former = pd.array(
            [1 if f else (0 if c else pd.NA) for c, f in zip(current_mask, former_mask)],
            dtype="Int8",
        )

    # Use reviewer_employment_status text as additional signal
    if "reviewer_employment_status" in df.columns:
        res = df["reviewer_employment_status"].astype("string").str.lower().str.strip()
        current_text_mask = res.str.contains("current", na=False)
        former_text_mask = res.str.contains("former|ex-", na=False, regex=True)

        for i, (ctxt, ftxt) in enumerate(zip(current_text_mask, former_text_mask)):
            if ctxt:
                is_current[i] = 1
                is_former[i] = 0
                status_clean[i] = "current"
            elif ftxt:
                is_current[i] = 0
                is_former[i] = 1
                status_clean[i] = "former"

    # Fill status_clean for cases already resolved via boolean
    if "reviewer_current_job" in df.columns and "reviewer_employment_status" not in df.columns:
        for i, val in enumerate(is_current):
            if val == 1:
                status_clean[i] = "current"
            elif val == 0:
                status_clean[i] = "former"

    df["is_current_employee"] = is_current
    df["is_former_employee"] = is_former
    df["employment_status_clean"] = status_clean
    return df


def clean_job_fields(df: pd.DataFrame) -> pd.DataFrame:
    if "job_title_raw" in df.columns:
        df["job_title_clean"] = (
            df["job_title_raw"].astype("string").str.lower().str.strip()
        )
    else:
        df["job_title_clean"] = pd.Series(pd.NA, index=df.index, dtype="string")

    if "role_k1500" in df.columns:
        df["role_k1500_clean"] = (
            df["role_k1500"].astype("string").str.lower().str.strip()
        )
    else:
        df["role_k1500_clean"] = pd.Series(pd.NA, index=df.index, dtype="string")

    if "seniority" in df.columns:
        df["seniority_clean"] = (
            df["seniority"].astype("string").str.lower().str.strip()
        )
    else:
        df["seniority_clean"] = pd.Series(pd.NA, index=df.index, dtype="string")

    return df


def create_job_category_dummies(df: pd.DataFrame) -> pd.DataFrame:
    # Combine all text signals into one lowercased search string per row
    title_col = df["job_title_clean"].fillna("").astype(str)
    role_col = df["role_k1500_clean"].fillna("").astype(str)
    seniority_col = df["seniority_clean"].fillna("").astype(str)
    combined = " " + title_col + " " + role_col + " " + seniority_col + " "

    for dummy_col, keywords in JOB_KEYWORD_RULES.items():
        pattern = "|".join(kw.lower() for kw in keywords)
        df[dummy_col] = combined.str.contains(pattern, na=False, regex=True).astype("Int8")

    return df


def build_conservative_keyword_pattern(keywords: List[str]) -> str:
    """Build conservative regex pattern with non-alnum boundaries."""
    cleaned = [kw.strip().lower() for kw in keywords if kw and kw.strip()]
    parts = [rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])" for kw in cleaned]
    return "|".join(parts)


def create_union_exclusion_role_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary dummies for union exclusion role categories.
    
    These dummies identify roles that are typically excluded from unionization
    under U.S. NLRA principles (managers, supervisors, confidential employees, etc.)
    and roles likely to be rank-and-file bargaining unit workers.
    
    This is HEURISTIC classification based on job title text analysis for empirical
    research purposes. It is NOT a legal determination of union eligibility.
    Ambiguous cases are left unclassified.
    """
    # Combine job title, role, and seniority into single search string
    title_col = df["job_title_clean"].fillna("").astype(str)
    role_col = df["role_k1500_clean"].fillna("").astype(str)
    seniority_col = df["seniority_clean"].fillna("").astype(str)
    combined = " " + title_col + " " + role_col + " " + seniority_col + " "

    # Create category dummies with conservative keyword matching
    for dummy_col, keywords in UNION_EXCLUSION_KEYWORD_RULES.items():
        pattern = build_conservative_keyword_pattern(keywords)
        df[dummy_col] = combined.str.contains(pattern, na=False, regex=True).astype("Int8")

    # Create summary classification variables
    exclusion_cols = [
        "role_management_supervisory", "role_legal", "role_hr_labor_relations",
        "role_strategy_corporate", "role_sales_commission", "role_high_level_professional",
        "role_owner_nonemployee",
    ]
    rank_file_col = "role_rank_and_file_likely"

    # role_likely_excluded_from_union: 1 if any exclusion category matches
    df["role_likely_excluded_from_union"] = (
        df[exclusion_cols].sum(axis=1) > 0
    ).astype("Int8")

    # role_likely_unionizable: 1 if rank-and-file AND not excluded
    df["role_likely_unionizable"] = (
        (df[rank_file_col] == 1) & (df["role_likely_excluded_from_union"] == 0)
    ).astype("Int8")

    # role_ambiguous_union_status: 1 if neither excluded nor unionizable
    df["role_ambiguous_union_status"] = (
        (df["role_likely_excluded_from_union"] == 0) & 
        (df[rank_file_col] == 0)
    ).astype("Int8")

    # role_exclusion_reason: semicolon-separated list of matched exclusion categories
    exclusion_reason_names = {
        "role_management_supervisory": "management_supervisory",
        "role_legal": "legal",
        "role_hr_labor_relations": "hr_labor_relations",
        "role_strategy_corporate": "strategy_corporate",
        "role_sales_commission": "sales_commission",
        "role_high_level_professional": "high_level_professional",
        "role_owner_nonemployee": "owner_nonemployee",
    }
    reasons = pd.Series("", index=df.index, dtype="object")
    for col in exclusion_cols:
        label = exclusion_reason_names[col]
        mask = df[col] == 1
        first_mask = mask & (reasons == "")
        add_mask = mask & (reasons != "")
        reasons.loc[first_mask] = label
        reasons.loc[add_mask] = reasons.loc[add_mask] + ";" + label
    reasons = reasons.mask(reasons == "", pd.NA)
    df["role_exclusion_reason"] = reasons.astype("string")

    # role_union_classification: categorical summary
    df["role_union_classification"] = pd.Series(
        np.select(
            [
                df["role_likely_excluded_from_union"] == 1,
                df["role_likely_unionizable"] == 1,
            ],
            ["likely_excluded", "likely_unionizable"],
            default="ambiguous",
        ),
        index=df.index,
        dtype="string",
    )

    return df


def create_text_length_vars(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = {
        "summary_len": "review_summary",
        "advice_len": "review_advice",
        "pros_len": "review_pros",
        "cons_len": "review_cons",
    }
    for new_col, src_col in text_cols.items():
        if src_col in df.columns:
            filled = df[src_col].astype("string").fillna("")
        else:
            filled = pd.Series("", index=df.index, dtype="string")
        df[new_col] = filled.str.len().astype("Int64")

    df["total_text_len"] = (
        df["summary_len"].fillna(0)
        + df["advice_len"].fillna(0)
        + df["pros_len"].fillna(0)
        + df["cons_len"].fillna(0)
    ).astype("Int64")

    for word_count_col, src_col in [("pros_word_count", "review_pros"), ("cons_word_count", "review_cons")]:
        if src_col in df.columns:
            filled = df[src_col].astype("string").fillna("")
            df[word_count_col] = filled.str.split().str.len().fillna(0).astype("Int64")
        else:
            df[word_count_col] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in FINAL_COLUMNS if c in df.columns]
    missing = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing expected output columns (will be skipped): {missing}")
    return df[existing].copy()


def align_table_to_schema(table: pa.Table, target_schema: pa.Schema) -> pa.Table:
    current = table

    missing_cols = [n for n in target_schema.names if n not in current.column_names]
    for name in missing_cols:
        current = current.append_column(name, pa.nulls(current.num_rows))

    current = current.select(target_schema.names)

    casted = []
    for field in target_schema:
        arr = current[field.name]
        if arr.type == field.type:
            casted.append(arr)
        else:
            try:
                casted.append(pc.cast(arr, target_type=field.type, safe=False))
            except Exception:
                # Fallback: cast through string then to target
                casted.append(pc.cast(
                    pc.cast(arr, pa.string(), safe=False),
                    target_type=field.type,
                    safe=False,
                ))

    return pa.Table.from_arrays(casted, schema=target_schema)


def iter_input_chunks(input_path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    pf = pq.ParquetFile(input_path)
    for batch in pf.iter_batches(batch_size=chunksize):
        yield batch.to_pandas()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 100)
    print("build_review_level_analysis_file.py")
    print(f"Input:     {INPUT_PARQUET_PATH}")
    print(f"Output:    {OUTPUT_PARQUET_PATH}")
    print(f"Chunksize: {CHUNKSIZE:,}")
    print("=" * 100)

    if not INPUT_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PARQUET_PATH}")

    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- Diagnostics accumulators ---
    total_rows_read: int = 0
    total_rows_written: int = 0
    rows_dropped_missing_gvkey: int = 0
    rows_dropped_invalid_date: int = 0
    unique_gvkeys: set[str] = set()
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    rating_nonmissing: Dict[str, int] = {c: 0 for c in RATING_RENAME.values()}
    rating_sum: Dict[str, float] = {c: 0.0 for c in RATING_RENAME.values()}
    invalid_rating_counts: Dict[str, int] = {}
    employment_status_dist: Dict[str, int] = {}
    job_dummy_counts: Dict[str, int] = {c: 0 for c in JOB_KEYWORD_RULES}
    union_role_dummy_counts: Dict[str, int] = {c: 0 for c in UNION_EXCLUSION_KEYWORD_RULES}
    union_classification_dist: Dict[str, int] = {}
    n_current_employee: int = 0
    n_valid_status: int = 0
    n_likely_excluded: int = 0
    n_likely_unionizable: int = 0
    n_ambiguous_union: int = 0

    writer: Optional[pq.ParquetWriter] = None
    writer_schema: Optional[pa.Schema] = None

    try:
        for chunk_idx, chunk in enumerate(iter_input_chunks(INPUT_PARQUET_PATH, CHUNKSIZE), start=1):
            total_rows_read += len(chunk)

            # 1. Drop missing gvkey (should not exist in input, but guard anyway)
            before = len(chunk)
            chunk = chunk[chunk["gvkey"].notna()].copy()
            rows_dropped_missing_gvkey += before - len(chunk)

            # 2. Standardize identifiers
            for col in ("gvkey", "rcid", "ultimate_parent_rcid"):
                if col in chunk.columns:
                    chunk[col] = normalize_string(chunk[col])
            for col in ("review_id", "company_id"):
                if col in chunk.columns:
                    chunk[col] = normalize_string(chunk[col].astype("string"))

            # 3. Parse dates
            chunk = parse_review_dates(chunk)

            before_date = len(chunk)
            chunk = chunk[chunk["review_date_clean"].notna()].copy()
            rows_dropped_invalid_date += before_date - len(chunk)

            if len(chunk) == 0:
                print(f"Chunk {chunk_idx}: all rows dropped (no valid date). Skipping.")
                continue

            # 4 & 5. Standardize ratings
            chunk = standardize_ratings(chunk, invalid_rating_counts)

            # 6. Employment status
            chunk = standardize_employment_status(chunk)

            # 7. Clean job fields
            chunk = clean_job_fields(chunk)

            # 8. Job category dummies
            chunk = create_job_category_dummies(chunk)

            # 8.5 Union exclusion role dummies (NLRA-based heuristic classification)
            chunk = create_union_exclusion_role_dummies(chunk)

            # 9. Text lengths
            chunk = create_text_length_vars(chunk)

            # 10. Select final columns
            chunk = select_final_columns(chunk)

            # --- Accumulate diagnostics ---
            unique_gvkeys.update(chunk["gvkey"].dropna().astype(str).tolist())

            years = chunk["review_year"].dropna()
            if len(years) > 0:
                y_min = int(years.min())
                y_max = int(years.max())
                min_year = y_min if min_year is None else min(min_year, y_min)
                max_year = y_max if max_year is None else max(max_year, y_max)

            for gd_col in RATING_RENAME.values():
                if gd_col in chunk.columns:
                    valid = chunk[gd_col].dropna()
                    rating_nonmissing[gd_col] += len(valid)
                    rating_sum[gd_col] += float(valid.sum())

            if "employment_status_clean" in chunk.columns:
                for val, cnt in chunk["employment_status_clean"].value_counts(dropna=True).items():
                    employment_status_dist[str(val)] = employment_status_dist.get(str(val), 0) + int(cnt)

            if "is_current_employee" in chunk.columns:
                n_current_employee += int((chunk["is_current_employee"] == 1).sum())
                n_valid_status += int(chunk["is_current_employee"].notna().sum())

            for dc in JOB_KEYWORD_RULES:
                if dc in chunk.columns:
                    job_dummy_counts[dc] += int((chunk[dc] == 1).sum())

            # Accumulate union exclusion role diagnostics
            for udc in UNION_EXCLUSION_KEYWORD_RULES:
                if udc in chunk.columns:
                    union_role_dummy_counts[udc] += int((chunk[udc] == 1).sum())

            if "role_union_classification" in chunk.columns:
                for val, cnt in chunk["role_union_classification"].value_counts(dropna=True).items():
                    union_classification_dist[str(val)] = union_classification_dist.get(str(val), 0) + int(cnt)

            if "role_likely_excluded_from_union" in chunk.columns:
                n_likely_excluded += int((chunk["role_likely_excluded_from_union"] == 1).sum())

            if "role_likely_unionizable" in chunk.columns:
                n_likely_unionizable += int((chunk["role_likely_unionizable"] == 1).sum())

            if "role_ambiguous_union_status" in chunk.columns:
                n_ambiguous_union += int((chunk["role_ambiguous_union_status"] == 1).sum())

            # --- Write parquet ---
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer_schema = table.schema
                writer = pq.ParquetWriter(OUTPUT_PARQUET_PATH, writer_schema, compression="snappy")
            else:
                table = align_table_to_schema(table, writer_schema)
            writer.write_table(table)
            total_rows_written += len(chunk)

            print(
                f"Chunk {chunk_idx}: read={len(chunk) + rows_dropped_missing_gvkey:,} -> "
                f"written={len(chunk):,} | "
                f"total_written={total_rows_written:,}"
            )
            if "role_union_classification" in chunk.columns:
                chunk_cls = chunk["role_union_classification"].value_counts(dropna=False)
                print(
                    "  role_union_classification: "
                    f"excluded={int(chunk_cls.get('likely_excluded', 0)):,}, "
                    f"unionizable={int(chunk_cls.get('likely_unionizable', 0)):,}, "
                    f"ambiguous={int(chunk_cls.get('ambiguous', 0)):,}"
                )

    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        print("[WARN] No rows were written. Writing empty parquet.")
        empty_schema = pa.schema([pa.field(c, pa.string()) for c in FINAL_COLUMNS])
        pq.write_table(pa.Table.from_pydict({c: [] for c in FINAL_COLUMNS}, schema=empty_schema),
                       OUTPUT_PARQUET_PATH, compression="snappy")

    # --- Build diagnostics ---
    rating_means = {
        col: round(rating_sum[col] / rating_nonmissing[col], 4) if rating_nonmissing[col] > 0 else None
        for col in RATING_RENAME.values()
    }
    share_current = (
        round(n_current_employee / n_valid_status, 4) if n_valid_status > 0 else None
    )
    share_likely_excluded = (
        round(n_likely_excluded / total_rows_written, 4) if total_rows_written > 0 else None
    )
    share_likely_unionizable = (
        round(n_likely_unionizable / total_rows_written, 4) if total_rows_written > 0 else None
    )
    share_ambiguous_union = (
        round(n_ambiguous_union / total_rows_written, 4) if total_rows_written > 0 else None
    )
    union_role_dummy_shares = {
        col: round(cnt / total_rows_written, 4) if total_rows_written > 0 else None
        for col, cnt in union_role_dummy_counts.items()
    }

    diagnostics: Dict[str, object] = {
        "total_rows_read": total_rows_read,
        "total_rows_written": total_rows_written,
        "rows_dropped_missing_gvkey": rows_dropped_missing_gvkey,
        "rows_dropped_invalid_review_date": rows_dropped_invalid_date,
        "unique_gvkeys": len(unique_gvkeys),
        "review_year_min": min_year,
        "review_year_max": max_year,
        "employment_status_distribution": employment_status_dist,
        "share_current_employee": share_current,
        "rating_nonmissing_counts": rating_nonmissing,
        "rating_means": rating_means,
        "invalid_rating_value_counts": invalid_rating_counts,
        "job_category_dummy_counts": job_dummy_counts,
        "union_exclusion_role_dummy_counts": union_role_dummy_counts,
        "union_exclusion_role_dummy_shares": union_role_dummy_shares,
        "union_classification_distribution": union_classification_dist,
        "share_likely_excluded_from_union": share_likely_excluded,
        "share_likely_unionizable": share_likely_unionizable,
        "share_ambiguous_union_status": share_ambiguous_union,
        "output_file_path": str(OUTPUT_PARQUET_PATH),
        "chunksize": CHUNKSIZE,
        "script_run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    with OUTPUT_DIAG_PATH.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print("\nDiagnostics:")
    print(json.dumps(diagnostics, indent=2))
    print(f"\nWrote parquet: {OUTPUT_PARQUET_PATH}")
    print(f"Wrote diagnostics: {OUTPUT_DIAG_PATH}")


if __name__ == "__main__":
    main()
