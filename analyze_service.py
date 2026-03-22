from pathlib import Path
from typing import Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data_hf" / "06_analytics" / "step6_hf_core_cases_light.csv"


def load_repo_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    path = csv_path or DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)

    expected_cols = [
        "hf_model_id",
        "hf_url",
        "github_url",
        "family_cluster",
        "case_cluster",
        "method_group",
        "domain_group",
        "core_case_label",
        "core_case_score",
        "analysis_include_flag",
        "is_family_representative",
        "priority_bucket",
        "candidate_tier",
        "case_study_priority",
        "primary_domain",
        "primary_task",
        "primary_family",
        "repo_style",
        "transition_archetype",
        "transition_signal_type",
        "artifact_change_focus",
        "transition_readiness",
        "sync_readiness",
        "documentation_level",
        "reproducibility_level",
        "deployment_level",
        "screen_total_score",
        "transition_value_score",
        "artifact_richness_score",
        "cross_platform_score",
        "file_count",
        "artifact_signature",
        "evidence_terms",
        "local_repo_json",
        "local_readme_txt",
        "local_manifest_json",
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    return df


def infer_product_label(row: pd.Series) -> str:
    core_label = str(row.get("core_case_label", "")).strip().lower()
    repro = str(row.get("reproducibility_level", "")).strip().lower()
    deploy = str(row.get("deployment_level", "")).strip().lower()
    readiness = str(row.get("transition_readiness", "")).strip().lower()
    repo_style = str(row.get("repo_style", "")).strip().lower()
    score = float(row.get("core_case_score", 0) or 0)
    artifact_score = float(row.get("artifact_richness_score", 0) or 0)

    if (
        core_label == "core_primary"
        and readiness in {"high", "medium"}
        and repro in {"medium", "high"}
        and artifact_score >= 4
        and score >= 24
    ):
        return "Production-ready"

    if (
        core_label in {"core_primary", "core_secondary"}
        and readiness in {"medium", "high"}
        and score >= 12
    ):
        return "Research-grade"

    if repo_style in {"notebook_project_repo", "lightweight_repo"}:
        return "Experimental"

    if deploy == "low" and repro == "low" and score < 12:
        return "Experimental"

    return "Research-grade"


def infer_recommendation(row: pd.Series) -> str:
    label = infer_product_label(row)
    repo_style = str(row.get("repo_style", "")).strip().lower()
    family = str(row.get("family_cluster", "")).strip().lower()
    task = str(row.get("primary_task", "")).strip().lower()

    if label == "Production-ready":
        return "✅ Use"

    if label == "Research-grade":
        if repo_style == "multi_artifact_repo":
            return "⚠️ Advanced"
        if family in {"patchtst", "nbeats", "lstm", "sarimax", "autoformer", "informer", "timesnet"}:
            return "⚠️ Evaluate first"
        return "⚠️ Research use"

    if task in {"forecasting", "time-series-forecasting"}:
        return "❌ Demo only"

    return "❌ Avoid"


def build_search_blob(df: pd.DataFrame) -> pd.Series:
    cols = [
        "hf_model_id",
        "family_cluster",
        "case_cluster",
        "method_group",
        "domain_group",
        "primary_domain",
        "primary_task",
        "primary_family",
        "repo_style",
        "transition_archetype",
        "transition_signal_type",
        "artifact_change_focus",
        "artifact_signature",
        "evidence_terms",
    ]

    blob = pd.Series("", index=df.index)
    for col in cols:
        blob = blob + " " + df[col].fillna("").astype(str)

    return blob.str.lower().str.strip()


def finalize_product_view(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["product_label"] = df.apply(infer_product_label, axis=1)
    df["recommendation"] = df.apply(infer_recommendation, axis=1)
    df["search_blob"] = build_search_blob(df)
    return df


def filter_ranked_results(
    df: pd.DataFrame,
    query: str = "",
    domain: str = "All",
    family: str = "All",
    label: str = "All",
    only_representatives: bool = False,
    top_n: int = 50,
) -> pd.DataFrame:
    df = finalize_product_view(df)

    if query.strip():
        q = query.strip().lower()
        terms = [t for t in q.split() if t]
        for term in terms:
            df = df[df["search_blob"].str.contains(term, na=False)]

    if domain != "All":
        df = df[df["domain_group"].fillna("") == domain]

    if family != "All":
        df = df[df["family_cluster"].fillna("") == family]

    if label != "All":
        df = df[df["product_label"] == label]

    if only_representatives and "is_family_representative" in df.columns:
        df = df[df["is_family_representative"].fillna(False) == True]

    sort_cols = [c for c in [
        "core_case_score",
        "screen_total_score",
        "artifact_richness_score",
        "transition_value_score",
        "file_count",
    ] if c in df.columns]

    if sort_cols:
        ascending = [False] * len(sort_cols)
        df = df.sort_values(by=sort_cols, ascending=ascending, na_position="last")

    return df.head(top_n).reset_index(drop=True)


def analyze_repos(
    query: str = "",
    domain: str = "All",
    family: str = "All",
    label: str = "All",
    only_representatives: bool = False,
    top_n: int = 50,
) -> pd.DataFrame:
    df = load_repo_data().copy()
    return filter_ranked_results(
        df=df,
        query=query,
        domain=domain,
        family=family,
        label=label,
        only_representatives=only_representatives,
        top_n=top_n,
    )


def get_filter_values() -> dict:
    df = load_repo_data()
    return {
        "domains": ["All"] + sorted([x for x in df["domain_group"].dropna().astype(str).unique() if x]),
        "families": ["All"] + sorted([x for x in df["family_cluster"].dropna().astype(str).unique() if x]),
        "labels": ["All", "Production-ready", "Research-grade", "Experimental"],
        "queries": [
            "load forecasting",
            "energy forecasting",
            "electricity demand forecasting",
            "pv forecasting",
            "solar forecasting",
            "energy prediction",
            "patchtst",
            "informer",
            "timesnet",
            "lstm forecasting",
            "xgboost energy prediction",
        ],
    }
