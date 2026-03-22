import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


HF_BASE_URL = "https://huggingface.co/api/models"


def normalize_text(value: Optional[Any]) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\u00a0", " ").split()).strip()


def safe_lower(value: Optional[Any]) -> str:
    return normalize_text(value).lower()


def first_non_empty(*values: Any) -> str:
    for v in values:
        t = normalize_text(v)
        if t:
            return t
    return ""


def clean_repo_name(model_id: str) -> Tuple[str, str]:
    text = normalize_text(model_id)
    if "/" in text:
        owner, repo = text.split("/", 1)
        return owner.strip(), repo.strip()
    return "", text.strip()


def stringify_list(values: Any, sep: str = "; ") -> str:
    if values is None:
        return ""
    if isinstance(values, list):
        out = []
        for v in values:
            t = normalize_text(v)
            if t:
                out.append(t)
        return sep.join(out)
    return normalize_text(values)


def extract_card_text(item: Dict[str, Any]) -> str:
    candidate_fields = [
        item.get("readme"),
        item.get("README"),
        item.get("modelCard"),
        item.get("description"),
    ]

    parts: List[str] = []

    card_data = item.get("cardData")
    if isinstance(card_data, dict):
        for _, v in card_data.items():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, str):
                        parts.append(x)

    for field in candidate_fields:
        if isinstance(field, str):
            parts.append(field)
        elif isinstance(field, dict):
            for _, v in field.items():
                if isinstance(v, str):
                    parts.append(v)

    return normalize_text("\n".join([normalize_text(x) for x in parts if normalize_text(x)]))


def extract_siblings(item: Dict[str, Any]) -> List[str]:
    siblings = item.get("siblings", [])
    out: List[str] = []
    if isinstance(siblings, list):
        for s in siblings:
            if isinstance(s, dict):
                out.append(normalize_text(s.get("rfilename")))
            else:
                out.append(normalize_text(s))
    return [x for x in out if x]


def extract_datasets(item: Dict[str, Any]) -> List[str]:
    datasets: List[str] = []

    direct = item.get("datasets")
    if isinstance(direct, list):
        datasets.extend([normalize_text(x) for x in direct if normalize_text(x)])

    card = item.get("cardData", {})
    if isinstance(card, dict):
        for key in ["datasets", "dataset", "train_datasets"]:
            value = card.get(key)
            if isinstance(value, list):
                datasets.extend([normalize_text(x) for x in value if normalize_text(x)])
            elif isinstance(value, str):
                datasets.append(normalize_text(value))

    tags = item.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            tag_text = normalize_text(tag)
            if tag_text.lower().startswith("dataset:"):
                datasets.append(tag_text.split(":", 1)[1].strip())

    cleaned: List[str] = []
    seen = set()
    for x in datasets:
        k = x.lower()
        if x and k not in seen:
            seen.add(k)
            cleaned.append(x)

    return cleaned


def extract_github_url_from_text(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
        r"www\.github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
        r"github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            url = match.group(0)
            if not url.startswith("http"):
                url = "https://" + url
            return url.rstrip("/").rstrip(").,;]")
    return ""


def extract_urls(value: Any) -> List[str]:
    urls: List[str] = []

    def _walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, str):
            found = re.findall(r"https?://[^\s<>\")\]]+", x)
            urls.extend(found)
        elif isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, list):
            for v in x:
                _walk(v)

    _walk(value)

    out: List[str] = []
    seen = set()
    for u in urls:
        u2 = u.strip().rstrip(").,;]")
        if u2 not in seen:
            seen.add(u2)
            out.append(u2)
    return out


def pick_github_url(item: Dict[str, Any], card_text: str) -> str:
    priority_fields = [
        item.get("github_url"),
        item.get("repo_url"),
        item.get("code_repository"),
        item.get("repository"),
        item.get("source"),
        item.get("homepage"),
    ]

    card = item.get("cardData", {})
    if isinstance(card, dict):
        priority_fields.extend([
            card.get("github"),
            card.get("repo"),
            card.get("repository"),
            card.get("source"),
            card.get("homepage"),
        ])

    for field in priority_fields:
        url = extract_github_url_from_text(normalize_text(field))
        if url:
            return url

    for url in extract_urls(item):
        if "github.com/" in url.lower():
            return url

    return extract_github_url_from_text(card_text)


def expand_query(query: str) -> List[str]:
    q = safe_lower(query)

    base_terms = [query.strip()]
    expansions: List[str] = []

    domain_map = {
        "energy": ["electricity forecasting", "load forecasting", "demand forecasting", "power forecasting"],
        "electricity": ["electricity forecasting", "load forecasting", "demand forecasting"],
        "load": ["load forecasting", "building load forecasting", "electricity demand forecasting"],
        "solar": ["solar forecasting", "pv forecasting", "photovoltaic forecasting", "solar irradiance forecasting"],
        "pv": ["pv forecasting", "photovoltaic forecasting", "solar forecasting"],
        "forecast": ["time series forecasting", "time-series-forecasting"],
        "prediction": ["energy prediction", "electricity prediction", "pv prediction"],
    }

    family_terms = [
        "patchtst",
        "informer",
        "timesnet",
        "autoformer",
        "nbeats",
        "lstm",
        "sarimax",
        "xgboost",
    ]

    for key, vals in domain_map.items():
        if key in q:
            expansions.extend(vals)

    if any(x in q for x in ["forecast", "prediction", "time series", "energy", "electricity", "solar", "pv", "load"]):
        expansions.extend(family_terms)

    expansions.extend([
        f"{query} model",
        f"{query} time series",
    ])

    deduped: List[str] = []
    seen = set()
    for x in base_terms + expansions:
        t = normalize_text(x)
        if t and t.lower() not in seen:
            seen.add(t.lower())
            deduped.append(t)

    return deduped[:15]


class HuggingFaceClient:
    def __init__(self, timeout: int = 60, max_retries: int = 4):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

        token = os.getenv("HF_TOKEN", "").strip()
        headers = {
            "User-Agent": os.getenv("HF_USER_AGENT", "ML-Repo-Intelligence/2.2").strip(),
            "Accept": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.session.headers.update(headers)

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response
            except Exception as e:
                last_error = e
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"HF request failed: {last_error}")

    def search_models(self, query: str, limit: int = 20, full: bool = True) -> List[Dict[str, Any]]:
        params = {
            "search": query,
            "limit": limit,
            "full": "true" if full else "false",
        }
        response = self._get(HF_BASE_URL, params=params)
        payload = response.json()
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "models" in payload:
            return payload["models"]
        return []

    def get_model_details(self, model_id: str) -> Dict[str, Any]:
        response = self._get(f"{HF_BASE_URL}/{model_id}")
        payload = response.json()
        return payload if isinstance(payload, dict) else {}


def merge_dicts(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        if k not in out or out[k] in [None, "", [], {}]:
            out[k] = v
        elif isinstance(out[k], dict) and isinstance(v, dict):
            merged = dict(out[k])
            merged.update(v)
            out[k] = merged
    return out


def score_live_repo(row: Dict[str, Any]) -> Dict[str, Any]:
    blob = " ".join([
        safe_lower(row.get("hf_model_id")),
        safe_lower(row.get("hf_repo_name")),
        safe_lower(row.get("pipeline_tag")),
        safe_lower(row.get("library_name")),
        safe_lower(row.get("tags")),
        safe_lower(row.get("datasets")),
        safe_lower(row.get("card_text_preview")),
        safe_lower(row.get("matched_query")),
    ])

    score = 0

    domain_terms = ["energy", "electricity", "load", "demand", "solar", "pv", "photovoltaic", "power", "wind", "irradiance"]
    deep_family_terms = ["patchtst", "autoformer", "informer", "timesnet", "dlinear", "nbeats"]
    baseline_terms = ["lstm", "sarimax", "xgboost", "linear regression", "random forest"]
    task_terms = ["forecast", "prediction", "regression", "time-series-forecasting", "tabular-regression", "time series"]

    domain_hits = [t for t in domain_terms if t in blob]
    deep_hits = [t for t in deep_family_terms if t in blob]
    baseline_hits = [t for t in baseline_terms if t in blob]
    task_hits = [t for t in task_terms if t in blob]

    artifact_count = (
        int(row.get("has_weights", False))
        + int(row.get("has_config", False))
        + int(row.get("has_training_args", False))
        + int(row.get("has_requirements", False))
        + int(row.get("has_example_code", False))
        + int(row.get("has_notebooks", False))
        + int(row.get("has_multi_level_structure", False))
    )

    if domain_hits:
        score += 4
    if task_hits:
        score += 4

    if deep_hits:
        score += 5
    elif baseline_hits:
        score += 3

    if "patchtst" in deep_hits or "autoformer" in deep_hits:
        score += 3

    if row.get("has_weights"):
        score += 3
    if row.get("has_config"):
        score += 3
    if row.get("has_training_args"):
        score += 3
    if row.get("has_requirements"):
        score += 2
    if row.get("has_example_code"):
        score += 2
    if row.get("has_notebooks"):
        score += 1
    if row.get("has_multi_level_structure"):
        score += 2
    if row.get("has_github_link"):
        score += 3
    if row.get("dataset_count", 0) > 0:
        score += 2

    if artifact_count >= 5:
        score += 5

    if row.get("file_count", 0) >= 20:
        score += 4
    elif row.get("file_count", 0) >= 8:
        score += 3
    elif row.get("file_count", 0) >= 3:
        score += 1

    if any(x in blob for x in ["mse", "mae", "rmse", "mape", "r2"]):
        score += 2

    row["core_case_score"] = score
    row["screen_total_score"] = score
    row["artifact_richness_score"] = artifact_count
    row["transition_value_score"] = 4 if deep_hits else 2 if baseline_hits else 0
    row["cross_platform_score"] = 3 if row.get("has_github_link") else 0

    if any(x in domain_hits for x in ["solar", "pv", "photovoltaic", "irradiance"]):
        row["domain_group"] = "solar_pv"
    elif domain_hits:
        row["domain_group"] = "energy_systems"
    else:
        row["domain_group"] = "other"

    if deep_hits:
        row["family_cluster"] = deep_hits[0]
        row["primary_family"] = deep_hits[0]
        row["method_group"] = "deep_time_series_family"
        row["case_cluster"] = "time_series_family_repo"
        row["transition_archetype"] = "time_series_family_repo"
        row["transition_signal_type"] = "family_transition_signal"
    elif baseline_hits:
        row["family_cluster"] = baseline_hits[0]
        row["primary_family"] = baseline_hits[0]
        row["method_group"] = "baseline_or_classical_forecasting"
        row["case_cluster"] = "classical_or_baseline_forecasting_repo"
        row["transition_archetype"] = "classical_or_baseline_forecasting_repo"
        row["transition_signal_type"] = "baseline_forecasting_signal"
    else:
        row["family_cluster"] = "other"
        row["primary_family"] = ""
        row["method_group"] = "forecasting_model"
        row["case_cluster"] = "forecasting_model_artifact_repo"
        row["transition_archetype"] = "forecasting_model_artifact_repo"
        row["transition_signal_type"] = "forecasting_repo_signal"

    if "time-series-forecasting" in blob:
        row["primary_task"] = "time-series-forecasting"
    elif "forecast" in blob:
        row["primary_task"] = "forecasting"
    elif "regression" in blob or "tabular-regression" in blob:
        row["primary_task"] = "regression"
    elif "prediction" in blob:
        row["primary_task"] = "prediction"
    else:
        row["primary_task"] = ""

    row["primary_domain"] = domain_hits[0] if domain_hits else ""

    if row.get("has_weights") and row.get("has_config") and row.get("has_training_args"):
        row["repo_style"] = "trainer_export_repo"
        row["artifact_change_focus"] = "training_and_model_artifacts"
        row["transition_readiness"] = "high"
    elif row.get("has_weights") and row.get("has_config") and row.get("has_multi_level_structure"):
        row["repo_style"] = "multi_artifact_repo"
        row["artifact_change_focus"] = "model_release_artifacts"
        row["transition_readiness"] = "high"
    elif row.get("has_weights") and row.get("has_config"):
        row["repo_style"] = "deployable_model_repo"
        row["artifact_change_focus"] = "model_release_artifacts"
        row["transition_readiness"] = "medium"
    elif row.get("has_example_code") and row.get("has_requirements"):
        row["repo_style"] = "app_or_demo_repo"
        row["artifact_change_focus"] = "application_or_runtime_artifacts"
        row["transition_readiness"] = "medium"
    elif row.get("has_notebooks") and not row.get("has_weights"):
        row["repo_style"] = "notebook_project_repo"
        row["artifact_change_focus"] = "notebook_centric"
        row["transition_readiness"] = "medium"
    elif row.get("has_weights"):
        row["repo_style"] = "weights_only_repo"
        row["artifact_change_focus"] = "weights_centric"
        row["transition_readiness"] = "medium"
    else:
        row["repo_style"] = "lightweight_repo"
        row["artifact_change_focus"] = "light_metadata_artifacts"
        row["transition_readiness"] = "low"

    if row.get("has_github_link") and row["artifact_richness_score"] >= 4:
        row["sync_readiness"] = "high"
    elif row.get("has_github_link") or row["artifact_richness_score"] >= 4:
        row["sync_readiness"] = "medium"
    else:
        row["sync_readiness"] = "low"

    row["documentation_level"] = (
        "high" if row.get("readme_len", 0) > 300
        else "medium" if row.get("readme_len", 0) > 80
        else "low"
    )
    row["reproducibility_level"] = (
        "high" if row.get("has_config") and row.get("has_training_args")
        else "medium" if row.get("has_config") or row.get("has_requirements")
        else "low"
    )
    row["deployment_level"] = (
        "high" if row.get("has_example_code") and row.get("has_requirements") and row.get("has_weights")
        else "medium" if row.get("has_example_code") or row.get("has_requirements")
        else "low"
    )

    if score >= 20:
        row["core_case_label"] = "core_primary"
        row["candidate_tier"] = "priority_transition_candidate"
        row["priority_bucket"] = "p1"
        row["case_study_priority"] = "primary_case"
    elif score >= 12:
        row["core_case_label"] = "core_secondary"
        row["candidate_tier"] = "strong_candidate"
        row["priority_bucket"] = "p2"
        row["case_study_priority"] = "secondary_case"
    elif score >= 8:
        row["core_case_label"] = "supporting_case"
        row["candidate_tier"] = "secondary_candidate"
        row["priority_bucket"] = "p3"
        row["case_study_priority"] = "secondary_case"
    else:
        row["core_case_label"] = "background_case"
        row["candidate_tier"] = "background_candidate"
        row["priority_bucket"] = "p4"
        row["case_study_priority"] = "support_case"

    row["analysis_include_flag"] = True
    row["is_family_representative"] = False
    return row


def build_live_row(item: Dict[str, Any], matched_query: str = "") -> Dict[str, Any]:
    model_id = first_non_empty(item.get("id"), item.get("modelId"), item.get("model_id"))
    owner, repo_name = clean_repo_name(model_id)
    card_text = extract_card_text(item)
    siblings = extract_siblings(item)
    datasets = extract_datasets(item)
    github_url = pick_github_url(item, card_text)
    files_blob = " ".join([safe_lower(x) for x in siblings])

    row: Dict[str, Any] = {
        "hf_model_id": model_id,
        "hf_owner": owner,
        "hf_repo_name": repo_name,
        "hf_url": f"https://huggingface.co/{model_id}" if model_id else "",
        "pipeline_tag": first_non_empty(item.get("pipeline_tag"), item.get("pipelineTag")),
        "library_name": first_non_empty(item.get("library_name"), item.get("libraryName")),
        "downloads": item.get("downloads", 0) or 0,
        "likes": item.get("likes", 0) or 0,
        "created_at": first_non_empty(item.get("createdAt"), item.get("created_at")),
        "last_modified": first_non_empty(item.get("lastModified"), item.get("last_modified")),
        "license": first_non_empty(item.get("license")),
        "datasets": "; ".join(datasets),
        "dataset_count": len(datasets),
        "tags": stringify_list(item.get("tags", [])),
        "siblings": "; ".join(siblings),
        "readme_len": len(card_text),
        "file_count": len(siblings),
        "has_weights": any(x in files_blob for x in ["model.safetensors", "pytorch_model.bin", ".pth", ".ckpt", ".pkl", ".joblib"]),
        "has_config": any(x in files_blob for x in ["config.json", "configuration.pkl"]),
        "has_training_args": "training_args.bin" in files_blob,
        "has_requirements": "requirements.txt" in files_blob,
        "has_example_code": any(x in files_blob for x in ["predict.py", "model.py", "example.py", "main.py", "app.py"]),
        "has_notebooks": ".ipynb" in files_blob,
        "has_multi_level_structure": any(x in files_blob for x in ["level0/", "level1/", "checkpoint", "weights/", "models/"]),
        "has_github_link": bool(github_url),
        "github_url": github_url,
        "artifact_signature": "",
        "evidence_terms": "",
        "local_repo_json": "",
        "local_readme_txt": "",
        "local_manifest_json": "",
        "card_text_preview": card_text[:1200],
        "matched_query": matched_query,
    }

    artifact_parts = []
    if row["has_weights"]:
        artifact_parts.append("weights")
    if row["has_config"]:
        artifact_parts.append("config")
    if row["has_training_args"]:
        artifact_parts.append("training_args")
    if row["has_requirements"]:
        artifact_parts.append("requirements")
    if row["has_example_code"]:
        artifact_parts.append("example_code")
    if row["has_notebooks"]:
        artifact_parts.append("notebooks")
    if row["has_multi_level_structure"]:
        artifact_parts.append("multi_level")
    if row["has_github_link"]:
        artifact_parts.append("github_link")
    row["artifact_signature"] = "; ".join(artifact_parts)

    evidence_terms: List[str] = []
    blob = safe_lower(card_text + " " + row["tags"] + " " + row["datasets"] + " " + matched_query)
    for t in [
        "forecast", "prediction", "energy", "electricity", "solar", "pv",
        "mse", "mae", "rmse", "requirements", "config", "time-series-forecasting",
        "patchtst", "informer", "timesnet", "nbeats", "lstm", "xgboost"
    ]:
        if t in blob:
            evidence_terms.append(t)
    row["evidence_terms"] = "; ".join(dict.fromkeys(evidence_terms))

    return score_live_repo(row)


def analyze_live_hf_query(query: str, limit_per_query: int = 12) -> pd.DataFrame:
    client = HuggingFaceClient()
    expanded_queries = expand_query(query)

    collected: List[Dict[str, Any]] = []
    seen_model_ids = set()

    for q in expanded_queries:
        try:
            search_items = client.search_models(query=q, limit=limit_per_query, full=True)
        except Exception:
            continue

        for item in search_items:
            model_id = first_non_empty(item.get("id"), item.get("modelId"), item.get("model_id"))
            if not model_id or model_id in seen_model_ids:
                continue

            seen_model_ids.add(model_id)

            try:
                detail = client.get_model_details(model_id)
            except Exception:
                detail = {}

            merged = merge_dicts(item, detail)
            collected.append(build_live_row(merged, matched_query=q))

    if not collected:
        return pd.DataFrame()

    df = pd.DataFrame(collected)

    df = df.sort_values(
        by=[
            "core_case_score",
            "artifact_richness_score",
            "transition_value_score",
            "cross_platform_score",
            "file_count",
            "downloads",
            "likes",
        ],
        ascending=[False, False, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    if "family_cluster" in df.columns and "core_case_score" in df.columns:
        df["family_rank"] = df.groupby("family_cluster")["core_case_score"].rank(method="dense", ascending=False)
        df["is_family_representative"] = df["family_rank"] == 1
        df = df.drop(columns=["family_rank"])

    return df
