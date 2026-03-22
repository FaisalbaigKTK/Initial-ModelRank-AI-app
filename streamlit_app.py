import os
from datetime import datetime

import pandas as pd
import streamlit as st

from analyze_service import get_filter_values
from pipeline_runner import run_pipeline
from report_generator import generate_pdf


st.set_page_config(page_title="ModelRank AI", layout="wide")

# -----------------------------------------------------------------------------
# Header / branding
# -----------------------------------------------------------------------------

st.title("🚀 ModelRank AI")
st.caption("Find production-ready ML models. Avoid research dead ends.")
st.info(
    "Analyze ML repositories and instantly identify which models are production-ready, "
    "research-grade, or not worth using."
)

with st.expander("How this works"):
    st.write("""
This platform evaluates ML repositories using:

- Artifact completeness (weights, configs, examples)
- Repository structure
- Model family signals (PatchTST, Autoformer, Informer, etc.)
- Transition readiness from research to production

The output is a ranked list with actionable recommendations.
""")


# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------

filters = get_filter_values()

with st.sidebar:
    st.header("Filters")

    mode = st.radio("Mode", ["live", "static"], index=0)

    query_choice = st.selectbox("Preset query", filters["queries"], index=0)
    custom_query = st.text_input("Or type your own query", value="")

    query = custom_query.strip() if custom_query.strip() else query_choice

    domain = st.selectbox("Domain", filters["domains"], index=0)
    family = st.selectbox("Family", filters["families"], index=0)
    label = st.selectbox("Readiness label", filters["labels"], index=0)
    only_representatives = st.checkbox("Only family representatives", value=False)
    top_n = st.slider("Top N results", min_value=5, max_value=100, value=20, step=5)


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_mode" not in st.session_state:
    st.session_state.last_mode = ""
if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False
if "pdf_file_path" not in st.session_state:
    st.session_state.pdf_file_path = ""


# -----------------------------------------------------------------------------
# Run analysis
# -----------------------------------------------------------------------------

if st.button("Run Analysis", type="primary"):
    with st.spinner("Running analysis..."):
        st.session_state.results = run_pipeline(
            query=query,
            domain=domain,
            family=family,
            label=label,
            only_representatives=only_representatives,
            top_n=top_n,
            mode=mode,
        )
        st.session_state.last_query = query
        st.session_state.last_mode = mode
        st.session_state.pdf_ready = False
        st.session_state.pdf_file_path = ""

df = st.session_state.results

if df is None:
    st.info("Choose a preset query or type your own, then click 'Run Analysis'.")
    st.stop()

st.success(f"Showing results for: '{st.session_state.last_query}' ({st.session_state.last_mode} mode)")

if df.empty:
    st.warning("No repositories matched your filters.")
    st.stop()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def safe_str(x):
    return "" if pd.isna(x) else str(x)


def make_family_summary(df_in: pd.DataFrame) -> pd.DataFrame:
    if "family_cluster" not in df_in.columns:
        return pd.DataFrame()

    rows = []
    for fam, g in df_in.groupby("family_cluster", dropna=False):
        fam_name = safe_str(fam) or "unknown"
        rows.append({
            "family_cluster": fam_name,
            "repo_count": len(g),
            "avg_score": round(pd.to_numeric(g.get("core_case_score"), errors="coerce").fillna(0).mean(), 2),
            "top_repo": g.sort_values(by=["core_case_score", "screen_total_score"], ascending=[False, False]).iloc[0].get("hf_model_id", ""),
            "production_ready": int((g.get("product_label", "") == "Production-ready").sum()),
            "research_grade": int((g.get("product_label", "") == "Research-grade").sum()),
            "experimental": int((g.get("product_label", "") == "Experimental").sum()),
        })

    return pd.DataFrame(rows).sort_values(
        by=["avg_score", "repo_count"], ascending=[False, False]
    ).reset_index(drop=True)


def make_domain_summary(df_in: pd.DataFrame) -> pd.DataFrame:
    if "domain_group" not in df_in.columns:
        return pd.DataFrame()

    rows = []
    for dom, g in df_in.groupby("domain_group", dropna=False):
        dom_name = safe_str(dom) or "unknown"
        rows.append({
            "domain_group": dom_name,
            "repo_count": len(g),
            "avg_score": round(pd.to_numeric(g.get("core_case_score"), errors="coerce").fillna(0).mean(), 2),
            "top_repo": g.sort_values(by=["core_case_score", "screen_total_score"], ascending=[False, False]).iloc[0].get("hf_model_id", ""),
            "production_ready": int((g.get("product_label", "") == "Production-ready").sum()),
            "research_grade": int((g.get("product_label", "") == "Research-grade").sum()),
            "experimental": int((g.get("product_label", "") == "Experimental").sum()),
        })

    return pd.DataFrame(rows).sort_values(
        by=["avg_score", "repo_count"], ascending=[False, False]
    ).reset_index(drop=True)


def build_executive_summary(df_in: pd.DataFrame, q: str, mode_name: str) -> str:
    total = len(df_in)
    prod = int((df_in["product_label"] == "Production-ready").sum()) if "product_label" in df_in.columns else 0
    research = int((df_in["product_label"] == "Research-grade").sum()) if "product_label" in df_in.columns else 0
    experimental = int((df_in["product_label"] == "Experimental").sum()) if "product_label" in df_in.columns else 0

    top_repo = ""
    if len(df_in) > 0 and "hf_model_id" in df_in.columns:
        top_repo = safe_str(df_in.iloc[0].get("hf_model_id", ""))

    top_family = ""
    if "family_cluster" in df_in.columns and not df_in["family_cluster"].dropna().empty:
        top_family = safe_str(df_in["family_cluster"].fillna("unknown").value_counts().index[0])

    top_domain = ""
    if "domain_group" in df_in.columns and not df_in["domain_group"].dropna().empty:
        top_domain = safe_str(df_in["domain_group"].fillna("unknown").value_counts().index[0])

    lines = [
        "# ML Repo Intelligence Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Query:** {q}",
        f"**Mode:** {mode_name}",
        "",
        "## Executive Summary",
        "",
        f"- Total repositories analyzed: **{total}**",
        f"- Production-ready: **{prod}**",
        f"- Research-grade: **{research}**",
        f"- Experimental: **{experimental}**",
        f"- Top ranked repository: **{top_repo}**",
        f"- Dominant family cluster: **{top_family}**",
        f"- Dominant domain group: **{top_domain}**",
        "",
    ]

    if prod == 0 and research > 0:
        lines.extend([
            "### Interpretation",
            "",
            "The current result set is dominated by research-grade repositories.",
            "This usually means the repositories have useful artifacts and method signals,",
            "but still require engineering effort before direct production use.",
            "",
        ])

    return "\n".join(lines)


def build_top_repos_markdown(df_in: pd.DataFrame, limit: int = 10) -> str:
    lines = ["## Top Ranked Repositories", ""]
    top = df_in.head(limit)

    for idx, (_, row) in enumerate(top.iterrows(), start=1):
        lines.extend([
            f"### {idx}. {safe_str(row.get('hf_model_id', ''))}",
            f"- Score: **{safe_str(row.get('core_case_score', ''))}**",
            f"- Type: **{safe_str(row.get('product_label', ''))}**",
            f"- Recommendation: **{safe_str(row.get('recommendation', ''))}**",
            f"- Family: **{safe_str(row.get('family_cluster', ''))}**",
            f"- Domain: **{safe_str(row.get('domain_group', ''))}**",
            f"- Task: **{safe_str(row.get('primary_task', ''))}**",
            f"- Repo style: **{safe_str(row.get('repo_style', ''))}**",
            f"- Signal: **{safe_str(row.get('transition_signal_type', ''))}**",
            f"- Artifacts: **{safe_str(row.get('artifact_signature', ''))}**",
            f"- Evidence: **{safe_str(row.get('evidence_terms', ''))}**",
            "",
        ])
    return "\n".join(lines)


def dataframe_to_markdown_table(df_in: pd.DataFrame, max_rows: int = 20) -> str:
    if df_in.empty:
        return "_No data available._"

    df_show = df_in.head(max_rows).copy()
    for col in df_show.columns:
        df_show[col] = df_show[col].astype(str)

    header = "| " + " | ".join(df_show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df_show.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in df_show.values.tolist()]
    return "\n".join([header, sep] + rows)


def build_full_report(df_in: pd.DataFrame, q: str, mode_name: str) -> str:
    family_summary = make_family_summary(df_in)
    domain_summary = make_domain_summary(df_in)

    report_parts = [
        build_executive_summary(df_in, q, mode_name),
        build_top_repos_markdown(df_in, limit=10),
        "## Family Summary",
        "",
        dataframe_to_markdown_table(family_summary, max_rows=15),
        "",
        "## Domain Summary",
        "",
        dataframe_to_markdown_table(domain_summary, max_rows=15),
        "",
        "## Notes",
        "",
        "- Production-ready means the repository exposes stronger signals of practical usability.",
        "- Research-grade means the repository is promising but may require engineering or validation.",
        "- Experimental means the repository is weakly structured or incomplete for direct use.",
        "",
    ]
    return "\n".join(report_parts)


def format_label(x):
    if x == "Production-ready":
        return "✅ Production"
    if x == "Research-grade":
        return "⚠️ Research"
    return "❌ Experimental"


# -----------------------------------------------------------------------------
# KPI row
# -----------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Results", len(df))
c2.metric("Production-ready", int((df["product_label"] == "Production-ready").sum()))
c3.metric("Research-grade", int((df["product_label"] == "Research-grade").sum()))
c4.metric("Experimental", int((df["product_label"] == "Experimental").sum()))

st.markdown("### 🧠 What should you do?")

prod_count = len(df[df["product_label"] == "Production-ready"])
research_count = len(df[df["product_label"] == "Research-grade"])

if prod_count > 0:
    st.success(f"Use top {prod_count} models as production candidates.")
else:
    st.warning("No production-ready models found. Proceed with caution.")

if research_count > 0:
    st.info(f"{research_count} models require validation before use.")


# -----------------------------------------------------------------------------
# Ranked results
# -----------------------------------------------------------------------------

st.subheader("📊 Ranked Model Intelligence")

display_cols = [
    "hf_model_id",
    "core_case_score",
    "product_label",
    "recommendation",
    "family_cluster",
    "domain_group",
    "primary_task",
    "repo_style",
    "transition_signal_type",
    "artifact_signature",
    "evidence_terms",
]
display_cols = [c for c in display_cols if c in df.columns]

show_df = df[display_cols].copy()
show_df = show_df.rename(
    columns={
        "hf_model_id": "Repo",
        "core_case_score": "Score",
        "product_label": "Type",
        "family_cluster": "Family",
        "domain_group": "Domain",
        "primary_task": "Task",
        "repo_style": "Repo Style",
        "transition_signal_type": "Signal",
        "artifact_signature": "Artifacts",
        "evidence_terms": "Evidence",
    }
)

if "Type" in show_df.columns:
    show_df["Type"] = show_df["Type"].apply(format_label)

st.dataframe(show_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Repo cards
# -----------------------------------------------------------------------------

st.subheader("Repository Cards")

for _, row in df.iterrows():
    with st.container(border=True):
        c_left, c_right = st.columns([3, 1])

        with c_left:
            st.markdown(f"### {row.get('hf_model_id', '')}")
            st.write(
                f"**Type:** {row.get('product_label', '')}  \n"
                f"**Recommendation:** {row.get('recommendation', '')}  \n"
                f"**Family:** {row.get('family_cluster', '')}  \n"
                f"**Domain:** {row.get('domain_group', '')}  \n"
                f"**Task:** {row.get('primary_task', '')}  \n"
                f"**Repo style:** {row.get('repo_style', '')}  \n"
                f"**Signal:** {row.get('transition_signal_type', '')}"
            )
            st.write(f"**Artifacts:** {row.get('artifact_signature', '')}")
            st.write(f"**Evidence:** {row.get('evidence_terms', '')}")

            hf_url = row.get("hf_url", "")
            github_url = row.get("github_url", "")
            if hf_url:
                st.markdown(f"[Open on Hugging Face]({hf_url})")
            if github_url:
                st.markdown(f"[Open GitHub]({github_url})")

        with c_right:
            st.metric("Score", int(row.get("core_case_score", 0) or 0))
            st.metric("Screen", int(row.get("screen_total_score", 0) or 0))
            st.metric("Files", int(row.get("file_count", 0) or 0))


# -----------------------------------------------------------------------------
# Premium report export
# -----------------------------------------------------------------------------

st.subheader("📄 Export Intelligence Report")
st.write("""
Download a professional report with:
- ranked models
- production readiness
- key insights
- actionable recommendations
""")

family_summary_df = make_family_summary(df)
domain_summary_df = make_domain_summary(df)
report_text = build_full_report(df, st.session_state.last_query, st.session_state.last_mode)

tab1, tab2, tab3 = st.tabs(["Executive Summary", "Summary Tables", "Downloads"])

with tab1:
    st.markdown(build_executive_summary(df, st.session_state.last_query, st.session_state.last_mode))
    st.markdown(build_top_repos_markdown(df, limit=5))

with tab2:
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("### Family Summary")
        if family_summary_df.empty:
            st.info("No family summary available.")
        else:
            st.dataframe(family_summary_df, use_container_width=True, hide_index=True)

    with c_right:
        st.markdown("### Domain Summary")
        if domain_summary_df.empty:
            st.info("No domain summary available.")
        else:
            st.dataframe(domain_summary_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### Download premium report artifacts")

    if st.button("📄 Generate Premium PDF Report"):
        file_path = "ml_report.pdf"
        generate_pdf(st.session_state.results, st.session_state.last_query, file_path)
        st.session_state.pdf_ready = True
        st.session_state.pdf_file_path = file_path
        st.success("PDF report generated successfully.")

    if st.session_state.pdf_ready and st.session_state.pdf_file_path and os.path.exists(st.session_state.pdf_file_path):
        with open(st.session_state.pdf_file_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="ML_Report.pdf",
                mime="application/pdf",
            )

    report_bytes = report_text.encode("utf-8")
    st.download_button(
        label="Download Report (Markdown)",
        data=report_bytes,
        file_name=f"ml_repo_intelligence_report_{st.session_state.last_query.replace(' ', '_')}.md",
        mime="text/markdown",
    )

    ranked_csv = show_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Ranked Results CSV",
        data=ranked_csv,
        file_name="ml_repo_intelligence_results.csv",
        mime="text/csv",
    )

    family_csv = family_summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Family Summary CSV",
        data=family_csv,
        file_name="ml_repo_family_summary.csv",
        mime="text/csv",
    )

    domain_csv = domain_summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Domain Summary CSV",
        data=domain_csv,
        file_name="ml_repo_domain_summary.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# Basic download
# -----------------------------------------------------------------------------

st.subheader("Quick Download")
csv_data = show_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="ml_repo_intelligence_results.csv",
    mime="text/csv",
)

st.markdown("---")
st.subheader("Why this matters")
st.write("""
Most ML repositories are not production-ready.

This tool helps you:
- avoid wasting time on incomplete repos
- identify deployable models instantly
- compare architectures (PatchTST, Autoformer, etc.)
- focus only on high-value candidates
""")

st.markdown("---")
st.subheader("💼 Use Cases")
st.write("""
- ML Engineers → find deployable models faster
- Startups → avoid wasting time on bad repos
- Companies → accelerate model selection

This tool replaces hours of manual repository evaluation.
""")

st.markdown("---")
st.subheader("🚀 Premium (Coming Soon)")
st.write("""
- Full PDF reports
- API access
- Domain-specific intelligence packs
- Weekly curated model rankings
""")
