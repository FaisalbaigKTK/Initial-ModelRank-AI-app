from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd


def _safe(value):
    if pd.isna(value):
        return ""
    return str(value)


def generate_pdf(df: pd.DataFrame, query: str, filename="report.pdf"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename)
    elements = []

    elements.append(Paragraph("ML Model Intelligence Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Query: {query}", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    total = len(df)
    prod = len(df[df["product_label"] == "Production-ready"]) if "product_label" in df.columns else 0
    research = len(df[df["product_label"] == "Research-grade"]) if "product_label" in df.columns else 0
    exp = len(df[df["product_label"] == "Experimental"]) if "product_label" in df.columns else 0

    elements.append(Paragraph(f"Total repos analyzed: {total}", styles["Normal"]))
    elements.append(Paragraph(f"Production-ready: {prod}", styles["Normal"]))
    elements.append(Paragraph(f"Research-grade: {research}", styles["Normal"]))
    elements.append(Paragraph(f"Experimental: {exp}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    table_data = [["Repo", "Score", "Type", "Recommendation"]]

    top_df = df.head(15).copy()

    for _, row in top_df.iterrows():
        table_data.append([
            _safe(row.get("hf_model_id", "")),
            _safe(row.get("core_case_score", "")),
            _safe(row.get("product_label", "")),
            _safe(row.get("recommendation", "")),
        ])

    table = Table(table_data, repeatRows=1)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Key Insights", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(
        "This report ranks machine learning repositories by practical usability, "
        "artifact completeness, and recommendation strength.",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 8))

    if total > 0:
        top_repo = _safe(top_df.iloc[0].get("hf_model_id", ""))
        elements.append(Paragraph(
            f"Top ranked repository: {top_repo}",
            styles["Normal"]
        ))
        elements.append(Spacer(1, 8))

    elements.append(Paragraph(
        "Production-ready repositories are stronger candidates for direct use. "
        "Research-grade repositories may need validation or engineering work. "
        "Experimental repositories are weaker for immediate adoption.",
        styles["Normal"]
    ))

    doc.build(elements)
    return filename
