"""Recompute catalog summaries and standalone interactive charts from the CSV."""

import json
from pathlib import Path
import plotly.express as px
from plotly.subplots import make_subplots
from product_pipeline import (
    clean_for_analysis,
    data_summary,
    file_sha256,
    DATASET,
    load_catalog,
)


def main():
    out = Path("reports")
    out.mkdir(exist_ok=True)
    raw = load_catalog()
    clean = clean_for_analysis(raw)
    categories = (
        clean.groupby("primary_category")
        .agg(
            products=("product_name", "count"),
            median_price_usd=("price_usd", "median"),
            mean_product_rating=("rating", "mean"),
            total_reviews=("reviews", "sum"),
            total_loves=("loves_count", "sum"),
        )
        .reset_index()
    )
    brands = (
        clean.groupby("brand_name")
        .agg(
            products=("product_name", "count"),
            median_price_usd=("price_usd", "median"),
            mean_product_rating=("rating", "mean"),
            total_reviews=("reviews", "sum"),
        )
        .reset_index()
    )
    categories.to_csv(out / "category_summary.csv", index=False)
    brands.to_csv(out / "brand_summary.csv", index=False)
    summary = {
        "dataset_sha256": file_sha256(DATASET),
        **data_summary(raw),
        "price_rating_pearson_correlation": float(clean.price_usd.corr(clean.rating)),
        "scope": "Static catalog; product means are unweighted; engagement is not sales.",
    }
    (out / "business_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    chart = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Catalog assortment", "Median listed price (USD)"],
    )
    for trace in px.bar(categories, x="primary_category", y="products").data:
        chart.add_trace(trace, row=1, col=1)
    for trace in px.bar(categories, x="primary_category", y="median_price_usd").data:
        chart.add_trace(trace, row=1, col=2)
    chart.update_layout(
        title="Analysis subset: categories, not sales or market share", showlegend=False
    )
    chart.write_html(out / "category_comparison.html", include_plotlyjs=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
