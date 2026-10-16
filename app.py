"""Additive business dashboard and customer product-discovery interface."""

import plotly.express as px
import streamlit as st
from product_pipeline import (
    DATASET,
    INDEX,
    clean_for_analysis,
    data_summary,
    filter_catalog,
    generate_recommendations,
    get_embedder,
    load_catalog,
    load_verified_index,
    retrieve,
)

st.set_page_config(page_title="Beauty Product Analytics & Search", layout="wide")
st.title("Beauty Product Analytics & Search")
st.caption(
    "Analyze a static product catalog and discover products using its recorded facts."
)


@st.cache_data
def read_data(path, modified):
    return load_catalog(path)


@st.cache_resource
def search_resources(index_modified, manifest_modified):
    raw, index, manifest = load_verified_index()
    return raw, index, get_embedder(), manifest


if not DATASET.exists():
    st.error("Dataset/product_info.csv is missing. See README.md.")
    st.stop()
raw = read_data(str(DATASET), DATASET.stat().st_mtime_ns)
clean = clean_for_analysis(raw)
business, customer, about = st.tabs(
    ["Business analysis", "Product discovery", "About the data"]
)

with business:
    st.subheader("Catalog overview")
    metrics = data_summary(raw)
    a, b, c = st.columns(3)
    a.metric("Catalog products", metrics["raw_rows"])
    b.metric("Analysis entries", metrics["analysis_rows"])
    c.metric("Brands in analysis", metrics["brands"])
    st.caption(
        f"{metrics['removed_rows']} entries are excluded by the notebook's duplicate/missing-value rules. "
        "Unrated products can still be available to search."
    )
    categories = st.multiselect(
        "Categories", sorted(clean.primary_category.dropna().unique())
    )
    data = clean[clean.primary_category.isin(categories)] if categories else clean
    if data.empty:
        st.info("No rows match the selected categories.")
    else:
        grouped = (
            data.groupby("primary_category")
            .agg(
                products=("product_name", "count"),
                median_price_usd=("price_usd", "median"),
                mean_product_rating=("rating", "mean"),
                total_loves=("loves_count", "sum"),
            )
            .reset_index()
        )
        left, right = st.columns(2)
        left.plotly_chart(
            px.bar(
                grouped,
                x="primary_category",
                y="products",
                title="Product assortment by category",
            ),
            use_container_width=True,
        )
        right.plotly_chart(
            px.bar(
                grouped,
                x="primary_category",
                y="median_price_usd",
                title="Median listed price by category (USD)",
            ),
            use_container_width=True,
        )
        hide_outliers = st.checkbox("Hide highest-priced 1% in scatterplot", True)
        plotted = (
            data[data.price_usd <= data.price_usd.quantile(0.99)]
            if hide_outliers
            else data
        )
        st.plotly_chart(
            px.scatter(
                plotted,
                x="price_usd",
                y="rating",
                color="primary_category",
                hover_data=["product_name", "brand_name", "reviews"],
                title="Listed price versus product rating",
            ),
            use_container_width=True,
        )
        st.caption(
            "The optional cutoff affects this chart only. Correlation is not causation."
        )
        min_products = st.number_input("Minimum products per brand", 1, value=10)
        min_reviews = st.number_input("Minimum total reviews per brand", 0, value=500)
        brands = (
            data.groupby("brand_name")
            .agg(
                products=("product_name", "count"),
                mean_product_rating=("rating", "mean"),
                reviews=("reviews", "sum"),
                median_price_usd=("price_usd", "median"),
            )
            .reset_index()
        )
        brands = brands[
            (brands.products >= min_products) & (brands.reviews >= min_reviews)
        ]
        top = brands.nlargest(10, "mean_product_rating").sort_values(
            "mean_product_rating"
        )
        if top.empty:
            st.info("No brands meet the selected thresholds.")
        else:
            st.plotly_chart(
                px.bar(
                    top,
                    x="mean_product_rating",
                    y="brand_name",
                    orientation="h",
                    title="Highest mean product ratings among qualifying brands",
                ),
                use_container_width=True,
            )
        st.caption(
            "Brand ratings are unweighted product means. Loves/reviews indicate engagement, not sales."
        )
        st.dataframe(grouped, hide_index=True)
        st.download_button(
            "Download category summary",
            grouped.to_csv(index=False),
            "category_summary.csv",
            "text/csv",
        )

with customer:
    st.subheader("Find products in the catalog")
    st.caption(
        "This is catalog-based discovery, not a prediction of personal effectiveness."
    )
    with st.form("search"):
        query = st.text_input(
            "What are you looking for?", placeholder="A moisturizer within my budget"
        )
        brand = st.selectbox(
            "Brand", ["Any brand", *sorted(raw.brand_name.dropna().unique())]
        )
        category = st.selectbox(
            "Category",
            ["Any category", *sorted(raw.primary_category.dropna().unique())],
        )
        use_budget = st.checkbox("Apply maximum price")
        budget = st.number_input(
            "Maximum listed price (USD)", min_value=0.0, value=50.0
        )
        use_rating = st.checkbox("Apply minimum rating")
        rating = st.slider("Minimum product rating", 0.0, 5.0, 4.0)
        top_k = st.slider("Number of products", 1, 10, 5)
        use_ai = st.checkbox(
            "Also request a Mistral explanation through local Ollama", False
        )
        submitted = st.form_submit_button("Search")
    if submitted:
        if not query.strip():
            st.warning("Enter a search question.")
        else:
            filters = dict(
                brand=None if brand == "Any brand" else brand,
                category=None if category == "Any category" else category,
                max_price=budget if use_budget else None,
                min_rating=rating if use_rating else None,
            )
            eligible = filter_catalog(raw, **filters)
            if eligible.empty:
                st.info("No catalog products satisfy all selected filters.")
            else:
                try:
                    manifest_path = INDEX.with_suffix(".manifest.json")
                    if not INDEX.exists() or not manifest_path.exists():
                        raise ValueError(
                            "Run python build_index.py --verify-existing, or rebuild using python build_index.py."
                        )
                    with st.spinner("Searching the validated catalog index..."):
                        indexed, index, embedder, manifest = search_resources(
                            INDEX.stat().st_mtime_ns, manifest_path.stat().st_mtime_ns
                        )
                        results = retrieve(
                            query, indexed, index, embedder, final_k=top_k, **filters
                        )
                    st.caption(
                        f"{len(eligible)} products satisfy the filters; matches below are ranked semantically."
                    )
                    st.dataframe(
                        results[
                            [
                                "product_id",
                                "product_name",
                                "brand_name",
                                "primary_category",
                                "price_usd",
                                "rating",
                                "reviews",
                                "similarity_score",
                            ]
                        ],
                        hide_index=True,
                    )
                    if use_ai:
                        with st.spinner("Requesting a local Mistral explanation..."):
                            answer = generate_recommendations(query, results)
                        for item in answer["recommendations"]:
                            product = results[
                                results.product_id.astype(str) == item["product_id"]
                            ].iloc[0]
                            st.markdown(
                                f"**{product['product_name']}** — {product['brand_name']}"
                            )
                            st.write(
                                f"Catalog ID: {product['product_id']} · Listed price (USD): {product['price_usd']:.2f}"
                            )
                            st.write(item["reason"])
                        if answer.get("note"):
                            st.write(answer["note"])
                        st.caption(
                            "IDs are checked against retrieval. Generated explanations still require factual review."
                        )
                except Exception as exc:
                    st.error(f"{type(exc).__name__}: {exc}")
                    st.info(
                        "See README.md for index setup and local Ollama/Mistral startup."
                    )

with about:
    st.subheader("Purpose and evidence")
    st.write(
        "Business users can compare assortment, price positioning, ratings and engagement. "
        "Customers can search the same catalog with product text and explicit filters."
    )
    st.write(
        "Analysis follows the existing notebook cleaning rules. Search uses the full catalog "
        "and represents missing values as unavailable. Prices and availability may be outdated."
    )
    st.write(
        "The data is not transaction history: revenue, profit, market share, causal price effects "
        "and individual customer suitability are not measured."
    )
    st.write(
        "The original analysis.ipynb and rag.py are preserved; this app is additive."
    )
    st.caption(
        "Document dataset provenance and usage terms from the original source before public redistribution."
    )
