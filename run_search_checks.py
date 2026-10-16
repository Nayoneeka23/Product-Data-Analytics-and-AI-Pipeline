"""Run actual semantic searches; optionally exercise local Mistral. No accuracy claim."""

import argparse
import json
from pathlib import Path
import time
from product_pipeline import (
    get_embedder,
    load_verified_index,
    retrieve,
    generate_recommendations,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-mistral", action="store_true")
    args = parser.parse_args()
    output = Path("reports/search_checks.json")
    output.parent.mkdir(exist_ok=True)
    report = {
        "purpose": "Functional checks, not a relevance or grounding benchmark",
        "semantic_search": {"status": "not_run"},
        "mistral": {"status": "not_run"},
    }
    try:
        raw, index, manifest = load_verified_index()
        embedder = get_embedder()
        cases = [
            ("moisturizer", {"category": "Skincare", "max_price": 50}),
            ("serum", {"brand": "The Ordinary", "min_rating": 4}),
            ("lipstick", {"category": "Makeup", "max_price": 30}),
        ]
        checks = []
        for query, filters in cases:
            start = time.perf_counter()
            result = retrieve(query, raw, index, embedder, final_k=3, **filters)
            if result.empty:
                raise ValueError("Expected nonempty search result")
            if "max_price" in filters:
                assert result.price_usd.le(filters["max_price"]).all()
            if "min_rating" in filters:
                assert result.rating.ge(filters["min_rating"]).all()
            if "brand" in filters:
                assert result.brand_name.eq(filters["brand"]).all()
            if "category" in filters:
                assert result.primary_category.eq(filters["category"]).all()
            checks.append(
                {
                    "query": query,
                    "filters": filters,
                    "elapsed_seconds": time.perf_counter() - start,
                    "results": result[
                        ["product_id", "product_name", "similarity_score"]
                    ].to_dict("records"),
                }
            )
        report["semantic_search"] = {
            "status": "passed",
            "cases": checks,
            "dataset_sha256": manifest["dataset_sha256"],
        }
        if args.with_mistral:
            report["mistral"] = {"status": "requested"}
            answer = generate_recommendations(query, result)
            report["mistral"] = {
                "status": "completed_ids_validated",
                "answer": answer,
                "factual_grounding": "not independently evaluated",
            }
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if report["semantic_search"]["status"] == "not_run":
            report["semantic_search"]["status"] = "blocked_or_failed"
        else:
            report["mistral"]["status"] = "failed"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    if "error" in report:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
