"""Inspect supplied index structure offline. This does not verify embedding alignment."""

import json
from pathlib import Path
import faiss
import numpy as np
from product_pipeline import INDEX, DATASET, file_sha256, load_catalog


def main():
    raw = load_catalog()
    index = faiss.read_index(str(INDEX))
    vectors = np.vstack([index.reconstruct(i) for i in range(index.ntotal)])
    norms = np.linalg.norm(vectors, axis=1)
    report = {
        "dataset_sha256": file_sha256(DATASET),
        "index_sha256": file_sha256(INDEX),
        "index_size_bytes": INDEX.stat().st_size,
        "index_type": type(index).__name__,
        "vectors": int(index.ntotal),
        "dimensions": int(index.d),
        "metric_type": int(index.metric_type),
        "vector_count_matches_csv": bool(index.ntotal == len(raw)),
        "all_vectors_finite": bool(np.isfinite(vectors).all()),
        "all_vectors_unit_normalized": bool(np.allclose(norms, 1, atol=1e-3)),
        "minimum_norm": float(norms.min()),
        "maximum_norm": float(norms.max()),
        "embedding_alignment": "not_verified; requires freshly encoding product documents",
        "semantic_search": "not_run",
        "live_mistral_generation": "not_run",
        "explanation_grounding_accuracy": "not_measured",
    }
    target = Path("reports/index_structure.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(
        report[k]
        for k in [
            "vector_count_matches_csv",
            "all_vectors_finite",
            "all_vectors_unit_normalized",
        ]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
