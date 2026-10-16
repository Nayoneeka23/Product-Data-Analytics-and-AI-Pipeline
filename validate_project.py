"""Write actual validation outcomes; unexecuted checks remain explicitly unrun."""

import argparse
import json
from pathlib import Path
from product_pipeline import (
    DATASET,
    data_summary,
    file_sha256,
    load_catalog,
    load_verified_index,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-index", action="store_true")
    parser.add_argument("--output", default="reports/validation.json")
    args = parser.parse_args()
    raw = load_catalog()
    summary = data_summary(raw)
    report = {
        "dataset_sha256": file_sha256(DATASET),
        "data": summary,
        "resume_count_checks": {
            "raw_8494": summary["raw_rows"] == 8494,
            "analysis_8216": summary["analysis_rows"] == 8216,
        },
        "index": {"status": "not_checked"},
        "live_mistral_generation": {"status": "not_run"},
        "answer_grounding_accuracy": {"status": "not_evaluated"},
    }
    if args.check_index:
        try:
            _, index, manifest = load_verified_index()
            report["index"] = {
                "status": "passed",
                "products": int(index.ntotal),
                "dimensions": int(index.d),
                "manifest": manifest,
            }
        except Exception as exc:
            report["index"] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)
    if (
        not all(report["resume_count_checks"].values())
        or report["index"]["status"] == "failed"
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
