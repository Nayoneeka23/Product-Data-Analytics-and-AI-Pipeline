"""Build or verify the supplied index without modifying the original RAG script."""

import argparse
import json
from product_pipeline import DATASET, INDEX, build_index, verify_provided_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--index", default=str(INDEX))
    args = parser.parse_args()
    manifest = (
        verify_provided_index(args.dataset, args.index)
        if args.verify_existing
        else build_index(args.dataset, args.index)
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
