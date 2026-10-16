import ast
import json
from pathlib import Path
from unittest.mock import Mock

import faiss
import numpy as np
import pandas as pd
import pytest

import product_pipeline as p


def test_original_files_preserved():
    hashes = json.loads((p.ROOT / "docs/original_files.sha256.json").read_text())
    for name in ["analysis.ipynb", "rag.py", "Dataset/product_info.csv"]:
        assert p.file_sha256(p.ROOT / name) == hashes[name]


def test_resume_counts_and_cleaning():
    raw = p.load_catalog()
    clean = p.clean_for_analysis(raw)
    assert len(raw) == 8494
    assert len(clean) == 8216
    assert not clean[["price_usd", "rating", "reviews"]].isna().any().any()
    assert not clean.duplicated().any()


def test_document_format_matches_original_for_every_product():
    tree = ast.parse((p.ROOT / "rag.py").read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("safe_text", "create_product_document")
    ]
    namespace = {"pd": pd}
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]), "original_functions", "exec"
        ),
        namespace,
    )
    for _, row in p.load_catalog()[p.RAG_COLUMNS].iterrows():
        assert p.product_document(row) == namespace["create_product_document"](row)


@pytest.mark.parametrize(
    "filters",
    [
        {"max_price": -1},
        {"max_price": float("nan")},
        {"min_rating": 6},
        {"min_rating": float("inf")},
    ],
)
def test_invalid_filters(filters):
    with pytest.raises(ValueError):
        p.filter_catalog(p.load_catalog(), **filters)


def test_unknown_brand_does_not_silently_ignore_filter():
    assert p.filter_catalog(p.load_catalog(), brand="not a real brand").empty


def test_combined_filters():
    result = p.filter_catalog(
        p.load_catalog(),
        brand=" the ordinary ",
        max_price=30,
        min_rating=4,
        category="Skincare",
    )
    assert not result.empty
    assert result.brand_name.eq("The Ordinary").all()
    assert result.price_usd.le(30).all() and result.rating.ge(4).all()
    assert result.primary_category.eq("Skincare").all()


def test_retrieval_filters_beyond_first_200():
    # Synthetic vectors test filtering logic only, not semantic retrieval quality.
    raw = pd.DataFrame({"brand_name": ["A"] * 250 + ["B"], "product_id": range(251)})
    vectors = np.tile(np.array([[1.0, 0.0]], dtype="float32"), (251, 1))
    vectors[-1] = [0.0, 1.0]
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    embedder = Mock()
    embedder.encode.return_value = np.array([[1.0, 0.0]], dtype="float32")
    result = p.retrieve("example", raw, index, embedder, brand="B")
    assert result.product_id.tolist() == [250]


def test_empty_filters_avoid_model_call():
    embedder = Mock()
    raw = pd.DataFrame({"brand_name": ["A"]})
    assert p.retrieve("example", raw, None, embedder, brand="B").empty
    embedder.encode.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        {"done": False, "response": "{}"},
        {"done": True, "response": "not json"},
        {
            "done": True,
            "response": '{"recommendations":[{"product_id":"invented","reason":"x"}]}',
        },
        {"done": True, "response": '{"recommendations":"invalid"}'},
    ],
)
def test_bad_generation_is_rejected(monkeypatch, body):
    response = Mock()
    response.json.return_value = body
    monkeypatch.setattr("requests.post", Mock(return_value=response))
    with pytest.raises((ValueError, json.JSONDecodeError)):
        p.generate_recommendations("example", p.load_catalog().head(1))


def test_generation_contract_and_http_timeout(monkeypatch):
    rows = p.load_catalog().head(1)
    pid = str(rows.iloc[0].product_id)
    response = Mock()
    response.json.return_value = {
        "done": True,
        "response": json.dumps(
            {
                "recommendations": [
                    {
                        "product_id": pid,
                        "reason": "Example fixture, not a real model answer.",
                    }
                ]
            }
        ),
    }
    post = Mock(return_value=response)
    monkeypatch.setattr("requests.post", post)
    assert (
        p.generate_recommendations("example", rows)["recommendations"][0]["product_id"]
        == pid
    )
    assert post.call_args.kwargs["timeout"] == (5, 180)
    response.raise_for_status.assert_called_once()


def test_dashboard_initial_and_category_filter():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(p.ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert [m.value for m in app.metric][:2] == ["8494", "8216"]
    app.multiselect[0].set_value(["Skincare"]).run()
    assert not app.exception
    assert app.dataframe[0].value.primary_category.tolist() == ["Skincare"]


def test_index_checksum_tampering_rejected(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_bytes(p.DATASET.read_bytes())
    index_path = tmp_path / "sample.index"
    index_path.write_bytes(b"corrupt index")
    manifest = {"dataset_sha256": p.file_sha256(dataset), "index_sha256": "wrong"}
    index_path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum changed"):
        p.load_verified_index(dataset, index_path)


def test_request_timeout_is_not_presented_as_answer(monkeypatch):
    import requests

    monkeypatch.setattr(
        "requests.post", Mock(side_effect=requests.Timeout("test fixture timeout"))
    )
    with pytest.raises(requests.Timeout):
        p.generate_recommendations("example", p.load_catalog().head(1))
