# Change scope

The original notebook, RAG script and CSV were preserved byte-for-byte. Their original SHA-256 checksums are in `original_files.sha256.json`, alongside checksums of the other original files for provenance.

The original `app.py`, `README.md` and `requirements.txt` were empty; they are now populated. `.gitignore` was updated to include the supplied 12.44 MiB index and exclude environments/caches. All other implementation, tests, documentation and generated reports are additions. The separately supplied index was copied unchanged; its current checksum is recorded in `reports/index_structure.json`.

The additive search path keeps the main architecture but validates the dataset/index relationship, applies exact metadata constraints, searches the whole small index before filtering, and rejects invalid generated IDs. The original interactive CLI retains its original logic, including its top-200 candidate pool and limitations.

No sales, ROI, segmentation, predictive accuracy, grounding accuracy, or successful live model output was invented. Business summaries were recomputed from the actual CSV. Vector fixtures and mocked HTTP responses exercise code behavior only.

Cloud setup followed the cloud-environment-onboarding:setup skill. Hugging Face access was denied by the proxy, including after an escalated network attempt. A network draft adding huggingface.co was saved while retaining the existing domains; a draft save does not apply or publish the setting. No secret or private API key is needed for the public embedding model. Live model checks remain pending and are not replaced by fabricated results.
