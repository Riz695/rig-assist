"""v1's retrieval pipeline, ported as the evaluation baseline (Phase 3).

Two configurations, so Phase 3 can split v2's gain into "better pages" vs "better retrieval":
- `v1_baseline`: v1 end to end. v1's parser (pypdf text + pdfplumber tables, no OCR, so
  scanned pages have no chunks), its 500-token windows, Okapi BM25, dense, RRF.
- `v1_retrieval_on_v2_pages`: the same chunking and retrieval on v2's parsed pages
  (text layer + OCR, v2's cleaning).

Known deviation: v1 embedded with sentence-transformers (PyTorch, fp32); this uses
fastembed's fp16 ONNX build of the same `BAAI/bge-small-en-v1.5`, like v2.
Chunk IDs use v2's doc_id (v1 used the file name), so ground truth matches across configs.
Source: https://github.com/Riz695/offshore-drilling-rag (src/rag/).
"""
