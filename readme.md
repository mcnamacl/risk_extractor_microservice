<!--
This README is intended for developers who want to run or contribute
to the risk_extractor_microservice project. Keep this file concise
and use the repository's other docs for detailed policies.
-->

# Risk Extractor Microservice

Lightweight gRPC microservice that extracts and summarizes financial
risk factors from SEC-style corporate filings (example: 10-K). The
service locates the risk section of a document, splits it into
focused chunks, runs a fine-tuned LLM (via a remote HF Space) to
identify risk categories and summaries, and returns structured
results with provenance metadata.

Badges
-------

- **Status**: Work in progress
- **Language**: Python

Why this project exists
-----------------------

- Help automation teams extract structured risk information from
	long financial documents (10-K, 10-Q style reports).
- Provide provenance and integrity metadata for compliance and
	auditability (audit id, lineage, integrity hash).
- Integrates local pre-processing (text extraction, clustering) with
	lightweight remote inference (HF Space / Gradio client).

Key Features
------------

- gRPC API: `ExtractRisks` RPC that accepts a simple query and
	returns categorized risk summaries.
- Document parsing: PDF text extraction via `PyMuPDF` and pattern
	matching for typical SEC item markers (Item 1A Risk Factors).
- Chunking: sentence-level clustering + lexical heuristics to form
	inference-friendly chunks.
- Remote inference: batch inference via a Hugging Face Space (using
	`gradio_client`) so the model can be hosted separately.
- Provenance: generation timestamps, audit id, data-lineage and a
	SHA-256 integrity hash are returned with results.

Repository Layout
-----------------

- `server.py` — gRPC server implementation. Binds to `:50051` by
	default and serves the `RiskExtractor` RPC.
- `client.py` — example client showing how to call the service.
- `utils.py` — core extraction, chunking, inference and provenance
	utilities.
- `proto/risk_extractor.proto` — canonical service and message
	definitions (used to generate the `risk_extractor_pb2*.py` files).
- `LLM_Model_Test_v8_gretel/` — local artifacts (adapter, tokenizer,
	optimizer) for a finetuned TinyLlama variant (not required for the
	remote Space-based inference approach).
- `requirements.txt` — pinned Python dependencies for development.

Quickstart (developer)
----------------------

Prerequisites

- Python 3.10+ (3.11 recommended)
- `git` and a working internet connection to install packages
- Optional: GPU + drivers if you plan to run local model inference

1. Clone and enter the repository

```powershell
git clone <repo-url>
cd risk_extractor_microservice
```

2. Create and activate a virtual environment

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

3. Install dependencies

```powershell
pip install -r requirements.txt
```

Note: `requirements.txt` includes many NLP and ML packages. If you
only need to run the gRPC server and call a remote HF Space for
inference, you can try installing a reduced set focused on gRPC,
`gradio_client`, and `pymupdf` first.

4. (Optional) Generate Python gRPC bindings from `.proto`

If you modify `proto/risk_extractor.proto`, regenerate the Python
stubs with:

```powershell
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/risk_extractor.proto
```

5. Run the server

```powershell
python server.py
```

The server listens on `localhost:50051` by default.

6. Run the example client

```powershell
python client.py
```

The `client.py` script demonstrates creating an `ExtractRisksRequest`
and printing a JSON-like result. Edit the `query` and `space_url`
fields in `client.py` to target a different document or inference
space.

Usage Example (from `client.py`)

```python
request = risk_extractor_pb2.ExtractRisksRequest(
		query="What are the risks in the report from 2022?",
		user_id="user123",
		query_timestamp=int(time.time()),
		space_url="mcnamacl/tinyllama-inference"
)
```

Implementation Notes
--------------------

- Document extraction uses `PyMuPDF` to read PDFs and then looks for
	common SEC markers (Item 1A). The heuristics are intentionally
	simple — adapt them if your document formats differ.
- Chunking uses `sentence-transformers` to embed sentences then
	clusters them with `KMeans` to select candidate "risky" sentences
	and pack them into token-limited chunks.
- Inference is performed by calling a remote Hugging Face Space via
	`gradio_client.Client`. The repo expects the Space to accept a
	list of prompts at the `/infer` api_name and return a list of
	outputs in a simple format: `[CATEGORIES] | SUMMARY`.
- The code includes utilities for loading a quantized TinyLlama base
	model plus a PEFT adapter; those heavy tasks require GPU and are
	optional if you use remote inference.

Security & Privacy
------------------

- Uploaded documents and queries may be sent to a remote HF Space
	depending on configuration. Do not send sensitive or regulated
	data to an external space unless you control it and understand the
	data handling policies.

Where to get help
-----------------

- File an issue in this repository's issue tracker for bugs or
	feature requests.
- For questions about the `.proto` service, see `proto/risk_extractor.proto`.
- For contribution guidance, see `CONTRIBUTING.md` (if present).

Maintainers & Contributing
--------------------------

- Maintainer: repository owner (see repo metadata).
- Contributions are welcome. Please open issues or pull requests.
- Keep PRs small, include tests where applicable, and describe the
	motivation and design in the PR description.
- See `CONTRIBUTING.md` for detailed contribution guidelines.

Development tips
----------------

- When experimenting with the model locally, prefer a controlled
	virtualenv and install optional heavy packages only when needed.
- To speed up prototyping, stub out `batch_generate` in
	`utils.py` to return canned outputs instead of calling a Space.

License
-------

See the project `LICENSE` file for license details.

Acknowledgements
----------------

- Built with `PyMuPDF`, `sentence-transformers`, `transformers`,
	`gradio_client` and other community libraries. See
	`requirements.txt` for full dependency details.

Next steps
----------

- Run the server and try `client.py` against a controlled HF Space.
- Add `CONTRIBUTING.md` and `docs/` materials if you want to grow
	the project documentation.

