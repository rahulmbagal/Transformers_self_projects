
# Beginner-friendly explanation of the finalized V2 GEO RAG notebook

This guide explains the notebook **cell by cell** and **line by line in plain English**.

Notebook explained:
`offline_geo_rag_qwen25_3b_hf_v2_full_finalized.ipynb`

---

# 1. What this notebook is doing overall

The notebook builds a **RAG system**:

1. Read PDFs from folders like `data/APAC`, `data/EMEA`, `data/AMER`
2. Extract text from them
3. Use OCR if normal text extraction fails
4. Split text into smaller chunks
5. Create embeddings
6. Build FAISS indexes
7. Save chunk data and registry metadata
8. At query time:
   - detect GEO(s) from the question
   - retrieve matching chunks
   - optionally use BM25 + reranker
   - give the final context to the LLM
   - answer only from the documents

---

# 2. Cell: package install

```python
# %pip install -qU langchain langchain-community langchain-text-splitters langchain-huggingface \
#     faiss-cpu pypdf sentence-transformers transformers accelerate torch tqdm rank-bm25 pdf2image pytesseract
```

Explanation:

- `#` means this is a comment, so it will not run unless you remove the `#`
- `%pip install` is a Jupyter way to install Python packages
- `-qU` means:
  - `-q` = quiet output
  - `-U` = upgrade package if already installed

Packages used:
- `langchain`, `langchain-community`, `langchain-text-splitters` → RAG building blocks
- `langchain-huggingface` → local Hugging Face embeddings
- `faiss-cpu` → vector index
- `pypdf` → PDF text extraction
- `sentence-transformers` → reranker model
- `transformers`, `accelerate`, `torch` → local LLM inference
- `tqdm` → progress bars
- `rank-bm25` → keyword retrieval
- `pdf2image`, `pytesseract` → OCR fallback

---

# 3. Cell: imports

```python
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import pipeline
```

Line by line:

- `from __future__ import annotations`
  - makes type hints behave more cleanly in modern Python
  - helps when classes refer to types that may be defined later

Standard library imports:
- `hashlib` → creates hashes like md5 and sha256
- `json` → save/load JSON files
- `logging` → write logs
- `math` → math helpers
- `re` → regular expressions
- `shutil` → file and folder operations like deleting directories
- `defaultdict` → dictionary that auto-creates missing values
- `dataclass` → easy way to define config objects
- `datetime` → timestamps
- `Path` → path handling like `data/APAC/file.pdf`
- `typing` imports are only for type hints

Other libraries:
- `tqdm` → progress bars in notebooks
- `PyPDFLoader` → extract text from PDFs
- `FAISS` → vector database
- `Document` → LangChain text object with `page_content` and `metadata`
- `HuggingFaceEmbeddings` → generate embeddings
- `RecursiveCharacterTextSplitter` → split long text into chunks
- `pipeline` → high-level Hugging Face inference API for the LLM

---

# 4. Cell: Settings dataclass

```python
@dataclass
class Settings:
    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage")
    geos: Tuple[str, ...] = ("APAC", "EMEA", "AMER")
```

- `@dataclass` tells Python to auto-create a class constructor for us
- `class Settings:` creates a config class
- `data_dir` is where your PDFs live
- `storage_dir` is where indexes, chunk files, registry, logs are saved
- `geos` is the allowed region list

Chunking settings:

```python
    chunk_size: int = 800
    chunk_overlap: int = 100
```

- `chunk_size` = target size of each chunk
- `chunk_overlap` = repeated text between chunks so meaning is not cut too hard

OCR settings:

```python
    min_page_chars: int = 40
    min_doc_chars_for_text_loader: int = 250
    min_usable_page_ratio: float = 0.30
    ocr_dpi: int = 200
    ocr_lang: str = "eng"
```

- `min_page_chars` → minimum characters to consider a page useful
- `min_doc_chars_for_text_loader` → if total text is too low, assume bad extraction
- `min_usable_page_ratio` → if too many pages are empty/noisy, use OCR
- `ocr_dpi` → image quality for OCR
- `ocr_lang` → Tesseract language

Registry setting:

```python
    registry_filename: str = "ingestion_registry.json"
```

- file that tracks what PDFs were already processed

Model settings:

```python
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

- embedding model turns text into vectors
- llm model generates answers
- reranker model reorders candidate chunks

Generation settings:

```python
    max_new_tokens: int = 180
    do_sample: bool = False
    temperature: float = 0.0
```

- `max_new_tokens` = max answer length
- `do_sample=False` = deterministic output
- `temperature=0.0` = even more deterministic

Retrieval settings:

```python
    retrieval_fetch_k: int = 10
    top_k: int = 4
    score_threshold: float = 1.15
    bm25_score_threshold: float = 0.05
    use_hybrid_retrieval: bool = True
    use_reranker: bool = True
    max_chars_per_chunk_in_prompt: int = 1200
    reranker_max_chars: int = 1200
```

- `retrieval_fetch_k` = get more candidates first
- `top_k` = final number of chunks sent to LLM
- `score_threshold` = only keep vector results good enough
- `bm25_score_threshold` = only keep keyword results good enough
- `use_hybrid_retrieval` = combine vector + BM25
- `use_reranker` = apply reranker
- `max_chars_per_chunk_in_prompt` = shorten each chunk before sending to LLM
- `reranker_max_chars` = shorten chunk text when reranking

RRF setting:

```python
    rrf_k: int = 60
```

- used in Reciprocal Rank Fusion to merge ranking lists

Create config object:

```python
settings = Settings()
settings.storage_dir.mkdir(parents=True, exist_ok=True)

settings
```

- `settings = Settings()` creates the config object
- `.mkdir(parents=True, exist_ok=True)` creates the `storage/` folder
- final `settings` displays the object in Jupyter

---

# 5. Cell: logging utilities

```python
def setup_logger(log_file: Path) -> logging.Logger:
```

- defines a function called `setup_logger`
- `log_file: Path` means input should be a path
- `-> logging.Logger` means it returns a logger object

```python
    logger = logging.getLogger("geo_rag_v2")
    logger.setLevel(logging.INFO)
```

- create/get a logger named `"geo_rag_v2"`
- set it to log INFO and above

```python
    if logger.handlers:
        return logger
```

- prevents duplicate handlers if you run the cell more than once

```python
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
```

- controls how log lines look

```python
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
```

- write logs to file

```python
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
```

- also print logs to notebook output

```python
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
```

- attach both handlers
- `propagate=False` avoids duplicate logs from parent loggers

```python
logger = setup_logger(settings.storage_dir / "rag_pipeline.log")
logger.info("Logger initialized")
```

- create logger and store it in `logger`
- write first test message

---

# 6. Cell: core helper functions

## normalize_geo

```python
def normalize_geo(value: str) -> str:
    value = value.strip().upper()
    if value not in settings.geos:
        raise ValueError(f"Unsupported GEO: {value}")
    return value
```

- remove extra spaces
- convert to uppercase
- check that GEO is one of APAC, EMEA, AMER
- throw an error if invalid

## detect_geos_from_query

```python
def detect_geos_from_query(query: str) -> List[str]:
    text = query.upper()
    found = []
```

- convert query to uppercase for matching
- prepare empty list for detected GEOs

```python
    for geo in settings.geos:
        if re.search(rf"\b{geo}\b", text):
            found.append(geo)
```

- loop through APAC/EMEA/AMER
- use regex to find whole-word match
- `rf""` means raw formatted string
- `\b` means word boundary

```python
    return found
```

- return all matched GEOs

## is_comparison_query

```python
def is_comparison_query(query: str, geos: Optional[List[str]] = None) -> bool:
```

- checks whether the question is a comparison question

```python
    text = query.lower()
    if geos and len(geos) > 1:
        return True
```

- if more than one GEO is already detected, treat as comparison

```python
    comparison_terms = [
        "compare", "comparison", "difference", "different",
        "vs", "versus", "similarities", "differences",
    ]
    return any(term in text for term in comparison_terms)
```

- if words like “compare” are present, return True

## make_doc_id

```python
def make_doc_id(path: Path) -> str:
    return hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()
```

- convert file path to absolute path
- encode to bytes
- hash with md5
- return unique-looking ID string

## clean_text

```python
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
```

- replace null characters with space
- collapse repeated spaces/tabs into one space
- collapse 3+ newlines into 2
- strip whitespace at beginning/end

## normalize_text_for_hash

- same idea as `clean_text`, but even stricter for stable deduplication hashes

## page_has_usable_text

```python
def page_has_usable_text(text: str, min_chars: int = settings.min_page_chars) -> bool:
    return len(clean_text(text)) >= min_chars
```

- returns True if cleaned text length is at least the minimum

## build_chunk_id

```python
def build_chunk_id(source: str, page: Optional[int], idx: int) -> str:
    raw = f"{source}|{page}|{idx}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()
```

- creates a unique ID using source path, page number, and chunk index

## chunk_text_hash

- creates a hash from normalized text
- used for deduplication

## tokenize_for_bm25

```python
return re.findall(r"\b\w+\b", text.lower())
```

- split text into simple lowercase tokens
- BM25 needs tokens, not raw full strings

---

# 7. Cell: OCR fallback helpers

## ocr_pdf_to_documents

```python
def ocr_pdf_to_documents(pdf_path: Path, geo: str) -> List[Document]:
```

- converts a PDF to LangChain `Document` objects using OCR

```python
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as e:
        raise ImportError(...)
```

- import OCR tools only when needed
- clearer error if missing

```python
    logger.info(f"OCR fallback started for {pdf_path.name}")
    images = convert_from_path(str(pdf_path), dpi=settings.ocr_dpi)
```

- log start
- convert each PDF page to an image

```python
    doc_id = make_doc_id(pdf_path)
    ocr_docs: List[Document] = []
```

- get document ID
- create empty result list

```python
    for page_idx, image in enumerate(images):
```

- loop over pages and their index

```python
        text = pytesseract.image_to_string(image, lang=settings.ocr_lang)
        cleaned = clean_text(text)
        if not page_has_usable_text(cleaned):
            continue
```

- OCR the page image
- clean the extracted text
- skip useless pages

```python
        metadata = {
            "geo": geo,
            "source": str(pdf_path),
            "doc_id": doc_id,
            "file_name": pdf_path.name,
            "page": page_idx,
            "loader": "ocr",
        }
```

- build metadata for this page

```python
        ocr_docs.append(Document(page_content=cleaned, metadata=metadata))
```

- create LangChain `Document` and add it to list

```python
    logger.info(...)
    return ocr_docs
```

- log result count
- return OCR documents

## load_pdf_with_fallback

```python
loader = PyPDFLoader(str(pdf_path))
pages = loader.load()
```

- try normal text extraction first

```python
doc_id = make_doc_id(pdf_path)
page_docs: List[Document] = []
total_chars = 0
usable_pages = 0
```

- initialize counters and container

```python
for page_doc in pages:
    raw_text = page_doc.page_content or ""
    cleaned = clean_text(raw_text)
    total_chars += len(cleaned)
```

- get page text
- clean it
- count total characters

```python
    if page_has_usable_text(cleaned):
        usable_pages += 1
```

- count usable pages

```python
    metadata = dict(page_doc.metadata)
    metadata.update({...})
```

- copy loader metadata
- add our own metadata like geo, source, doc_id, loader type

```python
    page_docs.append(Document(page_content=cleaned, metadata=metadata))
```

- save every page as a `Document`

```python
usable_ratio = usable_pages / max(1, len(page_docs))
```

- compute fraction of useful pages
- `max(1, len(page_docs))` avoids divide-by-zero crash

```python
if total_chars < settings.min_doc_chars_for_text_loader or usable_ratio < settings.min_usable_page_ratio:
    ...
    return ocr_pdf_to_documents(pdf_path, geo)
```

- if text extraction looks bad, use OCR instead

```python
usable_docs = [doc for doc in page_docs if page_has_usable_text(doc.page_content)]
```

- keep only usable pages from normal extraction

---

# 8. Cell: chunking, deduplication, and chunk store helpers

## text_splitter

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)
```

- creates the splitter object used later

## chunk_documents

```python
chunks = text_splitter.split_documents(docs)
```

- split input page documents into smaller chunks

```python
for idx, chunk in enumerate(chunks):
```

- loop through chunks with index

```python
    source = chunk.metadata.get("source", "")
    page = chunk.metadata.get("page")
    chunk.metadata["chunk_id"] = build_chunk_id(source, page, idx)
    chunk.metadata["text_hash"] = chunk_text_hash(chunk.page_content)
```

- attach unique chunk ID
- attach text hash for deduplication

## deduplicate_chunks

```python
unique_docs: List[Document] = []
seen_hashes = set()
```

- output list and set of already-seen hashes

```python
for doc in docs:
    text_hash = doc.metadata.get("text_hash") or chunk_text_hash(doc.page_content)
    if text_hash in seen_hashes:
        continue
```

- if chunk text already seen, skip it

```python
    seen_hashes.add(text_hash)
    doc.metadata["text_hash"] = text_hash
    unique_docs.append(doc)
```

- remember hash and keep document

## chunk_store_path

- returns a file path like `storage/chunks_apac.jsonl`

## serialize_document

- converts a `Document` into plain JSON-ready dict

## deserialize_document

- converts JSON dict back into `Document`

## save_chunk_store

```python
with path.open("w", encoding="utf-8") as f:
```

- open file for writing
- `with` automatically closes file safely

```python
    for doc in docs:
        f.write(json.dumps(serialize_document(doc), ensure_ascii=False) + "\n")
```

- write one JSON object per line
- this format is called JSONL

## load_chunk_store

- if file missing, return empty list
- otherwise read each line and convert back to `Document`

---

# 9. Cell: incremental ingestion registry

## registry_path

- returns `storage/ingestion_registry.json`

## load_registry

- if registry file does not exist, return empty dict
- else load JSON from file

## save_registry

- write the registry dict to disk as pretty JSON

## compute_file_hash

```python
h = hashlib.sha256()
with path.open("rb") as f:
```

- create SHA256 hasher
- open file in binary mode (`rb`)

```python
while True:
    chunk = f.read(chunk_size)
    if not chunk:
        break
    h.update(chunk)
```

- read file piece by piece
- update hash incrementally
- this avoids loading whole file into memory

## scan_pdf_inventory

- scans current PDF files on disk
- returns dictionary keyed by absolute path

Each entry stores:
- path
- geo
- mtime
- size

## plan_ingestion_changes

This is a key function.

It compares:
- `inventory` = current files on disk
- `registry` = last processed files

It finds:
- new files
- deleted files
- maybe updated files
- unchanged files

Important steps:
- compare path sets to find new and deleted
- compare `mtime` and `size` to find possible updates
- for possible updates, compute SHA256 to confirm actual content change
- group all results by GEO
- build `touched_geos`

This lets the notebook update only the changed GEOs.

---

# 10. Cell: log_change_plan

This function just writes the change summary into logs.

It logs:
- total new
- total updated
- total deleted
- total unchanged
- touched GEOs

Then it logs the same info for each GEO separately.

---

# 11. Cell: embeddings

```python
embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
    encode_kwargs={"normalize_embeddings": True},
)
embeddings
```

- load embedding model
- `normalize_embeddings=True` makes vectors unit-normalized
- this helps distance comparisons behave more consistently

---

# 12. Cell: incremental ingestion + per-GEO index rebuild

## geo_index_path

- returns directory like `storage/faiss_apac`

## remove_geo_index_if_exists

- deletes the FAISS folder for a GEO if it exists

## save_geo_index

```python
if not docs:
    remove_geo_index_if_exists(geo)
    return
```

- if there are no documents for that GEO, remove the old index

```python
vectorstore = FAISS.from_documents(docs, embeddings_model)
vectorstore.save_local(str(index_dir))
```

- build FAISS index from documents
- save it to disk

## load_geo_vectorstores

- loads saved FAISS indexes from disk into memory

## upsert_registry_entry

- create or update one registry entry for a PDF
- stores:
  - path
  - geo
  - mtime
  - size
  - sha256
  - doc_id
  - last_ingested_at

## remove_registry_entry

- delete entry from registry if file was removed

## rebuild_touched_geos_only

This is the main incremental ingestion function.

High-level logic:
1. look at touched GEOs only
2. remove chunks belonging to changed/deleted PDFs
3. re-read changed PDFs
4. chunk them
5. deduplicate
6. save chunk store
7. rebuild FAISS only for that GEO
8. update registry

Important lines:

```python
existing_chunks = load_chunk_store(geo)
remaining_chunks = [
    doc for doc in existing_chunks
    if doc.metadata.get("source") not in touched_paths
]
```

- load old chunks
- remove chunks belonging to changed or deleted PDFs

```python
for path_str in tqdm(changed_paths, desc=f"Ingesting {geo} changes"):
    pdf_path = Path(path_str)
    page_docs = load_pdf_with_fallback(pdf_path, geo)
    new_page_docs.extend(page_docs)
    upsert_registry_entry(registry, pdf_path, geo)
```

- reprocess only changed PDFs

```python
new_chunk_docs = deduplicate_chunks(chunk_documents(new_page_docs))
combined_chunks = deduplicate_chunks(remaining_chunks + new_chunk_docs)
```

- chunk changed PDFs
- merge with old untouched chunks
- deduplicate again

```python
save_chunk_store(geo, combined_chunks)
save_geo_index(geo, combined_chunks, embeddings_model)
```

- save JSONL store
- rebuild FAISS for this GEO only

---

# 13. Cell: run incremental ingestion

```python
registry = load_registry()
inventory = scan_pdf_inventory(settings.data_dir)
change_plan = plan_ingestion_changes(inventory, registry)
log_change_plan(change_plan)
```

- load previous registry
- scan current files
- compute changes
- log them

```python
geo_vectorstores = rebuild_touched_geos_only(change_plan, registry, embeddings)
```

- rebuild only touched GEOs

```python
if not change_plan["touched_geos"]:
    geo_vectorstores = load_geo_vectorstores(embeddings)
```

- if nothing changed, just load existing indexes

```python
print("Loaded vectorstores for GEOs:", list(geo_vectorstores.keys()))
```

- show which GEO indexes are ready

---

# 14. Cell: build BM25 indexes

This builds keyword search indexes from saved chunk stores.

## build_bm25_indexes

```python
from rank_bm25 import BM25Okapi
```

- import BM25 model

```python
for geo in settings.geos:
    docs = load_chunk_store(geo)
    if not docs:
        continue
```

- load chunks for each GEO
- skip empty GEOs

```python
tokenized_corpus = [tokenize_for_bm25(doc.page_content) for doc in docs]
bm25 = BM25Okapi(tokenized_corpus)
```

- tokenize each chunk
- build BM25 index

```python
indexes[geo] = {
    "bm25": bm25,
    "docs": docs,
}
```

- save BM25 model and original docs together

Then:

```python
bm25_indexes = build_bm25_indexes()
print("BM25 indexes:", list(bm25_indexes.keys()))
```

- build BM25 indexes now
- print available GEOs

---

# 15. Cell: load local instruction model

```python
generator = pipeline(
    "text-generation",
    model=settings.llm_model,
    torch_dtype="auto",
    device_map="auto",
)
```

- create a Hugging Face text generation pipeline
- `"text-generation"` means we want an LLM that generates text
- `model=settings.llm_model` uses the configured Qwen model
- `torch_dtype="auto"` lets Transformers choose suitable tensor type
- `device_map="auto"` lets it place model on GPU/CPU automatically

`generator` on final line displays the object in notebook.

---

# 16. Cell: load reranker

```python
from sentence_transformers import CrossEncoder
```

- CrossEncoder is used for reranking
- unlike embeddings, it scores query-document pairs directly

```python
reranker = CrossEncoder(settings.reranker_model)
```

- load reranker model

```python
reranker = load_reranker() if settings.use_reranker else None
```

- use reranker only when enabled
- otherwise set it to `None`

---

# 17. Cell: inspect retrieval scores

```python
def inspect_retrieval_scores(question: str, stores: Dict[str, FAISS], k: int = 5) -> None:
```

- debugging helper to see vector scores before full answer generation

```python
geos = detect_geos_from_query(question) or list(stores.keys())
```

- use GEOs found in query
- if none found, use all available GEOs

```python
docs_and_scores = stores[geo].similarity_search_with_score(question, k=k)
```

- get top-k FAISS results with scores

```python
print(f"{i}. score={score:.4f} | file=... ")
```

- print score and source info
- `:.4f` means show float with 4 decimals

---

# 18. Cell: BM25 search, RRF, reranking

## bm25_search

- gets BM25 scores for a query within one GEO
- sorts them descending
- keeps results above threshold
- returns top-k

Important point:
- higher BM25 score is better

## reciprocal_rank_fusion

This merges vector results and BM25 results.

```python
fused_scores = defaultdict(float)
doc_map: Dict[str, Document] = {}
```

- store fused score per chunk
- store actual doc by chunk ID

For each vector result:

```python
fused_scores[key] += 1.0 / (settings.rrf_k + rank)
doc.metadata["vector_score"] = float(score)
```

- add reciprocal rank contribution
- save vector score in metadata

For each BM25 result:
- same idea, but save `bm25_score`

Then:
- sort chunks by fused score
- write `rrf_score` into metadata
- return ranked docs

## rerank_documents

```python
if not docs or reranker_model is None:
    return docs[:top_n]
```

- if no reranker, just keep the first documents

```python
pairs = [[question, doc.page_content[: settings.reranker_max_chars]] for doc in docs]
scores = reranker_model.predict(pairs)
```

- prepare query-document pairs
- get reranker scores

```python
ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
```

- pair each doc with its score
- sort from best to worst

```python
doc.metadata["rerank_score"] = float(score)
```

- store reranker score in metadata

---

# 19. Cell: retrieve_documents

This is the main retrieval function.

```python
requested_geos = detect_geos_from_query(question)
target_geos = requested_geos if requested_geos else list(stores.keys())
```

- if query says APAC/AMER, use only those
- otherwise search all GEOs

```python
diagnostics: Dict[str, Any] = {"geos": target_geos, "by_geo": {}}
all_candidates: List[Document] = []
```

- create diagnostics dict for debugging
- create list for retrieval candidates

For each GEO:

```python
vector_raw = stores[geo].similarity_search_with_score(question, k=fetch_k)
vector_filtered = [(doc, score) for doc, score in vector_raw if score <= settings.score_threshold]
```

- get vector matches
- keep only good-enough matches
- note: for this FAISS setup, lower score is better

```python
bm25_filtered = bm25_search(...)
```

- get keyword matches

```python
if settings.use_hybrid_retrieval:
    fused_docs = reciprocal_rank_fusion(vector_filtered, bm25_filtered)
else:
    fused_docs = [doc for doc, _ in vector_filtered]
```

- combine vector + BM25 if hybrid enabled
- otherwise use vector only

```python
diagnostics["by_geo"][geo] = {...}
```

- save per-GEO counts for debugging

Then deduplication across all GEO candidates:

```python
deduped_candidates = []
seen = set()
for doc in all_candidates:
    key = doc.metadata.get("chunk_id")
    if key not in seen:
        seen.add(key)
        deduped_candidates.append(doc)
```

- prevents same chunk appearing twice

Finally rerank:

```python
reranked_docs = rerank_documents(question, deduped_candidates, reranker, top_n=k)
```

- final top-k goes through reranker

Return values:
- selected GEOs
- final documents
- diagnostics

---

# 20. Cell: prompt templates

## STANDARD_SYSTEM_PROMPT

- used for normal questions

## COMPARISON_SYSTEM_PROMPT

- used when query compares multiple GEOs
- asks model to structure answer into sections when possible

## format_context

```python
for i, doc in enumerate(docs[: settings.top_k], start=1):
```

- loop over top chunks only

```python
source = doc.metadata.get("file_name", "unknown")
page = doc.metadata.get("page", "unknown")
geo = doc.metadata.get("geo", "unknown")
text = doc.page_content[: settings.max_chars_per_chunk_in_prompt]
```

- read metadata
- trim content before sending to LLM

```python
blocks.append(
    f"[Chunk {i}]\n"
    f"GEO: {geo}\n"
    f"Source: {source}\n"
    f"Page: {page}\n"
    f"Content:\n{text}"
)
```

- create readable block for each chunk

```python
return "\n\n---\n\n".join(blocks)
```

- join all blocks with separators

---

# 21. Cell: structured source output

## build_structured_sources

- groups sources by GEO
- each source keeps:
  - file_name
  - page
  - chunk_id
  - loader
  - vector_score
  - bm25_score
  - rrf_score
  - rerank_score

This is helpful for debugging and trust.

## extract_assistant_text

This handles the output format from Hugging Face pipeline.

Sometimes pipeline returns chat-style list of messages.

```python
generated = generated_output[0]["generated_text"]
```

- get generated content from first result

```python
if isinstance(generated, list):
```

- if output is a list of chat messages, take the last one

Otherwise:
- convert output to string and return it

---

# 22. Cell: answer_question

This is the top-level function you call.

```python
geos, docs, diagnostics = retrieve_documents(question, stores, bm25_indexes)
```

- retrieve candidate chunks first

```python
if not docs:
    return {
        "question": question,
        "geo_used": geos,
        "answer": "I do not have the answer in the provided documents.",
        ...
    }
```

- if nothing passed retrieval, stop before LLM
- this is important guardrail behavior

```python
system_prompt = (
    COMPARISON_SYSTEM_PROMPT
    if is_comparison_query(question, geos)
    else STANDARD_SYSTEM_PROMPT
)
```

- choose prompt type based on question

```python
context = format_context(docs)
geo_label = ", ".join(geos) if geos else "ALL"
```

- build final context string
- create readable GEO label

```python
messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": (
            f"GEO filter: {geo_label}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        ),
    },
]
```

- build chat-style input for local model

```python
outputs = generator(
    messages,
    max_new_tokens=settings.max_new_tokens,
    do_sample=settings.do_sample,
    temperature=settings.temperature,
)
```

- call the local LLM

```python
answer = extract_assistant_text(outputs)
```

- turn model output into plain text

```python
flat_sources = [...]
```

- build flat list of source details

```python
result = {
    "question": question,
    "geo_used": geos,
    "answer": answer,
    "retrieval_diagnostics": diagnostics,
    "sources_by_geo": build_structured_sources(docs),
    "flat_sources": flat_sources,
}
```

- final result object returned to you

```python
logger.info(...)
return result
```

- log answer summary
- return the final dictionary

---

# 23. Example cells

## inspect retrieval scores

```python
inspect_retrieval_scores("What is the refund policy for APAC?", geo_vectorstores, k=5)
```

- this does not answer
- it only shows candidate scores

## normal answer

```python
result = answer_question("What is the refund policy for APAC?", geo_vectorstores, bm25_indexes)
print(result["answer"])
print(json.dumps(result["sources_by_geo"], indent=2))
```

- call the full pipeline
- print answer
- print grouped sources as formatted JSON

## comparison answer

```python
result = answer_question("Compare the refund policy for APAC and AMER", geo_vectorstores, bm25_indexes)
print(result["answer"])
print(json.dumps(result["sources_by_geo"], indent=2))
print(json.dumps(result["retrieval_diagnostics"], indent=2))
```

- asks multi-GEO question
- prints answer
- prints sources
- prints retrieval debug info

---

# 24. Validation helpers

These cells help you prove incremental ingestion works.

## get_geo_artifact_snapshot

For each GEO, it records:
- whether index exists
- latest modification time of index files
- whether chunk store exists
- chunk count

## print_geo_snapshot

- prints that snapshot nicely

## compare_snapshots

- compares before vs after
- shows:
  - whether index changed
  - chunk counts before/after
  - chunk count delta

## print_snapshot_comparison

- prints comparison nicely

---

# 25. Baseline cell

```python
baseline_registry = load_registry()
baseline_snapshot = get_geo_artifact_snapshot()

print(f"Registry entries: {len(baseline_registry)}")
print_geo_snapshot(baseline_snapshot, title="Baseline snapshot")
```

- captures current state before you change files
- lets you compare later

---

# 26. Rerun incremental ingestion after file change

```python
registry = load_registry()
inventory = scan_pdf_inventory(settings.data_dir)
change_plan = plan_ingestion_changes(inventory, registry)
log_change_plan(change_plan)
```

- recompute change plan after you added/updated/deleted a file

```python
geo_vectorstores = rebuild_touched_geos_only(change_plan, registry, embeddings)
```

- rebuild only touched GEOs

```python
if not change_plan["touched_geos"]:
    geo_vectorstores = load_geo_vectorstores(embeddings)
```

- if nothing changed, just load existing indexes

```python
after_snapshot = get_geo_artifact_snapshot()
...
comparison = compare_snapshots(baseline_snapshot, after_snapshot)
print_snapshot_comparison(comparison)
```

- compare before and after
- verify only one GEO changed

---

# 27. Big picture of how the code flows

The notebook has **two major phases**.

## Phase A: Ingestion

1. scan PDFs
2. compare them to registry
3. detect new / updated / deleted
4. read changed PDFs
5. use OCR if needed
6. split into chunks
7. deduplicate
8. save chunk JSONL
9. rebuild FAISS only for affected GEOs
10. update registry

## Phase B: Query answering

1. detect GEOs from question
2. search vector index
3. search BM25 index
4. fuse rankings
5. rerank results
6. apply threshold logic
7. build prompt
8. send prompt to local LLM
9. return answer + sources

---

# 28. Three Python syntax things that matter here

## a) Type hints

Examples:
- `question: str`
- `-> List[Document]`
- `Dict[str, Any]`

These do **not** change runtime behavior much.
They mostly help readability and editors.

## b) f-strings

Example:
```python
f"GEO filter: {geo_label}"
```

This means:
- put the value of `geo_label` into the string

## c) List comprehensions

Example:
```python
[tokenize_for_bm25(doc.page_content) for doc in docs]
```

This is a short way to write a loop that builds a list.

Equivalent long form:
```python
result = []
for doc in docs:
    result.append(tokenize_for_bm25(doc.page_content))
```

---

# 29. Most important functions to understand first

If you are new, read these first in this order:

1. `load_pdf_with_fallback`
2. `chunk_documents`
3. `deduplicate_chunks`
4. `plan_ingestion_changes`
5. `rebuild_touched_geos_only`
6. `retrieve_documents`
7. `answer_question`

Those are the backbone of the whole notebook.

---

# 30. Simple mental model for each main object

- `Document` = one piece of text + metadata
- `chunk store` = saved JSONL list of chunks
- `registry` = memory of which PDFs were processed before
- `FAISS index` = fast semantic search over chunk embeddings
- `BM25 index` = keyword search over chunk text
- `reranker` = relevance sorter
- `generator` = local LLM that writes the answer

---

# 31. Final summary

The notebook is organized like this:

- imports and settings
- logging
- helper functions
- OCR fallback
- chunking + deduplication
- registry and change detection
- incremental ingestion
- embeddings and FAISS
- BM25
- reranker
- retrieval
- prompts
- final answering
- validation for incremental updates

That is the complete pipeline.

