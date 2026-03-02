"""CLI script for batch ingestion of documents from a directory.

Usage::

    python scripts/ingest_batch.py --source data/raw/
    python scripts/ingest_batch.py --source data/raw/ --pattern "*.pdf" --strategy semantic
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.logging_config import setup_logging
from config.settings import settings
from core.cache.in_memory_cache import InMemoryCache
from core.cache.redis_cache import RedisCache
from core.embeddings.openai_embedder import OpenAIEmbedder
from core.vectorstore.vectorstore_factory import VectorStoreFactory
from services.ingestion_service import IngestionService

setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ingest documents into the RAG platform.")
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Directory containing documents to ingest.",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern to filter files (default: '*').",
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "semantic"],
        default=None,
        help="Chunking strategy override (default: from settings).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source.resolve()

    if not source_dir.is_dir():
        print(f"ERROR: '{source_dir}' is not a directory.")
        sys.exit(1)

    files = sorted(source_dir.glob(args.pattern))
    supported = {".pdf", ".docx", ".html", ".htm"}
    files = [f for f in files if f.suffix.lower() in supported]

    if not files:
        print(f"No supported files found in '{source_dir}' matching '{args.pattern}'.")
        sys.exit(0)

    print(f"Found {len(files)} file(s) to ingest.")

    cache = RedisCache() if settings.app.env == "prod" else InMemoryCache()
    service = IngestionService(
        embedder=OpenAIEmbedder(),
        vector_store=VectorStoreFactory.get_store(),
        cache=cache,
        chunking_strategy=args.strategy,
    )

    t0 = time.perf_counter()
    success, failed = 0, 0

    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Ingesting: {file_path.name} ...", end=" ", flush=True)
        result = service.ingest(file_path)
        if result.status == "success":
            print(
                f"OK  ({result.indexed_chunks} new / {result.skipped_chunks} skipped "
                f"/ {result.duration_s}s)"
            )
            success += 1
        else:
            print(f"FAILED ({result.duration_s}s)")
            failed += 1

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"
Done: {success} succeeded, {failed} failed — total {elapsed}s")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
