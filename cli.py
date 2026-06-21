"""Timonel RAG — Command-line interface.

Usage::

    python cli.py ingest              # Index all PDFs in the data directory.
    python cli.py query <question>    # Ask a question; print answer + sources.

Both commands read configuration from environment variables / .env file.
See .env.example for the full list of available settings.
"""

from __future__ import annotations

import argparse
import logging
import sys


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def cmd_ingest(_args: argparse.Namespace) -> None:
    """Run the ingest pipeline and exit."""
    from src.config import settings
    from src.core.ingest import ingest

    try:
        ingest(settings)
    except (FileNotFoundError, RuntimeError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        sys.exit(1)


def cmd_query(args: argparse.Namespace) -> None:
    """Run a retrieval query and print structured output."""
    from src.config import settings
    from src.core.retrieval import run_query

    question = " ".join(args.question)
    try:
        result = run_query(question, settings)
    except RuntimeError as exc:
        logging.getLogger(__name__).error("%s", exc)
        sys.exit(1)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"QUESTION: {result.question}")
    print(sep)
    print(f"\nANSWER:\n{result.answer}\n")
    print("-" * 60)
    print("SOURCES:")
    for i, doc in enumerate(result.sources):
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        print(f"  [{i + 1}] {src}  (page {page})")
    print(sep + "\n")


def main() -> None:
    # Import settings early only to read log_level; heavy imports stay lazy.
    from src.config import settings

    _configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(
        prog="timonel",
        description="Timonel RAG — nautical document question-answering system.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Index PDFs from the configured data directory.")

    q_parser = sub.add_parser("query", help="Ask a question and print the answer.")
    q_parser.add_argument("question", nargs="+", help="Question text (no quotes needed).")

    args = parser.parse_args()
    dispatch = {"ingest": cmd_ingest, "query": cmd_query}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
