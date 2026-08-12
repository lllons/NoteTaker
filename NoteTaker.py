#!/usr/bin/env python3
"""Backward-compatible CLI for the modular NoteTaker application."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-first high-fidelity knowledge capture")
    parser.add_argument("--model", default=os.getenv("NOTE_TAKER_MODEL", "small.en"))
    parser.add_argument("--draft", default=os.getenv("NOTE_TAKER_DRAFT_MODEL", "base.en"))
    parser.add_argument("--lang", default=os.getenv("NOTE_TAKER_LANGUAGE") or None)
    parser.add_argument("--hotwords", default=os.getenv("NOTE_TAKER_HOTWORDS") or None)
    parser.add_argument("--beam", type=int, default=int(os.getenv("NOTE_TAKER_BEAM_SIZE", "5")))
    parser.add_argument("--threads", type=int, default=int(os.getenv("NOTE_TAKER_THREADS", "0")))
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()

    os.environ["NOTE_TAKER_MODEL"] = args.model
    os.environ["NOTE_TAKER_DRAFT_MODEL"] = args.draft
    os.environ["NOTE_TAKER_BEAM_SIZE"] = str(args.beam)
    os.environ["NOTE_TAKER_THREADS"] = str(args.threads)
    if args.lang:
        os.environ["NOTE_TAKER_LANGUAGE"] = args.lang
    if args.hotwords:
        os.environ["NOTE_TAKER_HOTWORDS"] = args.hotwords

    import uvicorn
    from notetaker.app import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
