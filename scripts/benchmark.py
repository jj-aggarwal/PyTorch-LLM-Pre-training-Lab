"""Thin wrapper so `python scripts/benchmark.py` works. Logic lives in the package."""

from pretraining_lab.benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
