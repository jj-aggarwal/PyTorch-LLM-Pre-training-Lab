"""Thin wrapper so `python scripts/evaluate.py` works. Logic lives in the package."""

from pretraining_lab.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
