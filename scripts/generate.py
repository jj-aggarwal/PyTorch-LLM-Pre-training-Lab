"""Thin wrapper so `python scripts/generate.py` works. Logic lives in the package."""

from pretraining_lab.generate import main

if __name__ == "__main__":
    raise SystemExit(main())
