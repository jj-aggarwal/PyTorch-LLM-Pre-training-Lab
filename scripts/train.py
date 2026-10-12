"""Thin wrapper so `python scripts/train.py` works. Logic lives in the package."""

from pretraining_lab.train import main

if __name__ == "__main__":
    raise SystemExit(main())
