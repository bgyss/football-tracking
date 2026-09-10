#!/usr/bin/env python3
"""Run the repository's local benchmark command."""

from __future__ import annotations

from football_tracking.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["benchmark", *__import__("sys").argv[1:]]))
