"""The stable public command-line interface for VLBrandPhish."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence


def _dispatch(module_main: Callable[[], None], arguments: Sequence[str]) -> None:
    """Delegate legacy module parsers without duplicating their options."""
    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0], *arguments]
        module_main()
    finally:
        sys.argv = previous_argv


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vlbrandphish",
        description="VLBrandPhish: visual-language brand-aware phishing detection.",
    )
    commands = ("detect", "infer", "train-brand", "train-detector")
    parser.add_argument(
        "command",
        nargs="?",
        metavar="{detect,infer,train-brand,train-detector}",
        help="Run logo detection, inference, brand training, or detector training.",
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return

    command = arguments.pop(0)
    if command not in commands:
        parser.error(f"unknown command: {command}")

    if command == "detect":
        from inference.vlbrandphish.detection.command import main as command_main
    elif command == "infer":
        from inference.vlbrandphish.pipeline import main as command_main
    elif command == "train-brand":
        from training.vlbrandphish.brand_recognition.cli import main as command_main
    else:
        from training.vlbrandphish.logo_detection.train import main as command_main

    _dispatch(command_main, arguments)


if __name__ == "__main__":
    main()
