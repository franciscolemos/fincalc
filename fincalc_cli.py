#!/usr/bin/env python3
from __future__ import annotations

"""ConvFinQA command line interface.

This script wraps the common Retriever and Generator workflows for ConvFinQA
into a single menu-driven CLI.  It exposes both interactive and scripted
sub-commands and supports lightweight configuration via JSON files.

Example usage:

    python convfinqa_cli.py menu
    python convfinqa_cli.py train-retriever --config my_overrides.json
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional

# ---------------------------------------------------------------------------
# Ensure project paths are importable
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
RETRIEVER_DIR = CODE_DIR / "finqanet_retriever"

UTILS_DIR = CODE_DIR / "utils"

for p in (CODE_DIR, RETRIEVER_DIR, UTILS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# Import ConvFinQA utilities
from finqanet_retriever.finqa_utils import (
    get_json_keys,
    count_json_entries,
    preview_samples,
)


# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------

CONFIG_PATH_CANDIDATES: tuple[Path, ...] = (
    Path("convfinqa_cli.json"),
    Path("config/convfinqa_cli.json"),
)


@dataclass
class CLIConfig:
    """Holds configurable values for the CLI."""

    python_executable: str = sys.executable
    retriever_dir: str = "code/finqanet_retriever"
    generator_dir: str = "code/finqanet_generator"
    data_dir: str = "data"
    retriever_train_file: str = "data/train_turn.json"
    retriever_valid_file: str = "data/dev_turn.json"
    retriever_test_file: str = "data/dev_turn.json"
    retriever_output_dir: str = "code/finqanet_retriever/output"
    retriever_save_path: str = (
        "code/finqanet_retriever/output/retriever-roberta-base_*/saved_model/"
        "loads/1/model.pt"
    )
    retriever_predictions: str = "code/finqanet_retriever/retriever_outputs.json"
    generator_train_file: str = "code/finqanet_generator/dev_retrieve.json"
    generator_valid_file: str = "code/finqanet_generator/dev_retrieve.json"
    generator_test_file: str = "code/finqanet_generator/test_retrieve.json"
    generator_output_dir: str = "code/finqanet_generator/generator_ckpt"
    spinner_interval: float = 0.1
    spinner_frames: Iterable[str] = field(
        default_factory=lambda: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    )

    @classmethod
    def from_file(cls, path: Path) -> "CLIConfig":
        with path.open("r", encoding="utf8") as stream:
            overrides = json.load(stream)
        return cls(**{**cls().to_dict(), **overrides})

    def merge(self, overrides: Optional[Mapping[str, object]]) -> "CLIConfig":
        if not overrides:
            return self
        merged = {**self.to_dict()}
        merged.update(overrides)
        return CLIConfig(**merged)

    def to_dict(self) -> Dict[str, object]:
        return {
            "python_executable": self.python_executable,
            "retriever_dir": self.retriever_dir,
            "generator_dir": self.generator_dir,
            "data_dir": self.data_dir,
            "retriever_train_file": self.retriever_train_file,
            "retriever_valid_file": self.retriever_valid_file,
            "retriever_test_file": self.retriever_test_file,
            "retriever_output_dir": self.retriever_output_dir,
            "retriever_save_path": self.retriever_save_path,
            "retriever_predictions": self.retriever_predictions,
            "generator_train_file": self.generator_train_file,
            "generator_valid_file": self.generator_valid_file,
            "generator_test_file": self.generator_test_file,
            "generator_output_dir": self.generator_output_dir,
            "spinner_interval": self.spinner_interval,
            "spinner_frames": list(self.spinner_frames),
        }


def load_config(config_path: Optional[str]) -> CLIConfig:
    if config_path:
        return CLIConfig.from_file(Path(config_path))

    for candidate in CONFIG_PATH_CANDIDATES:
        if candidate.exists():
            return CLIConfig.from_file(candidate)

    return CLIConfig()

# ---------------------------------------------------------------------------
# Spinner utility
# ---------------------------------------------------------------------------


class Spinner:
    def __init__(self, message: str, frames: Iterable[str], interval: float) -> None:
        self._message = message
        self._frames = list(frames)
        if not self._frames:
            self._frames = ["-"]
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "Spinner":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        sys.stdout.write("\r" + " " * (len(self._message) + 4) + "\r")
        sys.stdout.flush()

    def _run(self) -> None:
        idx = 0
        while not self._stop_event.is_set():
            frame = self._frames[idx % len(self._frames)]
            sys.stdout.write(f"\r{frame} {self._message}")
            sys.stdout.flush()
            time.sleep(self._interval)
            idx += 1


# ---------------------------------------------------------------------------
# Command execution helpers
# ---------------------------------------------------------------------------

class CommandError(RuntimeError):
    pass


def run_subprocess(command: List[str], cwd: Optional[str], config: CLIConfig) -> None:
    display_cmd = " ".join(command)
    message = f"Running: {display_cmd}"
    with Spinner(message, config.spinner_frames, config.spinner_interval):
        process = subprocess.Popen(command, cwd=cwd)
        return_code = process.wait()
    if return_code != 0:
        raise CommandError(f"Command failed ({return_code}): {display_cmd}")
    print(f"✅ Completed: {display_cmd}")


def ensure_path_exists(path: str, kind: str = "file") -> None:
    expanded = os.path.expanduser(path)
    if "*" in expanded:
        return
    path_obj = Path(expanded)
    if kind == "file" and not path_obj.is_file():
        raise FileNotFoundError(f"Expected {kind} at '{path}'")
    if kind == "dir" and not path_obj.is_dir():
        raise FileNotFoundError(f"Expected {kind} at '{path}'")


# ---------------------------------------------------------------------------
# Task implementations
# ---------------------------------------------------------------------------

def train_retriever(config: CLIConfig, extra_args: List[str]) -> None:
    ensure_path_exists(config.retriever_train_file)
    ensure_path_exists(config.retriever_valid_file)
    command = [
        config.python_executable,
        "Main.py",
        "--train_file",
        config.retriever_train_file,
        "--valid_file",
        config.retriever_valid_file,
        "--output_dir",
        config.retriever_output_dir,
    ] + extra_args
    run_subprocess(command, config.retriever_dir, config)


def eval_retriever(config: CLIConfig, extra_args: List[str]) -> None:
    ensure_path_exists(config.retriever_test_file)
    command = [
        config.python_executable,
        "Test.py",
        "--model_path",
        config.retriever_save_path,
        "--test_file",
        config.retriever_test_file,
        "--save_path",
        config.retriever_predictions,
    ] + extra_args
    run_subprocess(command, config.retriever_dir, config)


def convert_split(config: CLIConfig, split: str, extra_args: List[str]) -> None:
    ensure_path_exists(config.retriever_predictions)
    command = [
        config.python_executable,
        "Convert.py",
        "--retriever_file",
        config.retriever_predictions,
        "--save_path",
        config.generator_train_file if split == "dev" else config.generator_test_file,
        "--split",
        split,
    ] + extra_args
    run_subprocess(command, config.generator_dir, config)


def train_generator(config: CLIConfig, extra_args: List[str]) -> None:
    ensure_path_exists(config.generator_train_file)
    ensure_path_exists(config.generator_valid_file)
    command = [
        config.python_executable,
        "Main.py",
        "--train_file",
        config.generator_train_file,
        "--valid_file",
        config.generator_valid_file,
        "--test_file",
        config.generator_test_file,
        "--output_dir",
        config.generator_output_dir,
        "--mode",
        "train",
    ] + extra_args
    run_subprocess(command, config.generator_dir, config)


def eval_generator(config: CLIConfig, extra_args: List[str]) -> None:
    ensure_path_exists(config.generator_test_file)
    command = [
        config.python_executable,
        "Main.py",
        "--test_file",
        config.generator_test_file,
        "--output_dir",
        config.generator_output_dir,
        "--mode",
        "test",
    ] + extra_args
    run_subprocess(command, config.generator_dir, config)


TASKS: Dict[str, Callable[[CLIConfig, List[str]], None]] = {
    "train-retriever": train_retriever,
    "eval-retriever": eval_retriever,
    "convert-dev": lambda cfg, args: convert_split(cfg, "dev", args),
    "convert-test": lambda cfg, args: convert_split(cfg, "test", args),
    "train-generator": train_generator,
    "eval-generator": eval_generator,
}

# ---------------------------------------------------------------------------
# Interactive menu helpers
# ---------------------------------------------------------------------------

def print_tree(root: Path, prefix: str = "", depth: int = 3, level: int = 0) -> None:
    """Recursively print a directory tree up to given depth."""
    if not root.exists():
        print(f"{prefix}(missing: {root})")
        return
    items = sorted(root.iterdir())
    for idx, item in enumerate(items):
        connector = "└── " if idx == len(items) - 1 else "├── "
        print(f"{prefix}{connector}{item.name}")
        if item.is_dir() and level < depth - 1:
            extension = "    " if idx == len(items) - 1 else "│   "
            print_tree(item, prefix + extension, depth, level + 1)


def view_models(config: CLIConfig) -> None:
    print("\n📂 Retriever Models Tree")
    print("=" * 40)
    retriever_path = Path(config.retriever_output_dir)
    print(f"{retriever_path}")
    print_tree(retriever_path)

    print("\n📂 Generator Models Tree")
    print("=" * 40)
    generator_path = Path(config.generator_output_dir)
    print(f"{generator_path}")
    print_tree(generator_path)


def view_config_retriever(config: CLIConfig) -> None:
    print("\n⚙️ Retriever Configuration")
    print("=" * 40)
    keys = [
        "retriever_dir", "retriever_train_file", "retriever_valid_file", "retriever_test_file",
        "retriever_output_dir", "retriever_save_path", "retriever_predictions"
    ]
    for k in keys:
        print(f"{k}: {getattr(config, k)}")


def view_config_generator(config: CLIConfig) -> None:
    print("\n⚙️ Generator Configuration")
    print("=" * 40)
    keys = [
        "generator_dir", "generator_train_file", "generator_valid_file", "generator_test_file",
        "generator_output_dir"
    ]
    for k in keys:
        print(f"{k}: {getattr(config, k)}")


def config_menu(config: CLIConfig) -> None:
    while True:
        print("\nConfiguration Menu")
        print("=" * 40)
        print("[1] Retriever Configuration")
        print("[2] Generator Configuration")
        print("[0] Back")
        choice = input("Select an option: ").strip()
        if choice == "0":
            return
        elif choice == "1":
            view_config_retriever(config)
        elif choice == "2":
            view_config_generator(config)
        else:
            print("❌ Invalid selection.")


def retriever_menu(config: CLIConfig) -> None:
    while True:
        print("\nRetriever Tasks")
        print("=" * 40)
        print("[1] Train Retriever")
        print("[2] Evaluate Retriever")
        print("[3] Convert Dev Split")
        print("[4] Convert Test Split")
        print("[0] Back")
        choice = input("Select an option: ").strip()
        if choice == "0":
            return
        elif choice == "1":
            train_retriever(config, [])
        elif choice == "2":
            eval_retriever(config, [])
        elif choice == "3":
            convert_split(config, "dev", [])
        elif choice == "4":
            convert_split(config, "test", [])
        else:
            print("❌ Invalid selection.")


def generator_menu(config: CLIConfig) -> None:
    while True:
        print("\nGenerator Tasks")
        print("=" * 40)
        print("[1] Train Generator")
        print("[2] Evaluate Generator")
        print("[0] Back")
        choice = input("Select an option: ").strip()
        if choice == "0":
            return
        elif choice == "1":
            train_generator(config, [])
        elif choice == "2":
            eval_generator(config, [])
        else:
            print("❌ Invalid selection.")


def workflow_menu(config: CLIConfig) -> None:
    while True:
        print("\nSuggested Workflows")
        print("=" * 40)
        print("[1] Full Retriever Workflow (train → eval → convert)")
        print("[2] Full Generator Workflow (train → eval)")
        print("[0] Back")
        choice = input("Select an option: ").strip()
        if choice == "0":
            return
        elif choice == "1":
            print("▶ Running full retriever workflow...")
            train_retriever(config, [])
            eval_retriever(config, [])
            convert_split(config, "dev", [])
            convert_split(config, "test", [])
        elif choice == "2":
            print("▶ Running full generator workflow...")
            train_generator(config, [])
            eval_generator(config, [])
        else:
            print("❌ Invalid selection.")


def inspect_data_files(config: CLIConfig) -> None:
    while True:
        print("\nInspect Data Files")
        print("=" * 40)
        print("[1] List JSON files")
        print("[2] Show keys of a file")
        print("[3] Count entries in a file")
        print("[4] Show input vs ground truth samples")
        print("[0] Back")
        choice = input("Select an option: ").strip()
        if choice == "0":
            return

        data_dir = Path(config.data_dir)
        files = sorted([f for f in data_dir.glob("*.json")])
        if not files:
            print("⚠️ No JSON files found.")
            continue

        for i, f in enumerate(files, 1):
            print(f"[{i}] {f.name}")
        try:
            idx = int(input("Select file #: ")) - 1
            file_path = files[idx]
        except (ValueError, IndexError):
            print("❌ Invalid selection.")
            continue

        if choice == "2":
            keys = get_json_keys(str(file_path))
            print("Top-level keys:", keys)
        elif choice == "3":
            count = count_json_entries(str(file_path))
            print(f"Number of entries: {count}")
        elif choice == "4":
            n = int(input("How many samples? ") or "3")
            samples = preview_samples(str(file_path), n=n)
            for i, (q, a) in enumerate(samples, 1):
                print(f"{i}. Input: {q}\n   Ground truth: {a}\n")
        elif choice == "1":
            # already listed files
            continue
        else:
            print("❌ Invalid selection.")


def menu(config: CLIConfig, extra_args: List[str]) -> None:
    if extra_args:
        print("⚠️  Extra arguments ignored in menu mode.")

    while True:
        print("\nConvFinQA Interactive CLI")
        print("=" * 40)
        print("[1] Retriever Tasks")
        print("[2] Generator Tasks")
        print("[3] View Models (tree)")
        print("[4] View Configuration")
        print("[5] Suggested Workflows")
        print("[6] View Project Tree")
        print("[7] Inspect Data Files")
        print("[0] Exit")

        choice = input("Select an option: ").strip()
        if choice == "0" or choice.lower() in {"q","quit","exit"}:
            print("👋 Goodbye!")
            return
        elif choice == "1":
            retriever_menu(config)
        elif choice == "2":
            generator_menu(config)
        elif choice == "3":
            view_models(config)
        elif choice == "4":
            config_menu(config)
        elif choice == "5":
            workflow_menu(config)
        elif choice == "6":
            depth = int(input("Depth (default=3): ") or "3")
            print_tree(PROJECT_ROOT, depth=depth)
        elif choice == "7":
            inspect_data_files(config)
        else:
            print("❌ Invalid selection.")


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ConvFinQA workflow helper")
    parser.add_argument(
        "command",
        choices=["menu", *TASKS.keys()],
        help="Action to perform",
    )
    parser.add_argument(
        "--config",
        help="Path to a JSON configuration file overriding defaults.",
    )
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        action="append",
        help="Override configuration entries inline (repeatable).",
    )
    parser.add_argument(
        "--pass",
        dest="forward",
        action="append",
        metavar="ARG",
        help="Extra arguments forwarded to the underlying script.",
    )
    return parser.parse_args(argv)


def parse_inline_overrides(pairs: Optional[Iterable[str]]) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    if not pairs:
        return overrides
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override '{pair}'. Expected KEY=VALUE.")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        inline_overrides = parse_inline_overrides(args.set)
        config = config.merge(inline_overrides)
    except (OSError, ValueError, TypeError) as err:
        print(f"Failed to load configuration: {err}")
        return 1

    command = args.command
    extra_args = args.forward or []

    if command == "menu":
        menu(config, extra_args)
        return 0

    task = TASKS[command]
    try:
        task(config, extra_args)
    except FileNotFoundError as err:
        print(f"❌ {err}")
        return 1
    except CommandError as err:
        print(f"❌ {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
