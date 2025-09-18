#!/usr/bin/env python3
"""ConvFinQA command line interface.

This script wraps the common Retriever and Generator workflows for ConvFinQA
into a single menu-driven CLI.  It exposes both interactive and scripted
sub-commands and supports lightweight configuration via JSON files.

Example usage:

    python convfinqa_cli.py menu
    python convfinqa_cli.py train-retriever --config my_overrides.json
"""

from __future__ import annotations

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
        # Skip glob validation to avoid shell-specific expansion.
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
# Interactive menu
# ---------------------------------------------------------------------------


def menu(config: CLIConfig, extra_args: List[str]) -> None:
    if extra_args:
        print("⚠️  Extra arguments ignored in menu mode.")

    entries = list(TASKS.keys())
    while True:
        print("\nConvFinQA CLI Menu")
        print("=" * 22)
        for idx, key in enumerate(entries, start=1):
            print(f"[{idx}] {key.replace('-', ' ').title()}")
        print("[0] Exit")

        choice = input("Select an option: ").strip()
        if choice == "0" or choice.lower() in {"q", "quit", "exit"}:
            print("Goodbye!")
            return

        if not choice.isdigit() or int(choice) not in range(1, len(entries) + 1):
            print("Invalid selection. Please try again.")
            continue

        task_key = entries[int(choice) - 1]
        try:
            TASKS[task_key](config, [])
        except FileNotFoundError as err:
            print(f"❌ {err}")
        except CommandError as err:
            print(f"❌ {err}")


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
