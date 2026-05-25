"""Entry point: parallel CoT synthesis with judge filtering.

Usage:
    python -m src.data_syn.run_synth \
        --input  datasets/raw_train/train.json \
        --output datasets/syn_train/train_cot.jsonl \
        --workers 3 \
        --limit -1
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .clients import ChatClient, load_endpoints
from .pipeline import SynthResult, synthesize_one
from .prompts import PromptBank


REPO_ROOT = Path(__file__).resolve().parents[2]


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_jsonl(path: Path, obj: dict, lock: threading.Lock) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--prompt-yaml",
        type=Path,
        default=REPO_ROOT / "configs" / "prompt" / "data_syn.yml",
    )
    p.add_argument(
        "--model-yaml",
        type=Path,
        default=REPO_ROOT / "configs" / "model" / "models.yml",
    )
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--limit", type=int, default=-1, help="max items to process (-1=all)")
    p.add_argument(
        "--log", type=Path, default=REPO_ROOT / "outputs" / "logs" / "data_syn.log"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log)
    log = logging.getLogger("data_syn")

    bank = PromptBank(args.prompt_yaml)
    endpoints = load_endpoints(args.model_yaml)

    gen_node = bank.get_node("generator")
    jud_node = bank.get_node("judge")
    generator = ChatClient(
        endpoints[gen_node["model_key"]],
        temperature=float(gen_node.get("temperature", 0.6)),
        max_tokens=int(gen_node.get("max_tokens", 1024)),
    )
    judge = ChatClient(
        endpoints[jud_node["model_key"]],
        temperature=float(jud_node.get("temperature", 0.0)),
        max_tokens=int(jud_node.get("max_tokens", 256)),
    )
    log.info("generator=%s judge=%s", generator.model, judge.model)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(args.output)
    todo = [it for it in data if str(it["id"]) not in done]
    log.info("total=%d done=%d todo=%d", len(data), len(done), len(todo))

    rej_path = args.output.with_suffix(".rejected.jsonl")
    pass_lock = threading.Lock()
    rej_lock = threading.Lock()

    counters = {"pass": 0, "rej": 0, "err": 0}
    counter_lock = threading.Lock()

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(synthesize_one, it, bank, generator, judge): it["id"]
            for it in todo
        }
        pbar = tqdm(total=len(futures), desc="synth", ncols=100)
        for fut in as_completed(futures):
            try:
                res: SynthResult = fut.result()
            except Exception as exc:  # noqa: BLE001
                with counter_lock:
                    counters["err"] += 1
                log.error("future failed for id=%s: %s", futures[fut], exc)
                pbar.update(1)
                continue

            record = dataclasses.asdict(res)
            if res.error:
                append_jsonl(rej_path, record, rej_lock)
                with counter_lock:
                    counters["err"] += 1
            elif res.passed == 1:
                append_jsonl(args.output, record, pass_lock)
                with counter_lock:
                    counters["pass"] += 1
            else:
                append_jsonl(rej_path, record, rej_lock)
                with counter_lock:
                    counters["rej"] += 1

            pbar.update(1)
            if pbar.n % 20 == 0:
                with counter_lock:
                    pbar.set_postfix(
                        pass_=counters["pass"], rej=counters["rej"], err=counters["err"]
                    )
        pbar.close()

    elapsed = time.time() - start
    log.info(
        "done in %.1fs | pass=%d rej=%d err=%d",
        elapsed,
        counters["pass"],
        counters["rej"],
        counters["err"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
