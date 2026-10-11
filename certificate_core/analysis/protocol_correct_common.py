"""Shared deterministic primitives for the protocol-correct replay."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def keyed_rng(key: bytes, domain: str, *parts: Any) -> np.random.Generator:
    message = "|".join([domain, *(str(value) for value in parts)]).encode("utf-8")
    digest = hmac.new(key, message, hashlib.sha256).digest()
    entropy = np.frombuffer(digest[:16], dtype=np.uint32).astype(np.uint64).tolist()
    return np.random.default_rng(np.random.SeedSequence(entropy))


def opaque_token(key: bytes, protocol: str, *parts: Any) -> str:
    message = "|".join([protocol, *(str(value) for value in parts)]).encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def ids_hash(values: np.ndarray) -> str:
    ordered = np.sort(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def load_key(path: Path) -> bytes:
    key = path.read_bytes()
    if len(key) != 32:
        raise RuntimeError("replay key must contain exactly 32 raw bytes")
    return key


def victim_seed(victim_id: str) -> int:
    suffix = str(victim_id).rsplit("seed", 1)
    if len(suffix) != 2 or not suffix[1].isdigit():
        raise RuntimeError(f"victim id lacks terminal seed: {victim_id}")
    return int(suffix[1])
