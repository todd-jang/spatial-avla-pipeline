#!/usr/bin/env python3
"""Fetch real, public-domain siren recordings for p_siren validation.

Why this exists: every DoA number in this repo is `source: "synthetic"`. The whole
result rests on an untested substitution -- a Woodworth-ITD harmonic stack standing
in for a real emergency siren. This script fetches actual recorded sirens so that
substitution can be checked against real spectra.

Sources are deliberately keyless. freesound.org search needs an API key and
original-quality download needs OAuth2 (see docs/INTERIM_POC_REPORT.md, open item
6); archive.org delivers original WAV under CC0 with no authentication, so the
acquisition step never blocks on a human registering for credentials.

Every file is recorded in a manifest with its source URL and licence, so a number
derived from one of these stays traceable -- the same `source` labelling rule the
synthetic sweep already follows.

Usage:
    python3 scripts/fetch_real_sirens.py
    python3 scripts/fetch_real_sirens.py --out exports/real_sirens --force
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.audio.wav_convert import (to_pipeline_mono_16k, read_wav_float_mono,  # noqa: E402
                                        write_wav_int16_mono_16k,
                                        write_wav_int16_stereo_16k)

# Public-domain collections with original-quality siren audio and direct links.
# (identifier, why it is here)
COLLECTIONS = [
    ("GOLD_TAPE_44_Sirens", "USC optical sound effects library, CC0, original WAV"),
]

MAX_BYTES = 12 * 1024 * 1024
SIREN_NAME_HINTS = ("siren", "ambulance", "air raid", "emergency")
ACCEPT_SUFFIX = (".wav", ".flac", ".mp3")
UA = "spatial-avla-pipeline/1.0 (research; contact: local)"


def iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def metadata(identifier: str) -> dict:
    raw = get(f"https://archive.org/metadata/{urllib.parse.quote(identifier)}")
    return json.loads(raw)


def pick_files(files: list[dict]) -> list[dict]:
    """Prefer original WAV, keep only siren-named audio under the size cap."""
    chosen: dict[str, dict] = {}
    for f in files:
        name = f.get("name", "")
        if not name.lower().endswith(ACCEPT_SUFFIX):
            continue
        if not any(h in name.lower() for h in SIREN_NAME_HINTS):
            continue
        try:
            size = int(f.get("size", 0))
        except (TypeError, ValueError):
            continue
        if size > MAX_BYTES:
            continue
        if f.get("source") not in (None, "original"):
            continue
        # One entry per base name; a .wav outranks an .mp3 of the same recording.
        stem = Path(name).stem.lower()
        rank = ACCEPT_SUFFIX.index(Path(name).suffix.lower())
        prev = chosen.get(stem)
        if prev is None or rank < ACCEPT_SUFFIX.index(Path(prev["name"]).suffix.lower()):
            chosen[stem] = f
    return sorted(chosen.values(), key=lambda x: x["name"])


def licence_of(meta: dict, file_name: str) -> tuple[str, str]:
    """Return (licence identifier, rights text) if the item declares a permissive one.

    archive.org nests item metadata under a `metadata` key; licence fields are not
    guaranteed at the response root.
    """
    item = meta.get("metadata") or meta
    if isinstance(item.get("collection"), list) or isinstance(item.get("collection"), str):
        item = meta.get("metadata") or {}
    rights = str(item.get("rights") or "").strip()
    lice = str(item.get("licenseurl") or "").strip()
    if "creativecommons.org/publicdomain" in lice or "CC0" in rights.upper():
        return "CC0-1.0", rights or lice
    if lice or rights:
        return "unverified", rights or lice
    return "unknown", ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="exports/real_sirens")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args(argv)

    out = ROOT / args.out
    raw = out / "raw"
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    manifest = {"version": "1.0", "generated_at": iso(), "entries": []}
    seen_ids: set[str] = set()
    failures: list[str] = []

    for identifier, why in COLLECTIONS:
        try:
            meta = metadata(identifier)
        except (urllib.error.URLError, TimeoutError) as exc:
            failures.append(f"{identifier}: metadata unreachable ({exc})")
            continue
        files = meta.get("files", [])
        chosen = pick_files(files)
        if not chosen:
            failures.append(f"{identifier}: no siren-named audio under {MAX_BYTES} bytes")
            continue
        for f in chosen:
            name = f["name"]
            lic, rights = licence_of(meta, name)
            if lic not in ("CC0-1.0",):
                failures.append(f"{name}: licence is '{lic}', refusing to store")
                continue
            dest = raw / name
            if dest.exists() and not args.force:
                print(f"  keep  {name} ({dest.stat().st_size} bytes)")
                downloaded = False
            else:
                url = f"https://archive.org/download/{urllib.parse.quote(identifier)}/" \
                      f"{urllib.parse.quote(name)}"
                try:
                    blob = get(url, timeout=300)
                except (urllib.error.URLError, TimeoutError) as exc:
                    failures.append(f"{name}: download failed ({exc})")
                    continue
                if len(blob) != int(f.get("size", len(blob))):
                    failures.append(f"{name}: size mismatch, refusing to store")
                    continue
                dest.write_bytes(blob)
                print(f"  fetch {name} ({len(blob)} bytes)")
                downloaded = True

            # Every pipeline consumer insists on 16 kHz int16 (wav_io refuses
            # 48 kHz 24-bit). Convert once, at ingest, so the strict loaders
            # accept the recording everywhere downstream.
            conv = out / f"{Path(name).stem}_16k.wav"
            stereo = out / f"{Path(name).stem}_stereo_16k.wav"
            if downloaded or not conv.exists() or not stereo.exists():
                mono, rate = read_wav_float_mono(dest)
                piped = to_pipeline_mono_16k(mono, rate)
                write_wav_int16_mono_16k(conv, piped)
                write_wav_int16_stereo_16k(stereo, piped)
                print(f"  conv  {conv.name} + {stereo.name} ({rate} Hz -> 16 kHz)")

            seen_ids.add(name)
            manifest["entries"].append({
                "file": name,
                "bytes": dest.stat().st_size,
                "converted": conv.name,
                "converted_bytes": conv.stat().st_size,
                "stereo": {"file": stereo.name, "bytes": stereo.stat().st_size},
                "identifier": identifier,
                "source_url": f"https://archive.org/details/{identifier}",
                "license": lic,
                "license_text": rights,
                "collection_note": why,
                "downloaded_now": downloaded,
            })

    if not manifest["entries"]:
        print("NO AUDIO FETCHED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 2

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest['entries'])} file(s) -> {out / 'manifest.json'}")
    for f in failures:
        print(f"  skipped: {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
