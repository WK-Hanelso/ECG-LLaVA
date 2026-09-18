#!/usr/bin/env python3
"""Scan the public evidence package for confidentiality risks.

Patterns are assembled from fragments so this scanner can scan its own source
without reporting its pattern definitions as findings.
"""

from __future__ import annotations

import re
import struct
import sys
import zlib
from pathlib import Path


PORTFOLIO_DIR = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".json", ".svg", ".py"}


def joined(*parts: str) -> str:
    return "".join(parts)


def build_patterns() -> dict[str, re.Pattern[str]]:
    path_terms = [joined("/", "home", "/"), joined("/", "mnt", "/"), joined("file", "://")]
    credential_terms = [
        joined("api", r"[_-]?", "key"),
        joined("client", r"[_-]?", "secret"),
        "password",
        "passwd",
        "authorization",
        joined("access", r"[_-]?", "token"),
        joined("refresh", r"[_-]?", "token"),
    ]
    identifier_terms = [
        joined("patient", r"[_-]?", "id"),
        joined("ecg", r"[_-]?", "id"),
        joined("vehicle", r"[_-]?", "id"),
        "vin",
        joined("vehicle", r"[_-]?", "parameter"),
        joined("topic", r"[_-]?", "name"),
        "hostname",
    ]
    return {
        "absolute_path": re.compile("|".join(re.escape(term) for term in path_terms), re.I),
        "ipv4": re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])"),
        "credential_assignment": re.compile(
            rf"(?:{'|'.join(credential_terms)})\s*[:=]\s*[^\s]+", re.I
        ),
        "private_endpoint": re.compile(
            joined(r"(?:https?://[^\s\"<]*(?:", "internal", "|", "private", r")|\.", "internal", r"\b)"),
            re.I,
        ),
        "private_identifier_assignment": re.compile(
            rf"(?:{'|'.join(identifier_terms)})\s*[:=]\s*[^\s]+", re.I
        ),
        "email": re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I),
    }


def png_text_chunks(path: Path) -> list[str]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return []
    offset = 8
    chunks: list[str] = []
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"tEXt":
            chunks.append(payload.decode("latin-1", errors="replace"))
        elif kind == b"zTXt" and b"\x00" in payload:
            _, compressed = payload.split(b"\x00", 1)
            if compressed[:1] == b"\x00":
                chunks.append(zlib.decompress(compressed[1:]).decode("latin-1", errors="replace"))
        elif kind == b"iTXt":
            chunks.append(payload.decode("utf-8", errors="replace"))
        if kind == b"IEND":
            break
    return chunks


def main() -> int:
    patterns = build_patterns()
    findings: list[tuple[str, str, str]] = []
    files_scanned = 0
    png_chunks_scanned = 0

    for path in sorted(PORTFOLIO_DIR.rglob("*")):
        if not path.is_file():
            continue
        texts: list[str] = []
        if path.suffix.lower() in TEXT_SUFFIXES:
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
            files_scanned += 1
        elif path.suffix.lower() == ".png":
            texts.extend(png_text_chunks(path))
            png_chunks_scanned += len(texts)
            files_scanned += 1
        else:
            continue
        for text in texts:
            for category, pattern in patterns.items():
                match = pattern.search(text)
                if match:
                    findings.append((path.relative_to(PORTFOLIO_DIR).as_posix(), category, match.group(0)))

    if findings:
        print("SECURITY_SCAN=FAIL")
        for filename, category, match in findings:
            print(f"{filename}: {category}: {match!r}")
        return 1

    print("SECURITY_SCAN=PASS")
    print(f"files_scanned={files_scanned}")
    print(f"png_metadata_chunks_scanned={png_chunks_scanned}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
