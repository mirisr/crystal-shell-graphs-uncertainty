#!/usr/bin/env python3
"""Fail when the Git candidate set crosses the documented public boundary."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".dcm", ".nii", ".nrrd", ".mha", ".mat", ".npz", ".parquet",
    ".h5", ".hdf5", ".pkl", ".pickle", ".joblib", ".xlsx", ".docx",
    ".pdf", ".mp4", ".gif", ".wav", ".mp3", ".pt", ".pth", ".ckpt",
}
FORBIDDEN_ROOTS = {"archive", "manuscript", "provenance", "runs"}
ALLOWED_BINARY_ARTIFACTS = {
    Path("splits/ispy2_folds.parquet"),
    *(Path(f"models/checkpoints/fold{fold}/best.pt") for fold in range(5)),
}
MAX_FILE_BYTES = 10 * 1024 * 1024
SECRET_PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "OpenAI key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "personal Mac path": re.compile(r"/Users/irisseaman/"),
    "personal cluster path": re.compile(r"/home/irisseaman[^/ ]*/"),
}


def candidates() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [Path(item.decode()) for item in result.stdout.split(b"\0") if item]


def main() -> int:
    errors: list[str] = []
    files = candidates()
    for rel in files:
        path = ROOT / rel
        if not path.is_file():
            continue
        if rel.parts and rel.parts[0] in FORBIDDEN_ROOTS:
            errors.append(f"forbidden release root: {rel}")
        if rel.suffix.lower() in FORBIDDEN_SUFFIXES and rel not in ALLOWED_BINARY_ARTIFACTS:
            errors.append(f"forbidden generated or patient-level suffix: {rel}")
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 10 MiB release limit: {rel}")
        if (
            path.stat().st_size <= 2 * 1024 * 1024
            and rel != Path("scripts/audit_public_release.py")
            and rel not in ALLOWED_BINARY_ARTIFACTS
        ):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{label} found in {rel}")
    if errors:
        print("public release audit failed", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"- {error}", file=sys.stderr)
        return 1
    total = sum((ROOT / rel).stat().st_size for rel in files if (ROOT / rel).is_file())
    print(f"public release audit passed: {len(files)} files, {total / 1024 / 1024:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
