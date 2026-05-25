import argparse
import hashlib
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_manifest(manifest: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{manifest}:{line_number}: expected '<sha256> <path>'")
        expected_hash, relative_path = parts
        if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
            raise ValueError(f"{manifest}:{line_number}: invalid sha256 digest")
        entries.append((expected_hash, Path(relative_path)))
    return entries


def verify_hashes(manifest: Path, root: Path) -> int:
    entries = parse_manifest(manifest)
    failed = False

    for expected_hash, relative_path in entries:
        target = root / relative_path
        if not target.is_file():
            print(f"MISSING  {relative_path}")
            failed = True
            continue

        actual_hash = sha256_file(target)
        if actual_hash != expected_hash:
            print(f"MISMATCH {relative_path}")
            print(f"  expected: {expected_hash}")
            print(f"  actual:   {actual_hash}")
            failed = True
            continue

        print(f"OK       {relative_path}")

    if failed:
        return 1

    print(f"Verified {len(entries)} file(s).")
    return 0


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify migrated files against a SHA-256 manifest.")
    parser.add_argument("--manifest", type=Path, default=repo_root / "scripts" / "hash_manifest.sha256")
    parser.add_argument("--root", type=Path, default=repo_root)
    args = parser.parse_args()

    try:
        return verify_hashes(args.manifest, args.root)
    except (OSError, ValueError) as error:
        print(f"ERROR    {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
