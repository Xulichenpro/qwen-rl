import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify_hashes.py"


def run_verify(manifest: Path, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            "--manifest",
            str(manifest),
            "--root",
            str(root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_verify_hashes_accepts_matching_files() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        target = root / "data.json"
        target.write_text('{"ok": true}\n', encoding="utf-8")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest = root / "manifest.sha256"
        manifest.write_text(f"{digest}  data.json\n", encoding="utf-8")

        result = run_verify(manifest, root)

        assert result.returncode == 0, result.stderr + result.stdout


def test_verify_hashes_rejects_mismatching_files() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        target = root / "data.json"
        target.write_text('{"ok": false}\n', encoding="utf-8")
        manifest = root / "manifest.sha256"
        manifest.write_text(f"{'0' * 64}  data.json\n", encoding="utf-8")

        result = run_verify(manifest, root)

        assert result.returncode != 0
        assert "MISMATCH" in result.stdout


if __name__ == "__main__":
    test_verify_hashes_accepts_matching_files()
    test_verify_hashes_rejects_mismatching_files()
