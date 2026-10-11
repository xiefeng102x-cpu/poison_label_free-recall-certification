from __future__ import annotations

from pathlib import Path
import sys


PAPER14 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER14 / "src"))

import run_pathmnist_terminal_staged_replay as staged


def test_staging_view_is_symlink_only_and_preserves_sources(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    output.mkdir()
    for name in staged.TOP_LEVEL_FILES:
        (output / name).write_bytes((name + "\n").encode("utf-8"))
    features = output / "features"
    features.mkdir()
    (features / "seed_103.npz").write_bytes(b"frozen-feature")

    before = {
        path.relative_to(output): staged.sha256_file(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    with staged.staging_view(output) as view:
        for name in staged.TOP_LEVEL_FILES:
            assert (view / name).is_symlink()
            assert (view / name).resolve() == (output / name).resolve()
        nested_features = view / staged.RECORDED_OUTPUT_PREFIX / "features"
        assert nested_features.is_symlink()
        assert nested_features.resolve() == features.resolve()
        assert (nested_features / "seed_103.npz").read_bytes() == b"frozen-feature"
    after = {
        path.relative_to(output): staged.sha256_file(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
