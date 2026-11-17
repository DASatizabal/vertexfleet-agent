"""Run receipt tests: every run records seed, config hash, git commit and versions."""
import json

from run_meta import write_run_meta


def test_receipt_has_all_items(tmp_path):
    path = write_run_meta("unit_test", out_dir=tmp_path, extra={"rmse": 3.53})
    meta = json.loads(path.read_text(encoding="utf-8"))
    for key in ["seed", "config_sha256", "git_commit", "python_version", "packages"]:
        assert key in meta
    assert set(meta["packages"]) == {"numpy", "pandas", "scikit-learn", "pydantic"}
    assert len(meta["config_sha256"]) == 64
    assert meta["extra"]["rmse"] == 3.53


def test_config_hash_is_stable(tmp_path):
    a = json.loads(write_run_meta("a", out_dir=tmp_path).read_text(encoding="utf-8"))
    b = json.loads(write_run_meta("b", out_dir=tmp_path).read_text(encoding="utf-8"))
    assert a["config_sha256"] == b["config_sha256"]