import json

from drpair import runtime


def test_exact_fingerprint_does_not_need_authorization(tmp_path):
    fingerprint = {
        "config_sha256": "config",
        "source_sha256": "source",
        "manifest_sha256": "manifest",
        "fold_manifest_sha256": "folds",
    }
    assert runtime.fingerprints_match(fingerprint, fingerprint, tmp_path)


def test_analysis_upgrade_requires_exact_authorization(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "provenance").mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps({"git_commit": "old-commit"}), encoding="utf-8"
    )
    saved = {
        "config_sha256": "config",
        "source_sha256": "old-source",
        "manifest_sha256": "manifest",
        "fold_manifest_sha256": "folds",
        "model_class": "SingleOrdinal",
    }
    expected = {
        **saved,
        "source_sha256": "new-source",
    }
    authorization = {
        "schema": 1,
        "from_git_commit": "old-commit",
        "to_git_commit": "new-commit",
        "from_source_sha256": "old-source",
        "to_source_sha256": "new-source",
        "config_sha256": "config",
        "manifest_sha256": "manifest",
        "fold_manifest_sha256": "folds",
    }
    path = run / "provenance" / "analysis_source_upgrade.json"
    path.write_text(json.dumps(authorization), encoding="utf-8")
    monkeypatch.setattr(runtime, "git_commit", lambda: "new-commit")
    assert runtime.fingerprints_match(saved, expected, run)

    authorization["to_source_sha256"] = "another-source"
    path.write_text(json.dumps(authorization), encoding="utf-8")
    assert not runtime.fingerprints_match(saved, expected, run)


def test_analysis_upgrade_never_ignores_training_identity(tmp_path):
    saved = {"source_sha256": "old", "model_class": "SingleOrdinal"}
    expected = {"source_sha256": "new", "model_class": "PairedOrdinal"}
    assert not runtime.fingerprints_match(saved, expected, tmp_path)
