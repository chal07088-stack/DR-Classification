"""Fetch and verify the official baseline repositories at locked commits."""

from __future__ import annotations

import subprocess
from pathlib import Path

SOURCES = {
    "Huang-2023": ("https://github.com/YijinHuang/pytorch-classification.git", "e6cd6b49a27612166f7e60f68bf511ac5030140b"),
    "DRGCNN": ("https://github.com/Fat-Hai/DRGCNN.git", "bf80220a5786bf52678519f381b742827447402f"),
    "CLIP-DR": ("https://github.com/Qinkaiyu/CLIP-DR.git", "ca7d88c44f13d0e77bea4b0c2381b51deb75d3c1"),
    "AOR-DR": ("https://github.com/Qinkaiyu/AOR-DR.git", "4b67c6892e0035829ec3719a8a15e3371654a5a4"),
}


def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def main():
    root = Path(__file__).resolve().parents[1] / "third_party"
    root.mkdir(parents=True, exist_ok=True)
    for name, (url, commit) in SOURCES.items():
        target = root / name
        if not target.exists():
            run("git", "clone", "--filter=blob:none", "--no-checkout", url, str(target))
        run("git", "fetch", "origin", commit, "--depth=1", cwd=target)
        run("git", "checkout", "--detach", commit, cwd=target)
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=target, text=True
        ).strip()
        if actual != commit:
            raise RuntimeError(f"{name}: expected {commit}, got {actual}")
        print(f"{name}: {actual}")


if __name__ == "__main__":
    main()
