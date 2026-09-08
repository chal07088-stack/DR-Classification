"""Register an existing immutable EyePACS tree in a local DVC object store.

Uses same-volume hardlinks: no image bytes are copied and the original paths
remain usable. The store doubles as the initial cache; other checkouts can pull
from it with independent caches. It is not a physical backup of the images.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(repo, *args):
    subprocess.run([sys.executable, "-m", "dvc", *args], cwd=repo, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source, store = args.source.resolve(strict=True), args.store.resolve()
    target = repo / "data" / "eyepacs"
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and destination must be separate directory trees")
    if repo == store or repo in store.parents:
        raise ValueError("Local DVC store must be outside the Git checkout")
    images = sorted(source.glob("*/*/*.jpeg"))
    if len(images) != 88702:
        raise ValueError(f"Expected 88702 images; found {len(images)}")
    for i, path in enumerate(images, 1):
        relative = path.relative_to(source)
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.link(path, dest)
        elif not os.path.samefile(path, dest):
            raise FileExistsError(f"Refusing to replace a different file: {dest}")
        if i % 10000 == 0:
            print(f"Linked {i}/{len(images)}", flush=True)
    if not (repo / ".dvc").exists():
        run(repo, "init")
    run(repo, "config", "core.analytics", "false")
    run(repo, "config", "cache.type", "hardlink,copy")
    run(repo, "config", "--local", "cache.dir", str(store))
    run(repo, "remote", "add", "--local", "--force", "--default", "local", str(store))
    run(repo, "add", "data/eyepacs")
    run(repo, "push", "data/eyepacs.dvc", "--remote", "local")
    print("Registered. Original paths and DVC store are hardlinks, not independent backups.")


if __name__ == "__main__":
    main()
