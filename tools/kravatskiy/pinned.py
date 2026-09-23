"""Detached checkouts of A. Kravatskiy's repository at PINNED commits.

The submodule external/kravatskiy follows his repository and its head moves; a verification of
"his claim as of commit X" must keep reading exactly commit X. This module creates, on demand, a
git worktree of the submodule's object store at the requested commit, under external/.pins/
(gitignored), and returns its path. Nothing is downloaded if the commit is already in the
submodule; otherwise the submodule is fetched once.

    python tools/kravatskiy/pinned.py 52fa09d          # prints the path of the pinned checkout

    from pinned import pinned_checkout
    pkg = os.path.join(pinned_checkout("52fa09d"), "verifications", "improved", "dim25-lens-heads")

Pinned commits used by this repository:
    52fa09d16e20394f06c1d19b7a1bdc967c865d9f   2026-09-18   K(25) >= 197569, K(26) >= 199632, K(27) >= 201010
    c349d565362f39f8492e55bda7129cd0787a1d6e   2026-09-20   two layers: K(26) >= 199806, K(27) >= 201509
    f0809165cb3c1cbcb817e7bbb051df27fcb15614   2026-09-22   K(25) >= 197579 (1016 lens heads) and its ceiling
    bfc28543fe89b10f5017bc58bfe4a98e1357b269   2026-09-23   K(27) >= 201557 (second layer re-solved to 303)

The environment variable KISS_ALEXEY, if set, names an existing clone and is returned unchanged
(the caller is then responsible for what it has checked out).
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUB = os.path.join(ROOT, "external", "kravatskiy")
PINS = os.path.join(ROOT, "external", ".pins")

PINNED = {
    "52fa09d": "52fa09d16e20394f06c1d19b7a1bdc967c865d9f",
    "c349d56": "c349d565362f39f8492e55bda7129cd0787a1d6e",
    "f080916": "f0809165cb3c1cbcb817e7bbb051df27fcb15614",
    "bfc2854": "bfc28543fe89b10f5017bc58bfe4a98e1357b269",
}


def _git(*args, cwd=SUB):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def pinned_checkout(commit, override_env="KISS_ALEXEY"):
    if override_env and os.environ.get(override_env):
        return os.environ[override_env]
    full = PINNED.get(commit, commit)
    if not os.path.exists(os.path.join(SUB, ".git")):
        r = subprocess.run(["git", "submodule", "update", "--init", "external/kravatskiy"], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("cannot initialise the submodule external/kravatskiy:\n" + r.stderr)
    if _git("cat-file", "-e", full + "^{commit}").returncode:
        _git("fetch", "origin")
        if _git("cat-file", "-e", full + "^{commit}").returncode:
            raise RuntimeError(f"commit {full} is not in external/kravatskiy, even after a fetch")
    dst = os.path.join(PINS, "kravatskiy-" + full[:7])
    if os.path.isdir(dst):
        head = _git("rev-parse", "HEAD", cwd=dst).stdout.strip()
        dirty = _git("status", "--porcelain", cwd=dst).stdout.strip()
        if not head.startswith(full) or dirty:        # `full` may be a short hash
            raise RuntimeError(f"{dst} is at {head[:7]}{' and modified' if dirty else ''}, expected {full[:7]}; remove it and rerun")
        return dst
    os.makedirs(PINS, exist_ok=True)
    _git("worktree", "prune")
    r = _git("worktree", "add", "--detach", dst, full)
    if r.returncode:
        raise RuntimeError("git worktree add failed:\n" + r.stderr)
    return dst


if __name__ == "__main__":
    print(pinned_checkout(sys.argv[1], override_env=None))
