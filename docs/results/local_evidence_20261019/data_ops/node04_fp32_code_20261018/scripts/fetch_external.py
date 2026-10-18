"""Fetch the official repositories listed in third_party/external_repos.json into external/.

    python scripts/fetch_external.py [--only NAME ...]

Shallow, single-commit fetches; sparse checkouts where listed. Pinned refs are fetched exactly;
unpinned ones take the default branch and the resolved commit is written back to the manifest so
later fetches are reproducible. Existing checkouts at the recorded commit are left untouched.
"""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "third_party" / "external_repos.json"
EXTERNAL = ROOT / "external"


def git(*args, cwd):
    return subprocess.run(["git", "-c", "core.longpaths=true", "-c", "core.autocrlf=false", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def fetch_raw(repo, target):
    """Per-file download at one commit, for repositories whose other paths are invalid on Windows."""
    import urllib.request
    owner_repo = repo["url"].removeprefix("https://github.com/")
    commit = repo["ref"]
    if commit is None:
        with urllib.request.urlopen(f"https://api.github.com/repos/{owner_repo}/commits/HEAD") as r:
            commit = json.load(r)["sha"]
    for path in repo["sparse"]:
        out = target / path
        out.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/{owner_repo}/{commit}/{path}") as r:
            out.write_bytes(r.read())
    (target / "COMMIT").write_text(commit + "\n")
    return commit, "fetched"


def fetch(repo):
    target = EXTERNAL / repo["name"]
    if repo.get("raw_files"):
        if (target / "COMMIT").exists():
            return (target / "COMMIT").read_text().strip(), "kept"
        target.mkdir(parents=True, exist_ok=True)
        return fetch_raw(repo, target)
    if (target / ".git").exists():
        head = git("rev-parse", "HEAD", cwd=target)
        if repo["ref"] in (None, head):
            return head, "kept"
        raise SystemExit(f"{target} is at {head}, manifest pins {repo['ref']}; remove it to refetch")
    target.mkdir(parents=True)
    git("init", "-q", cwd=target)
    git("remote", "add", "origin", repo["url"], cwd=target)
    if repo["sparse"]:
        git("sparse-checkout", "set", "--no-cone", *repo["sparse"], cwd=target)
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", repo["ref"] or "HEAD", cwd=target)
    git("checkout", "-q", "FETCH_HEAD", cwd=target)
    return git("rev-parse", "HEAD", cwd=target), "fetched"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    for repo in manifest["repos"]:
        if args.only and repo["name"] not in args.only:
            continue
        commit, action = fetch(repo)
        if repo["ref"] is None:
            repo["ref"] = commit
        print(f"{action:8s} {repo['name']:14s} {commit}", flush=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
