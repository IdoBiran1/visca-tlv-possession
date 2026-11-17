"""Check that your data matches the team's, and fetch the public external datasets.

    python data.py where             # the folders this machine uses
    python data.py check --quick     # every file present, right size (seconds)
    python data.py check             # ... and the right sha256 (22 GB: seconds to a few minutes)
    python data.py fetch             # download missing or wrong external files, verified

The manifest (data_manifest.json) lists every file we work from: the competition data, which each
of us downloads from the organisers (it is gated and may not be redistributed), and the public
external datasets, which `fetch` downloads from their original sources. See DATA.md.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "data_manifest.json")


def challenge_repo():
    """Same lookup as preprocess.find_challenge_repo, without importing numpy."""
    for path in (os.environ.get("MTAH_REPO"), os.path.join(HERE, "..", "..", "more-than-a-hack-2027"),
                 os.path.expanduser("~/more-than-a-hack-2027")):
        if path and os.path.isfile(os.path.join(path, "games.py")):
            return os.path.abspath(path)
    return None


def roots():
    repo = challenge_repo()
    comp = os.environ.get("MTAH_DATA_DIR") or (os.path.join(repo, "data") if repo else None)
    ext = os.path.expanduser(os.environ.get("MTAH_EXT", "~/Github/external_data"))
    return repo, comp, ext


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def verify(root, files, quick=False):
    """[(path, problem)] for every manifest file that is missing, the wrong size or the wrong hash."""
    problems = []
    for rel, want in files.items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            problems.append((rel, "missing"))
        elif os.path.getsize(p) != want["bytes"]:
            problems.append((rel, "size {} != {}".format(os.path.getsize(p), want["bytes"])))
        elif not quick and sha256(p) != want["sha256"]:
            problems.append((rel, "sha256 differs"))
    return problems


def download(url, dest, want):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
                for block in iter(lambda: r.read(1 << 20), b""):
                    f.write(block)
            if os.path.getsize(part) == want["bytes"] and sha256(part) == want["sha256"]:
                os.replace(part, dest)
                return True
            print("    checksum mismatch, retrying")
        except OSError as e:
            print("    {} ({}), retrying".format(e, attempt + 1))
    if os.path.exists(part):
        os.remove(part)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["where", "check", "fetch"])
    ap.add_argument("--quick", action="store_true", help="check: sizes only, no hashing")
    ap.add_argument("--only", choices=["competition", "external"], help="check one group only")
    args = ap.parse_args()
    m = json.load(open(MANIFEST))
    repo, comp, ext = roots()

    if args.command == "where":
        print("challenge repo   ", repo or "NOT FOUND: clone it and set MTAH_REPO")
        print("competition data ", comp)
        print("external data    ", ext)
        return 0

    if args.command == "fetch":
        files = m["external"]["files"]
        todo = verify(ext, files, quick=False)
        print("{} of {} external files need fetching into {}".format(len(todo), len(files), ext))
        failed = 0
        for rel, why in todo:
            want = files[rel]
            print("  {} ({:.0f} MB, {})".format(rel, want["bytes"] / 1e6, why))
            failed += not download(want["url"], os.path.join(ext, rel), want)
        print("done" if not failed else "{} files failed: run fetch again".format(failed))
        return 1 if failed else 0

    bad = 0
    if args.only == "external":
        pass  # the challenge repo only matters for the competition data
    elif repo is None:
        print("challenge repo: NOT FOUND (clone {} and set MTAH_REPO)".format(m["challenge_repo"]["url"]))
        bad += 1
    else:
        try:
            head = open(os.path.join(repo, ".git", "HEAD")).read().strip()
            if head.startswith("ref:"):
                head = open(os.path.join(repo, ".git", head.split()[1])).read().strip()
        except OSError:
            head = "unknown"
        pinned = m["challenge_repo"]["commit"]
        print("challenge repo: {} at {}{}".format(repo, head[:7], "" if head == pinned else
              "  (team is on {}; see DATA.md before updating)".format(pinned[:7])))
    for group, root in (("competition", comp), ("external", ext)):
        if args.only and args.only != group:
            continue
        files = m[group]["files"]
        problems = verify(root, files, args.quick) if root else [(r, "missing") for r in files]
        print("{}: {} of {} files OK in {}".format(group, len(files) - len(problems), len(files), root))
        for rel, why in problems:
            print("   {:<12} {}".format(why, rel))
        bad += len(problems)
    if bad:
        print("\nNot matching the team's data. Competition files: re-download from the organisers' link. "
              "External files: python data.py fetch.")
    else:
        print("\nAll data matches the manifest{}.".format(" (sizes only; drop --quick to hash)" if args.quick else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
