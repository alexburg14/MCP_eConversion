"""Cache build orchestrator.

The assistant runs entirely off local caches under ``data/``. Those caches have
a dependency order (embeddings need abstracts; the graph needs the PI list) that
was previously documented only in the README's prose. This expresses it as a
dependency graph and runs the build scripts in the right order, skipping any
target whose output is already newer than its inputs.

Usage:
    python build.py --list            # show every target and whether it's stale
    python build.py all               # build everything out-of-date, in order
    python build.py embeddings graph  # build specific targets (+ their deps)
    python build.py all --force       # rebuild everything unconditionally

Source inputs a target needs (e.g. the publications CSV) are the template's
onboarding contract: if one is missing, the build stops with an explicit message
naming the file to place under data/, rather than failing deep inside a script.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_DATA = _ROOT / "data"
_SCRIPTS = _ROOT / "src" / "scripts"


@dataclass(frozen=True)
class Target:
    script: str                              # build script under src/scripts/
    output: str                              # produced file under data/
    inputs: tuple[str, ...] = ()             # required source files under data/
    deps: tuple[str, ...] = ()               # other targets whose output is an input
    note: str = ""                           # shown in --list (e.g. "network, slow")


# The dependency graph. Order within the dict is a valid build order.
TARGETS: dict[str, Target] = {
    "abstracts": Target(
        script="build_abstracts_cache.py",
        output="abstracts_cache.json",
        inputs=("data_publication_dois.csv",),
        note="network (OpenAlex)",
    ),
    "pis": Target(
        script="build_pis_cache.py",
        output="pis_cache.json",
        note="network (scrapes e-conversion.de); no local input, rebuild with --force",
    ),
    "embeddings": Target(
        script="build_embeddings_cache.py",
        output="embeddings_cache.npz",
        deps=("abstracts",),
    ),
    "fulltext": Target(
        script="build_fulltext_cache.py",
        output="fulltext_cache.json",
        deps=("abstracts",),
        note="network, slow (~30 min); resumable, skips already-cached DOIs",
    ),
    "graph": Target(
        script="build_graph_cache.py",
        output="collaboration_graph.json",
        deps=("pis",),
    ),
    "proposal": Target(
        script="extract_proposal_summary.py",
        output="proposal_summary.md",
        inputs=("EXC_2089_e-conversion_A_Proposal_R.pdf",),
    ),
}


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _prereq_paths(name: str) -> list[Path]:
    """All files whose change should invalidate this target: source inputs + dep outputs."""
    t = TARGETS[name]
    paths = [_DATA / i for i in t.inputs]
    paths += [_DATA / TARGETS[d].output for d in t.deps]
    return paths


def is_stale(name: str) -> bool:
    """True if the target needs (re)building: output missing, or older than any prereq."""
    out = _DATA / TARGETS[name].output
    if not out.exists():
        return True
    return _mtime(out) < max((_mtime(p) for p in _prereq_paths(name)), default=0.0)


def _check_inputs(name: str) -> None:
    """Stop with an actionable message if a required source input is missing."""
    missing = [i for i in TARGETS[name].inputs if not (_DATA / i).exists()]
    if missing:
        for i in missing:
            print(f"  ! required input missing: place '{i}' under {_DATA}", file=sys.stderr)
        sys.exit(2)


def _resolve_order(names: list[str]) -> list[str]:
    """Expand the requested targets to include their dependencies, in build order."""
    ordered: list[str] = []

    def visit(n: str) -> None:
        for d in TARGETS[n].deps:
            visit(d)
        if n not in ordered:
            ordered.append(n)

    for n in names:
        visit(n)
    return ordered


def build(names: list[str], force: bool) -> None:
    for name in _resolve_order(names):
        if not force and not is_stale(name):
            print(f"= {name}: up to date ({TARGETS[name].output})")
            continue
        _check_inputs(name)
        print(f"+ {name}: building -> {TARGETS[name].output}")
        result = subprocess.run([sys.executable, str(_SCRIPTS / TARGETS[name].script)])
        if result.returncode != 0:
            print(f"  ! {name} failed (exit {result.returncode})", file=sys.stderr)
            sys.exit(result.returncode)


def list_targets() -> None:
    for name, t in TARGETS.items():
        status = "STALE" if is_stale(name) else "ok"
        deps = f" <- {', '.join(t.deps)}" if t.deps else ""
        note = f"  ({t.note})" if t.note else ""
        print(f"  [{status:>5}] {name}: {t.output}{deps}{note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the assistant's data caches in dependency order.")
    parser.add_argument("targets", nargs="*", help="target names, or 'all' (default: all)")
    parser.add_argument("--list", action="store_true", help="list targets and staleness, build nothing")
    parser.add_argument("--force", action="store_true", help="rebuild even if up to date")
    args = parser.parse_args()

    if args.list:
        list_targets()
        return

    requested = args.targets or ["all"]
    if requested == ["all"]:
        requested = list(TARGETS)

    unknown = [n for n in requested if n not in TARGETS]
    if unknown:
        parser.error(f"unknown target(s): {', '.join(unknown)}. Known: {', '.join(TARGETS)}")

    build(requested, force=args.force)


if __name__ == "__main__":
    main()
