"""System-prompt composition for the chat, independent of any UI framework.

The base prompt is parametrised by the cluster config and the live cache
counts so the numbers the model is told stay in sync with what is loaded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from config import Config

_REPO_ROOT = Path(__file__).resolve().parent.parent
PROPOSAL_SUMMARY_PATH = _REPO_ROOT / "data" / "cache" / "proposal_summary.md"


def _cluster_identity_line(cfg: Config) -> str:
    """One-line cluster identity (funder, ID, institutions) if configured, else empty."""
    parts = []
    if cfg.cluster.cluster_id:
        parts.append(f"Cluster ID {cfg.cluster.cluster_id}")
    if cfg.cluster.funding_body:
        parts.append(f"funded by {cfg.cluster.funding_body}")
    if cfg.cluster.host_institutions:
        parts.append("hosted at " + " and ".join(cfg.cluster.host_institutions))
    if not parts:
        return ""
    return "\n\n" + "; ".join(parts) + "."


def base_system_prompt(cfg: Config, n_papers: int, n_pis: int) -> str:
    return f"""\
You are a research assistant for {cfg.cluster.description}.\
{_cluster_identity_line(cfg)}

You have access to a local database of {n_papers} cluster publications (with \
abstracts and some full texts) and profiles of {n_pis} PIs. Use the tools to \
retrieve relevant information before answering. Always cite papers by title and DOI. If \
information is missing from the database, say so clearly — do not invent facts.

Two paper-search tools complement each other:
- search_papers (BM25, lexical): exact terms, acronyms, formulas, author names.
- semantic_search_papers (embeddings): conceptual queries where vocabulary may
  differ from abstracts. Run both when you're unsure which will hit — the
  union of results gives wider recall before you synthesize.

Use get_similar_papers(doi) for "what else is like this paper?" — it compares a
specific paper's own embedding to every other paper's, rather than taking a text query.

Use list_papers (exact metadata filter: author, year, journal) for exhaustive \
listings — "every paper by X", "papers in Nature", "what the cluster published in \
2022" — where the top-5 relevance results of the search tools are not enough.

Four collaboration-graph tools answer network questions that search cannot:
- get_collaborators(pi_query): who publishes with a given PI?
- joint_papers(pi_a, pi_b): which papers did two specific PIs co-author?
- collaboration_centrality(): which PIs bridge otherwise-separate groups?
- collaboration_communities(): which clusters of PIs work closely together?

When a question needs more than the abstract — specific methods, results, experimental \
details, or exact numbers — call get_paper_fulltext(doi) on the most relevant paper(s) \
from your search results before answering. Full text is cached for ~99% of the corpus; \
if it comes back missing, say so and answer from the abstract.

search_nomad queries the public NOMAD materials repository — data EXTERNAL to the \
cluster, not e-conversion papers. Use it when asked whether computed or measured data \
exists for a material, or what a PI has deposited. Report total_matches first; the \
entries it returns are a sample of a much larger set. NOMAD entries carry no DOI link \
back to cluster publications, so never present them as "the data behind" a paper.

For corpus-wide questions ("main open challenges", "trends over time", \
"complementary groups"), issue several complementary queries before answering: \
one tool call returns at most 5 papers, but a synthesis question needs evidence \
from many. Iterate with different phrasings, then summarize.

Answer in the same language as the question (German or English).\
"""


def build_system_prompt(cfg: Config, n_papers: int, n_pis: int,
                        summary_path: Path = PROPOSAL_SUMMARY_PATH) -> str:
    """Base prompt plus the DFG proposal summary (Section 2) when it has been extracted."""
    base = base_system_prompt(cfg, n_papers, n_pis)
    if not summary_path.exists():
        return base
    summary = summary_path.read_text(encoding="utf-8")
    return (
        base
        + "\n\nThe following is Section 2 of the e-conversion 2.0 DFG proposal "
        + "(\"Summary of the Proposal\"), describing the cluster's scope, motivation, "
        + "and research approach. Use it as background context.\n\n"
        + "<proposal_summary>\n"
        + summary
        + "\n</proposal_summary>"
    )


def remote_system_note(remote: Any) -> str:
    """One or two lines telling the model which live data sources are attached.

    Deliberately minimal: only appended when the user has connected a remote
    source, so the base system prompt stays untouched for everyone else.
    """
    if remote is None:
        return ""
    parts = []
    if getattr(remote, "elab_token", None):
        parts.append(
            "eLabFTW (ELN): tools prefixed elab_ read the lab notebook of the "
            "connected account - use them for questions about experiments, "
            "items, entries or inventory."
        )
    if getattr(remote, "dt_token", None):
        parts.append(
            "DataTagger: tools prefixed dt_ search the research data repository "
            "- use them for questions about datasets, versions or deposited "
            "research data."
        )
    if not parts:
        return ""
    return (
        "\n\nLive data sources are attached in this session; use their tools "
        "when the question concerns them:\n- " + "\n- ".join(parts)
    )
