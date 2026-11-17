"""Conditional source expansion and budgeted pair selection."""
from dataclasses import dataclass
from .memory import contains, merge, overlap, SYSTEM


@dataclass(frozen=True)
class RetrievalConfig:
    budget: int = 400
    max_anchors: int = 2
    source_expansion: int = 3
    anchor_gap: float = 2.0
    alpha: float = 1.2
    beta: float = 1.0
    mmr_weight: float = 0.7


def answer_messages(question, evidence):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Evidence:\n" + evidence + "\n\nQuestion: " + question + "\nAnswer:"},
    ]


def select_memory(task, nodes, relevance_scores, score_candidates, count_tokens,
                  config=None, mode="lmmr", prompt_fits=None):
    """Select original-text memory nodes.

    relevance_scores maps coarse node IDs to frozen relevance scores.
    score_candidates(anchor, candidates) returns one conditional/QE score per node.
    count_tokens uses the reference answer-model tokenizer.
    prompt_fits, when supplied, checks the complete reference prompt.
    """
    cfg = config or RetrievalConfig()
    coarse = sorted((n for n in nodes if n["grain"] == "coarse"),
                    key=lambda n: (-relevance_scores[n["id"]], n["id"]))
    if not coarse:
        raise ValueError("No coarse memory nodes")

    def fits(selected):
        text = merge(task, selected)
        return count_tokens(text) <= cfg.budget and (prompt_fits is None or prompt_fits(text))

    first = coarse[0]
    if mode in {"qe", "mmr"}:
        candidates = [n for n in coarse if n["paragraph"] != first["paragraph"]]
        values = list(score_candidates(first, candidates)) if candidates else []
        if len(values) != len(candidates):
            raise ValueError("Candidate and score counts differ")
        if mode == "mmr" and values:
            lo, hi = min(values), max(values)
            values = [cfg.mmr_weight * (v - lo) / (hi - lo + 1e-9)
                      - (1 - cfg.mmr_weight) * overlap(first["text"], n["text"])
                      for v, n in zip(values, candidates)]
        ranked = sorted(zip(values, candidates), key=lambda x: (-x[0], x[1]["id"]))
        for _, candidate in ranked:
            if fits([first, candidate]):
                return [first, candidate]
        if not fits([first]):
            raise ValueError("The leading anchor exceeds the reference budget")
        return [first]

    if mode not in {"lmmr", "frozen_hierarchy", "frozen_conditional", "coarse_matched"}:
        raise ValueError("Unknown retrieval mode: " + mode)
    anchors = coarse[:cfg.max_anchors]
    if len(anchors) > 1 and relevance_scores[anchors[0]["id"]] - relevance_scores[anchors[1]["id"]] > cfg.anchor_gap:
        anchors = anchors[:1]
    options = []
    for anchor in anchors:
        legal = [n for n in coarse if not contains(anchor, n)]
        values = list(score_candidates(anchor, legal)) if legal else []
        if len(values) != len(legal):
            raise ValueError("Candidate and score counts differ")
        scores = {n["id"]: float(v) for n, v in zip(legal, values)}
        sources = []
        for node in sorted(legal, key=lambda n: (-scores[n["id"]], n["id"])):
            if node["paragraph"] not in sources:
                sources.append(node["paragraph"])
            if len(sources) == cfg.source_expansion:
                break
        fine = [] if mode == "coarse_matched" else [
            n for n in nodes if n["grain"] == "fine"
            and n["paragraph"] in sources and not contains(anchor, n)]
        fine_scores = list(score_candidates(anchor, fine)) if fine else []
        if len(fine_scores) != len(fine):
            raise ValueError("Fine candidate and score counts differ")
        scores.update({n["id"]: float(v) for n, v in zip(fine, fine_scores)})
        for node in legal + fine:
            if fits([anchor, node]):
                value = relevance_scores[anchor["id"]] + cfg.alpha * scores[node["id"]] - cfg.beta * overlap(anchor["text"], node["text"])
                options.append((value, anchor, node))
    if options:
        _, anchor, candidate = max(options, key=lambda x: (x[0], x[1]["id"], x[2]["id"]))
        return [anchor, candidate]
    if not fits([first]):
        raise ValueError("The leading anchor exceeds the reference budget")
    return [first]
