"""Source-equivalent positive construction and the conditional learning loss."""
from .memory import contains, coverage, words
from collections import Counter
import math


def positive_views(task, nodes, anchor, coarse_positive):
    facts = task["supporting_facts"]
    gold = set(zip(facts["title"], facts["sent_id"]))
    remaining = gold - coverage(task, anchor, gold)
    required = coverage(task, coarse_positive, remaining)
    if not required:
        raise ValueError("The positive must add annotated support")
    fine = [n for n in nodes if n["grain"] == "fine" and contains(coarse_positive, n)
            and coverage(task, n, remaining) >= required]
    return [coarse_positive] + ([min(fine, key=lambda n: (n["tokens"], n["id"]))] if fine else [])


def select_hard_negatives(task, anchor, nodes, score_candidates, seeded=()):
    """QE mining over seeded/lexical pre-pools, four candidates per grain."""
    supporting_titles = set(task["supporting_facts"]["title"])
    query = Counter(words(task['question'] + ' ' + anchor['text']))
    def lexical(node):
        candidate = Counter(words(node['text']))
        return sum(min(v, candidate.get(k, 0)) for k, v in query.items()) / math.sqrt(max(1, sum(candidate.values())))
    result = {}
    for grain in ("coarse", "fine"):
        eligible = [n for n in nodes if n["grain"] == grain and n["title"] not in supporting_titles]
        initial = [n for n in seeded if n['grain'] == grain and n['title'] not in supporting_titles]
        candidates, seen = [], set()
        for node in initial + sorted(eligible, key=lambda n: (-lexical(n), n['id'])):
            if node['id'] not in seen:
                candidates.append(node)
                seen.add(node['id'])
            if len(candidates) == 4:
                break
        values = score_candidates(anchor, candidates) if candidates else []
        if len(values) != len(candidates):
            raise ValueError('Candidate and score counts differ')
        result[grain] = [n for _, n in sorted(zip(values, candidates), key=lambda x: (-x[0], x[1]["id"]))[:2]]
    return result


def conditional_loss(scores, margin=0.5, redundancy_weight=0.5, binary_weight=0.25,
                     consistency_weight=0.1):
    """scores has shape [positive views, 3]: positive, hard negative, redundant."""
    import torch
    import torch.nn.functional as F
    positive, negative, redundant = scores[:, 0], scores[:, 1], scores[:, 2]
    rank = F.softplus(negative - positive + margin).mean()
    rank = rank + redundancy_weight * F.softplus(redundant - positive + margin).mean()
    binary = F.binary_cross_entropy_with_logits(
        torch.cat([positive, negative]),
        torch.cat([torch.ones_like(positive), torch.zeros_like(negative)]))
    consistency = F.smooth_l1_loss(positive[0].sigmoid(), positive[1].sigmoid()) if len(positive) > 1 else positive.new_zeros(())
    return rank + binary_weight * binary + consistency_weight * consistency
