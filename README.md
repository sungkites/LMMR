# LMMR

**Learned Multigranular Memory Retrieval for Small Language Model Agents**

LMMR trains a conditional reranker on source-equivalent paragraph/fragment views, relevant distractors, and redundant memories. Retrieval selects original-text evidence and its granularity under a token budget. The answer model stays frozen.

## Results

Full official validation sets, a 400-token evidence budget, and a 2B answer model:

| Dataset | Questions | Query expansion F1 | LMMR F1 | F1 gain | Evidence reduction |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 7,405 | 58.42 | **61.47** | +3.06 | 23.1% |
| 2WikiMultihopQA | 12,576 | 41.12 | **43.26** | +2.14 | 26.6% |
| MuSiQue | 2,417 | 26.02 | **30.58** | +4.55 | 26.9% |

Gains use unrounded aggregates. Eight main small answer models are compared on HotpotQA; the additional Nemotron result is separate. All 18 manuscript data tables are in `results/tables/`, including ablations, question types, budget analysis, and published reference results. Published scores retain their original evaluation settings.

## Install and retrieve

Python 3.10+:

```bash
pip install -e ".[model]"
lmmr-retrieve --task data/example.json --mode qe \
  --model Qwen/Qwen3-Reranker-0.6B \
  --answer-tokenizer /path/to/reference-answer-tokenizer --output evidence.json
```

For learned retrieval, use `--mode lmmr --adapter /path/to/adapter.pt`. The reference answer tokenizer measures the evidence and complete-prompt budgets. `evidence.json` includes the selected spans and messages for a local answer model. `qe`, `mmr`, `frozen_hierarchy`, and `frozen_conditional` run the corresponding frozen selectors.

This release provides portable core code and aggregate research data. Trained checkpoints and model weights are not bundled. The paper used an 8-bit-reconstructed reranker initialization; the default Hugging Face model is a convenient starting point for training a new adapter.

## Learning and evaluation

`src/lmmr/training.py` builds source-equivalent positives, mines grain-matched hard negatives, and implements the conditional ranking, redundancy, binary, and consistency losses. Prepared training groups are JSONL objects with `question`, `anchor`, `positive_views`, `hard_by_grain`, and `red_views`; node objects follow `memory.py`.

```bash
lmmr-train --groups /path/to/training-groups.jsonl --output adapter.pt
lmmr-evaluate --predictions /path/to/predictions.jsonl --output scores.json
python -m unittest discover -s tests -v
```

Evaluation rows contain `prediction` and `answer`, or `answers` for alias-aware scoring. Output scores are percentages. Dataset links and validation IDs are in `data/`; raw benchmark data remain available from their original repositories. `configs/default.json` lists the manuscript parameters. `references.bib` contains the 55 manuscript references.

## Citation and contact

Use `CITATION.cff` for author and title metadata. Correspondence: Pengfeng Song, Zhejiang Wanli University, `sungkites@163.com`.

Original implementation: MIT license. Benchmark text and pretrained models retain their original licenses.
