"""Portable retrieval, short-answer scoring, and adapter-training entry points."""
import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def retrieve():
    from .memory import build_nodes, merge
    from .retrieval import select_memory, answer_messages, RetrievalConfig
    from .scoring import QwenScorer
    parser = argparse.ArgumentParser(description='Select source-preserving evidence for a task JSON')
    parser.add_argument('--task', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-Reranker-0.6B')
    parser.add_argument('--adapter', help='Trained native q/v LoRA checkpoint for LMMR')
    parser.add_argument('--answer-tokenizer', required=True, help='Reference answer-model tokenizer directory or HF ID')
    parser.add_argument('--mode', choices=['lmmr', 'qe', 'mmr', 'frozen_hierarchy', 'frozen_conditional'], default='lmmr')
    parser.add_argument('--budget', type=int, default=400)
    parser.add_argument('--prompt-limit', type=int, default=768)
    parser.add_argument('--device')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.mode == 'lmmr' and not args.adapter:
        parser.error('LMMR requires --adapter; use a frozen mode to run the base model')
    if args.mode != 'lmmr' and args.adapter:
        parser.error('Frozen modes must not receive an adapted checkpoint')
    from transformers import AutoTokenizer
    task = read_json(args.task)
    scorer = QwenScorer(args.model, device=args.device)
    nodes = build_nodes(task, scorer.tokenizer)
    coarse = [n for n in nodes if n['grain'] == 'coarse']
    relevance = dict(zip([n['id'] for n in coarse], scorer.score(task['question'], coarse)))
    # Independent anchor relevance always comes from the frozen base.
    if args.adapter:
        scorer.merge_adapter(args.adapter)
    reference = AutoTokenizer.from_pretrained(args.answer_tokenizer)
    count = lambda text: len(reference.encode(text, add_special_tokens=False))
    def fits(evidence):
        prompt = reference.apply_chat_template(answer_messages(task['question'], evidence),
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return count(prompt) <= args.prompt_limit
    conditional = args.mode in {'lmmr', 'frozen_conditional'}
    def score(anchor, candidates):
        return scorer.score(task['question'], candidates, anchor['text'], conditional)
    selected = select_memory(task, nodes, relevance, score, count,
        RetrievalConfig(budget=args.budget), args.mode, fits)
    evidence = merge(task, selected)
    write_json(args.output, {'task_id': task['id'], 'mode': args.mode, 'selected': selected,
        'evidence': evidence, 'evidence_tokens': count(evidence),
        'messages': answer_messages(task['question'], evidence)})


def evaluate():
    from .metrics import evaluate_rows
    parser = argparse.ArgumentParser(description='Score JSONL rows containing prediction and answer(s)')
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding='utf-8').splitlines() if line.strip()]
    result = evaluate_rows(rows)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


def train():
    from .scoring import QwenScorer
    from .training import conditional_loss
    import random
    import torch
    parser = argparse.ArgumentParser(description='Train q/v adapters on prepared source-equivalent memory groups')
    parser.add_argument('--groups', required=True, help='JSONL: question, anchor, positive_views, hard_by_grain, red_views')
    parser.add_argument('--model', default='Qwen/Qwen3-Reranker-0.6B')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--questions-per-step', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--device')
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.groups).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        parser.error('Empty training groups')
    scorer = QwenScorer(args.model, device=args.device, trainable=True)
    optimizer = torch.optim.AdamW(scorer.adapter_parameters(), lr=args.learning_rate, weight_decay=.01)
    order = list(range(len(rows)))
    random.Random(20261113).shuffle(order)
    scorer.model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for j in range(args.questions_per_step):
            row = rows[order[(step * args.questions_per_step + j) % len(order)]]
            records = []
            for view in row['positive_views']:
                grain = view['grain']
                negatives = row['hard_by_grain'].get(grain) or row['hard_by_grain']['coarse']
                negative = negatives[step % len(negatives)]
                redundant = row['red_views'].get(grain, row['anchor'])
                records.extend((row['question'], n['text'], row['anchor']['text'], True)
                               for n in (view, negative, redundant))
            values = scorer.forward(records).reshape(len(row['positive_views']), 3)
            loss = conditional_loss(values) / args.questions_per_step
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(scorer.adapter_parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 20 == 0:
            print(json.dumps({'step': step + 1, 'loss': total}), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scorer.save_adapter(args.output)
