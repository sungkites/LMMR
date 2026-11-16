"""Normalized short-answer metrics, including HotpotQA boolean handling."""
import collections
import re
import string


def normalize_answer(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def score_answer(prediction, answer):
    prediction, answer = normalize_answer(prediction), normalize_answer(answer)
    em = float(prediction == answer)
    if prediction in {'yes', 'no', 'noanswer'} or answer in {'yes', 'no', 'noanswer'}:
        if not em:
            return {'f1': 0.0, 'em': 0.0}
    pt, at = prediction.split(), answer.split()
    common = sum((collections.Counter(pt) & collections.Counter(at)).values())
    f1 = (2.0 * common / (len(pt) + len(at))) if common else 0.0
    return {'f1': f1, 'em': em}


def evaluate_rows(rows):
    if not rows:
        raise ValueError('No predictions')
    scores = []
    for row in rows:
        answers = row.get('answers', [row.get('answer')])
        if not answers or any(not isinstance(a, str) for a in answers):
            raise ValueError('Each row needs answer or a nonempty answers list')
        values = [score_answer(row['prediction'], a) for a in answers]
        scores.append({key: max(s[key] for s in values) for key in ('f1', 'em')})
    return {'questions': len(rows), **{key: 100 * sum(s[key] for s in scores) / len(scores)
                                     for key in ('f1', 'em')}}
