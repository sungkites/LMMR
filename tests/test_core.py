import unittest
from lmmr.memory import intervals, contains, merge, build_nodes
from lmmr.metrics import score_answer, evaluate_rows
from lmmr.retrieval import select_memory, RetrievalConfig
from lmmr.training import positive_views, select_hard_negatives


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def node(pi, start, end, grain='coarse', text=''):
    return {'id': f'{pi}-{start}-{end}-{grain}', 'paragraph': pi, 'title': str(pi),
            'grain': grain, 'text': text, 'tokens': end-start,
            'spans': [{'paragraph': pi, 'sentence': 0, 'start': start, 'end': end}]}


class CoreTests(unittest.TestCase):
    def test_source_interval_merge(self):
        task = {'paragraphs': [{'title': 'A', 'sentences': ['abcdefghij']}]}
        first, second = node(0, 0, 7), node(0, 4, 10)
        self.assertEqual(intervals(first['spans'] + second['spans'])[(0, 0)], [(0, 10)])
        self.assertEqual(merge(task, [first, second]), '[P0:S0] A: abcdefghij')
        self.assertTrue(contains(first, node(0, 2, 4)))
        self.assertFalse(contains(first, node(1, 2, 4)))

    def test_original_text_and_grain_budgets(self):
        text = ' '.join('word'+str(i) for i in range(400))
        task = {'paragraphs': [{'title': 'T', 'sentences': [text, 'Another fact.']}]}
        nodes = build_nodes(task, WordTokenizer())
        self.assertTrue(any(n['grain'] == 'fine' for n in nodes))
        for n in nodes:
            self.assertLessEqual(n['tokens'], 192 if n['grain'] == 'coarse' else 96)
            for span in n['spans']:
                source = task['paragraphs'][0]['sentences'][span['sentence']].strip()
                self.assertIn(source[span['start']:span['end']], n['text'])

    def test_hierarchy_selects_fine_when_coarse_exceeds_budget(self):
        task = {'paragraphs': [{'title': 'A', 'sentences': ['a']},
                               {'title': 'B', 'sentences': ['b ' * 20]}]}
        anchor = node(0, 0, 1, text='a')
        coarse = node(1, 0, 39, text='b ' * 20)
        fine = node(1, 0, 1, 'fine', 'b')
        values = {anchor['id']: 10., coarse['id']: 1.}
        selected = select_memory(task, [anchor, coarse, fine], values,
            lambda a, ns: [5.] * len(ns), lambda s: len(s.split()),
            RetrievalConfig(budget=8))
        self.assertEqual([n['id'] for n in selected], [anchor['id'], fine['id']])
        self.assertLessEqual(len(merge(task, selected).split()), 8)

    def test_source_equivalent_positives_and_hard_negatives(self):
        task = {'question': 'Which missing fact?', 'paragraphs': [
            {'title': '0', 'sentences': ['anchor']},
            {'title': '1', 'sentences': ['fact']}],
            'supporting_facts': {'title': ['0', '1'], 'sent_id': [0, 0]}}
        anchor, coarse, fine = node(0, 0, 6), node(1, 0, 4), node(1, 0, 4, 'fine')
        self.assertEqual(positive_views(task, [anchor, coarse, fine], anchor, coarse), [coarse, fine])
        distractors = [node(i, 0, 3, grain, 'missing fact') for i in range(2, 8)
                       for grain in ['coarse', 'fine']]
        seen = []
        def score(a, candidates):
            seen.append(len(candidates))
            return list(range(len(candidates)))
        negatives = select_hard_negatives(task, anchor, [coarse, fine] + distractors, score)
        self.assertEqual(seen, [4, 4])
        self.assertEqual([len(v) for v in negatives.values()], [2, 2])
        self.assertTrue(all(n['title'] not in {'0', '1'} for ns in negatives.values() for n in ns))

    def test_official_boolean_and_alias_scoring(self):
        self.assertEqual(score_answer('Yes, both did.', 'yes')['f1'], 0.)
        self.assertEqual(score_answer('The St. Louis Cardinals', 'St Louis Cardinals')['em'], 1.)
        self.assertEqual(evaluate_rows([{'prediction': 'NYC', 'answers': ['New York City', 'NYC']}])['em'], 100.)


if __name__ == '__main__':
    unittest.main()
