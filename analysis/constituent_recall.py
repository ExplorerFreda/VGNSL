import benepar
import collections
import spacy
from nltk import Tree


class BareTree(Tree):
    @classmethod
    def fromstring(cls, st):
        augmented_st = st.replace('(', '( NT ')
        return super().fromstring(augmented_st)


class RecallMetric:
    def __init__(self):
        self.cnt = 0
        self.recalled = 0

    def update(self, pred):
        self.cnt += 1
        self.recalled += pred

    @property
    def recall(self):
        return self.recalled / self.cnt

    def __str__(self):
        return f'Total Cnt: {self.cnt}, Recall: {self.recall}'

    def __repr__(self):
        return str(self)


def spans(node, left=0):
    if isinstance(node, str):
        return
    if len(node.leaves()) > 1:
        yield (node.label(), left, left + len(node.leaves()))
    child_left = left
    for child in node:
        for x in spans(child, child_left):
            yield x
        n_child_leaves = 1 if isinstance(child, str) else len(child.leaves())
        child_left += n_child_leaves


def constituent_recall(gold_captions, pred_trees, gold_trees=None):
    if gold_trees is None:
        nlp = spacy.load('en_core_web_md')
        if spacy.__version__.startswith('2'):
            nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
        else:
            nlp.add_pipe("benepar", config={"model": "benepar_en3"})
        gold_trees = [list(nlp(cap.lower()).sents)[0]._.parse_string for cap in gold_captions]
    recall = collections.defaultdict(RecallMetric)
    for gold_tree, pred_tree in zip(gold_trees, pred_trees):
        pred_spans = {tuple(x[1:]) for x in spans(BareTree.fromstring(pred_tree))}
        for label, l, r in spans(Tree.fromstring(gold_tree)):
            recalled = (l, r) in pred_spans
            recall[label].update(recalled)
    return recall


if __name__ == '__main__':
    gold_captions = ['This is a cat']
    pred_trees = ['( This ( is ( a cat ) ) )']
    recall = constituent_recall(gold_captions, pred_trees)
    from IPython import embed; embed(using=False)
