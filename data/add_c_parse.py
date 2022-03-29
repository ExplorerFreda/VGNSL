# add constituency parse tree to MSCOCO metadata with Benepar
import argparse
import benepar
import json
import spacy
from nltk import Tree
from tqdm import tqdm


def replace_leaves(tree, leaves):
    if isinstance(tree, str):
        return leaves[0]
    left = 0
    new_children = list()
    for child in tree:
        n_leaves = 1 if isinstance(child, str) else len(child.leaves())
        new_child = replace_leaves(child, leaves[left:left+n_leaves])
        new_children.append(new_child)
        left += n_leaves
    return Tree(tree.label(), new_children)


def remove_label(tree):
    if len(tree) == 1:
        if len(tree.leaves()) == 1:
            return tree.leaves()[0]
        return remove_label(tree[0])
    new_children = list()
    for child in tree:
        new_child = remove_label(child)
        new_children.append(new_child)
    return Tree('', new_children)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', type=str)
    parser.add_argument('--output-file', '-o', type=str)
    parser.add_argument('--no-label', dest='label', action='store_false', default=True)
    args = parser.parse_args()

    nlp = spacy.load('en_core_web_md')
    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    data = json.load(open(args.input_file))
    for item in tqdm(data['data']):
        for cap in item['captions']:
            caption = cap['text'].lower()
            doc = nlp(caption)
            sent = list(doc.sents)[0]
            tree = Tree.fromstring(sent._.parse_string)
            if not args.label:
                tree = remove_label(tree)
            tree = ' '.join(str(tree).replace('(', ' ( ').replace(')', ' ) ').split())
            cap['parse_tree'] = tree
    with open(args.output_file, 'w') as fout:
        json.dump(data, fout)
        fout.close()
