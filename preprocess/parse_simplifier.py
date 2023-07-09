import argparse
import tqdm
import nltk


def get_simplified_tree(tree: nltk.Tree) -> str:
    ''' Simplify a tree by removing all non-terminal labels and punctuation marks.

    Args:
        tree: An NLTK tree.
    
    Returns:
        A string representation of the simplified tree.
    '''
    if len(tree) == 1 and isinstance(tree[0], str):
        return tree[0]
    # calculate tree length (number of chlidren)
    tree_length = 0
    for child in tree:
        if isinstance(child, str):
            tree_length += 1
        else:
            assert isinstance(child, nltk.Tree)
            tree_length += not child.label().startswith('$')  # German-specific hack for punctuation marks
    if tree_length == 0:
        return '(TOP place_holder_for_empty_utterance)'
    return '( {} )'.format(' '.join([get_simplified_tree(child) for child in tree if not child.label().startswith("$")]))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input-path', type=str, required=True, help='Input file path')
    arg_parser.add_argument('--output-path', type=str, required=True, help='Output file path')
    args = arg_parser.parse_args()

    with open(args.input_path, 'r') as fin, open(args.output_path, 'w') as fout:
        for line in tqdm.tqdm(fin.readlines()):
            line = f'(TOP { (line.strip())})'
            tree = nltk.Tree.fromstring(line)
            simplified_tree = get_simplified_tree(tree)
            print(simplified_tree, file=fout)
