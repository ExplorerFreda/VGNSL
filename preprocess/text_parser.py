import argparse
import benepar
import spacy
import tqdm


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input-path', type=str, required=True, help='Input file path')
    arg_parser.add_argument('--output-path', type=str, required=True, help='Output file path')
    arg_parser.add_argument('--spacy-model', type=str, default='de_dep_news_trf', help='SpaCy model name')
    arg_parser.add_argument('--benepar-model', type=str, default='benepar_de2', help='Benepar model name')
    args = arg_parser.parse_args()

    nlp = spacy.load(args.spacy_model)
    nlp.add_pipe('benepar', config={'model': args.benepar_model})
    with open(args.input_path, 'r') as fin, open(args.output_path, 'w') as fout:
        for line in tqdm.tqdm(fin.readlines()):
            line = line.strip()
            doc = nlp(line)
            for sent in doc.sents:
                print(sent._.parse_string, file=fout)
