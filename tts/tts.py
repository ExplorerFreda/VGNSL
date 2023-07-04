import argparse
import tqdm
import os
from TTS.api import TTS


banned_chars = ['“', '„']


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input-path', type=str, required=True, help='Input file path')
    arg_parser.add_argument('--output-path', type=str, required=True, help='Output folder path')
    arg_parser.add_argument('--model', type=str, default='tts_models/de/thorsten/tacotron2-DDC', help='Model name for TTS, default to a German model')
    arg_parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = arg_parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    tts = TTS(args.model, gpu=args.gpu)
    with open(args.input_path, 'r') as f:
        for i, text in enumerate(tqdm.tqdm(f.readlines())):
            text = text.strip()
            text = ''.join(filter(lambda x: x not in banned_chars, text))
            speech_file_path = os.path.join(args.output_path, f'{i:05d}.wav')
            if not os.path.exists(speech_file_path):
                tts.tts_to_file(text, file_path=speech_file_path)
