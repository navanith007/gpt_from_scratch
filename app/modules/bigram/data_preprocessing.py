import os, dill, torch

from app.helpers import get_character_encoder, get_character_decoder


def get_bigram_encoder_decoder(charset):
    char_encoder = get_character_encoder(charset)
    char_decoder = get_character_decoder(char_encoder)

    def encode(s):
        return [char_encoder[c] for c in s]

    def decode(l):
        return ''.join([char_decoder[i] for i in l])

    return encode, decode


class CharTokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0

    def build_vocab(self, text):
        chars = set(text)
        self.vocab_size = len(chars)
        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for idx, char in enumerate(chars)}

    def text_to_tensor(self, text):
        return torch.tensor([self.char2idx[char] for char in text], dtype=torch.long)

    def tensor_to_text(self, tensor):
        return ''.join([self.idx2char[idx.item()] for idx in tensor])

    def get_char_set(self):
        return set(self.char2idx.keys())


def load_encoder_decoder_functions(file_path):
    """
    Load encoder and decoder functions from a file.

    Parameters:
    - file_path (str): The path to the file containing encoder and decoder functions.

    Returns:
    - encode (function): The encoder function.
    - decode (function): The decoder function.
    """
    if os.path.exists(os.path.abspath(file_path)):
        with open(os.path.abspath(file_path), 'rb') as file:
            encode, decode = dill.load(file)
            return encode, decode
    else:
        raise FileNotFoundError(f"Encoder and decoder functions file not found at {file_path}.")
