import json
import os
from google.cloud import storage
import dill
import torch


def read_text_file(file_path):
    """
    Read the content of a text file and return it as a string.

    Parameters:
    - file_path (str): The path to the text file.

    Returns:
    - str: The content of the text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_characters_set(corpus: str) -> list:
    """
    Reads the entire corpus and gets the list of unique characters
    :param corpus:
    :return: char_list
    """
    char_list = list(sorted(set(corpus)))
    return char_list


def get_character_encoder(char_list: list) -> dict:
    """
    Reads the list of unique characters and encode each character to an integer
    :param char_list:
    :return:
    """
    assert len(char_list) == len(list(set(char_list))), "List contains non-unique characters."
    encoder = {
        char: ix for ix, char in enumerate(char_list)
    }
    return encoder


def get_character_decoder(char_encoder: dict) -> dict:
    """
    Reads character encoder and returns the decoder of integers
    :param char_encoder:
    :return:
    """
    decoder = {value: key for key, value in char_encoder.items()}
    return decoder


def get_train_val_data(data, split_ratio=0.8):
    n = int(split_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


def get_data_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def download_blob(bucket_name, source_blob_name, destination_file_name):
    if not os.path.exists(destination_file_name):
        """Downloads a blob from the bucket."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        blob.download_to_filename(destination_file_name)

        print(f'Blob {source_blob_name} downloaded to {destination_file_name}.')
