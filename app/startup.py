import os

import torch

from app.modules import JsonConfigUpdater
from app.modules.bigram import GPT, load_encoder_decoder_functions
from config import GCP_BUCKET_NAME, GCP_SOURCE_GPT_MODEL_PATH, LOCAL_DESTINATION_GPT_MODEL_PATH, \
    GCP_SOURCE_GPT_ENCODER_DECODER_PATH, GCP_SOURCE_GPT_CONFIG_PATH, LOCAL_DESTINATION_GPT_ENCODER_DECODER_PATH, \
    LOCAL_DESTINATION_GPT_CONFIG_PATH
from app.helpers.utils import download_blob


def on_startup(app):
    # Create local directories if they don't exist
    local_directory = LOCAL_DESTINATION_GPT_MODEL_PATH.split('/')[0]
    print(local_directory)
    model_directory = LOCAL_DESTINATION_GPT_MODEL_PATH.split('/')[1]
    print(model_directory)
    os.makedirs(local_directory, exist_ok=True)
    os.makedirs(local_directory + "/" + model_directory, exist_ok=True)

    download_blob(GCP_BUCKET_NAME, GCP_SOURCE_GPT_MODEL_PATH, LOCAL_DESTINATION_GPT_MODEL_PATH)
    download_blob(GCP_BUCKET_NAME, GCP_SOURCE_GPT_ENCODER_DECODER_PATH,
                        LOCAL_DESTINATION_GPT_ENCODER_DECODER_PATH)
    download_blob(GCP_BUCKET_NAME, GCP_SOURCE_GPT_CONFIG_PATH, LOCAL_DESTINATION_GPT_CONFIG_PATH)

    # Load the pre-trained model
    json_config = JsonConfigUpdater(os.path.abspath(LOCAL_DESTINATION_GPT_CONFIG_PATH))
    BIGRAM_MODEL_CONFIG = json_config.load_json()
    app.state.gpt_bigram_model = GPT(BIGRAM_MODEL_CONFIG["VOCAB_SIZE"], BIGRAM_MODEL_CONFIG["EMB_DIM"],
                                     BIGRAM_MODEL_CONFIG.get("NUM_HEADS", 4),
                                     BIGRAM_MODEL_CONFIG.get("NUM_LAYERS", 1))
    model_path = os.path.abspath(LOCAL_DESTINATION_GPT_MODEL_PATH)

    app.state.gpt_bigram_encoder, app.state.gpt_bigram_decoder = load_encoder_decoder_functions(
        file_path=LOCAL_DESTINATION_GPT_ENCODER_DECODER_PATH)

    # Load the pre-trained model state dictionary
    try:
        app.state.gpt_bigram_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        raise Exception(f"Model file not found at {model_path}. Make sure to provide the correct path.")
