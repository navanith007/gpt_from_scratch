import os.path

import torch
from fastapi import APIRouter

from app.modules import JsonConfigUpdater
from app.modules.bigram import BigramModel, load_encoder_decoder_functions
from pydantic import BaseModel

bigram_router = APIRouter()

json_config = JsonConfigUpdater('configs/bigram_config.json')
BIGRAM_MODEL_CONFIG = json_config.load_json()

model = BigramModel(BIGRAM_MODEL_CONFIG["VOCAB_SIZE"], BIGRAM_MODEL_CONFIG["EMB_DIM"])
model_path = os.path.abspath("models/bigram_basic_model.pth")

# Load the pre-trained model state dictionary
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
except FileNotFoundError:
    raise Exception(f"Model file not found at {model_path}. Make sure to provide the correct path.")

encode, decode = load_encoder_decoder_functions()

model.eval()  # Set the model to evaluation mode


# Define a Pydantic model for request input validation
class InputData(BaseModel):
    prompt: str
    max_new_tokens: int


# Endpoint to make predictions
@bigram_router.post("/bigram_chat_complete")
def predict(data: InputData):
    input_text = data.prompt
    max_new_tokens = data.max_new_tokens

    # Perform inference
    with torch.no_grad():
        test_input = torch.tensor(encode(input_text), dtype=torch.long)
        batch_tensor = test_input.unsqueeze(0)
        output = model.generate(batch_tensor, max_new_tokens=max_new_tokens)
        output = output[0].tolist()
        output_generated_text = decode(output)

    return {"chat_output": output_generated_text}
