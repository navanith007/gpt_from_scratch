import os.path

import torch
from fastapi import APIRouter

from app.modules import JsonConfigUpdater
from app.modules.bigram import BigramModel, load_encoder_decoder_functions, GPT
from pydantic import BaseModel

bigram_gpt_router = APIRouter()

json_config = JsonConfigUpdater(os.path.abspath('configs/bigram_gpt.json'))
BIGRAM_MODEL_CONFIG = json_config.load_json()

model = GPT(BIGRAM_MODEL_CONFIG["VOCAB_SIZE"], BIGRAM_MODEL_CONFIG["EMB_DIM"], BIGRAM_MODEL_CONFIG.get("NUM_HEADS", 4),
            BIGRAM_MODEL_CONFIG.get("NUM_LAYERS", 1))
model_path = os.path.abspath("models/bigram_gpt_model.pth")

# Load the pre-trained model state dictionary
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
except FileNotFoundError:
    raise Exception(f"Model file not found at {model_path}. Make sure to provide the correct path.")

encode, decode = load_encoder_decoder_functions(file_path='models/bigram_gpt_encoder_decoder.pkl')

model.eval()  # Set the model to evaluation mode

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {total_params}")


# Define a Pydantic model for request input validation
class InputData(BaseModel):
    prompt: str
    max_new_tokens: int


# Endpoint to make predictions
@bigram_gpt_router.post("/bigram_gpt_chat_complete")
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
