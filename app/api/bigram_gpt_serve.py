import torch
from fastapi import APIRouter, Depends

from app.dependencies import get_gpt_model, get_gpt_encoder, get_gpt_decoder
from pydantic import BaseModel

bigram_gpt_router = APIRouter()


# Define a Pydantic model for request input validation
class InputData(BaseModel):
    prompt: str
    max_new_tokens: int


# Endpoint to make predictions
@bigram_gpt_router.post("/bigram_gpt_chat_complete")
def predict(data: InputData, model=Depends(get_gpt_model), encode=Depends(get_gpt_encoder),
            decode=Depends(get_gpt_decoder)):
    input_text = data.prompt
    max_new_tokens = data.max_new_tokens
    model.eval()  # Set the model to evaluation mode
    # Perform inference
    with torch.no_grad():
        test_input = torch.tensor(encode(input_text), dtype=torch.long)
        batch_tensor = test_input.unsqueeze(0)
        output = model.generate(batch_tensor, max_new_tokens=max_new_tokens)
        output = output[0].tolist()
        output_generated_text = decode(output)

    return {"chat_output": output_generated_text}
