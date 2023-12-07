import os

import torch, dill
from torch.utils.tensorboard import SummaryWriter

from app.helpers import (read_text_file,
                         get_characters_set,
                         get_train_val_data,
                         get_data_batch)
from app.modules import JsonConfigUpdater
from app.modules.bigram import GPT, get_bigram_encoder_decoder, load_encoder_decoder_functions, CharTokenizer

json_config = JsonConfigUpdater('configs/bigram_gpt.json')
BIGRAM_GPT_MODEL_CONFIG = json_config.load_json()

# For tensor board
writer = SummaryWriter()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Example usage:
tokenizer = CharTokenizer()

text_data_path = "datasets/tinyshakespeare.txt"
text_data = read_text_file(text_data_path)
tokenizer.build_vocab(text_data)
vocab_size = tokenizer.vocab_size

json_config.add_new_key_value("VOCAB_SIZE", value=vocab_size)
json_config.write_to_file()

char_set = tokenizer.get_char_set()

encoder_decoder_functions = get_bigram_encoder_decoder(char_set)

if not os.path.exists(os.path.abspath('models/bigram_gpt_encoder_decoder.pkl')):
    with open(os.path.abspath('models/bigram_gpt_encoder_decoder.pkl'), 'wb') as file:
        dill.dump(encoder_decoder_functions, file)
        print("Encoder and decoder functions saved.")
else:
    print("Encoder and decoder functions file already exists.")

encode, decode = load_encoder_decoder_functions(file_path='models/bigram_gpt_encoder_decoder.pkl')

encoded_input_text_data = torch.tensor(encode(text_data), dtype=torch.long)

train_data, val_data = get_train_val_data(encoded_input_text_data, split_ratio=0.8)
eval_iters = BIGRAM_GPT_MODEL_CONFIG.get("EVAL_ITERS", 10)
model = GPT(vocab_size, BIGRAM_GPT_MODEL_CONFIG.get("EMB_DIM", 64), BIGRAM_GPT_MODEL_CONFIG.get("NUM_HEADS", 4),
            BIGRAM_GPT_MODEL_CONFIG.get("NUM_LAYERS", 1))


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split_type in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split_type == 'train':
                data = train_data
            else:
                data = val_data
            X, Y = get_data_batch(data, block_size=BIGRAM_GPT_MODEL_CONFIG["BLOCK_SIZE"],
                                  batch_size=BIGRAM_GPT_MODEL_CONFIG["BATCH_SIZE"], device=device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split_type] = losses.mean()
    model.train()
    return out


# Training the model
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=BIGRAM_GPT_MODEL_CONFIG["LEARNING_RATE"])

for epoch in range(BIGRAM_GPT_MODEL_CONFIG.get("EPOCHS", 100)):
    total_loss = 0
    for iter in range(BIGRAM_GPT_MODEL_CONFIG.get("MAX_ITER", 128)):
        if iter % eval_iters == 0:
            losses = estimate_loss()
            print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
        inputs, targets = get_data_batch(train_data, block_size=BIGRAM_GPT_MODEL_CONFIG["BLOCK_SIZE"],
                                         batch_size=BIGRAM_GPT_MODEL_CONFIG["BATCH_SIZE"], device=device)

        # evaluate the loss
        _, loss = model.forward(inputs, targets)
        total_loss += loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    epoch_loss = total_loss / BIGRAM_GPT_MODEL_CONFIG.get("MAX_ITER", 128)
    # # Log the loss to TensorBoard
    writer.add_scalar('Training Loss', epoch_loss, epoch)




# Save the model
torch.save(model.state_dict(), 'models/bigram_gpt_model.pth')

# Close the writer at the end of the script
writer.close()
