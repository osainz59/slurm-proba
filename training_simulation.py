from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from tqdm import tqdm


print("CUDA available:", torch.cuda.is_available())

model = AutoModelForMaskedLM.from_pretrained('ixa-ehu/ixambert-base-cased')
tokenizer = AutoTokenizer.from_pretrained('ixa-ehu/ixambert-base-cased')

print('Model loaded: ixa-ehu/ixambert-base-cased')

model.to("cuda" if torch.cuda.is_available() else "cpu")
model.train()


# Optimizer
# Split weights in two groups, one with weight decay and the other not.
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-4)

for epoch in range(1000):
    progress_bar = tqdm(range(10), desc=f"Epoch {epoch + 1} - Loss: -")
    for steps in progress_bar:
        batch = ["Niri sagarrak gustatzen zaizkit"*32]
        batch = tokenizer(batch, padding=True, return_tensors='pt')
        batch = {key: value.cuda() for key, value in batch.items()}
        batch['labels'] = batch['input_ids']
        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        progress_bar.set_description(f"Epoch {epoch + 1} - Loss: {loss.item():.2f}")




