import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "bert4rec_event_best.pth")
LAST_MODEL_PATH = os.path.join(SAVE_DIR, "bert4rec_event_last.pth")

best_loss = float("inf")

df = pd.read_csv("clean_data.csv")


PAD = 0
MASK = 1

df_items = df.drop_duplicates("product_id")
item2idx = {pid: i+2 for i, pid in enumerate(df_items["product_id"])}

df_events = df.drop_duplicates("event_type")
event2idx = {ev: i for i, ev in enumerate(df_events["event_type"])}

num_items = len(item2idx) + 2
num_events = len(event2idx)

EVENT_WEIGHT = {
    "view": 0.5,
    "add_to_cart": 1.0,
    "purchase": 2.0
}

class BERT4RecDataset(Dataset):
    def __init__(self, df, max_len=50, mask_prob=0.15):
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.seqs = []

        for _, g in df.groupby("user_id"):
            items = g["product_id"].map(item2idx).dropna().tolist()
            events = g["event_type"].map(event2idx).dropna().tolist()
            weights = g["event_type"].map(EVENT_WEIGHT).dropna().tolist()

            if len(items) >= 2:
                self.seqs.append((
                    items[-max_len:],
                    events[-max_len:],
                    weights[-max_len:]
                ))
    def __getitem__(self, idx):
        items, events, weights = self.seqs[idx]
        input_items, input_events, input_weights, labels = [], [], [], []

        for it, ev, w in zip(items, events, weights):
            if np.random.rand() < self.mask_prob:
                input_items.append(MASK)
                input_events.append(ev)
                input_weights.append(w)
                labels.append(it)
            else:
                input_items.append(it)
                input_events.append(ev)
                input_weights.append(w)
                labels.append(-100)

        pad = self.max_len - len(input_items)
        return (
            torch.tensor([PAD]*pad + input_items),
            torch.tensor([0]*pad + input_events),
            torch.tensor([0.0]*pad + input_weights),
            torch.tensor([-100]*pad + labels)
        )

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        items, events, weights = self.seqs[idx]
        input_items, input_events, input_weights, labels = [], [], [], []

        for it, ev, w in zip(items, events, weights):
            if np.random.rand() < self.mask_prob:
                input_items.append(MASK)
                input_events.append(ev)
                input_weights.append(w)
                labels.append(it)
            else:
                input_items.append(it)
                input_events.append(ev)
                input_weights.append(w)
                labels.append(-100)

        pad = self.max_len - len(input_items)
        return (
            torch.tensor([PAD]*pad + input_items),
            torch.tensor([0]*pad + input_events),
            torch.tensor([0.0]*pad + input_weights),
            torch.tensor([-100]*pad + labels)
    )

class BERT4Rec(nn.Module):
    def __init__(self, num_items, num_events, hidden=128, max_len=50):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, hidden, padding_idx=PAD)
        self.event_emb = nn.Embedding(num_events, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)

        encoder = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder, 2)
        self.fc = nn.Linear(hidden, num_items)

    def forward(self, items, events, weights):
        B, T = items.size()
        pos = torch.arange(T, device=items.device).unsqueeze(0)

        x = (
            self.item_emb(items)
            + self.event_emb(events) * weights.unsqueeze(-1)
            + self.pos_emb(pos)
        )

        out = self.encoder(x)
        return self.fc(out)

dataset = BERT4RecDataset(df)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
num_epochs = 5

model = BERT4Rec(num_items, num_events)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0

    for items, events, weights, labels in loader:
        optimizer.zero_grad()

        logits = model(items, events, weights)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": best_loss,
            "item2idx": item2idx,
            "event2idx": event2idx
        }, BEST_MODEL_PATH)
        print("Saved BEST model")