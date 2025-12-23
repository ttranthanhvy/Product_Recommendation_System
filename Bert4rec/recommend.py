import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
DATA_PATH = "clean_data.csv"
MODEL_PATH = "checkpoints/bert4rec_event_best.pth"
MAX_LEN = 50
TOP_K = 10
PAD = 0
MASK = 1

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["user_id", "timestamp"])

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

    def forward(self, items, events):
        B, T = items.size()
        pos = torch.arange(T, device=items.device).unsqueeze(0)

        x = (
            self.item_emb(items)
            + self.event_emb(events)
            + self.pos_emb(pos)
        )

        out = self.encoder(x)
        return self.fc(out)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

item2idx = checkpoint["item2idx"]
event2idx = checkpoint["event2idx"]

idx2item = {v: k for k, v in item2idx.items()}

num_items = len(item2idx) + 2
num_events = len(event2idx)

model = BERT4Rec(num_items, num_events, max_len=MAX_LEN)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def build_user_input(df, user_id):
    user_df = df[df["user_id"] == user_id].sort_values("timestamp")

    items = user_df["product_id"].map(item2idx).dropna().tolist()
    events = user_df["event_type"].map(event2idx).dropna().tolist()

    if len(items) == 0:
        return None, None, None

    items = items[-MAX_LEN:]
    events = events[-MAX_LEN:]


    items[-1] = MASK

    pad_len = MAX_LEN - len(items)

    item_tensor = torch.tensor(
        [PAD] * pad_len + items
    ).unsqueeze(0)

    event_tensor = torch.tensor(
        [0] * pad_len + events
    ).unsqueeze(0)

    seen_items = set(user_df["product_id"].tolist())

    return item_tensor, event_tensor, seen_items


@torch.no_grad()
def recommend_top_k(user_id, k=10):
    items, events, seen_items = build_user_input(df, user_id)

    if items is None:
        return []

    logits = model(items, events)     
    scores = logits[0, -1]              

    topk_idx = torch.topk(scores, k * 2).indices.tolist()

    recs = []
    for idx in topk_idx:
        pid = idx2item.get(idx)
        if pid and pid not in seen_items:
            recs.append(pid)
        if len(recs) == k:
            break

    return recs


if __name__ == "__main__":
    user_id = "#4317"
    print(f"\n🎯 Recommend for user: {user_id}\n")

    recs = recommend_top_k(user_id, TOP_K)

    for i, r in enumerate(recs, 1):
        print(f"{i}. {r}")

