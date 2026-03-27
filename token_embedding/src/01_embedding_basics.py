"""
01 — Embedding Basics
=====================
Covers:
  - Tokenization: raw text → token IDs
  - The embedding matrix: a table with one row per vocabulary token
  - Static embedding lookup: token ID → fixed vector (no context)
  - Embedding dimensions: what 768 numbers actually are

Model: bert-base-uncased  (vocab size: 30,522 | embedding dim: 768)
"""

import torch
from transformers import BertTokenizer, BertModel

# ── 1. Load tokenizer and model ──────────────────────────────────────────────
print("Loading BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# ── 2. Grab the embedding matrix directly from the model ─────────────────────
# model.embeddings.word_embeddings is an nn.Embedding layer
# Its .weight attribute is the full matrix: shape [vocab_size, hidden_dim]
embedding_matrix = model.embeddings.word_embeddings.weight  # [30522, 768]

print("\n── Embedding Matrix ─────────────────────────────────────────────────")
print(f"  Vocabulary size : {embedding_matrix.shape[0]:,} tokens")
print(f"  Embedding dims  : {embedding_matrix.shape[1]}")
print(f"  Total params    : {embedding_matrix.numel():,}")
print(f"  Memory (float32): ~{embedding_matrix.numel() * 4 / 1e6:.1f} MB")


# ── 3. Tokenise a movie title ─────────────────────────────────────────────────
title = "Sardar Udham"
token_ids = tokenizer.encode(title, add_special_tokens=True)

print(f"\n── Tokenisation: '{title}' ──────────────────────────────────────────")
print(f"  Token IDs : {token_ids}")
print(f"  Tokens    : {tokenizer.convert_ids_to_tokens(token_ids)}")
print()
print("  ID      Token          First 6 dims of static vector")
print("  ─" * 14)
for tid in token_ids:
    token = tokenizer.convert_ids_to_tokens([tid])[0]
    # Static lookup: read the row directly from the matrix
    vec = embedding_matrix[tid].detach().numpy()
    print(f"  {tid:<7} {token:<15} {vec[:6].round(3)}")


# ── 4. Dimensions explained ───────────────────────────────────────────────────
print("\n── What are 768 dimensions? ─────────────────────────────────────────")
print("  Think of it as a point in 768-dimensional space.")
print("  1-D  → position on a line      e.g. temperature     [37.5]")
print("  2-D  → position on a map       e.g. GPS             [13.08, 80.27]")
print("  768-D→ position in meaning space e.g. a BERT token  [0.23, -0.11, ...]")
print()

sample_token   = "udham"
sample_token_id = tokenizer.convert_tokens_to_ids(sample_token)
vec = embedding_matrix[sample_token_id].detach().numpy()
print(f"  Token '{sample_token}' (ID {sample_token_id}) — first 10 dims:")
print(f"  {vec[:10].round(4)}")
print(f"  Shape: {vec.shape}  ← one row of the 30,522 × 768 matrix")


# ── 5. Key takeaway ───────────────────────────────────────────────────────────
print("\n── Key Rule ─────────────────────────────────────────────────────────")
print("  These are STATIC embeddings: 'udham' always maps to the same vector,")
print("  no matter which sentence it appears in.  Context does not affect them.")
print("  The next script (02) shows how contextual embeddings fix this.")
