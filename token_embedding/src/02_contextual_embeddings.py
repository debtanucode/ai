"""
02 — Static vs Contextual Embeddings
=====================================
Covers:
  - Static embedding: same token → always the same vector
  - Contextual embedding: same token → different vector per sentence
  - Why contextual > static (the 'bank' / 'spy' problem)
  - The full pipeline: raw text → token IDs → static vectors → BERT → contextual vectors
  - Special tokens: [CLS] and [SEP], output shape [batch, tokens, dims]

We use the word "spy" which appears in multiple movie descriptions with
different surrounding words — demonstrating how BERT changes its vector
based on context.
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading BERT...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertModel.from_pretrained("bert-base-uncased")
model.eval()

embedding_matrix = model.embeddings.word_embeddings.weight


def get_token_position(tokens: list, target: str) -> int:
    """Return the index of the first occurrence of target in tokens list."""
    for i, t in enumerate(tokens):
        if t == target:
            return i
    return -1


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Two sentences where 'spy' carries different meanings ─────────────────────
sent_a = "he operates as a spy knowing he will die at the hands of terrorists"
sent_b = "intelligence spy thriller about recovering a stolen nuclear bomb"

TARGET = "spy"

print(f"\n── Target word: '{TARGET}' ──────────────────────────────────────────")
print(f"  Sentence A: {sent_a}")
print(f"  Sentence B: {sent_b}")


# ── Step 1: Static embedding (raw matrix lookup — no context) ─────────────────
spy_id         = tokenizer.convert_tokens_to_ids(TARGET)
static_vec     = embedding_matrix[spy_id].detach().numpy()

print(f"\n── Step 1 — Static Embedding (matrix row lookup) ────────────────────")
print(f"  Token ID for '{TARGET}': {spy_id}")
print(f"  Static vector (first 8 dims): {static_vec[:8].round(4)}")
print(f"  Same vector both sentences. No context. Static.")


# ── Step 2: Contextual embeddings through BERT ────────────────────────────────
def get_contextual_vector(sentence: str, target_token: str) -> np.ndarray:
    """Run sentence through BERT and return the contextual vector for target_token."""
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    with torch.no_grad():
        output = model(**inputs)  # last_hidden_state: [1, seq_len, 768]

    pos = get_token_position(tokens, target_token)
    if pos == -1:
        raise ValueError(f"'{target_token}' not found in tokens: {tokens}")

    return output.last_hidden_state[0, pos, :].numpy(), tokens, output.last_hidden_state.shape


vec_a, tokens_a, shape_a = get_contextual_vector(sent_a, TARGET)
vec_b, tokens_b, shape_b = get_contextual_vector(sent_b, TARGET)

print(f"\n── Step 2 — Contextual Embedding (BERT output) ──────────────────────")
print(f"\n  Sentence A tokens : {tokens_a}")
print(f"  Output shape      : {shape_a}  ← [batch=1, seq_len, dims=768]")
print(f"  '{TARGET}' contextual vector A (first 8 dims): {vec_a[:8].round(4)}")

print(f"\n  Sentence B tokens : {tokens_b}")
print(f"  Output shape      : {shape_b}")
print(f"  '{TARGET}' contextual vector B (first 8 dims): {vec_b[:8].round(4)}")


# ── Step 3: Compare static vs contextual similarity ──────────────────────────
sim_static      = cosine_similarity(static_vec, static_vec)          # always 1.0
sim_contextual  = cosine_similarity(vec_a, vec_b)

print(f"\n── Step 3 — Cosine Similarity Comparison ────────────────────────────")
print(f"  Static   '{TARGET}' vs '{TARGET}'  (same matrix row)   : {sim_static:.4f}  ← always 1.0")
print(f"  Contextual A vs B  (different sentence context) : {sim_contextual:.4f}  ← <1.0, context matters")


# ── Step 4: Special tokens [CLS] and [SEP] ────────────────────────────────────
demo = "Best movie ever !"
inputs = tokenizer(demo, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
with torch.no_grad():
    out = model(**inputs)

print(f"\n── Step 4 — Special Tokens: [CLS] and [SEP] ─────────────────────────")
print(f"  Input text   : '{demo}'")
print(f"  Tokens       : {tokens}")
print(f"  Output shape : {tuple(out.last_hidden_state.shape)}")
print(f"             ↑ [batch=1, tokens={len(tokens)} ([CLS]+words+[SEP]), dims=768]")
print(f"  [CLS] token is at index 0 — often used as the sentence-level vector.")


# ── Key takeaway ──────────────────────────────────────────────────────────────
print(f"\n── Key Takeaway ─────────────────────────────────────────────────────")
print(f"  Static  : '{TARGET}' always has the same vector. Context-blind.")
print(f"  BERT    : '{TARGET}' gets a different vector in every sentence.")
print(f"  The same token can mean espionage, danger, or intrigue —")
print(f"  BERT bakes the surrounding words into each token's output vector.")
