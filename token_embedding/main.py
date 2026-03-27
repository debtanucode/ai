"""
Token Embeddings — Main Entry Point
=====================================
Runs all four features in sequence:

  Feature 1 — Embedding Basics          (embedding matrix, token IDs, dims)
  Feature 2 — Static vs Contextual      (same word, different context)
  Feature 3 — Semantic Movie Search     (interactive query loop)
  Feature 4 — Visualise Embedding Space (PCA cluster plot)

Usage:
  python main.py
"""

import os
import csv
import sys
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

matplotlib.rcParams["font.family"] = "DejaVu Sans"

CSV_PATH  = os.path.join(os.path.dirname(__file__), "indian_movies.csv")
PLOT_PATH = os.path.join(os.path.dirname(__file__), "embedding_clusters.png")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def banner(title: str):
    line = "═" * 62
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")


def pause():
    input("\n  Press Enter to continue to the next feature...\n")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_movies(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (done once, shared across all features)
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    print("  Loading models (this takes a moment on first run)...")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model     = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()

    mini_lm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("  bert-base-uncased     ✓")
    print("  all-MiniLM-L6-v2      ✓")
    return bert_tokenizer, bert_model, mini_lm


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1 — Embedding Basics
# ─────────────────────────────────────────────────────────────────────────────

def feature_1(tokenizer: BertTokenizer, model: BertModel):
    banner("FEATURE 1 — Embedding Basics")

    # The embedding matrix
    embedding_matrix = model.embeddings.word_embeddings.weight

    print("── Embedding Matrix ─────────────────────────────────────────────────")
    print(f"  Vocabulary size : {embedding_matrix.shape[0]:,} tokens")
    print(f"  Embedding dims  : {embedding_matrix.shape[1]}")
    print(f"  Total params    : {embedding_matrix.numel():,}")
    print(f"  Memory (float32): ~{embedding_matrix.numel() * 4 / 1e6:.1f} MB")

    # Tokenise a movie title
    title     = "Sardar Udham"
    token_ids = tokenizer.encode(title, add_special_tokens=True)

    print(f"\n── Tokenisation: '{title}' ──────────────────────────────────────────")
    print(f"  Token IDs : {token_ids}")
    print(f"  Tokens    : {tokenizer.convert_ids_to_tokens(token_ids)}")
    print()
    print("  ID      Token          First 6 dims of static vector")
    print("  " + "─" * 56)
    for tid in token_ids:
        token = tokenizer.convert_ids_to_tokens([tid])[0]
        vec   = embedding_matrix[tid].detach().numpy()
        print(f"  {tid:<7} {token:<15} {vec[:6].round(3)}")

    # Dimension analogy
    print("\n── What are 768 dimensions? ─────────────────────────────────────────")
    print("  1-D  → position on a line        e.g. temperature  [37.5]")
    print("  2-D  → position on a map         e.g. GPS          [13.08, 80.27]")
    print("  768-D→ position in meaning space  e.g. a BERT token [0.23, -0.11, ...]")

    sample_id  = tokenizer.convert_tokens_to_ids("udham")
    sample_vec = embedding_matrix[sample_id].detach().numpy()
    print(f"\n  Token 'udham' (ID {sample_id}) — first 10 dims:")
    print(f"  {sample_vec[:10].round(4)}")
    print(f"  Shape: {sample_vec.shape}  ← one row of the 30,522 × 768 matrix")

    print("\n── Key Rule ─────────────────────────────────────────────────────────")
    print("  Static embeddings: the same token always returns the same vector.")
    print("  Context has zero effect at this stage — it is a pure table lookup.")


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2 — Static vs Contextual Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def feature_2(tokenizer: BertTokenizer, model: BertModel):
    banner("FEATURE 2 — Static vs Contextual Embeddings")

    embedding_matrix = model.embeddings.word_embeddings.weight

    sent_a = "he operates as a spy knowing he will die at the hands of terrorists"
    sent_b = "intelligence spy thriller about recovering a stolen nuclear bomb"
    TARGET = "spy"

    print(f"  Target word : '{TARGET}'")
    print(f"  Sentence A  : {sent_a}")
    print(f"  Sentence B  : {sent_b}")

    # Static lookup
    spy_id     = tokenizer.convert_tokens_to_ids(TARGET)
    static_vec = embedding_matrix[spy_id].detach().numpy()

    print(f"\n── Step 1 — Static Embedding (matrix row lookup) ────────────────────")
    print(f"  Token ID for '{TARGET}' : {spy_id}")
    print(f"  Static vector (first 8 dims): {static_vec[:8].round(4)}")
    print(f"  This vector is identical in both sentences. No context.")

    # Contextual through BERT
    def get_contextual_vec(sentence: str, target: str):
        inputs = tokenizer(sentence, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
        with torch.no_grad():
            out = model(**inputs)
        pos = next((i for i, t in enumerate(tokens) if t == target), -1)
        return out.last_hidden_state[0, pos, :].numpy(), tokens, out.last_hidden_state.shape

    vec_a, tokens_a, shape_a = get_contextual_vec(sent_a, TARGET)
    vec_b, tokens_b, shape_b = get_contextual_vec(sent_b, TARGET)

    print(f"\n── Step 2 — Contextual Embedding (BERT output) ──────────────────────")
    print(f"\n  Sentence A tokens : {tokens_a}")
    print(f"  Output shape      : {shape_a}   ← [batch=1, seq_len, dims=768]")
    print(f"  '{TARGET}' vector A (first 8 dims): {vec_a[:8].round(4)}")
    print(f"\n  Sentence B tokens : {tokens_b}")
    print(f"  Output shape      : {shape_b}")
    print(f"  '{TARGET}' vector B (first 8 dims): {vec_b[:8].round(4)}")

    sim_static      = cosine_similarity(static_vec, static_vec)
    sim_contextual  = cosine_similarity(vec_a, vec_b)

    print(f"\n── Step 3 — Cosine Similarity Comparison ────────────────────────────")
    print(f"  Static   '{TARGET}' vs '{TARGET}'  (same matrix row)     : {sim_static:.4f}  ← always 1.0")
    print(f"  Contextual A vs B  (different sentence context)  : {sim_contextual:.4f}  ← context changed it")

    # [CLS] and [SEP]
    demo   = "Best movie ever !"
    inputs = tokenizer(demo, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    with torch.no_grad():
        out = model(**inputs)

    print(f"\n── Step 4 — Special Tokens: [CLS] and [SEP] ─────────────────────────")
    print(f"  Input   : '{demo}'")
    print(f"  Tokens  : {tokens}")
    print(f"  Shape   : {tuple(out.last_hidden_state.shape)}")
    print(f"          ↑ [batch=1, tokens={len(tokens)} ([CLS]+words+[SEP]), dims=768]")

    print(f"\n── Key Takeaway ─────────────────────────────────────────────────────")
    print(f"  Static  : '{TARGET}' always has the same vector regardless of context.")
    print(f"  BERT    : '{TARGET}' gets a different vector shaped by its surroundings.")


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3 — Semantic Movie Search
# ─────────────────────────────────────────────────────────────────────────────

def feature_3(mini_lm: SentenceTransformer):
    banner("FEATURE 3 — Semantic Movie Search  (type 'quit' to exit)")

    movies       = load_movies(CSV_PATH)
    descriptions = [m["Description"] for m in movies]

    print(f"  Encoding {len(movies)} movie descriptions...")
    embeddings = mini_lm.encode(descriptions, normalize_embeddings=True)

    print(f"\n── Text Embedding Shape ─────────────────────────────────────────────")
    print(f"  Each description → 1 vector of {embeddings.shape[1]} dims")
    print(f"  All movies matrix : {embeddings.shape}   ← [20 movies × 384 dims]")
    print(f"\n  Mean-pooling (how one vector is made from many tokens):")
    print(f"  'Best'  → [0.23, 0.81, ...]")
    print(f"  'movie' → [0.54, 0.12, ...]")
    print(f"  'ever'  → [0.11, 0.94, ...]")
    print(f"  ───────────────────────────")
    print(f"  Average → [0.29, 0.62, ...]   ← one vector = text embedding")

    print(f"\n  Try queries like:")
    print(f"    'spy who sacrifices everything for the nation'")
    print(f"    'soldier fighting in a war'")
    print(f"    'nuclear weapons and secret operation'")
    print(f"    'freedom fighter executed by the British'\n")

    while True:
        try:
            query = input("  Query: ").strip()
        except EOFError:
            break
        if query.lower() in ("quit", "exit", "q", ""):
            if query.lower() in ("quit", "exit", "q"):
                break
            continue

        query_vec = mini_lm.encode(query, normalize_embeddings=True)
        scores    = [cosine_similarity(query_vec, embeddings[i]) for i in range(len(movies))]
        ranked    = sorted(range(len(movies)), key=lambda i: scores[i], reverse=True)

        print(f"\n  Top 3 matches for: '{query}'")
        print(f"  {'#':<3} {'Score':<8} {'IMDB':<6} Movie")
        print(f"  {'─'*3} {'─'*7} {'─'*5} {'─'*38}")
        for rank, idx in enumerate(ranked[:3], 1):
            m       = movies[idx]
            desc    = m["Description"]
            snippet = desc[:85] + "..." if len(desc) > 85 else desc
            print(f"  {rank:<3} {scores[idx]:.4f}   {m['IMDB Rating']:<6} {m['Movie Name']}")
            print(f"      ↳ {snippet}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4 — Visualise Embedding Space
# ─────────────────────────────────────────────────────────────────────────────

GENRE_MAP = {
    "Spy / Espionage":   ["Dhurandhar", "Dhurandhar: The Revenge", "16 December",
                          "Madras Cafe", "Parmanu: The Story of Pokhran", "Article 370"],
    "War / Military":    ["Uri: The Surgical Strike", "Shershaah", "Border",
                          "Sam Bahadur", "1971", "Major", "The Ghazi Attack", "Lakshya"],
    "Freedom Fighter":   ["Sardar Udham", "The Legend of Bhagat Singh"],
    "Civilian / Social": ["Swades", "Airlift", "A Wednesday", "Sarfarosh"],
}
COLORS = {
    "Spy / Espionage":   "#e74c3c",
    "War / Military":    "#2980b9",
    "Freedom Fighter":   "#27ae60",
    "Civilian / Social": "#f39c12",
}


def feature_4(mini_lm: SentenceTransformer):
    banner("FEATURE 4 — Visualise Embedding Space (PCA 2-D)")

    movies       = load_movies(CSV_PATH)
    names        = [m["Movie Name"] for m in movies]
    descriptions = [m["Description"] for m in movies]

    print(f"  Encoding {len(movies)} descriptions ({384}-dims each)...")
    embeddings = mini_lm.encode(descriptions, normalize_embeddings=True)

    pca    = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    var    = pca.explained_variance_ratio_

    print(f"  PCA: reduced {embeddings.shape[1]}-D → 2-D")
    print(f"  Variance explained: PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  Total={sum(var)*100:.1f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    plotted = set()
    for i, name in enumerate(names):
        genre = next((g for g, titles in GENRE_MAP.items() if name in titles), "Other")
        color = COLORS.get(genre, "#95a5a6")
        ax.scatter(coords[i, 0], coords[i, 1], c=color, s=120, zorder=3,
                   label=genre if genre not in plotted else None,
                   edgecolors="white", linewidth=0.8)
        plotted.add(genre)
        ax.annotate(name, xy=(coords[i, 0], coords[i, 1]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color="#2c3e50", ha="left")

    ax.legend(title="Genre", fontsize=9, title_fontsize=10, framealpha=0.9, loc="upper left")
    ax.set_title("Indian Movies — Embedding Space (PCA 2-D projection)\n"
                 "Movies with similar descriptions cluster together",
                 fontsize=13, pad=14, color="#2c3e50")
    ax.set_xlabel(f"PC1  ({var[0]*100:.1f}% variance)", fontsize=9)
    ax.set_ylabel(f"PC2  ({var[1]*100:.1f}% variance)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axhline(0, color="#bdc3c7", linewidth=0.8)
    ax.axvline(0, color="#bdc3c7", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {PLOT_PATH}")

    # Nearest neighbours
    print(f"\n── Nearest Neighbour in Embedding Space ─────────────────────────────")
    print(f"  {'Movie':<40} Nearest neighbour")
    print(f"  {'─'*39} {'─'*35}")
    for i, name in enumerate(names):
        others  = [j for j in range(len(names)) if j != i]
        nearest = names[others[int(np.argmin([np.linalg.norm(coords[i] - coords[j]) for j in others]))]]
        print(f"  {name:<40} {nearest}")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    line = "═" * 62
    print(f"\n{line}")
    print(f"  TOKEN EMBEDDINGS — Hands-on Learning Project")
    print(f"  Dataset : indian_movies.csv  (20 Indian movies)")
    print(f"{line}")
    print(f"\n  Features:")
    print(f"    1 — Embedding Basics          (matrix, token IDs, dims)")
    print(f"    2 — Static vs Contextual      (same word, different context)")
    print(f"    3 — Semantic Movie Search     (interactive query loop)")
    print(f"    4 — Visualise Embedding Space (PCA cluster plot)")

    print(f"\n{'─' * 62}")
    bert_tokenizer, bert_model, mini_lm = load_models()

    feature_1(bert_tokenizer, bert_model)
    pause()

    feature_2(bert_tokenizer, bert_model)
    pause()

    feature_3(mini_lm)
    pause()

    feature_4(mini_lm)

    print(f"\n{'═' * 62}")
    print(f"  All features complete.")
    print(f"  Cluster plot saved → embedding_clusters.png")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()
