"""
04 — Visualising the Embedding Space
======================================
Covers:
  - Embedding space: tokens/sentences as points in high-dimensional space
  - PCA: collapsing 384-dim vectors → 2-D so we can plot them
  - Clusters: movies with similar meaning should sit close together
  - The geometric intuition: similar meaning = short distance

Dataset : indian_movies.csv  (20 Indian movies)
Model   : sentence-transformers/all-MiniLM-L6-v2  (384-dim)

Output  : saves  embedding_clusters.png  in the project root
"""

import csv
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

matplotlib.rcParams["font.family"] = "DejaVu Sans"

CSV_PATH  = os.path.join(os.path.dirname(__file__), "..", "indian_movies.csv")
OUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "embedding_clusters.png")

# Manually grouped for colour-coding
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


def load_movies(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_genre(name: str) -> str:
    for genre, titles in GENRE_MAP.items():
        if name in titles:
            return genre
    return "Other"


def main():
    print("Loading model...")
    model  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    movies = load_movies(CSV_PATH)

    names        = [m["Movie Name"] for m in movies]
    descriptions = [m["Description"] for m in movies]

    print(f"Encoding {len(movies)} movie descriptions ({384}-dims each)...")
    embeddings = model.encode(descriptions, normalize_embeddings=True)
    print(f"  Embedding matrix shape: {embeddings.shape}  ← [movies × dims]")

    # Reduce 384-D → 2-D with PCA so we can visualise
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    var    = pca.explained_variance_ratio_
    print(f"\n  PCA: reduced {embeddings.shape[1]}-D → 2-D")
    print(f"  Variance explained: PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  "
          f"Total={sum(var)*100:.1f}%")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    plotted_genres = set()
    for i, name in enumerate(names):
        genre = get_genre(name)
        color = COLORS.get(genre, "#95a5a6")
        label = genre if genre not in plotted_genres else None
        ax.scatter(coords[i, 0], coords[i, 1],
                   c=color, s=120, zorder=3, label=label, edgecolors="white", linewidth=0.8)
        plotted_genres.add(genre)

        # Annotate with movie name — offset to avoid overlap
        ax.annotate(
            name,
            xy=(coords[i, 0], coords[i, 1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#2c3e50",
            ha="left",
        )

    ax.legend(title="Genre", fontsize=9, title_fontsize=10,
              framealpha=0.9, loc="upper left")
    ax.set_title(
        "Indian Movies — Embedding Space (PCA 2-D projection)\n"
        "Movies with similar descriptions cluster together",
        fontsize=13, pad=14, color="#2c3e50"
    )
    ax.set_xlabel(f"PC1  ({var[0]*100:.1f}% variance)", fontsize=9)
    ax.set_ylabel(f"PC2  ({var[1]*100:.1f}% variance)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axhline(0, color="#bdc3c7", linewidth=0.8)
    ax.axvline(0, color="#bdc3c7", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {OUT_PATH}")

    # ── Print nearest neighbour for each movie ────────────────────────────────
    print("\n── Nearest Neighbour in Embedding Space ─────────────────────────────")
    print(f"  {'Movie':<40} Nearest neighbour")
    print(f"  {'─'*39} {'─'*35}")
    for i, name in enumerate(names):
        dists = [
            np.linalg.norm(coords[i] - coords[j])
            for j in range(len(names)) if j != i
        ]
        idx_of_dists = [j for j in range(len(names)) if j != i]
        nearest = names[idx_of_dists[int(np.argmin(dists))]]
        print(f"  {name:<40} {nearest}")

    plt.show()


if __name__ == "__main__":
    main()
