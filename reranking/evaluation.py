"""
evaluation.py
-------------
Retrieval Evaluation Metrics  (§5 of docx, Figures 8-16 to 8-23 of book)

Three metrics — each fixing the weakness of the previous one:

  Metric            | What it measures              | Weakness
  ------------------|-------------------------------|------------------------------
  Precision@k       | Relevant docs in top-k / k    | Position-blind (#1 == #k)
  Average Precision | Precision at relevant ranks   | Only one query at a time
  MAP               | Mean AP across all queries    | None — standard benchmark

Test suite (Figure 8-16):
  A pre-prepared set of queries where the correct (relevant) documents are
  already known. It is the answer key the system is scored against.

  Here we assign relevance by movie index in the CSV:
    Query 1: "covert spy mission in Pakistan"
      Relevant: Dhurandhar(0), Dhurandhar:Revenge(1), 16 December(2),
                Article 370(8), Madras Cafe(10)
    Query 2: "soldiers battle war sacrifice"
      Relevant: Uri(12), Shershaah(13), Border(14), Sam Bahadur(15),
                1971(16), Major(17), Ghazi(18), Lakshya(19)
    Query 3: "freedom fighter revolutionary assassination"
      Relevant: Sardar Udham(3), Bhagat Singh(11), Sarfarosh(7)
"""

# ---------------------------------------------------------------------------
# Test Suite — answer key (Figure 8-16)
# ---------------------------------------------------------------------------
TEST_SUITE = [
    {
        "query": "covert spy mission in Pakistan",
        "relevant": {0, 1, 2, 8, 10},
    },
    {
        "query": "soldiers battle war sacrifice",
        "relevant": {12, 13, 14, 15, 16, 17, 18, 19},
    },
    {
        "query": "freedom fighter revolutionary assassination",
        "relevant": {3, 7, 11},
    },
]


# ---------------------------------------------------------------------------
# Metric 1: Precision@k  (Figure 8-21)
# ---------------------------------------------------------------------------
def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """
    Of the top-k results returned, what fraction are actually relevant?

      Precision@k = (relevant hits in top-k) / k

    Example (Figure 8-22):
      top-3 = [relevant, not-relevant, relevant]
      Precision@3 = 2/3 = 0.67

    Weakness: Does not care WHICH position the relevant doc is at.
    A relevant doc at #1 and at #3 score the same → Average Precision fixes this.
    """
    top_k = retrieved[:k]
    hits = sum(1 for idx in top_k if idx in relevant)
    return hits / k if k > 0 else 0.0


# ---------------------------------------------------------------------------
# Metric 2: Average Precision (AP)  (Figures 8-20, 8-22)
# ---------------------------------------------------------------------------
def average_precision(retrieved: list, relevant: set) -> float:
    """
    Calculate precision ONLY at positions where a relevant document appears,
    then average those values. Rewards systems that rank relevant docs HIGHER.

    Algorithm:
      For each position i in the retrieved list:
        If retrieved[i] is relevant:
          record Precision@i = (total relevant hits so far) / i
      AP = mean of all recorded precision values

    Examples from the docx:
      System A: [✓, ✗, ✓]  → P@1=1.0, P@3=0.67  → AP = (1.0+0.67)/2 = 0.83
      System B: [✗, ✗, ✓]  → P@3=0.33            → AP = 0.33/1      = 0.33
    """
    if not relevant:
        return 0.0

    hits = 0
    precision_at_relevant_ranks = []

    for rank, idx in enumerate(retrieved, start=1):
        if idx in relevant:
            hits += 1
            precision_at_relevant_ranks.append(hits / rank)

    # Denominator = total number of relevant docs (not just retrieved ones)
    return sum(precision_at_relevant_ranks) / len(relevant)


# ---------------------------------------------------------------------------
# Metric 3: Mean Average Precision (MAP)  (Figure 8-23)
# ---------------------------------------------------------------------------
def mean_average_precision(results_per_query: list) -> float:
    """
    AP works for a single query. MAP extends it across ALL queries in the test
    suite — one number to compare two search systems.

      MAP = (AP_q1 + AP_q2 + ... + AP_qn) / n

    Example (Figure 8-23):
      Query 1 AP = 0.80
      Query 2 AP = 0.67
      Query 3 AP = 0.30
      MAP = (0.80 + 0.67 + 0.30) / 3 = 0.59
    """
    aps = [average_precision(retrieved, relevant) for retrieved, relevant in results_per_query]
    return sum(aps) / len(aps) if aps else 0.0


# ---------------------------------------------------------------------------
# Helper: run a search function against the full test suite
# ---------------------------------------------------------------------------
def evaluate_system(search_fn, test_suite: list, top_n: int = 3) -> dict:
    """
    Run search_fn on every query in test_suite and compute
    Precision@top_n, AP per query, and MAP across all queries.

    Parameters
    ----------
    search_fn  : callable(query: str) → list of dicts with 'index' key
    test_suite : list of {query, relevant} dicts
    top_n      : k for Precision@k

    Returns
    -------
    dict with 'per_query' list and 'MAP' float
    """
    map_inputs = []
    per_query_results = []

    for item in test_suite:
        query = item["query"]
        relevant = item["relevant"]

        results = search_fn(query)
        retrieved_indices = [r["index"] for r in results]

        p_at_k = precision_at_k(retrieved_indices, relevant, k=top_n)
        ap = average_precision(retrieved_indices, relevant)

        map_inputs.append((retrieved_indices, relevant))
        per_query_results.append({
            "query": query,
            "retrieved": retrieved_indices,
            "precision_at_k": p_at_k,
            "ap": ap,
        })

    return {
        "per_query": per_query_results,
        "MAP": mean_average_precision(map_inputs),
    }
