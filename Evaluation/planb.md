# DDRR Plan B (TF-IDF Similarity)

Goal: provide a fast, training-free fallback to compute DDRR without WordNet or GNN, using TF-IDF similarity between dialog text and risk pattern keyword expansions.

## Method Summary
- Build a single dialog text by concatenating all rounds (question + answers).
- For each risk pattern, build a keyword list from `detection_rules`, then expand with `expanded_keywords.json`.
- Convert dialog text and each pattern keyword set into TF-IDF vectors.
- Compute cosine similarity between dialog vector and each pattern vector.
- Map similarity to a soft hit score `p_k` with piecewise thresholds:
  - `sim <= sim_low` => `p_k = 0`
  - `sim >= sim_high` => `p_k = 1`
  - otherwise linear interpolation
- Compute DDRR as weighted average of `p_k`.

## Advantages
- No NLTK / subgraph isomorphism.
- No model training required.
- Fast enough for large pattern libraries.

## Configuration
In `evaluation.py`, change:
- `ddrr_method = "A"` to use embedding similarity.
- `ddrr_method = "B"` to use this TF-IDF method.

Optional tuning (in `cal_indicator.py`):
- `sim_low`, `sim_high` (similarity thresholds)
- `max_units_per_pattern` (upper bound on expanded keyword size)
