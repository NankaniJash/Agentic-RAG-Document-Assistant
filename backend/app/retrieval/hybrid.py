def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]], k: int = 60
) -> list[tuple[int, float]]:
    """Merge multiple ranked result lists (e.g. vector search + BM25) into one
    ranking using Reciprocal Rank Fusion. RRF only looks at rank position, not
    raw scores, which sidesteps the problem that cosine similarity and BM25
    scores live on completely different, incomparable scales.

    score(doc) = sum over lists containing doc of 1 / (k + rank_in_that_list)
    """
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
