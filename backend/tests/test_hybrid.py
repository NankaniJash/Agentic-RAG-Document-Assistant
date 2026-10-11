from app.retrieval.hybrid import reciprocal_rank_fusion


def test_item_ranked_highly_in_both_lists_wins():
    vector_hits = [(1, 0.9), (2, 0.8), (3, 0.7)]
    keyword_hits = [(2, 15.0), (4, 10.0), (1, 5.0)]

    fused = reciprocal_rank_fusion([vector_hits, keyword_hits])

    assert fused[0][0] == 2


def test_empty_lists_return_empty():
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_list_preserves_relative_order():
    hits = [(1, 0.9), (2, 0.8), (3, 0.1)]

    fused = reciprocal_rank_fusion([hits])

    assert [doc_id for doc_id, _ in fused] == [1, 2, 3]


def test_raw_scores_never_compared_across_lists():
    # BM25 scores can be huge (double digits+) while cosine similarity is
    # bounded [-1, 1]. RRF must ignore magnitude and use rank position only,
    # or a mediocre keyword match would always crush a strong vector match.
    vector_hits = [(1, 0.99)]
    keyword_hits = [(2, 500.0), (1, 1.0)]

    fused = reciprocal_rank_fusion([vector_hits, keyword_hits])
    fused_ids = [doc_id for doc_id, _ in fused]

    assert fused_ids[0] == 1
