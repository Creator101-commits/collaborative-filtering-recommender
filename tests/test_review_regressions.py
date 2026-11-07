"""Review regressions for the item-oriented sparse index."""

from recommender.models import ItemKNNRecommender


def test_item_knn_excludes_actual_seen_item_rows():
    rows = [
        {"user_idx": 0, "item_idx": 2, "rating": 5},
        {"user_idx": 0, "item_idx": 4, "rating": 4},
        {"user_idx": 1, "item_idx": 1, "rating": 5},
        {"user_idx": 1, "item_idx": 2, "rating": 4},
    ]
    model = ItemKNNRecommender(2, 6).fit(rows)
    ranked = model.recommend(0, range(6), 6)
    assert set(ranked) == {0, 1, 3, 5}
    assert 2 not in ranked and 4 not in ranked
