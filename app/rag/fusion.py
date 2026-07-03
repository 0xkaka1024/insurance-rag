"""RRF（Reciprocal Rank Fusion）：融合多路检索的排名，k=60 为经验标准值。

只看名次不看原始分数，天然解决「余弦相似度 vs BM25 分数不可比」的问题。
"""


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """rankings 中每路是按相关性降序的 id 列表；返回 {id: 融合分}。"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
