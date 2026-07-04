"""BM25 关键词索引：jieba 分词 + rank_bm25，按切片策略分文件持久化。

兜住精确术语匹配（产品名、条号、专有名词）——这类 query 向量检索容易漂。
存储为本项目自产的 pickle 文件（仅加载 data/index 下自己写的文件）。
"""

import pickle
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from app.ingest.chunker import Chunk

jieba.setLogLevel(60)  # 静默初始化日志

# 保险领域词典：jieba 默认词典会把「等待期为90天」切成 等待/期为/90/天，
# 精确术语召回全毁；先内置高频术语，后续可换 data 驱动的词表文件。
DOMAIN_TERMS = (
    "等待期", "等候期", "犹豫期", "冷静期", "免赔额", "自付费", "垫底费",
    "身故赔偿", "现金价值", "保证现金价值", "保单周年日", "保单年度",
    "不保事项", "投保人", "受保人", "保单持有人", "受益人",
    "危疾", "轻症", "重疾", "标体", "核保", "既往症", "已有病症",
    "住院现金", "医疗保障", "终身保障限额", "每年保障限额", "分项赔偿限额",
    "自愿医保", "认可产品", "税务扣减", "保证续保", "误导销售",
)
for _term in DOMAIN_TERMS:
    jieba.add_word(_term, freq=2_000_000)  # 高频压过默认词典的组合


def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text.lower()) if t.strip()]


class BM25Index:
    def __init__(self, index_dir: Path, strategy: str):
        self._path = index_dir / f"bm25_{strategy}.pkl"
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        if self._path.exists():
            data = pickle.loads(self._path.read_bytes())
            self._ids = data["ids"]
            self._texts = data["texts"]
            self._metas = data["metas"]
            self._tokens = data["tokens"]
            self._rebuild()

    def _rebuild(self) -> None:
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None

    def upsert(self, chunks: list[Chunk]) -> None:
        """按 chunk_id 覆盖写入，与 Chroma upsert 保持同样的幂等语义。"""
        if not chunks:
            return
        pos = {cid: i for i, cid in enumerate(self._ids)}
        for c in chunks:
            meta = {
                "product": c.product,
                "page_start": c.page_start,
                "page_end": c.page_end,
                **c.meta,
            }
            tokens = tokenize(c.text)
            if c.chunk_id in pos:
                i = pos[c.chunk_id]
                self._texts[i], self._metas[i], self._tokens[i] = c.text, meta, tokens
            else:
                pos[c.chunk_id] = len(self._ids)
                self._ids.append(c.chunk_id)
                self._texts.append(c.text)
                self._metas.append(meta)
                self._tokens.append(tokens)
        self._rebuild()
        self._save()

    def delete_by_product(self, product: str) -> int:
        """删除某产品的全部条目（清场式重入库用），返回删除数。"""
        keep = [i for i, m in enumerate(self._metas) if m.get("product") != product]
        removed = len(self._ids) - len(keep)
        if not removed:
            return 0
        self._ids = [self._ids[i] for i in keep]
        self._texts = [self._texts[i] for i in keep]
        self._metas = [self._metas[i] for i in keep]
        self._tokens = [self._tokens[i] for i in keep]
        self._rebuild()
        self._save()
        return removed

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(
            pickle.dumps(
                {"ids": self._ids, "texts": self._texts, "metas": self._metas,
                 "tokens": self._tokens}
            )
        )

    def search(self, query: str, k: int) -> list[tuple[str, str, dict, float]]:
        """返回 [(chunk_id, text, meta, score)]，按 BM25 分数降序。"""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._ids[i], self._texts[i], self._metas[i], float(scores[i])) for i in order]

    def __len__(self) -> int:
        return len(self._ids)
