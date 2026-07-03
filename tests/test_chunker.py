from app.ingest.chunker import FixedChunker, _atomize
from app.ingest.parser import Page


def _pages(*texts: str) -> list[Page]:
    return [
        Page(product="Demo", page_no=i, text=t, raw_text=t) for i, t in enumerate(texts, start=1)
    ]


def _sentences(n: int, prefix: str = "句") -> str:
    return "".join(f"第{i}{prefix}的内容补充说明若干字。" for i in range(n))


def test_atomize_splits_on_sentence_end_and_hard_splits_long():
    pieces = _atomize("第一句。第二句；" + "长" * 30, max_len=10)
    assert pieces[0] == "第一句。"
    assert pieces[1] == "第二句；"
    assert all(len(p) <= 10 for p in pieces)
    assert "".join(pieces) == "第一句。第二句；" + "长" * 30


def test_chunk_length_bounded():
    chunker = FixedChunker(chunk_size=100, overlap_ratio=0.15)
    chunks = chunker.split(_pages(_sentences(40)))
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 + chunker.overlap for c in chunks)


def test_consecutive_chunks_overlap():
    chunker = FixedChunker(chunk_size=100, overlap_ratio=0.15)
    chunks = chunker.split(_pages(_sentences(40)))
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.text[:10] in prev.text  # 下一块开头来自上一块尾部


def test_no_content_lost():
    text = _sentences(40)
    chunks = FixedChunker(chunk_size=100).split(_pages(text))
    # 每块都是原文连续片段；首块起于文首，末块止于文尾；相邻块间无缝隙
    assert all(c.text in text for c in chunks)
    assert text.startswith(chunks[0].text)
    assert text.endswith(chunks[-1].text)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        prev_end = text.index(prev.text) + len(prev.text)
        assert text.index(nxt.text, text.index(prev.text)) <= prev_end


def test_page_range_metadata():
    chunker = FixedChunker(chunk_size=100, overlap_ratio=0.1)
    chunks = chunker.split(_pages(_sentences(8, "甲"), _sentences(8, "乙")))
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert all(c.page_start <= c.page_end for c in chunks)


def test_chunk_ids_deterministic_and_unique():
    pages = _pages(_sentences(20))
    a = FixedChunker(chunk_size=100).split(pages)
    b = FixedChunker(chunk_size=100).split(pages)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
    assert a[0].chunk_id == "Demo:fixed:0000"
    assert a[0].strategy == "fixed"


def test_empty_pages():
    assert FixedChunker().split([]) == []
