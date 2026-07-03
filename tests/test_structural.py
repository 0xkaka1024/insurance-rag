from app.ingest.parser import Page
from app.ingest.structural import StructuralChunker, _is_heading


def _pages(*texts: str) -> list[Page]:
    return [
        Page(product="Demo", page_no=i, text=t, raw_text=t) for i, t in enumerate(texts, start=1)
    ]


CLAUSE_DOC = """第一章 总则
第一条 本合同由保险单及所附条款构成，对合同双方均有约束力，签署后生效。
第二条 投保人应如实告知健康状况，故意隐瞒的我们有权解除合同并不退还保费。
第二章 保险责任
第三条 等待期为90天，等待期内确诊的疾病不承担给付保险金的责任，保费全额退还。
"""

BROCHURE_DOC = """住院保障
病房及膳食、专科医生费、外科医生费，全数赔偿不设分项限额，须为医疗所需。
诊断保障
订明诊断成像检测包括CT扫描、MRI扫描及PET扫描，按保障表所列金额赔偿。
"""


def test_clause_mode_splits_per_clause():
    chunks = StructuralChunker().split(_pages(CLAUSE_DOC))
    clause_metas = [c.meta.get("clause", "") for c in chunks]
    assert any(m.startswith("第一条") for m in clause_metas)
    assert any(m.startswith("第三条") for m in clause_metas)
    third = next(c for c in chunks if c.meta.get("clause", "").startswith("第三条"))
    assert "等待期为90天" in third.text
    assert third.meta["chapter"] == "第二章 保险责任"


def test_clause_not_split_mid_clause():
    chunks = StructuralChunker().split(_pages(CLAUSE_DOC))
    for c in chunks:
        if c.meta.get("clause", "").startswith("第二条"):
            assert "如实告知" in c.text and "解除合同" in c.text


def test_brochure_mode_splits_on_headings():
    chunks = StructuralChunker().split(_pages(BROCHURE_DOC))
    sections = [c.meta["section"] for c in chunks]
    assert "住院保障" in sections
    assert "诊断保障" in sections
    hosp = next(c for c in chunks if c.meta["section"] == "住院保障")
    assert "病房及膳食" in hosp.text
    assert "CT扫描" not in hosp.text  # 下一节内容不混入


def test_heading_heuristic():
    assert _is_heading("住院保障")
    assert _is_heading("其他计划特点")
    assert not _is_heading("病房及膳食、专科医生费，全数赔偿。")  # 有句读
    assert not _is_heading("港元 0 16,000 25,000 50,000")  # 数字表格行
    assert not _is_heading("这是一个非常非常非常长的不可能是标题的行超过二十个字")


def test_oversized_section_secondary_split_shares_meta():
    long_body = "保障内容详述，" * 200  # 远超 max_chunk_size
    doc = f"住院保障\n{long_body}"
    chunks = StructuralChunker().split(_pages(doc))
    assert len(chunks) > 1
    assert all(c.meta["section"] == "住院保障" for c in chunks)


def test_cross_page_section_page_range():
    page1 = "住院保障\n第一页的保障内容说明，涵盖病房及膳食费用与手术费。"
    page2 = "接续上一页的更多保障内容说明，包括深切治疗与药费。"
    chunks = StructuralChunker().split(_pages(page1, page2))
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2


def test_deterministic_ids_and_strategy():
    a = StructuralChunker().split(_pages(BROCHURE_DOC))
    b = StructuralChunker().split(_pages(BROCHURE_DOC))
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert all(c.strategy == "structural" for c in a)
    assert a[0].chunk_id.startswith("Demo:structural:")


def test_empty_pages():
    assert StructuralChunker().split([]) == []
