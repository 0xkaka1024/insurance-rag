"""语料质量报告：边界 lint 规则与报告结构。"""

from app.ingest.chunker import Chunk
from app.ingest.parser import Page
from app.ingest.report import build_report, lint_chunks


def _chunk(seq: int, text: str, strategy: str = "structural", meta: dict | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"Demo:{strategy}:{seq:04d}",
        product="Demo",
        strategy=strategy,
        text=text,
        page_start=1,
        page_end=1,
        meta=meta or {},
    )


def test_cut_midsentence_flag_and_boundary_snippets():
    chunks = [_chunk(0, "等候期为90天，期内确诊"), _chunk(1, "不予赔付。")]
    recs = lint_chunks(chunks)
    assert "cut_midsentence" in recs[0]["flags"]
    assert recs[0]["boundary"]["cut"] is True
    assert recs[0]["boundary"]["tail"].endswith("期内确诊")
    assert recs[0]["boundary"]["next_head"].startswith("不予赔付")
    assert recs[1]["boundary"] is None  # 末块无接缝


def test_sentence_end_boundary_not_flagged():
    recs = lint_chunks([_chunk(0, "等候期为90天。"), _chunk(1, "危疾定义如下。")])
    assert "cut_midsentence" not in recs[0]["flags"]
    assert recs[0]["boundary"]["cut"] is False


def test_closing_bracket_after_sentence_end_is_clean():
    recs = lint_chunks([_chunk(0, "详见附表。）"), _chunk(1, "以下条款适用。")])
    assert "cut_midsentence" not in recs[0]["flags"]


def test_no_clause_flag_only_for_structural():
    structural = _chunk(0, "本条款正文说明如下，内容与等候期相关，共计足够长度。")
    fixed = _chunk(0, "本条款正文说明如下，内容与等候期相关，共计足够长度。", strategy="fixed")
    assert "no_clause" in lint_chunks([structural])[0]["flags"]
    assert "no_clause" not in lint_chunks([fixed])[0]["flags"]


def test_merged_clauses_flag_only_for_structural():
    text = "第1条 等候期\n等候期为90天，自保单生效日起计算适用。\n第2条 冷静期\n冷静期为21天。"
    merged = _chunk(0, text, meta={"clause": "第1条 等候期"})
    fixed = _chunk(0, text, strategy="fixed")
    assert "merged_clauses" in lint_chunks([merged])[0]["flags"]
    assert "merged_clauses" not in lint_chunks([fixed])[0]["flags"]  # fixed 跨条属正常


def test_length_outlier_flags():
    assert "overshort" in lint_chunks([_chunk(0, "太短。")])[0]["flags"]
    assert "overlong" in lint_chunks([_chunk(0, "长" * 1300 + "。")])[0]["flags"]


def test_build_report_coverage_flag_counts_and_empty_pages():
    pages = [Page(product="Demo", page_no=1, text="等候期为90天。", raw_text="等候期為90天。")]
    with_clause = _chunk(
        0, "第1条 等候期\n等候期为90天，自保单生效日起计算，期内确诊不予赔付。",
        meta={"clause": "第1条 等候期", "section": "第1条 等候期"},
    )
    without_clause = _chunk(1, "以上内容仅供参考，具体以保单条款正文为准，详情见附录。")
    report = build_report(
        product="Demo",
        filename="Demo_tc.pdf",
        sha256="ab12",
        total_pages=2,
        pages=pages,
        chunks_by_strategy={"structural": [with_clause, without_clause], "fixed": []},
    )
    assert report["parsed_pages"] == 1
    assert report["empty_pages"] == [2]  # 第 2 页无文本，暴露而非静默
    assert report["pages"][0]["cjk_ratio"] > 0.5
    assert report["strategies"]["structural"]["clause_coverage"] == 0.5
    assert report["strategies"]["structural"]["flag_counts"]["no_clause"] == 1
    assert "clause_coverage" not in report["strategies"]["fixed"]
    assert report["strategies"]["fixed"]["n_chunks"] == 0
    assert report["vlm_fallback"] is False  # 未传参默认：未启用，可归因


def test_build_report_vlm_attribution_and_flat_table_red_flags():
    pages = [
        Page(product="Demo", page_no=1, text="表格页拍平文本", raw_text="表格頁", has_table=True),
        Page(product="Demo", page_no=2, text="| 已转写 |", raw_text="| 已轉寫 |",
             has_table=True, vlm_used=True),
        Page(product="Demo", page_no=3, text="普通文本页。", raw_text="普通文本頁。"),
    ]
    report = build_report(
        product="Demo", filename="Demo_tc.pdf", sha256="ab12", total_pages=3,
        pages=pages, chunks_by_strategy={}, vlm_fallback=True,
    )
    assert report["vlm_fallback"] is True
    assert report["table_pages_flat"] == [1]  # 有表格且未转写的页才是红旗
    assert report["pages"][0]["has_table"] is True
    assert report["pages"][0]["vlm_used"] is False
    assert report["pages"][1]["vlm_used"] is True
    assert report["pages"][2]["has_table"] is False
