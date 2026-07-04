from pathlib import Path

from app.ingest.parser import _is_low_quality, normalize, parse_pdf, product_from_filename
from tests.pdf_fixture import make_pdf, table_grid


def test_normalize_traditional_to_simplified():
    assert normalize("等待期內確診之疾病") == "等待期内确诊之疾病"


def test_normalize_keeps_simplified_and_latin():
    assert normalize("Waiting Period 90 days 等待期") == "Waiting Period 90 days 等待期"


def test_product_from_filename_strips_tc_suffix():
    assert product_from_filename(Path("newVHISmedical-tc.pdf")) == "newVHISmedical"
    assert product_from_filename(Path("SmartEliteUltra_tc.pdf")) == "SmartEliteUltra"
    assert product_from_filename(Path("NoSuffix.pdf")) == "NoSuffix"


def test_parse_pdf_extracts_pages_with_metadata(tmp_path):
    pdf = tmp_path / "DemoProduct_tc.pdf"
    pdf.write_bytes(make_pdf("Waiting Period is 90 days", "Second page content"))
    pages = parse_pdf(pdf)
    assert len(pages) == 2
    assert pages[0].product == "DemoProduct"
    assert pages[0].page_no == 1
    assert "Waiting Period is 90 days" in pages[0].text
    assert pages[1].page_no == 2


def test_parse_pdf_skips_blank_pages(tmp_path):
    pdf = tmp_path / "Blank.pdf"
    pdf.write_bytes(make_pdf(" "))
    assert parse_pdf(pdf) == []


def _table_page(text: str = "Benefit waiting 300 days") -> dict:
    """带真实网格线的页：pdfplumber find_tables 能检出，非 mock。"""
    return {"texts": [(80.0, 680.0, text)], "lines": table_grid()}


def test_parse_pdf_flags_table_page(tmp_path):
    pdf = tmp_path / "Demo.pdf"
    pdf.write_bytes(make_pdf(_table_page(), "plain text page"))
    pages = parse_pdf(pdf)
    assert pages[0].has_table is True
    assert pages[1].has_table is False
    assert not any(p.vlm_used for p in pages)  # 未注入 vlm 只打标记，不改文本


def test_parse_pdf_vlm_rewrites_flagged_pages_only(tmp_path):
    pdf = tmp_path / "Demo.pdf"
    pdf.write_bytes(make_pdf(_table_page(), "plain text page"))
    calls: list[int] = []

    def fake_vlm(page):
        calls.append(page.page_number)
        return "| 保障 | 等候期 |\n| 危疾 | 300日（確診後起計） |"

    pages = parse_pdf(pdf, vlm=fake_vlm)
    assert calls == [1]  # 纯文本页不调 VLM，不花冤枉钱
    assert pages[0].vlm_used is True
    assert pages[0].raw_text.startswith("| 保障 |")  # 引用展示也基于转写结果
    assert "确诊" in pages[0].text  # 繁简归一照常作用于 VLM 输出
    assert pages[1].vlm_used is False
    assert "plain text page" in pages[1].text


def test_parse_pdf_vlm_failure_falls_back_to_plumber_text(tmp_path):
    pdf = tmp_path / "Demo.pdf"
    pdf.write_bytes(make_pdf(_table_page()))

    def boom(page):
        raise RuntimeError("api down")

    pages = parse_pdf(pdf, vlm=boom)
    assert len(pages) == 1  # 调用失败页不丢
    assert pages[0].vlm_used is False
    assert "Benefit" in pages[0].text  # 回退 pdfplumber 文本
    assert pages[0].has_table is True  # 红旗保留，报告可见


def test_parse_pdf_vlm_empty_result_falls_back(tmp_path):
    pdf = tmp_path / "Demo.pdf"
    pdf.write_bytes(make_pdf(_table_page()))
    pages = parse_pdf(pdf, vlm=lambda page: "  ")
    assert pages[0].vlm_used is False
    assert "Benefit" in pages[0].text


def test_low_quality_detection_by_cid_placeholder():
    assert _is_low_quality("(cid:123) (cid:456)")
    assert not _is_low_quality("等候期为90天。")
