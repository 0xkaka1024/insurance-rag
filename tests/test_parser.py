from pathlib import Path

from app.ingest.parser import normalize, parse_pdf, product_from_filename
from tests.pdf_fixture import make_pdf


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
