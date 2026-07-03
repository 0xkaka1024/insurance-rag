import pdfplumber

from tests.pdf_fixture import make_pdf

# 页宽 612pt，中线 306。左栏 x=72，右栏 x=340。
TWO_COL_PAGE = [
    (72, 740, "Full Width Title Spanning Both Columns Of This Page Layout"),
    (72, 700, "left one"),
    (340, 700, "right one"),
    (72, 680, "left two"),
    (340, 680, "right two"),
    (72, 660, "left three"),
    (340, 660, "right three"),
]

SINGLE_COL_PAGE = [
    (72, 700, "alpha line"),
    (72, 680, "beta line"),
    (72, 660, "gamma line"),
]


def _extract(spec, tmp_path):
    from app.ingest.parser import extract_page_text

    f = tmp_path / "t.pdf"
    f.write_bytes(make_pdf(spec))
    with pdfplumber.open(f) as pdf:
        return extract_page_text(pdf.pages[0])


def test_naive_extraction_interleaves_columns(tmp_path):
    """基线事实：pdfplumber 默认阅读顺序把两栏按行交错（这就是要修的问题）。"""
    f = tmp_path / "naive.pdf"
    f.write_bytes(make_pdf(TWO_COL_PAGE))
    with pdfplumber.open(f) as pdf:
        naive = pdf.pages[0].extract_text()
    assert "left one right one" in naive


def test_two_column_page_reads_left_then_right(tmp_path):
    text = _extract(TWO_COL_PAGE, tmp_path)
    assert text.index("left three") < text.index("right one")
    assert "left one right one" not in text  # 不再交错


def test_full_width_title_stays_first(tmp_path):
    text = _extract(TWO_COL_PAGE, tmp_path)
    assert text.index("Full Width Title") < text.index("left one")


def test_single_column_order_preserved(tmp_path):
    text = _extract(SINGLE_COL_PAGE, tmp_path)
    assert text.index("alpha") < text.index("beta") < text.index("gamma")


def test_parse_pdf_uses_column_aware_extraction(tmp_path):
    from app.ingest.parser import parse_pdf

    f = tmp_path / "Demo_tc.pdf"
    f.write_bytes(make_pdf(TWO_COL_PAGE))
    [page] = parse_pdf(f)
    assert page.text.index("left three") < page.text.index("right one")
