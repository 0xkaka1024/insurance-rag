"""手工构造最小合法 PDF（Type1 Helvetica，仅支持 latin 文本），测试解析不依赖真实条款。

每页入参可以是 str（单文本，画在 (72, 720)），
也可以是 list[tuple[x, y, text]]（定位文本，用于模拟双栏等版面），
也可以是 dict {"texts": [...], "lines": [(x0, y0, x1, y1), ...]}——
画出的直线能被 pdfplumber find_tables 检出，用于模拟表格页。
"""

PageSpec = str | list[tuple[float, float, str]] | dict

Line = tuple[float, float, float, float]


def table_grid(
    x0: float = 72.0, y0: float = 620.0, x1: float = 300.0, y1: float = 700.0,
    rows: int = 2, cols: int = 2,
) -> list[Line]:
    """rows×cols 网格线：足以让 pdfplumber 的 lines 策略识别为表格。"""
    lines: list[Line] = []
    for i in range(rows + 1):
        y = y0 + (y1 - y0) * i / rows
        lines.append((x0, y, x1, y))
    for j in range(cols + 1):
        x = x0 + (x1 - x0) * j / cols
        lines.append((x, y0, x, y1))
    return lines


def make_pdf(*page_specs: PageSpec) -> bytes:
    pages: list[tuple[list[tuple[float, float, str]], list[Line]]] = []
    for spec in page_specs:
        if isinstance(spec, str):
            pages.append(([(72.0, 720.0, spec)], []))
        elif isinstance(spec, dict):
            pages.append((list(spec.get("texts", [])), list(spec.get("lines", []))))
        else:
            pages.append((list(spec), []))
    return _build(pages)


def _build(pages: list[tuple[list[tuple[float, float, str]], list[Line]]]) -> bytes:
    page_count = len(pages)
    objs: list[bytes] = []
    # 1: catalog, 2: pages
    kids = " ".join(f"{3 + i} 0 R" for i in range(page_count))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode())
    font_obj_no = 3 + 2 * page_count
    for i in range(page_count):
        content_no = 3 + page_count + i
        objs.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_no} 0 R "
                f"/Resources << /Font << /F1 {font_obj_no} 0 R >> >> >>"
            ).encode()
        )
    for texts, lines in pages:
        line_ops = "".join(f"{x0} {y0} m {x1} {y1} l S " for x0, y0, x1, y1 in lines)
        text_ops = "".join(f"BT /F1 12 Tf {x} {y} Td ({text}) Tj ET " for x, y, text in texts)
        stream = (line_ops + text_ops).encode()
        objs.append(
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF"
    ).encode()
    return bytes(out)
