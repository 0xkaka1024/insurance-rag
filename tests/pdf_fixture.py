"""手工构造最小合法 PDF（Type1 Helvetica，仅支持 latin 文本），测试解析不依赖真实条款。

每页入参可以是 str（单文本，画在 (72, 720)），
也可以是 list[tuple[x, y, text]]（定位文本，用于模拟双栏等版面）。
"""

PageSpec = str | list[tuple[float, float, str]]


def make_pdf(*page_specs: PageSpec) -> bytes:
    page_texts: list[list[tuple[float, float, str]]] = [
        [(72.0, 720.0, spec)] if isinstance(spec, str) else list(spec) for spec in page_specs
    ]
    return _build(page_texts)


def _build(page_texts: list[list[tuple[float, float, str]]]) -> bytes:
    page_count = len(page_texts)
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
    for items in page_texts:
        ops = "".join(f"BT /F1 12 Tf {x} {y} Td ({text}) Tj ET " for x, y, text in items)
        stream = ops.encode()
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
