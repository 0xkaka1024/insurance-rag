"""手工构造最小合法 PDF（Type1 Helvetica，仅支持 latin 文本），测试解析不依赖真实条款。"""


def make_pdf(*page_texts: str) -> bytes:
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
    for text in page_texts:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
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
