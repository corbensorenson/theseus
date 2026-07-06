"""Small QR SVG helper for Project Theseus setup.

This intentionally supports the small invite URLs used by the Hive setup
wizard: byte-mode, error-correction level L, QR versions 1-5. If an invite URL
gets too large, callers should fall back to showing the copyable link.
"""

from __future__ import annotations

from dataclasses import dataclass


DATA_CODEWORDS_L = {
    1: 19,
    2: 34,
    3: 55,
    4: 80,
    5: 108,
}

ECC_CODEWORDS_L = {
    1: 7,
    2: 10,
    3: 15,
    4: 20,
    5: 26,
}

ALIGNMENT_POSITIONS = {
    1: [],
    2: [6, 18],
    3: [6, 22],
    4: [6, 26],
    5: [6, 30],
}


@dataclass
class QrMatrix:
    version: int
    modules: list[list[bool]]


def qr_svg(text: str, *, scale: int = 6, border: int = 4) -> str:
    matrix = encode_qr(text.encode("utf-8"))
    size = len(matrix.modules)
    total = (size + border * 2) * scale
    rects = []
    for y, row in enumerate(matrix.modules):
        for x, dark in enumerate(row):
            if dark:
                rects.append(
                    f'<rect x="{(x + border) * scale}" y="{(y + border) * scale}" '
                    f'width="{scale}" height="{scale}"/>'
                )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total} {total}" '
        f'width="{total}" height="{total}" role="img" aria-label="Project Theseus Hive join QR">'
        f'<rect width="100%" height="100%" fill="#fff"/>'
        f'<g fill="#000">{"".join(rects)}</g></svg>'
    )


def encode_qr(payload: bytes) -> QrMatrix:
    version = choose_version(payload)
    size = 21 + 4 * (version - 1)
    modules: list[list[bool | None]] = [[None for _ in range(size)] for _ in range(size)]
    reserved = [[False for _ in range(size)] for _ in range(size)]

    add_function_patterns(modules, reserved, version)
    data = make_data_codewords(payload, version)
    ecc = reed_solomon_ecc(data, ECC_CODEWORDS_L[version])
    bits = bytes_to_bits(data + ecc)
    place_data(modules, reserved, bits, mask=0)
    add_format_bits(modules, reserved, mask=0)

    return QrMatrix(
        version=version,
        modules=[[bool(value) for value in row] for row in modules],
    )


def choose_version(payload: bytes) -> int:
    for version in range(1, 6):
        capacity_bits = DATA_CODEWORDS_L[version] * 8
        count_bits = 8
        needed = 4 + count_bits + len(payload) * 8
        if needed <= capacity_bits:
            return version
    raise ValueError("QR payload too large for built-in setup QR")


def make_data_codewords(payload: bytes, version: int) -> bytes:
    capacity_bits = DATA_CODEWORDS_L[version] * 8
    bits: list[int] = []
    append_bits(bits, 0b0100, 4)
    append_bits(bits, len(payload), 8)
    for byte in payload:
        append_bits(bits, byte, 8)
    terminator = min(4, capacity_bits - len(bits))
    append_bits(bits, 0, terminator)
    while len(bits) % 8:
        bits.append(0)
    pad_bytes = [0xEC, 0x11]
    pad_index = 0
    while len(bits) < capacity_bits:
        append_bits(bits, pad_bytes[pad_index % 2], 8)
        pad_index += 1
    return bits_to_bytes(bits)


def add_function_patterns(
    modules: list[list[bool | None]],
    reserved: list[list[bool]],
    version: int,
) -> None:
    size = len(modules)
    add_finder(modules, reserved, 0, 0)
    add_finder(modules, reserved, size - 7, 0)
    add_finder(modules, reserved, 0, size - 7)

    for i in range(8, size - 8):
        set_module(modules, reserved, i, 6, i % 2 == 0, reserve=True)
        set_module(modules, reserved, 6, i, i % 2 == 0, reserve=True)

    for y in range(size):
        mark_reserved(reserved, 8, y)
    for x in range(size):
        mark_reserved(reserved, x, 8)

    for ay in ALIGNMENT_POSITIONS[version]:
        for ax in ALIGNMENT_POSITIONS[version]:
            if overlaps_finder(ax, ay, size):
                continue
            add_alignment(modules, reserved, ax, ay)

    set_module(modules, reserved, 8, 4 * version + 9, True, reserve=True)


def add_finder(
    modules: list[list[bool | None]],
    reserved: list[list[bool]],
    x0: int,
    y0: int,
) -> None:
    size = len(modules)
    for y in range(y0 - 1, y0 + 8):
        for x in range(x0 - 1, x0 + 8):
            if 0 <= x < size and 0 <= y < size:
                reserved[y][x] = True
                if x0 <= x < x0 + 7 and y0 <= y < y0 + 7:
                    dx = x - x0
                    dy = y - y0
                    modules[y][x] = (
                        dx in {0, 6}
                        or dy in {0, 6}
                        or (2 <= dx <= 4 and 2 <= dy <= 4)
                    )
                else:
                    modules[y][x] = False


def add_alignment(
    modules: list[list[bool | None]],
    reserved: list[list[bool]],
    cx: int,
    cy: int,
) -> None:
    for y in range(cy - 2, cy + 3):
        for x in range(cx - 2, cx + 3):
            dark = abs(x - cx) == 2 or abs(y - cy) == 2 or (x == cx and y == cy)
            set_module(modules, reserved, x, y, dark, reserve=True)


def overlaps_finder(cx: int, cy: int, size: int) -> bool:
    return (
        (cx <= 8 and cy <= 8)
        or (cx >= size - 9 and cy <= 8)
        or (cx <= 8 and cy >= size - 9)
    )


def place_data(
    modules: list[list[bool | None]],
    reserved: list[list[bool]],
    bits: list[int],
    *,
    mask: int,
) -> None:
    size = len(modules)
    bit_index = 0
    upward = True
    x = size - 1
    while x > 0:
        if x == 6:
            x -= 1
        rows = range(size - 1, -1, -1) if upward else range(size)
        for y in rows:
            for xx in [x, x - 1]:
                if reserved[y][xx]:
                    continue
                bit = bits[bit_index] if bit_index < len(bits) else 0
                if mask_condition(mask, xx, y):
                    bit ^= 1
                modules[y][xx] = bool(bit)
                bit_index += 1
        upward = not upward
        x -= 2


def add_format_bits(
    modules: list[list[bool | None]],
    reserved: list[list[bool]],
    *,
    mask: int,
) -> None:
    size = len(modules)
    bits = format_bits(mask)

    first = [
        (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7), (8, 8),
        (7, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8),
    ]
    second = [
        (size - 1, 8), (size - 2, 8), (size - 3, 8), (size - 4, 8),
        (size - 5, 8), (size - 6, 8), (size - 7, 8), (8, size - 8),
        (8, size - 7), (8, size - 6), (8, size - 5), (8, size - 4),
        (8, size - 3), (8, size - 2), (8, size - 1),
    ]
    for i, (x, y) in enumerate(first):
        set_module(modules, reserved, x, y, bool((bits >> i) & 1), reserve=True)
    for i, (x, y) in enumerate(second):
        set_module(modules, reserved, x, y, bool((bits >> i) & 1), reserve=True)


def format_bits(mask: int) -> int:
    data = (0b01 << 3) | mask  # ECL L.
    value = data << 10
    generator = 0b10100110111
    for shift in range(14, 9, -1):
        if (value >> shift) & 1:
            value ^= generator << (shift - 10)
    return ((data << 10) | value) ^ 0b101010000010010


def mask_condition(mask: int, x: int, y: int) -> bool:
    if mask == 0:
        return (x + y) % 2 == 0
    raise ValueError("unsupported mask")


def set_module(
    modules: list[list[bool | None]],
    reserved: list[list[bool]],
    x: int,
    y: int,
    value: bool,
    *,
    reserve: bool,
) -> None:
    modules[y][x] = value
    if reserve:
        reserved[y][x] = True


def mark_reserved(reserved: list[list[bool]], x: int, y: int) -> None:
    if 0 <= y < len(reserved) and 0 <= x < len(reserved):
        reserved[y][x] = True


def append_bits(bits: list[int], value: int, count: int) -> None:
    for i in range(count - 1, -1, -1):
        bits.append((value >> i) & 1)


def bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits), 8):
        value = 0
        for bit in bits[i:i + 8]:
            value = (value << 1) | bit
        out.append(value)
    return bytes(out)


def bytes_to_bits(data: bytes) -> list[int]:
    bits: list[int] = []
    for byte in data:
        append_bits(bits, byte, 8)
    return bits


def reed_solomon_ecc(data: bytes, degree: int) -> bytes:
    generator = rs_generator_poly(degree)
    message = [0] * degree
    for byte in data:
        factor = byte ^ message.pop(0)
        message.append(0)
        if factor:
            for i, coefficient in enumerate(generator):
                message[i] ^= gf_mul(coefficient, factor)
    return bytes(message)


def rs_generator_poly(degree: int) -> list[int]:
    result = [1]
    for i in range(degree):
        result = poly_mul(result, [1, gf_pow(2, i)])
    return result[1:]


def poly_mul(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            out[i + j] ^= gf_mul(av, bv)
    return out


def gf_pow(base: int, exponent: int) -> int:
    value = 1
    for _ in range(exponent):
        value = gf_mul(value, base)
    return value


def gf_mul(a: int, b: int) -> int:
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        if a & 0x100:
            a ^= 0x11D
        b >>= 1
    return result & 0xFF
