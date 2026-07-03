import lzma
import struct

# == modify osr ==
# Refer to: https://osu.ppy.sh/wiki/en/Client/File_formats/osr_(file_format)

def _read_uleb128(data: bytes, offset: int):
    result, shift = 0, 0
    while True:
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
    return result, offset


def _skip_string(data: bytes, offset: int) -> int:
    marker = data[offset]
    offset += 1
    if marker == 0x00:
        return offset
    if marker == 0x0B:
        length, offset = _read_uleb128(data, offset)
        return offset + length
    raise ValueError(f"Unexpected string marker {marker} at offset {offset}")

def _find_replay_data_offset(raw: bytes) -> int:
    offset = 1 + 4 # mode (byte) + game_version (int)
    for _ in range(3): # beatmap_hash, username, replay_hash
        offset = _skip_string(raw, offset)
    offset += 2 * 6 + 4 + 2 + 1 + 4 # 6 counts (short) + score (int) + max_combo (short) + perfect (byte) + mods (int)
    offset = _skip_string(raw, offset) # life_bar_graph
    offset += 8 # timestamp (long)
    return offset # (int) "Length in bytes of compressed replay data"

def _decrompress_replay_data(raw: bytes, offset: int):
    length = struct.unpack_from("<I", raw, offset)[0]
    comrpessed = raw[offset + 4 : offset + 4 + length]
    text = lzma.decompress(comrpessed, format=lzma.FORMAT_AUTO).decode("ascii")
    return text, offset + 4 + length

def _parse_frames(text: str):
    text = text.strip(",")
    frames = []
    for chunk in text.split(","):
        t, x, y, k = chunk.split("|")
        frames.append([int(t), float(x), float(y), int(k)])
    return frames

def _format_frames(frames) -> str:
    return ",".join(f"{t}|{x}|{y}|{k}" for t, x, y, k in frames) + ","


def _recompress(text: str) -> bytes:
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 21, "mode": lzma.MODE_FAST}]
    return lzma.compress(text.encode("ascii"), format=lzma.FORMAT_ALONE, filters=filters)

def _mods_offset(raw: bytes) -> int:
    offset = 1 + 4  # mode + game_version
    for _ in range(3):  # beatmap_hash, username, replay_hash
        offset = _skip_string(raw, offset)
    offset += 2 * 6 + 4 + 2 + 1  # 6 counts + score + max_combo + perfect
    return offset  # mods (int32)

def _read_string(raw: bytes, offset: int):
    marker = raw[offset]
    offset += 1
    if marker == 0x00:
        return None, offset
    length, offset = _read_uleb128(raw, offset)
    s = raw[offset : offset + length]
    return s, offset + length


def patch_osr(raw: bytes, transform) -> bytes:
    """
    transform(frames: list[[t, x, y, k]]) -> frames modified only
    """
    header_end = _find_replay_data_offset(raw)
    text, data_end = _decrompress_replay_data(raw, header_end)
    tail = raw[data_end:]

    frames = _parse_frames(text)
    frames = transform(frames)

    compressed = _recompress(_format_frames(frames))
    new_blob = struct.pack("<I", len(compressed)) + compressed

    return raw[:header_end] + new_blob + tail

def set_mods(raw: bytes, mods_value: int) -> bytes:
    off = _mods_offset(raw)
    return raw[:off] + struct.pack("<I", mods_value) + raw[off + 4:]

def set_replay_hash(raw: bytes, new_hash: str) -> bytes:
    offset = 1 + 4  # mode + game_version
    offset = _skip_string(raw, offset)  # skip beatmap_hash
    offset = _skip_string(raw, offset)  # skip username
    
    _, end_offset = _read_string(raw, offset)
    
    encoded_hash = new_hash.encode('utf-8')
    val = len(encoded_hash)
    res = bytearray()
    while True:
        byte = val & 0x7F
        val >>= 7
        if val != 0:
            res.append(byte | 0x80)
        else:
            res.append(byte)
            break
            
    new_str_bytes = b'\x0b' + bytes(res) + encoded_hash
    return raw[:offset] + new_str_bytes + raw[end_offset:]

def get_beatmap_replay_hash(raw: bytes) -> list[str, str]:
    """
    Returns:
        beatmap_hash (str): the beatmap hash
        replay_hash  (str): the replay hash
    """
    offset = 1 + 4 # mode + game_version

    b_hash, offset = _read_string(raw, offset)
    _, offset = _read_string(raw, offset) # username
    r_hash, offset = _read_string(raw, offset)

    return b_hash.decode(), r_hash.decode()

 