"""Download one real RadarScenes sequence without fetching the 11 GB archive.

The official Zenodo object supports HTTP byte ranges.  This utility reads the
ZIP central directory remotely and extracts only the selected sequence's HDF5
and metadata files.  Sequence 158 is about 8 MB after extraction.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import ssl
import struct
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

import certifi


ARCHIVE_URL = "https://zenodo.org/api/records/4559821/files/RadarScenes.zip/content"
ARCHIVE_MD5 = "db4874b43e45faa0d635bafc58d56579"
DOI = "10.5281/zenodo.4559821"
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int


def _request(request: urllib.request.Request, attempts: int = 5) -> bytes:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120, context=SSL_CONTEXT) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt + 1 == attempts:
                raise
            time.sleep(float(error.headers.get("Retry-After", 2 ** attempt)))
    raise RuntimeError("unreachable")


def _content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "radar-tracker/1.0"})
    with urllib.request.urlopen(request, timeout=60, context=SSL_CONTEXT) as response:
        return int(response.headers["Content-Length"])


def _range(url: str, begin: int, end: int) -> bytes:
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={begin}-{end}", "User-Agent": "radar-tracker/1.0"},
    )
    data = _request(request)
    expected = end - begin + 1
    if len(data) != expected:
        raise RuntimeError(f"server returned {len(data)} bytes for a {expected}-byte range")
    return data


def _zip_entries(url: str) -> dict[str, ZipEntry]:
    length = _content_length(url)
    tail_begin = max(0, length - 65536)
    tail = _range(url, tail_begin, length - 1)
    locator_index = tail.rfind(b"PK\x06\x07")
    if locator_index < 0:
        raise RuntimeError("ZIP64 locator not found")
    _, _, zip64_offset, _ = struct.unpack_from("<4sLQL", tail, locator_index)
    zip64 = _range(url, zip64_offset, zip64_offset + 55)
    if zip64[:4] != b"PK\x06\x06":
        raise RuntimeError("ZIP64 end record not found")
    values = struct.unpack_from("<4sQ2H2L4Q", zip64)
    central_size, central_offset = values[-2], values[-1]
    central = _range(url, central_offset, central_offset + central_size - 1)

    entries: dict[str, ZipEntry] = {}
    offset = 0
    while offset + 46 <= len(central) and central[offset:offset + 4] == b"PK\x01\x02":
        values = struct.unpack_from("<4s6H3L5H2L", central, offset)
        flags, method, crc32 = values[3], values[4], values[7]
        compressed, uncompressed = values[8], values[9]
        name_size, extra_size, comment_size = values[10:13]
        local_offset = values[-1]
        name_bytes = central[offset + 46:offset + 46 + name_size]
        name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437")
        extra = central[offset + 46 + name_size:offset + 46 + name_size + extra_size]
        extra_offset = 0
        while extra_offset + 4 <= len(extra):
            field_id, field_size = struct.unpack_from("<HH", extra, extra_offset)
            field = extra[extra_offset + 4:extra_offset + 4 + field_size]
            extra_offset += 4 + field_size
            if field_id == 1:
                cursor = 0
                if uncompressed == 0xFFFFFFFF:
                    uncompressed = struct.unpack_from("<Q", field, cursor)[0]
                    cursor += 8
                if compressed == 0xFFFFFFFF:
                    compressed = struct.unpack_from("<Q", field, cursor)[0]
                    cursor += 8
                if local_offset == 0xFFFFFFFF:
                    local_offset = struct.unpack_from("<Q", field, cursor)[0]
        entries[name] = ZipEntry(name, method, crc32, compressed, uncompressed, local_offset)
        offset += 46 + name_size + extra_size + comment_size
    return entries


def _extract(url: str, entry: ZipEntry) -> bytes:
    header = _range(url, entry.local_offset, entry.local_offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"local ZIP header missing for {entry.name}")
    values = struct.unpack_from("<4s5H3L2H", header)
    name_size, extra_size = values[-2], values[-1]
    begin = entry.local_offset + 30 + name_size + extra_size
    compressed = _range(url, begin, begin + entry.compressed_size - 1)
    if entry.method == 0:
        data = compressed
    elif entry.method == 8:
        data = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"unsupported ZIP method {entry.method} for {entry.name}")
    if len(data) != entry.uncompressed_size:
        raise RuntimeError(f"size mismatch for {entry.name}")
    if binascii.crc32(data) & 0xFFFFFFFF != entry.crc32:
        raise RuntimeError(f"CRC mismatch for {entry.name}")
    return data


def download_sample(output_dir: str | Path, sequence: int = 158, force: bool = False) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = {
        "DATASET_README.md": "RadarScenes/Readme.md",
        "sensors.json": "RadarScenes/data/sensors.json",
        "scenes.json": f"RadarScenes/data/sequence_{sequence}/scenes.json",
        "radar_data.h5": f"RadarScenes/data/sequence_{sequence}/radar_data.h5",
    }
    entries = _zip_entries(ARCHIVE_URL)
    hashes: dict[str, str] = {}
    for output_name, archive_name in names.items():
        target = output / output_name
        if force or not target.exists():
            target.write_bytes(_extract(ARCHIVE_URL, entries[archive_name]))
        hashes[output_name] = hashlib.sha256(target.read_bytes()).hexdigest()
        print(f"{target} ({target.stat().st_size} bytes)")
    provenance = {
        "dataset": "RadarScenes",
        "doi": DOI,
        "record_url": "https://zenodo.org/records/4559821",
        "archive_url": ARCHIVE_URL,
        "archive_md5_from_zenodo": ARCHIVE_MD5,
        "sequence": sequence,
        "license": "CC BY-NC-SA 4.0",
        "extracted_sha256": hashes,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=158)
    parser.add_argument("--output-dir", type=Path, default=Path("real_data/radarscenes_sequence_158"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_sample(args.output_dir, args.sequence, args.force)


if __name__ == "__main__":
    main()
