"""The backdrop map, served from this machine: vector tiles and label glyphs.

Both are fetched once by ``pf fetch basemap_dc basemap_fonts`` (see ``sources.py``). Serving
them here means the page asks no outside service for map tiles, so nobody else learns where
a user is looking. When they have not been fetched the page falls back to OpenStreetMap's
public tiles, which are for light development use only.
"""

from __future__ import annotations

import re
from pathlib import Path

from .fetch import raw_path
from .sources import REGION, SOURCES

_RANGE = re.compile(r"^\d+-\d+\.pbf$")


class Basemap:
    def __init__(self, tiles: Path | None = None, fonts: Path | None = None):
        from pmtiles.reader import MmapSource, Reader
        from pmtiles.tile import Compression

        self._file = (tiles or raw_path(REGION.basemap)).open("rb")
        self._reader = Reader(MmapSource(self._file))
        header = self._reader.header()
        self.max_zoom: int = header["max_zoom"]
        self.gzipped: bool = header["tile_compression"] == Compression.GZIP
        self.fonts = fonts or raw_path("basemap_fonts")

    @staticmethod
    def available() -> bool:
        if REGION.basemap is None or "basemap_fonts" not in SOURCES:
            return False
        return raw_path(REGION.basemap).is_file() and raw_path("basemap_fonts").is_dir()

    def tile(self, z: int, x: int, y: int) -> bytes | None:
        """One tile as stored (gzip-compressed when ``gzipped``), or None outside the extract."""
        if not (0 <= z <= self.max_zoom and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
            return None
        data = self._reader.get(z, x, y)
        return bytes(data) if data else None

    def glyphs(self, stack: str, name: str) -> bytes:
        """A glyph range for one font. A range that was not fetched is answered with no
        glyphs rather than an error, so one unusual character does not blank a tile's labels."""
        if not _RANGE.match(name):
            raise ValueError(name)
        folder = (self.fonts / stack).resolve()
        if folder.parent != self.fonts.resolve() or not folder.is_dir():
            raise ValueError(stack)
        path = folder / name
        return path.read_bytes() if path.is_file() else b""
