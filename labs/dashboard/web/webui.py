"""
Shared assets for the lab dashboards: styles, fonts, logos, icons, charts.

Vendored from racecar_neo_ros2_driver scripts/web/ and scripts/webui.py
(upstream 7491b0d) so the sim dashboards match the car's. Each dashboard
serves this folder under /ui/ through static_file(), so pages work offline.
Only the files the lab pages load are copied; battery.js, lost.js and
override.js talk to the car's 8080 home page, which the simulator lacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import unquote, urlparse

WEB_DIR = Path(__file__).resolve().parent
PREFIX = "/ui/"
CACHE_CONTROL = "max-age=300"
# Pages link the shared files as /ui/<file>?v=ASSET_VERSION; bump it when
# rcn.css or charts.js changes, so browsers fetch the new copy at once.
ASSET_VERSION = "13"
TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}


def static_file(request_path: str, root: Path = WEB_DIR) -> Optional[Tuple[bytes, str]]:
    """Return (body, content type) for a /ui/ path, or None when it is not one we serve."""
    path = unquote(urlparse(request_path).path)
    if not path.startswith(PREFIX):
        return None
    target = (root / path[len(PREFIX):]).resolve()
    if root.resolve() not in target.parents or not target.is_file():
        return None
    kind = TYPES.get(target.suffix)
    if kind is None:
        return None
    return target.read_bytes(), kind
