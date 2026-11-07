"""Static asset routing for labs/web/webui.py.

    pytest -q labs/web/tests
"""

import re
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEB))
import webui  # noqa: E402


def test_serves_a_stylesheet_with_its_type() -> None:
    body, kind = webui.static_file("/ui/rcn.css?v=13")
    assert kind.startswith("text/css")
    assert b"--rcn-orange" in body


def test_ignores_paths_outside_the_prefix() -> None:
    assert webui.static_file("/") is None
    assert webui.static_file("/state") is None


def test_refuses_traversal_and_unknown_types() -> None:
    assert webui.static_file("/ui/../wallfollow_dashboard/wallfollow.yaml") is None
    assert webui.static_file("/ui/%2e%2e/web/webui.py") is None
    assert webui.static_file("/ui/webui.py") is None     # .py is not a served type
    assert webui.static_file("/ui/missing.css") is None


def test_every_font_the_stylesheet_names_exists() -> None:
    css = (WEB / "rcn.css").read_text()
    fonts = re.findall(r'url\("([^"]+)"\)', css)
    assert fonts
    for f in fonts:
        assert webui.static_file(f"/ui/{f}") is not None, f
