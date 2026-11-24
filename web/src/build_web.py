"""
Builds the single-page web course:
    web/src/{styles.css, content.html, *.js}  ->  web/dist/index.html
    web/model/*.bin                           ->  web/dist/model/*.b64.txt (+ manifest.json)
Run:  python3 web/src/build_web.py
"""
import base64
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent
WEB = SRC.parent
DIST = WEB / "dist"
JS = ["engine.js", "core.js", "widgets_basics.js", "widgets_attn.js", "widgets_model.js", "python.js", "learn.js", "toy.js", "lessons_engine.js", "lessons_content.js"]

HEAD = """<meta charset="utf-8">
<title>Inside the Transformer</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="description" content="An interactive course on embeddings and attention, built around a real Transformer running in your browser.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,650;12..96,700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Serif:ital@1&display=swap">
"""


def main():
    DIST.mkdir(exist_ok=True)
    css = (SRC / "styles.css").read_text()
    body = (SRC / "content.html").read_text()
    js = "\n".join((SRC / f).read_text() for f in JS)
    boot = "\ndocument.addEventListener('DOMContentLoaded', () => Router.init());\nif (document.readyState !== 'loading') Router.init();\n"
    page = HEAD + "<style>\n" + css + "</style>\n" + body + "\n<script>\n" + js + boot + "</script>\n"
    (DIST / "index.html").write_text(page)
    (DIST / "probe.json").unlink(missing_ok=True)
    out = DIST / "model"
    out.mkdir(exist_ok=True)
    files = {}
    for b in sorted((WEB / "model").glob("*.bin")):
        name = b.stem + ".b64.txt"
        (out / name).write_text(base64.b64encode(b.read_bytes()).decode())
        files["model/" + name] = str(out / name)
    (out / "manifest.json").write_text((WEB / "model" / "manifest.json").read_text())
    files["model/manifest.json"] = str(out / "manifest.json")
    (DIST / "files.json").write_text(json.dumps(files, indent=1))
    print(f"index.html {len(page) / 1024:.0f} KB; {len(files)} model files")


if __name__ == "__main__":
    main()
