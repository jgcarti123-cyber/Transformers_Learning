"""
Builds the course from tools/src/  (you never need to run this - it was used to generate the project).

    tools/src/minitf/*.py         ->  minitf/ (your TODO version)  +  solutions/minitf/ + minitf/_reference/
    tools/src/notebooks/*.nbsrc   ->  notebooks/*.ipynb (TODO)     +  solutions/notebooks/*.ipynb

Solution blocks in the sources look like:

    ### TODO: message shown to the student
    solution code ...
    #> optional starter line(s) shown to the student instead of `raise NotImplementedError`
    ### END

Your own files (minitf/attention.py etc. and notebooks/*.ipynb) are NEVER overwritten unless you pass --force.
"""
import argparse
import re
import shutil
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tools" / "src"
STUDENT_MODULES = ["attention.py", "positional.py", "transformer.py"]

TODO_RE = re.compile(r"^(\s*)### TODO(?::\s*(.*))?$")
END_RE = re.compile(r"^\s*### END\s*$")
STUB_RE = re.compile(r"^(\s*)#> ?")


def render(text: str, mode: str) -> str:
    assert mode in ("student", "solution")
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        m = TODO_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent, msg = m.group(1), (m.group(2) or "your code here").strip()
        block, i = [], i + 1
        while not END_RE.match(lines[i]):
            block.append(lines[i])
            i += 1
        i += 1
        stub = [l for l in block if STUB_RE.match(l)]
        sol = [l for l in block if not STUB_RE.match(l)]
        if mode == "solution":
            out += sol
        elif stub:
            out += [STUB_RE.sub(r"\1", l) for l in stub]
        else:
            out.append(f'{indent}raise NotImplementedError("TODO: {msg}")')
    return "\n".join(out)


def parse_nbsrc(text: str):
    cells, kind, buf = [], None, []

    def flush():
        if kind is not None:
            src = "\n".join(buf).strip("\n")
            if src.strip():
                cells.append((kind, src))

    for line in text.split("\n"):
        if line.startswith("#%% md") or line.startswith("#%% code"):
            flush()
            kind, buf = ("markdown" if line.startswith("#%% md") else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def make_notebook(cells, mode):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for kind, src in cells:
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(render(src, mode)))
    return nb


def write(path: Path, content: str, protect: bool, force: bool):
    if protect and path.exists() and not force:
        print(f"  skip (exists, yours): {path.relative_to(ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  wrote {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite YOUR files too (destroys your work!)")
    args = ap.parse_args()

    for f in sorted((SRC / "minitf").glob("*.py")):
        text = f.read_text()
        if f.name in STUDENT_MODULES:
            write(ROOT / "minitf" / f.name, render(text, "student"), protect=True, force=args.force)
            sol = render(text, "solution")
            write(ROOT / "minitf" / "_reference" / f.name, sol, protect=False, force=True)
            write(ROOT / "solutions" / "minitf" / f.name, sol, protect=False, force=True)
            write(ROOT / "solutions" / "minitf" / "_reference" / f.name, sol, protect=False, force=True)
        else:
            for dest in (ROOT / "minitf", ROOT / "solutions" / "minitf"):
                write(dest / f.name, text, protect=False, force=True)
    ref_init = '"""Reference solutions used by minitf.check. Try not to read them before you finish each part!"""\n'
    for dest in (ROOT / "minitf" / "_reference", ROOT / "solutions" / "minitf" / "_reference"):
        write(dest / "__init__.py", ref_init, protect=False, force=True)

    for f in sorted((SRC / "notebooks").glob("*.nbsrc")):
        cells = parse_nbsrc(f.read_text())
        name = f.stem + ".ipynb"
        write(ROOT / "notebooks" / name, nbformat.writes(make_notebook(cells, "student")),
              protect=True, force=args.force)
        write(ROOT / "solutions" / "notebooks" / name, nbformat.writes(make_notebook(cells, "solution")),
              protect=False, force=True)


if __name__ == "__main__":
    main()
