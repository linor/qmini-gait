"""
usda_min.py -- a small, tolerant reader for USD ASCII.

Only what the Blender round-trip needs: the prim tree, scalar/array attributes,
and attribute timeSamples.

This is the FALLBACK path.  When pxr is importable, blender_io uses it instead --
``UsdGeom.Xformable.GetLocalTransformation`` composes whatever ops a DCC wrote
and handles binary layers, which this cannot.  Keep this only for environments
without pxr; ``blender_io.py selftest`` checks the two against each other.

Deliberately forgiving about layout -- Blender may write an ``xformOp:transform``
matrix where we wrote translate + rotateXYZ, may reorder the ops, and may add
prims of its own.  Binary .usdc is NOT supported here.
"""

import re
import numpy as np

_NUM = r"[-+0-9.eEinf]+"


class Prim:
    def __init__(self, spec="", name="", parent=None):
        self.spec, self.name, self.parent = spec, name, parent
        self.type = spec.split()[1] if len(spec.split()) > 1 else ""
        self.children, self.attrs, self.samples = [], {}, {}

    @property
    def path(self):
        if self.parent is None or not self.parent.name and self.parent.parent is None:
            return "/" + self.name
        return self.parent.path + "/" + self.name

    def find(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def __repr__(self):
        return f"<Prim {self.path} {self.type} attrs={len(self.attrs)} ts={len(self.samples)}>"


def _split_top(s):
    """split a bracketed list on commas that are not inside ( ) or [ ]"""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def parse_value(txt):
    txt = txt.strip()
    if not txt:
        return None
    if txt.startswith('"'):
        return txt.strip('"')
    if txt[0] == "[":
        return [parse_value(x) for x in _split_top(txt[1:-1])]
    if txt[0] == "(":
        inner = txt[1:-1].strip()
        parts = _split_top(inner)
        if parts and parts[0].strip().startswith("("):        # matrix
            return np.array([[float(v) for v in re.findall(_NUM, p)] for p in parts])
        return np.array([float(v) for v in parts])
    if txt in ("true", "false"):
        return txt == "true"
    try:
        return float(txt)
    except ValueError:
        return txt


def _balanced(src, i):
    """index just past the block whose opening brace is at/after i"""
    j = src.index("{", i)
    d = 0
    while True:
        if src[j] == "{":
            d += 1
        elif src[j] == "}":
            d -= 1
            if d == 0:
                return j
        j += 1


def parse(path_or_text):
    """-> (root Prim with children = the top-level prims, layer metadata dict)"""
    src = path_or_text
    if "\n" not in src and len(src) < 4096:
        src = open(src).read()
    if src.lstrip().startswith("#usda") is False and "def " not in src:
        raise ValueError("not a USD ASCII layer (binary .usdc is not supported)")

    meta = {}
    m = re.match(r"#usda[^\n]*\n\s*\((.*?)\n\)\n", src, re.S)
    if m:
        for line in m.group(1).splitlines():
            kv = re.match(r'\s*(\w+)\s*=\s*(.+?)\s*$', line)
            if kv:
                meta[kv.group(1)] = parse_value(kv.group(2))

    root = Prim("root", "", None)

    def body(text, parent):
        i = 0
        while True:
            m = re.compile(r'\b(?:def|over|class)\b[^\n{]*"[^"\n]+"[^\n{]*').search(text, i)
            if not m:
                break
            spec = m.group(0)
            nm = re.findall(r'"([^"]+)"', spec)[-1]
            try:
                end = _balanced(text, m.end())
            except ValueError:
                break
            start = text.index("{", m.end()) + 1
            p = Prim(spec, nm, parent if parent.name or parent.parent else parent)
            p.parent = parent
            parent.children.append(p)
            inner = text[start:end]
            attrs(inner, p)
            body(inner, p)
            i = end + 1

    def attrs(text, p):
        # strip nested prim blocks so their attributes are not stolen
        clean, i = "", 0
        while True:
            m = re.compile(r'\b(?:def|over|class)\b[^\n{]*"[^"\n]+"[^\n{]*').search(text, i)
            if not m:
                clean += text[i:]
                break
            clean += text[i:m.start()]
            try:
                i = _balanced(text, m.end()) + 1
            except ValueError:
                break
        # timeSamples
        for m in re.finditer(r'([\w:\[\]]+)\.timeSamples\s*=\s*\{', clean):
            name = m.group(1)
            e = _balanced(clean, m.end() - 1)
            d = {}
            for line in clean[clean.index("{", m.end() - 1) + 1:e].splitlines():
                line = line.strip().rstrip(",")
                if not line:
                    continue
                k, _, v = line.partition(":")
                try:
                    d[float(k)] = parse_value(v)
                except ValueError:
                    pass
            p.samples[name.split(".")[0]] = d
        # plain attributes.  A value may wrap over several lines, so consume
        # lines until the brackets balance rather than guessing with a regex.
        lines = clean.splitlines()
        i = 0
        pat = re.compile(r'^\s*(?:custom\s+)?(?:uniform\s+)?[\w\[\]]+\s+([\w:.]+)\s*=\s*(.*)$')
        bal = lambda t: (t.count("(") - t.count(")")) + (t.count("[") - t.count("]"))
        while i < len(lines):
            m = pat.match(lines[i])
            i += 1
            if not m:
                continue
            nm, val = m.group(1), m.group(2)
            if nm.endswith(".timeSamples") or "timeSamples" in nm:
                continue
            while bal(val) > 0 and i < len(lines):
                val += " " + lines[i].strip()
                i += 1
            try:
                p.attrs[nm] = parse_value(val)
            except Exception:
                p.attrs[nm] = val.strip()

    body(src, root)
    return root, meta


# --------------------------------------------------------------------- writing
def fmt(v):
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return "true" if v else "false"
    a = np.asarray(v)
    if a.ndim == 0:
        return f"{float(a):.9g}"
    if a.ndim == 1:
        return "(" + ", ".join(f"{float(x):.9g}" for x in a) + ")"
    return "( " + ", ".join(fmt(r) for r in a) + " )"


def time_samples(name, kind, d, indent):
    pad = " " * indent
    out = [f"{pad}{kind} {name}.timeSamples = {{"]
    for t in sorted(d):
        out.append(f"{pad}    {t:g}: {fmt(d[t])},")
    out.append(pad + "}")
    return "\n".join(out)
