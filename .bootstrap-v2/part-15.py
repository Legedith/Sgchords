from pathlib import Path

path = Path("src/sgchords/analyzer.py")
text = path.read_text(encoding="utf-8")
old = "    radius = max(1, int(round(median_gap * ratio))\n"
new = "    radius = max(1, int(round(median_gap * ratio)))\n"
if text.count(old) != 1:
    raise RuntimeError(f"Expected one malformed boundary-refinement line, found {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
