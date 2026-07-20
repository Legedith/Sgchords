from pathlib import Path

manifest = Path(".bootstrap-v2/expected-sha256.txt")
skip = {
    "src/sgchords/service.py",
    "src/sgchords/ui.py",
}
manifest.write_text(
    "\n".join(
        line
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if not any(line.endswith(f"  {path}") for path in skip)
    )
    + "\n",
    encoding="utf-8",
)
