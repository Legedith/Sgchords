import os
import shutil
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match in {path}, found {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "scripts/benchmark_known_chords.py",
    "import json\nimport shutil\n",
    "import json\nimport os\nimport shutil\n",
)
replace_once(
    "scripts/benchmark_known_chords.py",
    '''    media = downloads / f"{song.slug}.ogg"
    download(song.file_url, media)

    started = time.perf_counter()
''',
    '''    media = downloads / f"{song.slug}.ogg"
    download(song.file_url, media)
    if os.getenv("SGCHORDS_BENCHMARK_KEEP_AUDIO") == "1":
        audio_dir = output / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(media, audio_dir / media.name)

    started = time.perf_counter()
''',
)
