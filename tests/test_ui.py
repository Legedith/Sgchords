from sgchords.ui import SyncedChordViewer, build_demo


def test_synced_viewer_and_ui_build() -> None:
    viewer = SyncedChordViewer(
        value={
            "segments": [
                {
                    "start": 0,
                    "end": 2,
                    "time": "00:00.0",
                    "display": "C",
                    "secondary": "bar 1",
                    "confidence": 0.9,
                }
            ]
        }
    )
    assert viewer is not None
    assert build_demo() is not None
