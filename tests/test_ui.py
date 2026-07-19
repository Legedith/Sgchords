from sgchords.ui import build_demo


def test_ui_builds() -> None:
    demo = build_demo()
    assert demo is not None
