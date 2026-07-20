from sgchords.audio import is_youtube_url


def test_url_validation() -> None:
    assert is_youtube_url("https://youtu.be/abc")
    assert is_youtube_url("https://www.youtube.com/watch?v=abc")
    assert not is_youtube_url("https://example.com/video")
