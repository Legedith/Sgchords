from sgchords.cli import build_parser


def test_cli_options() -> None:
    args = build_parser().parse_args(
        [
            "song.wav",
            "--detail",
            "detailed",
            "--meter",
            "6",
            "--transpose",
            "2",
            "--capo",
            "3",
            "--notation",
            "nashville",
            "--instrument",
            "ukulele",
        ]
    )
    assert (args.detail, args.meter, args.transpose, args.capo, args.notation, args.instrument) == (
        "detailed",
        "6",
        2,
        "3",
        "nashville",
        "ukulele",
    )
