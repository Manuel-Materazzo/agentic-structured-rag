import argparse


def get_submission_parser():
    parser = argparse.ArgumentParser(
        description="Export submission files for the dish mapping task."
    )
    parser.add_argument(
        "--output_path",
        help="Path where the output CSV will be written.",
    )
    parser.add_argument(
        "--questions_path",
        help="Path to the questions CSV file.",
    )
    parser.add_argument(
        "--answers_path",
        default=None,
        help="Optional path to the answers CSV file. If omitted, an empty submission is generated.",
    )

    return parser
