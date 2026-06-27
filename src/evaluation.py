import sys

import click
import pandas as pd

from metrics.jaccard_similarity import score


@click.command()
@click.option(
    "--submission",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    help="Path to the submission CSV file to evaluate.",
)
@click.option(
    "--row-id-column",
    default="row_id",
    show_default=True,
    help=("Name of the column used to align rows between solution and submission."),
)
def main(submission: str, row_id_column: str) -> None:
    """Evaluate a submission CSV using the Jaccard similarity metric used in Kaggle."""

    solution_path = "./dataset/ground_truth/ground_truth_mapped.csv"

    # Load data
    solution_df = pd.read_csv(solution_path)
    submission_df = pd.read_csv(submission)

    try:
        jaccard_score = score(
            solution=solution_df,
            submission=submission_df,
            row_id_column_name=row_id_column,
        )
    except ValueError as e:
        click.echo(f"Invalid submission: {e}", err=True)
        sys.exit(1)

    click.echo(f"Jaccard similarity score: {jaccard_score:.4f}")


if __name__ == "__main__":
    main()
