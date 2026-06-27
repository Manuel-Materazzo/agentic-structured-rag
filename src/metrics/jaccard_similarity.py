import pandas as pd
import pandas.api.types


def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str
) -> float:
    """
    This metric calculates the average Jaccard similarity between corresponding list columns
    in the solution and submission dataframes. The submission values are expected to be
    comma-separated strings that will be converted to lists of integers.

    The Jaccard similarity is computed as:
    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
    - solution (pd.DataFrame): The dataframe containing the ground truth values.
    - submission (pd.DataFrame): The dataframe containing predictions as comma-separated strings.
    - row_id_column_name (str): The name of the column used to align rows in the dataframes.

    Returns:
    - float: The average Jaccard similarity over all rows.
    """
    # Make copies to avoid modifying the original dataframes
    solution = solution.copy()
    submission = submission.copy()

    # Remove the row ID column to avoid affecting the metric
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    def string_to_list(value):
        """Convert a string value to a list of integers."""
        if pd.isna(value) or value == "":
            return []
        if isinstance(value, (int, float)):
            return [int(value)]
        return [int(x.strip()) for x in str(value).split(",")]

    # Convert submission strings to lists
    for col in submission.columns:
        submission[col] = submission[col].apply(string_to_list)

    # Convert solution values to lists if they aren't already
    for col in solution.columns:
        if col != "Usage":
            if not all(isinstance(x, list) for x in solution[col].dropna()):
                solution[col] = solution[col].apply(string_to_list)

    # Check that all submission columns contain valid lists
    for col in submission.columns:
        if not pandas.api.types.is_object_dtype(submission[col]):
            raise ValueError(f"Submission column {col} must contain lists of integers")
        try:
            # Check that all non-empty values are lists of integers
            valid_lists = submission[col].apply(
                lambda x: isinstance(x, list) and all(isinstance(i, int) for i in x)
            )
            if not valid_lists.all():
                raise ValueError(
                    f"Submission column {col} must contain valid lists of integers"
                )
        except Exception as e:
            raise ValueError(f"Error processing column {col}: {str(e)}")

    # Define a helper function for Jaccard similarity
    def jaccard_similarity(list1, list2):
        if not list1 and not list2:  # Both empty
            return 1.0
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union != 0 else 0.0

    # Compute Jaccard similarity row-wise for all columns
    similarities = []
    for col in submission.columns:
        col_similarities = submission[col].combine(
            solution[col],
            func=lambda sub_list, sol_list: jaccard_similarity(sub_list, sol_list),
        )
        similarities.append(col_similarities)

    # Calculate the average similarity across all rows and columns
    overall_similarity = pd.concat(similarities, axis=1).mean().mean()
    return float(overall_similarity) * 100
