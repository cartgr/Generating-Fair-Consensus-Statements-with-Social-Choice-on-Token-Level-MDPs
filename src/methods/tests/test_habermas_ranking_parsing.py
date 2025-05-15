import pytest
import numpy as np
from ..habermas_machine import (
    _hm_check_response_format,
    _hm_check_arrow_format,
    _hm_extract_arrow_ranking,
    _hm_process_ranking_response,
)

# --- Tests for _hm_check_response_format ---


# Updated to match the two specific cases in the Google test file
def test_check_response_format():
    """Tests the _check_response_format method (aligned with Google test)."""
    assert (
        _hm_check_response_format("<answer>Explanation\n<sep>\nA > B > C</answer>")
        is True
    )
    assert _hm_check_response_format("Explanation\nA > B > C") is False  # Missing tags


# --- Tests for _hm_check_arrow_format ---
# Note: Aligned with Google test cases.
# Assumes num_statements based on the letters present for the user's function.


@pytest.mark.parametrize(
    "ranking_str, num_statements, expected",
    [
        # Correct formats from Google test
        ("A>B>C", 3, True),
        ("A=B>C>D", 4, True),
        ("A>B=C=D>E", 5, True),
        ("A=B=C", 3, True),
        # Incorrect formats from Google test
        ("A<B>C", 3, False),
        ("A>>B>C", 3, False),
        ("A>B>A", 3, False),  # Duplicate letter (Set check should fail)
        ("A>B=B>C", 3, False),  # Duplicate letter (Set check should fail)
        ("A>B>C>B", 4, False),  # Duplicate letter (Set check should fail)
        ("A>>B", 2, False),
        ("A>B>>C", 3, False),
        ("A=>B", 2, False),
        ("A>B>", 2, False),  # Check should fail (trailing >)
        (">A>B", 2, False),  # Check should fail (leading >)
        ("A=B=>C", 3, False),
        ("A>B=", 2, False),  # Check should fail (trailing =)
        ("A=>B>C", 3, False),
        # Add cases for letter mismatch (implicitly tested by Google's set logic, explicit here)
        ("A>C", 3, False),  # Missing B
        ("A>B>C>D", 3, False),  # Extra D
        (
            "",
            0,
            False,
        ),  # Empty string (Google test doesn't explicitly test num_statements=0)
    ],
)
def test_check_arrow_format(ranking_str, num_statements, expected):
    """Tests the _hm_check_arrow_format method (aligned with Google test cases)."""
    # Note: The Google test's _check_arrow_format likely doesn't take num_statements.
    # We provide it here because the user's _hm_check_arrow_format requires it.
    assert _hm_check_arrow_format(ranking_str, num_statements) is expected


# --- Tests for _hm_extract_arrow_ranking ---


@pytest.mark.parametrize(
    "text, expected_ranking",
    [
        # Cases from Google test file
        (
            "Explanation\nA > B > C",
            "A>B>C",
        ),
        (
            "Explanation\n  A  >  B  >  C",
            "A>B>C",
        ),
        (
            "Explanation\n  A  =  B  >  C",
            "A=B>C",
        ),
        (
            "Explanation\nA > B < C > D",  # Google test expects partial match
            "A>B",
        ),
        (
            "Explanation",
            None,
        ),
    ],
)
def test_extract_arrow_ranking(text, expected_ranking):
    """Tests the _hm_extract_arrow_ranking method (aligned with Google test)."""
    extracted = _hm_extract_arrow_ranking(text)
    assert extracted == expected_ranking


# --- Tests for _hm_process_ranking_response ---


@pytest.mark.parametrize(
    "response, num_statements, expected_ranking_arr, expected_explanation",
    [
        # Cases from Google test file
        (
            "<answer>Explanation\n<sep>\nB>A=D>C</answer>",
            4,  # num_statements derived from expected_ranking length in Google test
            np.array([1, 0, 2, 1]),  # B=0, A=1, C=2, D=1
            "<answer>Explanation\n<sep>\nB>A=D>C</answer>",
        ),
        (
            "Explanation\nB>A=D>C",  # Incorrect template
            4,  # num_statements needed for check if template were correct
            None,
            "INCORRECT_TEMPLATE: Explanation\nB>A=D>C",  # Google format: prefix + original
        ),
        (
            "<answer>Explanation\n<sep>\nB<A=D>C</answer>",  # Incorrect arrow ranking
            4,  # num_statements needed for check if ranking were correct
            None,
            "INCORRECT_ARROW_RANKING: <answer>Explanation\n<sep>\nB<A=D>C</answer>",  # Google format: prefix + original
        ),
        (
            "Final ranking: B>A=D>C",  # Backup template
            4,  # num_statements derived from expected_ranking length in Google test
            np.array([1, 0, 2, 1]),  # B=0, A=1, C=2, D=1
            "Final ranking: B>A=D>C",
        ),
        (
            "<answer>Explanation\n<sep>\nA=B=C=D</answer>",  # All tied
            4,  # num_statements derived from expected_ranking length in Google test
            np.array([0, 0, 0, 0]),
            "<answer>Explanation\n<sep>\nA=B=C=D</answer>",
        ),
        # Add a case corresponding to Google's 'Incorrect arrow ranking' with duplicate/missing letters
        # This wasn't explicitly in the Google _process_model_response test parameters,
        # but is implied by the _check_arrow_format test.
        (
            "<answer>Explanation\n<sep>\nB>A>B</answer>",  # Duplicate B, missing C/D
            4,  # Assuming A,B,C,D were expected
            None,
            "INCORRECT_ARROW_RANKING: <answer>Explanation\n<sep>\nB>A>B</answer>",  # Google format: prefix + original
        ),
        (
            "<answer>Explanation\n<sep>\nA>C</answer>",  # Missing B/D
            4,  # Assuming A,B,C,D were expected
            None,
            "INCORRECT_ARROW_RANKING: <answer>Explanation\n<sep>\nA>C</answer>",  # Google format: prefix + original
        ),
    ],
)
def test_process_model_response(  # Renamed test function
    response, num_statements, expected_ranking_arr, expected_explanation
):
    """Tests _hm_process_ranking_response (aligned with Google test logic)."""
    # Note: Google test checks result.ranking and result.explanation.
    # User's function returns a tuple (ranking, explanation).
    ranking, explanation = _hm_process_ranking_response(response, num_statements)

    # Use np.testing for array comparison, allow None comparison
    if expected_ranking_arr is None:
        assert ranking is None, f"Expected ranking None but got {ranking}."
    else:
        assert ranking is not None, f"Expected ranking array but got None."
        np.testing.assert_array_equal(ranking, expected_ranking_arr)

    # Check explanation matches exactly (including prefix + original for errors)
    assert (
        explanation == expected_explanation
    ), f"Explanation mismatch.\nExpected: {expected_explanation}\nGot:      {explanation}"


# Remove the main execution block if present, as pytest handles test discovery
# if __name__ == '__main__':
#     pytest.main() # Or however tests are run in this project
