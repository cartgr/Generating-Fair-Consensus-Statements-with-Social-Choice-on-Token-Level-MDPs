# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for the Schulze method implementation in habermas_machine.py"""

import pytest
import numpy as np
import dataclasses

# Import the Schulze helper functions directly from habermas_machine
from ..habermas_machine import (
    _schulze_compute_pairwise_defeats,
    _schulze_compute_strongest_path_strengths,
    _schulze_rank_candidates,
    _schulze_aggregate_with_ties,
    _schulze_check_rankings,  # Import check function for error tests if needed
    _aggregate_schulze,  # Import the main aggregation function with tie-breaking
)

# Import non-Schulze helpers being tested
from ..habermas_machine import (
    _hm_check_response_format,
    _hm_check_arrow_format,
    _hm_extract_arrow_ranking,
    _hm_process_ranking_response,
)


# --- Test Data Structures ---
@dataclasses.dataclass
class SchulzeTestData:
    """Structure for holding Schulze test case data."""

    testcase_name: str
    rankings: np.ndarray
    pairwise_defeats: np.ndarray
    strongest_path_strengths: np.ndarray
    social_ranking: np.ndarray  # Expected result *before* tie-breaking
    num_candidates: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.num_candidates = self.rankings.shape[1]


# --- Core Schulze Test Cases (Adapted from Google's schulze_method_test.py) ---
# Test cases taken from https://electowiki.org/wiki/Schulze_method.
SCHULZE_TEST_CASES = [
    SchulzeTestData(
        testcase_name="Example 1 (30 voters, 4 candidates)",
        rankings=np.int32(
            5 * [[0, 2, 1, 3]]
            + 2 * [[0, 3, 1, 2]]
            + 3 * [[0, 3, 2, 1]]
            + 4 * [[1, 0, 2, 3]]
            + 3 * [[3, 1, 0, 2]]
            + 3 * [[3, 2, 0, 1]]
            + 1 * [[1, 3, 2, 0]]
            + 5 * [[2, 1, 3, 0]]
            + 4 * [[3, 2, 1, 0]]
        ),
        pairwise_defeats=np.int32(
            [
                [0, 11, 20, 14],
                [19, 0, 9, 12],
                [10, 21, 0, 17],
                [16, 18, 13, 0],
            ]
        ),
        strongest_path_strengths=np.int32(
            [
                [0, 20, 20, 17],
                [19, 0, 19, 17],
                [19, 21, 0, 17],
                [18, 18, 18, 0],
            ]
        ),
        social_ranking=np.int32([1, 3, 2, 0]),  # Expected: D > C > A > B
    ),
    SchulzeTestData(
        testcase_name="Example 2 (9 voters, 4 candidates)",
        rankings=np.int32(
            3 * [[0, 1, 2, 3]]
            + 2 * [[1, 2, 3, 0]]
            + 2 * [[3, 1, 2, 0]]
            + 2 * [[3, 1, 0, 2]]
        ),
        pairwise_defeats=np.int32(
            [
                [0, 5, 5, 3],
                [4, 0, 7, 5],
                [4, 2, 0, 5],
                [6, 4, 4, 0],
            ]
        ),
        strongest_path_strengths=np.int32(
            [
                [0, 5, 5, 5],
                [5, 0, 7, 5],
                [5, 5, 0, 5],
                [6, 5, 5, 0],
            ]
        ),
        social_ranking=np.int32([1, 0, 1, 0]),  # Expected Ties: B=D > A=C
    ),
    SchulzeTestData(
        testcase_name="Example 3 (2 voters, 4 candidates)",
        rankings=np.int32([[0, 0, 1, 2], [0, 1, 3, 2]]),
        pairwise_defeats=np.int32(
            [
                [0, 1, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]
        ),
        strongest_path_strengths=np.int32(
            [
                [0, 1, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ),
        social_ranking=np.int32([0, 1, 2, 2]),  # Expected: A > B > C=D
    ),
    SchulzeTestData(
        testcase_name="Example from MH (5 voters, 4 candidates)",
        rankings=np.int32(
            2 * [[0, 1, 3, 2]] + [[1, 3, 2, 0]] + [[2, 3, 0, 1]] + [[2, 0, 3, 1]]
        ),
        pairwise_defeats=np.int32(
            [
                [0, 4, 4, 2],
                [1, 0, 3, 3],
                [1, 2, 0, 1],
                [3, 2, 4, 0],
            ]
        ),
        strongest_path_strengths=np.int32(
            [
                [0, 4, 4, 3],
                [3, 0, 3, 3],
                [0, 0, 0, 0],
                [3, 3, 4, 0],
            ]
        ),
        social_ranking=np.int32([0, 1, 2, 0]),  # Expected Ties: A=D > B > C
    ),
    SchulzeTestData(
        testcase_name="Example for TBRC check (2 voters, 2 candidates)",
        rankings=np.int32([[0, 1], [1, 0]]),
        pairwise_defeats=np.int32([[0, 1], [1, 0]]),
        strongest_path_strengths=np.int32([[0, 0], [0, 0]]),
        social_ranking=np.int32([0, 0]),  # Expected Ties: A=B
    ),
]

# --- Figure 1 Test Cases ---
# Data is already 0-based normalized
figure_1_cases = [
    dict(
        testcase_name="Figure 1 opinion round",
        # Original ranks: [[1, 2, 3, 4], [2, 1, 4, 3], [4, 1, 2, 3], [2, 3, 4, 1], [3, 2, 4, 1]]
        rankings=np.int32(
            [
                [0, 1, 2, 3],  # A>B>C>D
                [1, 0, 3, 2],  # B>A>D>C
                [3, 0, 1, 2],  # D>A>B>C - Corrected from original comment
                [1, 2, 3, 0],  # B>C>D>A - Corrected from original comment
                [2, 1, 3, 0],  # C>B>D>A - Corrected from original comment
            ]
        ),
        # Expected Original: [3, 1, 4, 2] -> B > D > A > C
        # Expected Normalized: [2, 0, 3, 1] -> B=0, D=1, A=2, C=3
        expected_social_ranking=np.int32([2, 0, 3, 1]),
    ),
    dict(
        testcase_name="Figure 1 critique round",
        # Original ranks: [[3, 1, 2, 2], [1, 3, 2, 2], [3, 2, 2, 1], [2, 3, 1, 1], [4, 2, 1, 3]]
        rankings=np.int32(
            [
                [2, 0, 1, 1],  # B > C=D > A
                [0, 2, 1, 1],  # A > C=D > B
                [2, 1, 1, 0],  # D > C=B > A
                [1, 2, 0, 0],  # C=D > A > B
                [3, 1, 0, 2],  # C > B > D > A
            ]
        ),
        # Expected Original: [3, 2, 1, 1] -> C=D > B > A
        # Expected Normalized: [2, 1, 0, 0] -> C=0, D=0, B=1, A=2
        expected_social_ranking=np.int32([2, 1, 0, 0]),
    ),
]


# --- Tests for Schulze Intermediate Steps and Tied Aggregation ---
@pytest.mark.parametrize(
    "test_data", SCHULZE_TEST_CASES, ids=[d.testcase_name for d in SCHULZE_TEST_CASES]
)
def test_schulze_method_steps(test_data: SchulzeTestData):
    """Tests the intermediate Schulze steps and tied aggregation.
    Mirrors Google's test_schulze_method structure.
    """
    # Test _schulze_compute_pairwise_defeats.
    calculated_defeats = _schulze_compute_pairwise_defeats(test_data.rankings)
    np.testing.assert_array_equal(
        calculated_defeats,
        test_data.pairwise_defeats,
        err_msg=f"[{test_data.testcase_name}] Pairwise defeats mismatch",
    )

    # Test _schulze_compute_strongest_path_strengths.
    # Use expected defeats as input for isolation, like Google's test.
    calculated_strengths = _schulze_compute_strongest_path_strengths(
        test_data.pairwise_defeats
    )
    np.testing.assert_array_equal(
        calculated_strengths,
        test_data.strongest_path_strengths,
        err_msg=f"[{test_data.testcase_name}] Strongest path strengths mismatch",
    )

    # Test _schulze_rank_candidates.
    # Use expected strengths as input for isolation.
    calculated_ranking_tied = _schulze_rank_candidates(
        test_data.strongest_path_strengths
    )
    np.testing.assert_array_equal(
        calculated_ranking_tied,
        test_data.social_ranking,
        err_msg=(
            f"[{test_data.testcase_name}] Tied social ranking mismatch (from"
            " _rank_candidates)"
        ),
    )

    # Test _schulze_aggregate_with_ties (end-to-end tied).
    calculated_aggregate_tied = _schulze_aggregate_with_ties(test_data.rankings)
    np.testing.assert_array_equal(
        calculated_aggregate_tied,
        test_data.social_ranking,
        err_msg=(
            f"[{test_data.testcase_name}] Tied social ranking mismatch (from"
            " _schulze_aggregate_with_ties)"
        ),
    )


# --- Tests for Schulze Aggregation with Random Tie-Breaking ---
# Define parameters based on Google's test_schulze_aggregate for RANDOM method
# and include the extra case from the original user file.
aggregate_random_params = [
    # Case 1: Example 1 (No ties initially) - From Google Test
    dict(
        testcase_name=SCHULZE_TEST_CASES[0].testcase_name + "_RANDOM",
        rankings=SCHULZE_TEST_CASES[0].rankings,
        seed=0,
        target_untied_ranking=SCHULZE_TEST_CASES[
            0
        ].social_ranking,  # Untied (same as tied)
        num_candidates=SCHULZE_TEST_CASES[0].num_candidates,
    ),
    # Case 2: Example from MH (Ties A=D > B > C) - From Google Test
    dict(
        testcase_name=SCHULZE_TEST_CASES[3].testcase_name + "_RANDOM",
        rankings=SCHULZE_TEST_CASES[3].rankings,
        seed=1,
        # Google expected: [0, 2, 3, 1] (A > D > B > C)
        target_untied_ranking=np.int32([0, 2, 3, 1]),
        num_candidates=SCHULZE_TEST_CASES[3].num_candidates,
    ),
    # Case 3: Example for TBRC check (Ties A=B) - Seed 0 - From Google Test
    dict(
        testcase_name=SCHULZE_TEST_CASES[4].testcase_name + "_RANDOM_0",
        rankings=SCHULZE_TEST_CASES[4].rankings,
        seed=0,
        # Google expected: [0, 1] (A > B)
        target_untied_ranking=np.int32([0, 1]),
        num_candidates=SCHULZE_TEST_CASES[4].num_candidates,
    ),
    # Case 4: Example for TBRC check (Ties A=B) - Seed 3 - From Google Test
    dict(
        testcase_name=SCHULZE_TEST_CASES[4].testcase_name + "_RANDOM_3",
        rankings=SCHULZE_TEST_CASES[4].rankings,
        seed=3,
        # Google expected: [1, 0] (B > A)
        target_untied_ranking=np.int32([1, 0]),
        num_candidates=SCHULZE_TEST_CASES[4].num_candidates,
    ),
    # Case 5: Example 2 (Ties B=D > A=C) - Seed 1 - From original user file
    dict(
        testcase_name=SCHULZE_TEST_CASES[1].testcase_name
        + "_RANDOM_seed1",  # Example 2
        rankings=SCHULZE_TEST_CASES[1].rankings,
        seed=1,
        # Expected from original user file calc: [2, 0, 3, 1] (D > B > A > C)
        target_untied_ranking=np.int32([2, 0, 3, 1]),
        num_candidates=SCHULZE_TEST_CASES[1].num_candidates,
    ),
    # Case 6: Example 3 (Ties C=D) - Seed 2 - From original user file
    dict(
        testcase_name=SCHULZE_TEST_CASES[2].testcase_name
        + "_RANDOM_seed2",  # Example 3
        rankings=SCHULZE_TEST_CASES[2].rankings,
        seed=2,
        # Expected from original user file calc: [0, 1, 2, 3] (A > B > C > D)
        target_untied_ranking=np.int32([0, 1, 2, 3]),
        num_candidates=SCHULZE_TEST_CASES[2].num_candidates,
    ),
]


@pytest.mark.parametrize(
    "test_params",
    aggregate_random_params,
    ids=[d["testcase_name"] for d in aggregate_random_params],
)
def test_aggregate_schulze_random_tie_breaking(test_params):
    """Tests the _aggregate_schulze function with random tie-breaking.
    Mirrors Google's test_schulze_aggregate structure for RANDOM method.
    Checks only the final untied ranking, as that's what _aggregate_schulze returns.
    """
    # Prepare agent_rankings dict for the function
    agent_rankings_dict = {
        f"agent_{i}": rank for i, rank in enumerate(test_params["rankings"])
    }

    # Call the main function that performs aggregation and tie-breaking
    calculated_untied_ranking = _aggregate_schulze(
        agent_rankings=agent_rankings_dict,
        num_candidates=test_params["num_candidates"],
        seed=test_params["seed"],
        tie_breaking_method="random",  # Explicitly testing random
    )

    # Check that the result is not None
    assert (
        calculated_untied_ranking is not None
    ), f"[{test_params['testcase_name']}] Aggregation returned None unexpectedly"

    # Check the final untied ranking against the expected result
    np.testing.assert_array_equal(
        calculated_untied_ranking,
        test_params["target_untied_ranking"],
        err_msg=(
            f"[{test_params['testcase_name']}] Untied social ranking mismatch"
            " after random tie-breaking"
        ),
    )
    # Verify that the result is indeed untied (unless the input had only 1 candidate)
    if test_params["num_candidates"] > 1:
        assert (
            np.unique(calculated_untied_ranking).size == calculated_untied_ranking.size
        ), (
            f"[{test_params['testcase_name']}] Resulting ranking"
            f" {calculated_untied_ranking} is not untied"
        )


# --- Test for Figure 1 Examples ---
@pytest.mark.parametrize(
    "fig1_data", figure_1_cases, ids=[d["testcase_name"] for d in figure_1_cases]
)
def test_aggregate_schulze_figure_1(fig1_data):
    """Tests the Schulze method for the examples from Figure 1.
    Mirrors Google's test_for_fig_1 structure, checking the tied result.
    Uses _schulze_aggregate_with_ties as the target function returns tied ranks.
    Assumes input rankings are already 0-based normalized.
    """
    calculated_tied_ranking = _schulze_aggregate_with_ties(fig1_data["rankings"])

    np.testing.assert_array_equal(
        calculated_tied_ranking,
        fig1_data["expected_social_ranking"],
        err_msg=f"[{fig1_data['testcase_name']}] Tied social ranking mismatch",
    )


# --- Tests for Schulze Error Conditions ---
# These tests already matched Google's structure reasonably well.
@pytest.mark.parametrize(
    "test_name, invalid_defeats",
    [
        ("Non-zero diagonal", np.int32([[0, 1, 1], [1, 1, 1], [1, 1, 0]])),
        ("Wrong dimensions", np.int32([[0, 1, 1], [1, 0, 1]])),
    ],
    ids=lambda x: x[0] if isinstance(x, tuple) else None,
)
def test_schulze_compute_strongest_path_strengths_errors(test_name, invalid_defeats):
    """Tests ValueErrors for _schulze_compute_strongest_path_strengths.
    Mirrors Google's test_rank_compute_strongest_path_strengths.
    """
    with pytest.raises(ValueError):
        _schulze_compute_strongest_path_strengths(invalid_defeats)


@pytest.mark.parametrize(
    "test_name, invalid_strengths",
    [
        ("Non-zero diagonal", np.int32([[0, 1, 1], [1, 1, 1], [1, 1, 0]])),
        ("Wrong dimensions", np.int32([[0, 1, 1], [1, 0, 1]])),
    ],
    ids=lambda x: x[0] if isinstance(x, tuple) else None,
)
def test_schulze_rank_candidates_errors(test_name, invalid_strengths):
    """Tests ValueErrors for _schulze_rank_candidates.
    Mirrors Google's test_rank_candidates.
    """
    with pytest.raises(ValueError):
        _schulze_rank_candidates(invalid_strengths)


# --- Tests for Non-Schulze Helper Functions (Keep As Is) ---
# (Assuming these tests exist below this point and should be retained)
# ... existing tests for _hm_check_response_format etc. ...
