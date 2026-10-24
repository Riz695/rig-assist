"""Retrieval and refusal metrics, checked against hand-computed examples.

Ground truth is (doc_id, page), so a hit is any retrieved chunk on an expected page,
whichever parser or chunker produced it.
"""

import pytest

from rig_assist.evaluation.metrics import (
    PageRef,
    RefusalCase,
    first_relevant_rank,
    hit_at_k,
    reciprocal_rank_at_k,
    refusal_metrics,
)

MANUAL = "drilling-manual"


def pages(*numbers: int, doc_id: str = MANUAL) -> list[PageRef]:
    return [PageRef(doc_id, n) for n in numbers]


RELEVANT = {PageRef(MANUAL, 12)}


# --- first_relevant_rank -------------------------------------------------


def test_rank_is_one_based() -> None:
    assert first_relevant_rank(pages(12, 3), RELEVANT) == 1
    assert first_relevant_rank(pages(3, 7, 12), RELEVANT) == 3


def test_no_relevant_result_gives_none() -> None:
    assert first_relevant_rank(pages(1, 2, 3), RELEVANT) is None


def test_empty_result_list_gives_none() -> None:
    assert first_relevant_rank([], RELEVANT) is None


def test_same_page_number_in_another_document_is_not_a_hit() -> None:
    assert first_relevant_rank(pages(12, doc_id="other-manual"), RELEVANT) is None


def test_several_chunks_from_one_page_count_once_at_the_first_rank() -> None:
    assert first_relevant_rank(pages(3, 3, 12, 12), RELEVANT) == 3


def test_with_several_relevant_pages_the_best_ranked_one_counts() -> None:
    relevant = {PageRef(MANUAL, 12), PageRef(MANUAL, 13)}
    assert first_relevant_rank(pages(5, 13, 12), relevant) == 2


def test_no_relevant_pages_is_an_error() -> None:
    # Unanswerable questions have no expected page: they go to the refusal metrics.
    with pytest.raises(ValueError, match="relevant"):
        first_relevant_rank(pages(1), set())


# --- hit_at_k ------------------------------------------------------------


def test_hit_at_1_when_the_relevant_page_is_first() -> None:
    assert hit_at_k(pages(12, 3), RELEVANT, k=1) == 1.0


def test_hit_at_1_misses_when_the_relevant_page_is_second() -> None:
    assert hit_at_k(pages(3, 12), RELEVANT, k=1) == 0.0


def test_hit_at_5_counts_ranks_up_to_and_including_5() -> None:
    assert hit_at_k(pages(1, 2, 3, 4, 12), RELEVANT, k=5) == 1.0
    assert hit_at_k(pages(1, 2, 3, 4, 5, 12), RELEVANT, k=5) == 0.0


def test_hit_with_no_relevant_result_is_zero() -> None:
    assert hit_at_k(pages(1, 2), RELEVANT, k=5) == 0.0


def test_hit_at_k_rejects_k_below_one() -> None:
    with pytest.raises(ValueError, match="k"):
        hit_at_k(pages(12), RELEVANT, k=0)


# --- reciprocal_rank_at_k (MRR@k is the mean of this over questions) -----


def test_reciprocal_rank_hand_computed() -> None:
    assert reciprocal_rank_at_k(pages(12), RELEVANT, k=10) == 1.0
    assert reciprocal_rank_at_k(pages(1, 12), RELEVANT, k=10) == pytest.approx(1 / 2)
    assert reciprocal_rank_at_k(pages(1, 2, 3, 12), RELEVANT, k=10) == pytest.approx(1 / 4)


def test_reciprocal_rank_is_zero_beyond_k() -> None:
    assert reciprocal_rank_at_k(pages(1, 2, 3, 12), RELEVANT, k=3) == 0.0


def test_reciprocal_rank_with_no_relevant_result_is_zero() -> None:
    assert reciprocal_rank_at_k(pages(1, 2, 3), RELEVANT, k=10) == 0.0


def test_reciprocal_rank_uses_the_first_of_several_relevant_pages() -> None:
    relevant = {PageRef(MANUAL, 12), PageRef(MANUAL, 13)}
    assert reciprocal_rank_at_k(pages(5, 5, 13, 12), relevant, k=10) == pytest.approx(1 / 3)


def test_reciprocal_rank_rejects_k_below_one() -> None:
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_at_k(pages(12), RELEVANT, k=0)


def test_mean_reciprocal_rank_over_three_questions() -> None:
    # Ranks 1, 2 and "not found": MRR = (1 + 1/2 + 0) / 3 = 0.5.
    runs = [pages(12), pages(1, 12), pages(1, 2)]
    rrs = [reciprocal_rank_at_k(run, RELEVANT, k=10) for run in runs]
    assert sum(rrs) / len(rrs) == pytest.approx(0.5)


# --- refusal_metrics ("not found" is the positive class) -----------------
#
# A question is refused when its top score is below the threshold, or when retrieval
# returned nothing. Precision: of the refusals, how many were right (unanswerable).
# Recall: of the unanswerable questions, how many were refused.


def test_refusal_hand_computed_precision_and_recall() -> None:
    cases = [
        RefusalCase(top_score=0.40, answerable=False),  # refused, correct    (TP)
        RefusalCase(top_score=0.50, answerable=True),  # refused, wrong       (FP)
        RefusalCase(top_score=0.70, answerable=False),  # answered, wrong     (FN)
        RefusalCase(top_score=0.80, answerable=True),  # answered, correct    (TN)
        RefusalCase(top_score=0.30, answerable=False),  # refused, correct    (TP)
    ]
    result = refusal_metrics(cases, threshold=0.6)
    assert (result.true_pos, result.false_pos, result.false_neg, result.true_neg) == (2, 1, 1, 1)
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)


def test_score_equal_to_the_threshold_is_answered() -> None:
    # v1 kept chunks with dense_score >= 0.6, so exactly 0.6 is not a refusal.
    result = refusal_metrics([RefusalCase(top_score=0.6, answerable=False)], threshold=0.6)
    assert result.false_neg == 1
    assert result.true_pos == 0


def test_nothing_retrieved_counts_as_a_refusal() -> None:
    result = refusal_metrics([RefusalCase(top_score=None, answerable=False)], threshold=0.6)
    assert result.true_pos == 1


def test_precision_is_undefined_when_nothing_is_refused() -> None:
    cases = [RefusalCase(top_score=0.9, answerable=False), RefusalCase(0.9, answerable=True)]
    result = refusal_metrics(cases, threshold=0.6)
    assert result.precision is None
    assert result.recall == 0.0


def test_recall_is_undefined_without_unanswerable_questions() -> None:
    result = refusal_metrics([RefusalCase(top_score=0.1, answerable=True)], threshold=0.6)
    assert result.recall is None
    assert result.precision == 0.0


def test_raising_the_threshold_trades_precision_for_recall() -> None:
    cases = [
        RefusalCase(top_score=0.55, answerable=False),
        RefusalCase(top_score=0.65, answerable=True),
        RefusalCase(top_score=0.75, answerable=False),
        RefusalCase(top_score=0.85, answerable=True),
    ]
    low = refusal_metrics(cases, threshold=0.6)
    high = refusal_metrics(cases, threshold=0.8)
    assert (low.precision, low.recall) == (1.0, 0.5)
    assert high.recall == 1.0
    assert high.precision == pytest.approx(2 / 3)


def test_no_cases_is_an_error() -> None:
    with pytest.raises(ValueError, match="case"):
        refusal_metrics([], threshold=0.6)
