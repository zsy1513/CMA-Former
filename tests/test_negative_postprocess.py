import numpy as np

from experiments.postprocess_negative_events import _scores


def test_negative_event_scores_include_false_positive_rate():
    truth = np.array([True, True, False, False])
    prediction = np.array([True, False, True, False])
    scores = _scores(truth, prediction)
    assert scores['precision'] == 0.5
    assert scores['recall'] == 0.5
    assert scores['f1'] == 0.5
    assert scores['false_positive_rate'] == 0.5
    assert (scores['tp'], scores['fp'], scores['fn'], scores['tn']) == (1, 1, 1, 1)
