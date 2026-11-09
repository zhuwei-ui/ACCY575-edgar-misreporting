import math

import pytest

from src.features_style import sentences_of, style_features, syllables, words_of

TEXT = "The Company sold 2,345 units to U.S. Steel Corp. last year. Revenue increased significantly! Did costs rise?"


def test_words_and_sentences():
    words = words_of(TEXT)
    assert len(words) == 17
    assert "2,345" in words and "U.S" in words and words_of("a $ b") == ["a", "b"]
    sents = sentences_of(TEXT)  # U.S. and Corp. are protected
    assert sents == ["The Company sold 2,345 units to U.S. Steel Corp. last year.",
                     "Revenue increased significantly!", "Did costs rise?"]


def test_syllables_pyphen():
    assert syllables("Company") == 3
    assert syllables("significantly") == 5
    assert syllables("2,345") == 1


def test_readability_by_hand():
    f = style_features(TEXT)
    # 17 words, 3 sentences of 11, 3, 3 words.
    # Complex (3+ syllables, pyphen): Company (3), significantly (5) -> 2.
    # Fog = 0.4 * (17/3 + 100 * 2/17) = 0.4 * (5.6667 + 11.7647) = 6.9725
    assert f["fog"] == pytest.approx(0.4 * (17 / 3 + 200 / 17))
    assert f["fog"] == pytest.approx(6.9725, abs=1e-4)
    # Letters and digits: The 3, Company 7, sold 4, 2345 4, units 5, to 2, US 2, Steel 5, Corp 4, last 4,
    # year 4, Revenue 7, increased 9, significantly 13, Did 3, costs 5, rise 4 = 85.
    # CL = 5.88 * 85/17 - 29.6 * 3/17 - 15.8 = 29.4 - 5.2235 - 15.8 = 8.3765
    assert f["coleman_liau"] == pytest.approx(5.88 * 85 / 17 - 29.6 * 3 / 17 - 15.8)
    assert f["coleman_liau"] == pytest.approx(8.3765, abs=1e-4)
    assert f["log_words"] == pytest.approx(math.log(17))
    assert f["sent_len_mean"] == pytest.approx(17 / 3)
    assert f["pct_short_sent"] == 1.0 and f["pct_long_sent"] == 0.0
    assert f["n_allcaps"] == 1  # "U.S" -> letters "US"
    assert f["n_exclaim"] == 1 and f["n_question"] == 1


def test_repetition():
    f = style_features("Item one applies. Item one applies. Something else here.")
    assert f["repetition"] == pytest.approx(3 / 2)
