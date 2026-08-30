from quant_investor.market.fundamental_mart import _percent_to_ratio


def test_percent_to_ratio_never_guesses_small_percent_values_are_ratios():
    assert _percent_to_ratio(66.0) == 0.66
    assert _percent_to_ratio(1.9997) == 0.019997
    assert _percent_to_ratio(-1.25) == -0.0125
