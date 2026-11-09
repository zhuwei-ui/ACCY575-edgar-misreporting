import numpy as np
import pandas as pd
import pytest

from src.features_financial import compute, noac


def full_noa_components(ca, che, ivst, ivao, at, cl, dlc, lt, dltt, pstk):
    """Richardson et al. (2005): WC + NCO + FIN, each from its full definition."""
    wc = (ca - che - ivst) - (cl - dlc)
    nco = (at - ca - ivao) - (lt - cl - dltt)
    fin = (ivst + ivao) - (dltt + dlc + pstk)
    return wc + nco + fin


def test_rsst_reduced_form_equals_full_definition():
    # Made-up balance sheets, two years. The reduced form uses only at, che, lt, pstk.
    y0 = dict(ca=400.0, che=90.0, ivst=25.0, ivao=60.0, at=1000.0, cl=210.0, dlc=35.0, lt=580.0, dltt=240.0, pstk=15.0)
    y1 = dict(ca=470.0, che=70.0, ivst=40.0, ivao=55.0, at=1130.0, cl=260.0, dlc=50.0, lt=640.0, dltt=230.0, pstk=10.0)
    for y in (y0, y1):
        assert full_noa_components(**y) == pytest.approx(noac(y["at"], y["che"], y["lt"], y["pstk"]))
    delta_full = full_noa_components(**y1) - full_noa_components(**y0)
    delta_reduced = noac(y1["at"], y1["che"], y1["lt"], y1["pstk"]) - noac(y0["at"], y0["che"], y0["lt"], y0["pstk"])
    assert delta_full == pytest.approx(delta_reduced)


def panel(rows):
    cols = ["cik", "fyear", "at", "at_lag", "che", "che_lag", "lt", "lt_lag", "pstk", "pstk_lag", "rect",
            "rect_lag", "invt", "invt_lag", "ppent", "sale", "sale_lag", "pi", "pi_lag", "sstk", "dltis", "fincf"]
    base = dict(at=100.0, at_lag=80.0, che=10.0, che_lag=8.0, lt=50.0, lt_lag=40.0, pstk=0.0, pstk_lag=0.0,
                rect=20.0, rect_lag=15.0, invt=5.0, invt_lag=5.0, ppent=30.0, sale=200.0, sale_lag=150.0,
                pi=12.0, pi_lag=9.0, sstk=0.0, dltis=0.0, fincf=4.0)
    return pd.DataFrame([{**base, **r} for r in rows])[cols]


def test_ratios_and_cash_sales_lag():
    df = panel([
        dict(cik="1", fyear=2012),
        dict(cik="1", fyear=2013, rect=30.0, rect_lag=20.0, sale=260.0, sstk=5.0),
        dict(cik="1", fyear=2015, sale_lag=0.0),            # gap: no 2014, falls back to sale_lag (0 -> missing)
        dict(cik="2", fyear=2012, at=0.0),                   # dropped
    ])
    out, stats = compute(df)
    assert stats["dropped_nonpos_at"] == 1
    r = out.set_index(["cik", "fyear"])
    a = r.loc[("1", 2012)]
    assert a["log_at"] == pytest.approx(np.log(100))
    assert a["rsst_acc"] == pytest.approx(((100 - 10 - 50) - (80 - 8 - 40)) / 90)
    assert a["soft_assets"] == pytest.approx((100 - 30 - 10) / 100)
    assert a["ch_roa"] == pytest.approx(3 / 90)
    assert a["cs_lag_source"] == "sale_lag"                  # first year: fallback
    assert a["ch_cash_sales"] == pytest.approx((200 - 5) / 150 - 1)
    assert a["issue"] == 0
    b = r.loc[("1", 2013)]
    assert b["cs_lag_source"] == "prior_row"                 # cs_lag = 2012's cs = 195
    assert b["ch_cash_sales"] == pytest.approx((260 - 10) / 195 - 1)
    assert b["issue"] == 1
    c = r.loc[("1", 2015)]
    assert c["cs_lag_source"] == "sale_lag" and np.isnan(c["ch_cash_sales"])


def test_issue_missing_only_when_unknown():
    df = panel([dict(cik="1", fyear=2012, sstk=np.nan, dltis=3.0),
                dict(cik="2", fyear=2012, sstk=np.nan, dltis=0.0)])
    r = compute(df)[0].set_index("cik")
    assert r.loc["1", "issue"] == 1 and np.isnan(r.loc["2", "issue"])
