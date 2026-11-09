import pandas as pd


def test_overflow_files_were_read():
    panel = pd.read_parquet("data/interim/tenk_panel.parquet")
    assert {2011, 2012, 2013, 2014} <= set(panel.loc[panel["cik"] == "320193", "fyear"])
