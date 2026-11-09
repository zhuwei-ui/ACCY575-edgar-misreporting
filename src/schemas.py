"""pandera schemas. Each build script validates its output before writing it."""

import pandera.pandas as pa

tenk_panel_schema = pa.DataFrameSchema(
    {
        "cik": pa.Column(str, pa.Check.str_matches(r"^\d+$"), nullable=False),
        "fyear": pa.Column(int, pa.Check.in_range(2011, 2021)),
        "report_date": pa.Column("datetime64[ns]", nullable=False),
        "filing_date": pa.Column("datetime64[ns]", nullable=False),
        "accession": pa.Column(str, pa.Check.str_matches(r"^\d{10}-\d{2}-\d{6}$")),
        "primary_document": pa.Column(str, nullable=True),
        "sic": pa.Column(str, nullable=True),
        "n_amendments": pa.Column(int, pa.Check.ge(0)),
        "combined": pa.Column(bool),
        "late_filer": pa.Column(bool),
    },
    checks=[
        pa.Check(lambda df: ~df.duplicated(["cik", "accession"]), error="(cik, accession) not unique"),
        pa.Check(lambda df: ~df.duplicated(["cik", "fyear"]), error="(cik, fyear) not unique"),
        pa.Check(lambda df: df["report_date"] <= df["filing_date"], error="report_date after filing_date"),
    ],
    strict=True,
    coerce=True,
)


def _iff(a, b):
    return a == b


panel_schema = pa.DataFrameSchema(
    {
        "cik": pa.Column(str, pa.Check.str_matches(r"^\d+$")),
        "fyear": pa.Column(int, pa.Check.in_range(2011, 2021)),
        "accession": pa.Column(str, pa.Check.str_matches(r"^\d{10}-\d{2}-\d{6}$")),
        "filing_date": pa.Column("datetime64[ns]"),
        "at": pa.Column(float, pa.Check.ge(0), nullable=True),
        "restated": pa.Column(int, pa.Check.isin([0, 1])),
        "reveal_date": pa.Column("datetime64[ns]", nullable=True),
        "irregularity": pa.Column(bool),
        "irregularity_reveal_date": pa.Column("datetime64[ns]", nullable=True),
        "revealed_before_10k": pa.Column(bool, nullable=False),
        "blank_check": pa.Column(bool, nullable=False),
        "sampled_as": pa.Column(str, pa.Check.isin(["restated", "control"]), nullable=True),
        "w": pa.Column(float, pa.Check.isin([1.0, 5.0]), nullable=True),
        "random20": pa.Column(bool, nullable=False),
        "text_path": pa.Column(str, nullable=True),
    },
    checks=[
        pa.Check(lambda df: ~df.duplicated(["cik", "fyear"]), error="(cik, fyear) not unique"),
        pa.Check(lambda df: ~(df["fin_source"].notna() & df.duplicated("accession", keep=False)
                              & df.assign(_f=df["fin_source"].notna()).groupby("accession")["_f"].transform("sum").gt(1)),
                 error="accession repeated among rows with financials"),
        pa.Check(lambda df: _iff(df["reveal_date"].notna(), df["restated"] == 1),
                 error="reveal_date present exactly when restated == 1"),
        pa.Check(lambda df: df["irregularity_reveal_date"].isna() | df["irregularity"],
                 error="irregularity_reveal_date present only when irregularity"),
        pa.Check(lambda df: df["irregularity_reveal_date"].isna()
                 | (df["irregularity_reveal_date"] >= df["reveal_date"]),
                 error="irregularity_reveal_date before reveal_date"),
        pa.Check(lambda df: ~df["irregularity"] | (df["restated"] == 1), error="irregularity without restated"),
        pa.Check(lambda df: _iff(df["w"].notna(), df["sampled_as"].notna()),
                 error="w present exactly when sampled_as is"),
        pa.Check(lambda df: df["text_path"].isna() | df["sampled_as"].notna(), error="text for unsampled row"),
    ],
    strict=False,
    coerce=True,
)
