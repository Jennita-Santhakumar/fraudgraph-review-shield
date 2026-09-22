import pandas as pd

from src.synthetic_data import GeneratorConfig, generate_dataset


def test_generate_dataset_basic_shape():
    df = generate_dataset(GeneratorConfig(seed=1))
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    expected_cols = {"review_id", "reviewer_id", "product_id", "rating",
                      "text", "timestamp", "is_shill", "reviewer_is_shill"}
    assert expected_cols.issubset(df.columns)


def test_generate_dataset_has_both_classes():
    df = generate_dataset(GeneratorConfig(seed=1))
    assert set(df["is_shill"].unique()) == {0, 1}
    assert df["is_shill"].mean() < 0.5  # shills should be a minority


def test_generate_dataset_deterministic_with_seed():
    df1 = generate_dataset(GeneratorConfig(seed=7))
    df2 = generate_dataset(GeneratorConfig(seed=7))
    pd.testing.assert_frame_equal(df1, df2)


def test_shill_reviewers_have_shill_reviews():
    df = generate_dataset(GeneratorConfig(seed=3))
    shill_reviewers = df[df.reviewer_is_shill == 1]["reviewer_id"].unique()
    assert len(shill_reviewers) > 0
    for rid in shill_reviewers[:5]:
        sub = df[df.reviewer_id == rid]
        assert sub["reviewer_is_shill"].max() == 1
