"""Tests for Day 1 data pipeline — dataset loading, splitting, scaling, EDA."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Minimal config matching config/config.yaml structure
CONFIG = {
    "data": {
        "url": "https://example.com/fake.zip",
        "resample_freq": "1h",
        "target_col": "T (degC)",
        "train_split": 0.70,
        "val_split": 0.15,
        "test_split": 0.15,
    }
}

# Post-load columns (after wd (deg) is dropped and sin/cos added)
FEATURE_COLS = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd_sin",
    "wd_cos",
]

# Raw columns as they appear in the CSV (wd (deg) not yet encoded)
RAW_COLS = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
]


def _make_dummy_df(n: int = 200, raw: bool = False) -> pd.DataFrame:
    """Return a minimal DataFrame indexed by hourly timestamps.

    Args:
        n: Number of rows.
        raw: If True, use raw CSV columns (with wd (deg)); otherwise post-load columns.
    """
    rng = np.random.default_rng(0)
    idx = pd.date_range("2009-01-01", periods=n, freq="1h")
    cols = RAW_COLS if raw else FEATURE_COLS
    data = {col: rng.standard_normal(n) for col in cols}
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# chronological_split_and_scale
# ---------------------------------------------------------------------------


class TestChronologicalSplitAndScale:
    def test_split_sizes(self) -> None:
        """70/15/15 split sizes must be correct (integer rounding)."""
        from src.data.preprocessing import chronological_split_and_scale

        df = _make_dummy_df(200)
        with (
            patch("src.data.preprocessing.joblib.dump"),
            patch("pandas.DataFrame.to_csv"),
        ):
            train, val, test, _ = chronological_split_and_scale(df, CONFIG)

        assert len(train) == 140
        assert len(val) == 30
        assert len(test) == 30
        assert len(train) + len(val) + len(test) == len(df)

    def test_chronological_order(self) -> None:
        """No overlap — each split starts after the previous ends."""
        from src.data.preprocessing import chronological_split_and_scale

        df = _make_dummy_df(200)
        with (
            patch("src.data.preprocessing.joblib.dump"),
            patch("pandas.DataFrame.to_csv"),
        ):
            train, val, test, _ = chronological_split_and_scale(df, CONFIG)

        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]

    def test_scaler_fit_on_train_only(self) -> None:
        """Train feature means must be ~0 after scaling (scaler was fit on train)."""
        from src.data.preprocessing import chronological_split_and_scale

        df = _make_dummy_df(200)
        feature_cols = [c for c in FEATURE_COLS if c != "T (degC)"]
        with (
            patch("src.data.preprocessing.joblib.dump"),
            patch("pandas.DataFrame.to_csv"),
        ):
            train, _, _, _ = chronological_split_and_scale(df, CONFIG)

        means = train[feature_cols].mean()
        assert (means.abs() < 0.05).all(), f"Train feature means not ~0: {means}"

    def test_scaler_saved(self) -> None:
        """joblib.dump must be called exactly once."""
        from src.data.preprocessing import chronological_split_and_scale

        df = _make_dummy_df(200)
        with (
            patch("src.data.preprocessing.joblib.dump") as mock_dump,
            patch("pandas.DataFrame.to_csv"),
        ):
            chronological_split_and_scale(df, CONFIG)
        assert mock_dump.call_count == 1

    def test_csvs_written(self) -> None:
        """to_csv must be called three times (train, val, test)."""
        from src.data.preprocessing import chronological_split_and_scale

        df = _make_dummy_df(200)
        with (
            patch("src.data.preprocessing.joblib.dump"),
            patch("pandas.DataFrame.to_csv") as mock_csv,
        ):
            chronological_split_and_scale(df, CONFIG)
        assert mock_csv.call_count == 3

    def test_target_col_not_scaled(self) -> None:
        """Target column T (degC) must NOT be standardised."""
        from src.data.preprocessing import chronological_split_and_scale

        df = _make_dummy_df(200)
        original_train_target = df["T (degC)"].iloc[:140].values.copy()
        with (
            patch("src.data.preprocessing.joblib.dump"),
            patch("pandas.DataFrame.to_csv"),
        ):
            train, _, _, _ = chronological_split_and_scale(df, CONFIG)

        np.testing.assert_array_equal(train["T (degC)"].values, original_train_target)


# ---------------------------------------------------------------------------
# load_jena — offline tests (mocked I/O)
# ---------------------------------------------------------------------------


class TestLoadJena:
    def test_raises_data_load_error_on_non_https_url(self) -> None:
        """load_jena must raise DataLoadError for non-https URLs."""
        from src.data.dataset import load_jena
        from src.exceptions import DataLoadError

        bad_cfg = {**CONFIG, "data": {**CONFIG["data"], "url": "http://0.0.0.0/bad"}}
        with patch("src.data.dataset.Path.exists", return_value=False):
            with pytest.raises(DataLoadError, match="Only https://"):
                load_jena(bad_cfg)

    def test_happy_path_file_exists(self) -> None:
        """When CSV exists, load_jena must parse, encode, validate, and return."""
        from src.data.dataset import load_jena

        raw_df = _make_dummy_df(100, raw=True)
        # After resample+dropna index stays the same; raw_df already has DatetimeIndex
        mock_open = MagicMock()
        with (
            patch("src.data.dataset.Path.exists", return_value=True),
            patch("pandas.read_csv", return_value=raw_df),
            patch("src.data.validation.CLIMATE_SCHEMA.validate", return_value=None),
            patch("json.dump"),
            patch("builtins.open", mock_open),
        ):
            result = load_jena(CONFIG)

        # wd (deg) dropped, wd_sin/wd_cos added → 15 columns
        assert "wd (deg)" not in result.columns
        assert "wd_sin" in result.columns
        assert "wd_cos" in result.columns
        assert result.shape[1] == 15

    def test_raises_data_load_error_on_csv_parse_failure(self) -> None:
        """load_jena must raise DataLoadError when pd.read_csv fails."""
        from src.data.dataset import load_jena
        from src.exceptions import DataLoadError

        with (
            patch("src.data.dataset.Path.exists", return_value=True),
            patch("pandas.read_csv", side_effect=OSError("disk error")),
        ):
            with pytest.raises(DataLoadError, match="Failed to parse"):
                load_jena(CONFIG)

    def test_raises_data_validation_error_on_schema_fail(self) -> None:
        """load_jena must raise DataValidationError when pandera validation fails."""
        from src.data.dataset import load_jena
        from src.exceptions import DataValidationError

        raw_df = _make_dummy_df(100, raw=True)
        with (
            patch("src.data.dataset.Path.exists", return_value=True),
            patch("pandas.read_csv", return_value=raw_df),
            patch(
                "src.data.validation.CLIMATE_SCHEMA.validate",
                side_effect=Exception("schema mismatch"),
            ),
            patch("json.dump"),
            patch("builtins.open", MagicMock()),
        ):
            with pytest.raises(DataValidationError, match="Schema validation failed"):
                load_jena(CONFIG)

    def test_checksum_written(self) -> None:
        """json.dump must be called once to write checksums.json."""
        from src.data.dataset import load_jena

        raw_df = _make_dummy_df(100, raw=True)
        with (
            patch("src.data.dataset.Path.exists", return_value=True),
            patch("pandas.read_csv", return_value=raw_df),
            patch("src.data.validation.CLIMATE_SCHEMA.validate", return_value=None),
            patch("json.dump") as mock_jdump,
            patch("builtins.open", MagicMock()),
        ):
            load_jena(CONFIG)
        assert mock_jdump.call_count == 1
        call_args = mock_jdump.call_args[0][0]
        assert "jena_climate" in call_args


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------


class TestRunEda:
    def test_four_figures_saved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_eda must save exactly 4 PNG files."""
        import src.data.eda as eda_module

        monkeypatch.setattr(eda_module, "FIGURES_DIR", tmp_path)

        train = _make_dummy_df(140)
        val = _make_dummy_df(30)
        val.index = pd.date_range("2015-03-01", periods=30, freq="1h")
        test = _make_dummy_df(30)
        test.index = pd.date_range("2015-04-01", periods=30, freq="1h")

        eda_module.run_eda(train, val, test, CONFIG)

        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) == 4

    def test_expected_filenames(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_eda must produce the four canonical figure filenames."""
        import src.data.eda as eda_module

        monkeypatch.setattr(eda_module, "FIGURES_DIR", tmp_path)

        train = _make_dummy_df(140)
        val = _make_dummy_df(30)
        val.index = pd.date_range("2015-03-01", periods=30, freq="1h")
        test = _make_dummy_df(30)
        test.index = pd.date_range("2015-04-01", periods=30, freq="1h")

        eda_module.run_eda(train, val, test, CONFIG)

        names = {p.name for p in tmp_path.glob("*.png")}
        assert "temperature_timeseries.png" in names
        assert "feature_distributions.png" in names
        assert "correlation_heatmap.png" in names
        assert "train_val_test_split.png" in names
