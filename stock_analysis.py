#!/usr/bin/env python3
"""
Example:
    python stock_analysis.py --symbol AAPL --period 5y --freq D --lags 5 --test-size 0.2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


FREQ_MAP = {
    "D": "B",  # Business-day frequency
    "W": "W-FRI",
    "M": "ME",  # Month-end
    "Y": "YE",  # Year-end
}

ANNUALIZATION_MAP = {
    "D": 252,
    "W": 52,
    "M": 12,
    "Y": 1,
}


@dataclass
class EvaluationResults:
    rmse: float
    mae: float
    annual_return_buy_hold: float
    sharpe_buy_hold: float
    sortino_buy_hold: float
    annual_return_strategy: float
    sharpe_strategy: float
    sortino_strategy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock return modeling and metric computation")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Ticker symbol, e.g., AAPL")
    parser.add_argument(
        "--period",
        type=str,
        default="5y",
        help="Lookback period supported by yfinance (e.g., 1y, 2y, 5y, 10y, max)",
    )
    parser.add_argument(
        "--freq",
        type=str,
        choices=["D", "W", "M", "Y"],
        default="D",
        help="Return sampling frequency: D (daily), W (weekly), M (monthly), Y (yearly)",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=5,
        help="Number of lagged returns used as model features",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of observations for test split (time-ordered)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for saved artifacts",
    )
    return parser.parse_args()


def download_prices(symbol: str, period: str) -> pd.Series:
    data = yf.download(symbol, period=period, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for symbol={symbol} and period={period}.")

    if "Adj Close" in data.columns:
        prices = data["Adj Close"].copy()
    elif "Close" in data.columns:
        prices = data["Close"].copy()
    else:
        raise ValueError("Could not find 'Adj Close' or 'Close' in downloaded data.")

    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError(
                "Expected one price series for a single symbol, but got multiple columns."
            )
        prices = prices.iloc[:, 0]

    prices = prices.sort_index().astype(float)
    prices = prices[~prices.index.duplicated(keep="last")]
    prices.name = "price"
    return prices


def preprocess_returns(prices: pd.Series, freq: str) -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError("Expected a single price series for return computation.")
        prices = prices.iloc[:, 0]

    rule = FREQ_MAP[freq]
    prices_resampled = prices.resample(rule).last()

    prices_resampled = prices_resampled.ffill().dropna()
    returns = prices_resampled.pct_change().dropna()
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise ValueError("Expected a single return series after preprocessing.")
        returns = returns.iloc[:, 0]
    returns.name = "return"
    return returns


def build_lagged_dataset(returns: pd.Series, lags: int) -> pd.DataFrame:
    if lags < 1:
        raise ValueError("lags must be >= 1")

    frame = pd.DataFrame({"target": returns})
    for i in range(1, lags + 1):
        frame[f"lag_{i}"] = returns.shift(i)

    frame = frame.dropna()
    if frame.empty:
        raise ValueError("Not enough data after creating lagged features. Reduce lags or increase period.")
    return frame


def train_test_split_time_ordered(
    frame: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    split_idx = int(len(frame) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(frame):
        raise ValueError("test_size creates an invalid split for the available data.")

    train = frame.iloc[:split_idx].copy()
    test = frame.iloc[split_idx:].copy()
    return train, test


def safe_sharpe_ratio(returns: pd.Series, periods_per_year: int) -> float:
    std = returns.std(ddof=1)
    if np.isclose(std, 0.0):
        return np.nan
    return np.sqrt(periods_per_year) * returns.mean() / std


def safe_sortino_ratio(returns: pd.Series, periods_per_year: int) -> float:
    downside = returns[returns < 0]
    if downside.empty:
        return np.inf

    downside_std = downside.std(ddof=1)
    if np.isclose(downside_std, 0.0):
        return np.nan

    return np.sqrt(periods_per_year) * returns.mean() / downside_std


def annualized_return_from_periodic(returns: pd.Series, periods_per_year: int) -> float:
    compounded = (1.0 + returns).prod()
    n = len(returns)
    if n == 0:
        return np.nan
    return compounded ** (periods_per_year / n) - 1.0


def evaluate_model_and_metrics(
    model: LinearRegression,
    test: pd.DataFrame,
    periods_per_year: int,
) -> EvaluationResults:
    x_test = test.drop(columns=["target"])
    y_test = test["target"]
    y_pred = pd.Series(model.predict(x_test), index=test.index, name="predicted_return")

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    annual_return_buy_hold = annualized_return_from_periodic(y_test, periods_per_year)
    sharpe_buy_hold = safe_sharpe_ratio(y_test, periods_per_year)
    sortino_buy_hold = safe_sortino_ratio(y_test, periods_per_year)


    strategy_returns = y_test.where(y_pred > 0.0, 0.0)
    annual_return_strategy = annualized_return_from_periodic(strategy_returns, periods_per_year)
    sharpe_strategy = safe_sharpe_ratio(strategy_returns, periods_per_year)
    sortino_strategy = safe_sortino_ratio(strategy_returns, periods_per_year)

    return EvaluationResults(
        rmse=rmse,
        mae=mae,
        annual_return_buy_hold=annual_return_buy_hold,
        sharpe_buy_hold=sharpe_buy_hold,
        sortino_buy_hold=sortino_buy_hold,
        annual_return_strategy=annual_return_strategy,
        sharpe_strategy=sharpe_strategy,
        sortino_strategy=sortino_strategy,
    )


def save_outputs(
    symbol: str,
    returns: pd.Series,
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: pd.Series,
    results: EvaluationResults,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    split_frame = pd.DataFrame(
        {
            "actual_return": test["target"],
            "predicted_return": predictions,
            "strategy_return": test["target"].where(predictions > 0.0, 0.0),
        }
    )
    split_frame.to_csv(output_dir / f"{symbol}_test_predictions.csv", index=True)

    summary = pd.DataFrame(
        {
            "metric": [
                "n_observations_total",
                "n_train",
                "n_test",
                "rmse",
                "mae",
                "annual_return_buy_hold",
                "sharpe_buy_hold",
                "sortino_buy_hold",
                "annual_return_strategy",
                "sharpe_strategy",
                "sortino_strategy",
            ],
            "value": [
                len(returns),
                len(train),
                len(test),
                results.rmse,
                results.mae,
                results.annual_return_buy_hold,
                results.sharpe_buy_hold,
                results.sortino_buy_hold,
                results.annual_return_strategy,
                results.sharpe_strategy,
                results.sortino_strategy,
            ],
        }
    )
    summary.to_csv(output_dir / f"{symbol}_metrics_summary.csv", index=False)


def main() -> None:
    args = parse_args()

    prices = download_prices(args.symbol, args.period)
    returns = preprocess_returns(prices, args.freq)

    frame = build_lagged_dataset(returns, args.lags)
    train, test = train_test_split_time_ordered(frame, args.test_size)

    x_train = train.drop(columns=["target"])
    y_train = train["target"]

    model = LinearRegression()
    model.fit(x_train, y_train)

    periods_per_year = ANNUALIZATION_MAP[args.freq]
    results = evaluate_model_and_metrics(model, test, periods_per_year)

    predictions = pd.Series(
        model.predict(test.drop(columns=["target"])),
        index=test.index,
        name="predicted_return",
    )

    save_outputs(args.symbol, returns, train, test, predictions, results, args.output_dir)

    print("=" * 72)
    print(f"Stock Analysis Summary: {args.symbol}")
    print("=" * 72)
    print(f"Observations (returns): {len(returns)}")
    print(f"Train/Test split: {len(train)} / {len(test)}")
    print(f"RMSE: {results.rmse:.6f}")
    print(f"MAE: {results.mae:.6f}")
    print()
    print("Financial Metrics on Unseen (Test) Data")
    print(f"Buy & Hold Annual Return: {results.annual_return_buy_hold:.4%}")
    print(f"Buy & Hold Sharpe Ratio: {results.sharpe_buy_hold:.4f}")
    print(f"Buy & Hold Sortino Ratio: {results.sortino_buy_hold:.4f}")
    print(f"Strategy Annual Return: {results.annual_return_strategy:.4%}")
    print(f"Strategy Sharpe Ratio: {results.sharpe_strategy:.4f}")
    print(f"Strategy Sortino Ratio: {results.sortino_strategy:.4f}")
    print()
    print(f"Saved artifacts to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
