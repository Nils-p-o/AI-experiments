# TODO potentially use yahoo_fin instead (or RapidAPI) (or polygon.io for proffesional)
import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict


# V2 What i want
# TODO add indicators and other metrics (diluted EPS, etc.)
# TODO calculate additional metrics
# TODO add return versions of metrics (SMA, EMA, etc.)

class FinancialNumericalDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        seq_len: Optional[int] = None,
        preload: bool = True,
        time_file: Optional[str] = None,
    ):
        self.seq_len = seq_len
        self.preload = preload
        self.file_path = file_path
        if preload:
            self.sequences_data = torch.load(file_path)
            # self.targets_data = None
        else:
            self.sequences_data = None

    def __len__(self) -> int:
        if self.preload:
            return self.sequences_data.size(1) - self.seq_len
        else:  # TODO?
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.preload:
            input_sequence = self.sequences_data[:, idx : idx + self.seq_len, 0]
            target_sequence = self.sequences_data[
                0, idx + 1 : idx + self.seq_len + 1, :
            ]
            return input_sequence, target_sequence
        else:  # TODO?
            raise NotImplementedError


class FinancialNumericalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        test_file: str,
        metadata_file: str,
        batch_size: int,
        num_workers: int = 6,
        seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.metadata_file = metadata_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len

        self.train_dataset: Optional[FinancialNumericalDataset] = None
        self.val_dataset: Optional[FinancialNumericalDataset] = None
        self.test_dataset: Optional[FinancialNumericalDataset] = None

        self._metadata: Optional[Dict[str, Any]] = None
        # self._load_metadata()

    def _load_metadata(self):
        with open(self.metadata_file, "r") as f:
            self._metadata = json.load(f)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = FinancialNumericalDataset(
                self.train_file, self.seq_len
            )
            self.val_dataset = FinancialNumericalDataset(self.val_file, self.seq_len)
        if stage == "test" or stage is None:
            self.test_dataset = FinancialNumericalDataset(self.test_file, self.seq_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers - 3,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def is_already_downloaded(tickers, output_dir):
    file_path = os.path.join(output_dir, "metadata.json")
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            metadata = json.load(f)
        downloaded = True
        for ticker in tickers:
            if ticker not in metadata["tickers"]:
                downloaded = False
                break
        return downloaded
    else:
        return False


def save_indexes_to_csv(indexes, output_file):
    with open(output_file, "w") as f:
        f.write("\n".join(str(s) for s in indexes))


def download_numerical_financial_data(
    tickers: List[str],
    seq_len: int = 64,
    target_dates: List[str] = [1],
    output_dir: str = "time_series_data",
    start_date: str = "1990-01-01",
    end_date: str = "2025-01-01",
    val_split_ratio: float = 0.15,
    test_split_ratio: float = 0.0,  # dont need it for now
    check_if_already_downloaded: bool = True,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if check_if_already_downloaded and is_already_downloaded(tickers, output_dir):
        print("Data already downloaded.")
        return
    # download all available data by default
    raw_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=False,
        back_adjust=False,
    )
    if raw_data.empty:
        print("No data downloaded.")
        return

    unique_tickers = sorted(list(set(tickers)))
    ticker_to_id = {ticker: i for i, ticker in enumerate(unique_tickers)}

    # TODO future removing tickers without data (maybe replace with missing data tokens?)
    indexes = raw_data.index
    # can get day of week, month, year, etc.
    columns = list(raw_data.columns.get_level_values(0))

    # names and tickers
    raw_data = torch.tensor(raw_data.values, dtype=torch.float32)  # (Time, Features)
    raw_data = raw_data.transpose(0, 1)  # (Features, Time series)

    # removing adj close
    raw_data = raw_data[1:, :]
    columns = columns[1:]

    returns = (raw_data[:, 1:] - raw_data[:, :-1]) / raw_data[:, :-1]
    returns = torch.cat((torch.zeros(raw_data.shape[0], 1), returns), dim=1)

    volatility = calculate_volatility_log_ret(raw_data, lookback=10).unsqueeze(0)
    data = torch.cat((returns, volatility), dim=0)
    columns.append("Volatility_10")

    volatility = calculate_volatility_log_ret(raw_data, lookback=20).unsqueeze(0)
    data = torch.cat((data, volatility), dim=0)
    columns.append("Volatility_20")

    volatility = calculate_volatility_log_ret(raw_data, lookback=50).unsqueeze(0)
    data = torch.cat((data, volatility), dim=0)
    columns.append("Volatility_50")

    adr = (raw_data[2, :] - raw_data[3, :]) / raw_data[1, :]  # average daily range
    data = torch.cat((data, adr.unsqueeze(0)), dim=0)
    columns.append("ADR")

    atr = calculate_average_true_range(data, lookback=10).unsqueeze(0)
    data = torch.cat((data, atr), dim=0)
    columns.append("ATR_10")

    atr = calculate_average_true_range(data, lookback=21).unsqueeze(0)
    data = torch.cat((data, atr), dim=0)
    columns.append("ATR_21")

    sma = calculate_moving_average(raw_data, lookback=10).unsqueeze(0)
    data = torch.cat((data, sma), dim=0)
    columns.append("SMA_10")

    sma = calculate_moving_average(raw_data, lookback=20).unsqueeze(0)
    data = torch.cat((data, sma), dim=0)
    columns.append("SMA_20")

    sma = calculate_moving_average(raw_data, lookback=50).unsqueeze(0)
    data = torch.cat((data, sma), dim=0)
    columns.append("SMA_50")

    smar = calculate_moving_average_returns(raw_data, lookback=10).unsqueeze(0)
    data = torch.cat((data, smar), dim=0)
    columns.append("SMAR_10")

    smar = calculate_moving_average_returns(raw_data, lookback=20).unsqueeze(0)
    data = torch.cat((data, smar), dim=0)
    columns.append("SMAR_20")

    smar = calculate_moving_average_returns(raw_data, lookback=50).unsqueeze(0)
    data = torch.cat((data, smar), dim=0)
    columns.append("SMAR_50")
    # adding this, because maybe this is more informative for the data format im using, as both are from returns, but before was from log returns
    volatility = calculate_volatility_returns(raw_data, lookback=10).unsqueeze(0)
    data = torch.cat((data, volatility), dim=0)
    columns.append("Volatility_returns_10")

    volatility = calculate_volatility_returns(raw_data, lookback=20).unsqueeze(0)
    data = torch.cat((data, volatility), dim=0)
    columns.append("Volatility_returns_20")

    volatility = calculate_volatility_returns(raw_data, lookback=50).unsqueeze(0)
    data = torch.cat((data, volatility), dim=0)
    columns.append("Volatility_returns_50")
    # getting bollinger bands for multiple horizons (for the multipliers for volatility, TODO)
    bollinger = data[14,:] + 1.5 * data[17,:] 
    data = torch.cat((data, bollinger.unsqueeze(0)), dim=0)
    columns.append("Bollinger_10_up")

    bollinger = data[14,:] - 1.5 * data[17,:] 
    data = torch.cat((data, bollinger.unsqueeze(0)), dim=0)
    columns.append("Bollinger_10_down")

    bollinger = data[15,:] + 2 * data[18,:]
    data = torch.cat((data, bollinger.unsqueeze(0)), dim=0)
    columns.append("Bollinger_20_up")

    bollinger = data[15,:] - 2 * data[18,:]
    data = torch.cat((data, bollinger.unsqueeze(0)), dim=0)
    columns.append("Bollinger_20_down")

    bollinger = data[16,:] + 2.5 * data[19,:]
    data = torch.cat((data, bollinger.unsqueeze(0)), dim=0)
    columns.append("Bollinger_50_up")

    bollinger = data[16,:] - 2.5 * data[19,:]
    data = torch.cat((data, bollinger.unsqueeze(0)), dim=0)
    columns.append("Bollinger_50_down")

    ema = calculate_ema_returns(raw_data, lookback=10).unsqueeze(0)
    data = torch.cat((data, ema), dim=0)
    columns.append("EMA_10")

    ema = calculate_ema_returns(raw_data, lookback=20).unsqueeze(0)
    data = torch.cat((data, ema), dim=0)
    columns.append("EMA_20")

    ema = calculate_ema_returns(raw_data, lookback=50).unsqueeze(0)
    data = torch.cat((data, ema), dim=0)
    columns.append("EMA_50")

    ppo = (data[26,:] - data[27,:]) / data[27,:]
    data = torch.cat((data, ppo.unsqueeze(0)), dim=0)
    columns.append("PPO_10_20_10")

    macd = data[26,:] - data[27,:]
    data = torch.cat((data, macd.unsqueeze(0)), dim=0)
    columns.append("MACD_10_20_10")
    # time_related_data = np.array([indexes.day_of_week, indexes.day, indexes.day_of_year, indexes.month, indexes.quarter, indexes.is_leap_year, indexes.is_month_start, indexes.is_month_end, indexes.is_quarter_start, indexes.is_quarter_end], dtype=np.float32)
    # time_related_data = torch.tensor(time_related_data, dtype=torch.float32)
    # data = torch.cat((time_related_data, data), dim=0)
    # columns = ["day_of_week", "day", "day_of_year", "month", "quarter", "is_leap_year", "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end"] + columns

    column_to_id = {column: i for i, column in enumerate(columns)}
    # getting returns for multiple horizons
    data = data[:, 1:]  # for ease and compatibility with getting multiple targets
    if max(target_dates) != 1:
        temp_data = data[: raw_data.shape[0], : -(max(target_dates) - 1)].unsqueeze(
            -1
        )  # (Features, Time series, target dates)
        temp_indicators = data[
            raw_data.shape[0] :, : -(max(target_dates) - 1)
        ].unsqueeze(
            -1
        )  # (Features, Time series, target dates)
        for i in range(1, len(target_dates)):
            if target_dates[i] == max(target_dates):
                temp_returns = (
                    raw_data[: raw_data.shape[0], target_dates[i] :]
                    - raw_data[: raw_data.shape[0], : -(max(target_dates))]
                ) / raw_data[: raw_data.shape[0], : -(max(target_dates))]
                temp_indicators = torch.cat(
                    (
                        temp_indicators,
                        data[raw_data.shape[0] :, target_dates[i] - 1 :].unsqueeze(-1),
                    ),
                    dim=-1,
                )
            else:
                temp_returns = (
                    raw_data[
                        : raw_data.shape[0],
                        target_dates[i] : -(max(target_dates) - target_dates[i]),
                    ]
                    - raw_data[: raw_data.shape[0], : -(max(target_dates))]
                ) / raw_data[: raw_data.shape[0], : -(max(target_dates))]
                temp_indicators = torch.cat(
                    (
                        temp_indicators,
                        data[
                            raw_data.shape[0] :,
                            target_dates[i]
                            - 1 : -(max(target_dates) - target_dates[i]),
                        ].unsqueeze(-1),
                    ),
                    dim=-1,
                )
            temp_data = torch.cat((temp_data, temp_returns.unsqueeze(-1)), dim=-1)
        data = torch.cat((temp_data, temp_indicators), dim=0)

    data = data[:, 20:, :]  # getting rid of some trashy-ish data points

    data_length = data.shape[1]

    if (torch.isnan(data)).any() or (torch.isinf(data)).any():
        print("Data contains NaN or Inf values. Skipping.")

    train_data_length = int(data_length * (1 - val_split_ratio - test_split_ratio))
    val_data_length = int(data_length * val_split_ratio)
    test_data_length = data_length - train_data_length - val_data_length
    train_data, val_data, test_data = torch.split(
        data, [train_data_length, val_data_length, test_data_length], dim=1
    )
    train_indexes, val_indexes, test_indexes = (
        indexes[:train_data_length],
        indexes[train_data_length : train_data_length + val_data_length],
        indexes[train_data_length + val_data_length :],
    )

    # no normalization (will do that on the fly, per sequence)

    save_indexes_to_csv(train_indexes, os.path.join(output_dir, "train.csv"))
    save_indexes_to_csv(val_indexes, os.path.join(output_dir, "val.csv"))
    save_indexes_to_csv(test_indexes, os.path.join(output_dir, "test.csv"))

    torch.save(train_data, os.path.join(output_dir, "train.pt"))
    torch.save(val_data, os.path.join(output_dir, "val.pt"))
    torch.save(test_data, os.path.join(output_dir, "test.pt"))

    collected_data_start_date = str(train_indexes[0])[:10]
    collected_data_end_date = str(val_indexes[-1])[:10]

    metadata = {
        "tickers": unique_tickers,
        "ticker_to_id": ticker_to_id,
        "start_date": collected_data_start_date,
        "end_date": collected_data_end_date,
        "val_split_ratio": val_split_ratio,
        "test_split_ratio": test_split_ratio,
        "columns": columns,
        "column_to_id": column_to_id,
        "indexes": data_length,
        "target_dates": target_dates,
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    tickers = ["AAPL"]  # Reduced for faster testing
    output_dir = "time_series_data"
    seq_len = 64

    download_numerical_financial_data(
        tickers=tickers,
        seq_len=seq_len,
        output_dir=output_dir,
        check_if_already_downloaded=False,
    )


def calculate_volatility_log_ret(
    data: torch.Tensor, lookback: int = 30, dim: int = 0
) -> torch.Tensor:
    log_returns = torch.log(data[dim, 1:] / data[dim, :-1])  # from adj. Close for now
    volatility = [0, 0]
    for i in range(2, data.shape[1]):
        if i < lookback:
            volatility.append(log_returns[:i].std())
        else:
            volatility.append(log_returns[i - lookback : i].std())
    volatility[0] = volatility[2]  # maybe wrong, but its just two datapoints
    volatility[1] = volatility[2]
    return torch.tensor(volatility)


def calculate_average_true_range(
    data: torch.Tensor, lookback: int = 14
) -> torch.Tensor:
    ADRr = data[6, :]
    ATRr = []
    for i in range(data.shape[1]):
        ATRr.append(ADRr[max(i - lookback, 0) : i + 1].sum() / min(max(i - lookback,1), i + 1))
    return torch.tensor(ATRr)


def calculate_moving_average(
    data: torch.Tensor, lookback: int = 10, dim: int = 0
) -> torch.Tensor:
    price = data[dim, :]
    SMAr = []
    for i in range(data.shape[1]):
        SMAr.append(price[max(i - lookback, 0) : i + 1].mean())
    return torch.tensor(SMAr)

def calculate_moving_average_returns(
    data: torch.Tensor, lookback: int = 10, dim: int = 0) -> torch.Tensor:
    price = data[dim, :]
    SMAR = (price[lookback:] - price[:-lookback])/lookback
    temp = torch.zeros(lookback)
    for i in range(1,lookback):
        temp[i] = price[i] - price[0]/i
    temp[0] = temp[1]
    return torch.cat((temp,SMAR))

def calculate_volatility_returns(
    data: torch.Tensor, lookback: int = 30, dim: int = 0
) -> torch.Tensor:
    returns = (data[dim, 1:] - data[dim, :-1])/data[dim, :-1]  # from adj. Close for now
    volatility = [0, 0]
    for i in range(2, data.shape[1]):
        if i < lookback:
            volatility.append(returns[:i].std())
        else:
            volatility.append(returns[i - lookback : i].std())
    volatility[0] = volatility[2]  # maybe wrong, but its just two datapoints
    volatility[1] = volatility[2]
    return torch.tensor(volatility)

def calculate_ema_returns(
    data: torch.Tensor, lookback: int = 10, dim: int = 0
) -> torch.Tensor:
    price = data[dim, :]
    ema = torch.zeros(data.shape[1])
    multiplier = 2 / (lookback + 1)
    for i in range(data.shape[1]):
        if i == 0:
            ema[i] = price[i] * multiplier
        else:
            ema[i] = price[i] * multiplier + ema[i - 1] * (1 - multiplier)
    return ema

# 1. Major Equity Market Indexes (Broad Market Sentiment & Benchmarks):
# ^GSPC (S&P 500): Tracks 500 large-cap U.S. stocks. Excellent for overall U.S. market health.
# ^IXIC (NASDAQ Composite): Tracks most stocks listed on the Nasdaq exchange, heavily tech-weighted. Good for tech sector sentiment.
# ^DJI (Dow Jones Industrial Average): Tracks 30 large, publicly-owned U.S. companies. More of a historical/popular index but still watched.
# ^RUT (Russell 2000): Tracks 2000 small-cap U.S. stocks. Indicator for smaller companies and broader economic health.
# International Indexes (if your target stocks have global exposure):
# ^FTSE (FTSE 100 - UK): Large-cap UK stocks.
# ^GDAXI (DAX - Germany): Major German stocks.
# ^N225 (Nikkei 225 - Japan): Major Japanese stocks.
# ^HSI (Hang Seng Index - Hong Kong): Major Hong Kong stocks.
# ^STOXX50E (EURO STOXX 50): Blue-chip stocks from Eurozone countries.
# 2. Volatility Indexes (Market Fear/Uncertainty):
# ^VIX (CBOE Volatility Index): Often called the "fear index," it measures market expectations of 30-day volatility based on S&P 500 index options. Highly influential.
# ^VVIX (CBOE VIX of VIX Index): Measures the volatility of the VIX itself. Can indicate nervousness about future volatility.
# ^VXN (CBOE Nasdaq 100 Volatility Index): Similar to VIX, but for the Nasdaq 100.
# 3. Interest Rates & Bonds (Cost of Capital, Economic Outlook):
# Treasury Yields (proxy for risk-free rates and economic expectations):
# ^TNX (CBOE Interest Rate 10 Year T Note): 10-Year U.S. Treasury yield. Very important.
# ^TYX (CBOE Interest Rate 30 Year T Bond): 30-Year U.S. Treasury yield.
# ^FVX (CBOE Interest Rate 5 Year T Note): 5-Year U.S. Treasury yield.
# ^IRX (CBOE Interest Rate 13 Week T Bill): 13-Week (3-Month) U.S. Treasury Bill yield.
# Yield Curve Spreads (often leading indicators of recessions): You'd calculate these yourself after fetching individual yields. Common ones:
# 10-Year minus 2-Year (^TNX - ^FVX or directly fetch 2-year if available)
# 10-Year minus 3-Month (^TNX - ^IRX)
# 4. Commodities (Inflation, Industrial Demand, Global Growth):
# Crude Oil:
# CL=F (Crude Oil Futures): WTI crude oil.
# BZ=F (Brent Crude Oil Futures): Brent crude oil.
# Gold:
# GC=F (Gold Futures): Often seen as a safe-haven asset and inflation hedge.
# Silver:
# SI=F (Silver Futures): Both an industrial metal and a precious metal.
# Copper:
# HG=F (Copper Futures): "Dr. Copper," often seen as an indicator of global economic health due to its industrial uses.
# Broad Commodity Index ETFs (easier than individual futures sometimes):
# DBC (Invesco DB Commodity Index Tracking Fund)
# GSG (iShares S&P GSCI Commodity-Indexed Trust)
# 5. Currencies (Global Capital Flows, Relative Economic Strength):
# U.S. Dollar Index:
# DX-Y.NYB (U.S. Dollar Index Futures): Measures the value of the USD against a basket of foreign currencies.
# Major Currency Pairs (relative to USD, or against each other if relevant):
# EURUSD=X (Euro to USD)
# JPY=X (USD to Japanese Yen) - Note: some are quoted as USD/Other, others Other/USD.
# GBPUSD=X (British Pound to USD)
# AUDUSD=X (Australian Dollar to USD)
# CNY=X (USD to Chinese Yuan Renminbi)
# 6. Sector-Specific ETFs (If your target stock is in a specific sector):
# Technology: XLK, VGT, QQQ (Nasdaq 100, very tech-heavy)
# Financials: XLF, VFH
# Healthcare: XLV, VHT
# Energy: XLE, VDE
# Consumer Discretionary: XLY, VCR
# Consumer Staples: XLP, VDC
# Industrials: XLI, VIS
# Utilities: XLU, VPU
# Real Estate: XLRE, VNQ
# Materials: XLB, VAW
# (Search for "(Sector Name) Sector SPDR ETF" or "Vanguard (Sector Name) ETF" to find tickers)
# Tips for Incorporating These:
# Relevance: Choose indicators most relevant to the stocks you are trying to predict. A tech stock will be more influenced by ^IXIC and ^VXN than an oil company.
# Lagged Features: Consider using lagged values of these indicators (e.g., the VIX from yesterday, or the average VIX over the last week) as input features. The market often reacts with a delay.
# Transformations: Just like your primary stock data, you'll likely want to use percentage changes or standardized values for these indicators rather than raw price levels to ensure stationarity and comparable scales.
# Feature Engineering: Create new features, like the yield curve spreads mentioned above.
# Data Availability & Alignment: yfinance is great, but be mindful of:
# Trading Hours: Different markets have different trading hours. Daily data (interval="1d") is usually the easiest to align.
# Holidays: Market holidays can lead to missing data points. pandas can help with ffill() or bfill() or interpolation if needed carefully.
# Start Dates: Ensure all chosen indicators have historical data covering your desired training period.
# Curse of Dimensionality: Don't add too many features without reason. Feature selection or dimensionality reduction techniques might be necessary if you include a very large number. Start with a core set and expand.
# Causality vs. Correlation: Remember that correlation doesn't imply causation. While these indicators can improve predictive power, they are part of a complex system.
# When adding these to your download_and_process_numerical_financial_data function, you'd fetch them alongside your primary tickers and then process their percentage changes and normalize them using the training set statistics of those specific indicators. They would become additional columns in your input_data_np.
