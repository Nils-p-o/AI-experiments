# TODO potentially use yahoo_fin instead (or RapidAPI) (or polygon.io for proffesional)
import os
import json
import numpy as np
import yfinance as yf
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd

# TODO add indicators and other metrics (diluted EPS, etc.)


class FinancialNumericalDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        seq_len: Optional[int] = None,
        preload: bool = True,
        num_targets: int = 3,
    ):
        self.seq_len = seq_len
        self.preload = preload
        self.file_path = file_path
        self.num_targets = num_targets
        if preload:
            self.sequences_data = torch.load(file_path)
        else:
            self.sequences_data = None

    def __len__(self) -> int:
        if self.preload:
            return self.sequences_data.size(1) - self.seq_len - 1
        else:  # TODO?
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.preload:
            input_sequence = self.sequences_data[
                self.num_targets :, idx : idx + self.seq_len, :
            ]
            target_sequence = self.sequences_data[
                : self.num_targets, idx : idx + self.seq_len, :
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
        num_workers: int = 0,
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
        self._load_metadata()

    def _load_metadata(self):
        with open(self.metadata_file, "r") as f:
            self._metadata = json.load(f)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = FinancialNumericalDataset(
                self.train_file,
                self.seq_len,
                num_targets=len(self._metadata["target_dates"]),
            )
            self.val_dataset = FinancialNumericalDataset(
                self.val_file,
                self.seq_len,
                num_targets=len(self._metadata["target_dates"]),
            )
        if stage == "test" or stage is None:
            self.test_dataset = FinancialNumericalDataset(
                self.test_file,
                self.seq_len,
                num_targets=len(self._metadata["target_dates"]),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # persistent_workers=True,
            # prefetch_factor=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # persistent_workers=True,
            # prefetch_factor=4,
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
    start_date: str = "2000-09-01",  # check if all stocks have data from this date
    end_date: str = "2025-01-01",
    val_split_ratio: float = 0.21,  # to keep the same val dataset
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
    aligned_raw_data = pd.DataFrame(columns=raw_data.columns)
    for i in range(len(tickers)):
        for j in range(len(raw_data.columns.levels[0])):
            aligned_raw_data[raw_data.columns.levels[0][j], tickers[i]] = (
                align_financial_dataframes(
                    {tickers[i]: raw_data[raw_data.columns.levels[0][j]]},
                    target_column=tickers[i],
                    fill_method="ffill",
                    min_date=start_date,
                    max_date=end_date,
                )
            )
    raw_data = aligned_raw_data
    if raw_data.empty:
        print("No data downloaded.")
        return

    unique_tickers = sorted(list(set(tickers)))
    ticker_to_id = {ticker: i for i, ticker in enumerate(unique_tickers)}

    indexes = raw_data.index
    columns = list(raw_data.columns.levels[0])

    raw_data = torch.tensor(raw_data.values, dtype=torch.float32).reshape(
        -1, len(columns), len(tickers)
    )  # (Time, Features, tickers)
    raw_data = raw_data.transpose(0, 1)  # (Features, Time series, tickers)

    # TODO write somewhere else, maybe in another function

    # new structure - targets, then input features in the same dimension
    # (targets+features, time series, tickers)
    # TODO rewrite without for loop over tickers (adapt calculate functions for multiple tickers at once)
    raw_data = raw_data[1:, :, :]
    columns = columns[1:]

    data = []

    for i in range(len(tickers)):
        temp_raw_data = raw_data[:, :, i]
        # getting targets
        temp_raw_data = temp_raw_data[:, : -max(target_dates)]

        # getting features
        returns = (temp_raw_data[:, 1:] - temp_raw_data[:, :-1]) / temp_raw_data[:, :-1]
        returns = torch.cat((torch.zeros(temp_raw_data.shape[0], 1), returns), dim=1)
        temp_data = returns

        volatility = calculate_volatility_log_ret(temp_raw_data, lookback=10).unsqueeze(
            0
        )
        temp_data = torch.cat((temp_data, volatility), dim=0)
        if i == 0:
            columns.append("Volatility_10")

        volatility = calculate_volatility_log_ret(temp_raw_data, lookback=20).unsqueeze(
            0
        )
        temp_data = torch.cat((temp_data, volatility), dim=0)
        if i == 0:
            columns.append("Volatility_20")

        volatility = calculate_volatility_log_ret(temp_raw_data, lookback=50).unsqueeze(
            0
        )
        temp_data = torch.cat((temp_data, volatility), dim=0)
        if i == 0:
            columns.append("Volatility_50")

        adr = (temp_raw_data[1, :] - temp_raw_data[2, :]) / temp_raw_data[
            3, :
        ]  # average daily range
        temp_data = torch.cat((temp_data, adr.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("ADR")

        atr = calculate_average_true_range(temp_data, lookback=10).unsqueeze(0)
        temp_data = torch.cat((temp_data, atr), dim=0)
        if i == 0:
            columns.append("ATR_10")

        atr = calculate_average_true_range(temp_data, lookback=21).unsqueeze(0)
        temp_data = torch.cat((temp_data, atr), dim=0)
        if i == 0:
            columns.append("ATR_21")

        sma = calculate_moving_average(temp_raw_data, lookback=10).unsqueeze(0)
        temp_data = torch.cat((temp_data, sma), dim=0)
        if i == 0:
            columns.append("SMA_10")

        sma = calculate_moving_average(temp_raw_data, lookback=20).unsqueeze(0)
        temp_data = torch.cat((temp_data, sma), dim=0)
        if i == 0:
            columns.append("SMA_20")

        sma = calculate_moving_average(temp_raw_data, lookback=50).unsqueeze(0)
        temp_data = torch.cat((temp_data, sma), dim=0)
        if i == 0:
            columns.append("SMA_50")

        smar = calculate_moving_average_returns(temp_raw_data, lookback=10).unsqueeze(0)
        temp_data = torch.cat((temp_data, smar), dim=0)
        if i == 0:
            columns.append("SMAR_10")

        smar = calculate_moving_average_returns(temp_raw_data, lookback=20).unsqueeze(0)
        temp_data = torch.cat((temp_data, smar), dim=0)
        if i == 0:
            columns.append("SMAR_20")

        smar = calculate_moving_average_returns(temp_raw_data, lookback=50).unsqueeze(0)
        temp_data = torch.cat((temp_data, smar), dim=0)
        if i == 0:
            columns.append("SMAR_50")
        volatility = calculate_volatility_returns(temp_raw_data, lookback=10).unsqueeze(
            0
        )
        temp_data = torch.cat((temp_data, volatility), dim=0)
        if i == 0:
            columns.append("Volatility_returns_10")

        volatility = calculate_volatility_returns(temp_raw_data, lookback=20).unsqueeze(
            0
        )
        temp_data = torch.cat((temp_data, volatility), dim=0)
        if i == 0:
            columns.append("Volatility_returns_20")

        volatility = calculate_volatility_returns(temp_raw_data, lookback=50).unsqueeze(
            0
        )
        temp_data = torch.cat((temp_data, volatility), dim=0)
        if i == 0:
            columns.append("Volatility_returns_50")
        # getting bollinger bands for multiple horizons (for the multipliers for volatility, TODO)
        bollinger = temp_data[14, :] + 1.5 * temp_data[17, :]
        temp_data = torch.cat((temp_data, bollinger.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("Bollinger_10_up")

        bollinger = temp_data[14, :] - 1.5 * temp_data[17, :]
        temp_data = torch.cat((temp_data, bollinger.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("Bollinger_10_down")

        bollinger = temp_data[15, :] + 2 * temp_data[18, :]
        temp_data = torch.cat((temp_data, bollinger.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("Bollinger_20_up")

        bollinger = temp_data[15, :] - 2 * temp_data[18, :]
        temp_data = torch.cat((temp_data, bollinger.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("Bollinger_20_down")

        bollinger = temp_data[16, :] + 2.5 * temp_data[19, :]
        temp_data = torch.cat((temp_data, bollinger.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("Bollinger_50_up")

        bollinger = temp_data[16, :] - 2.5 * temp_data[19, :]
        temp_data = torch.cat((temp_data, bollinger.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("Bollinger_50_down")

        ema = calculate_ema_returns(temp_raw_data, lookback=10).unsqueeze(0)
        temp_data = torch.cat((temp_data, ema), dim=0)
        if i == 0:
            columns.append("EMA_10")

        ema = calculate_ema_returns(temp_raw_data, lookback=20).unsqueeze(0)
        temp_data = torch.cat((temp_data, ema), dim=0)
        if i == 0:
            columns.append("EMA_20")

        ema = calculate_ema_returns(temp_raw_data, lookback=50).unsqueeze(0)
        temp_data = torch.cat((temp_data, ema), dim=0)
        if i == 0:
            columns.append("EMA_50")

        ppo = (temp_data[26, :] - temp_data[27, :]) / temp_data[27, :]
        temp_data = torch.cat((temp_data, ppo.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("PPO_10_20_10")

        macd = temp_data[26, :] - temp_data[27, :]
        temp_data = torch.cat((temp_data, macd.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("MACD_10_20")

        # ################################ shorter metrics
        volatility = calculate_volatility_returns(temp_raw_data, lookback=5).unsqueeze(
            0
        )
        temp_data = torch.cat((temp_data, volatility), dim=0)
        if i == 0:
            columns.append("Volatility_returns_5")

        smar = calculate_moving_average_returns(temp_raw_data, lookback=5).unsqueeze(0)
        temp_data = torch.cat((temp_data, smar), dim=0)
        if i == 0:
            columns.append("SMAR_5")

        bollinger = temp_data[31, :] + 1 * temp_data[32, :]
        temp_data = torch.cat((temp_data, bollinger.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("Bollinger_5_up")

        bollinger = temp_data[31, :] - 1 * temp_data[32, :]
        temp_data = torch.cat((temp_data, bollinger.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("Bollinger_5_down")

        ema = calculate_ema_returns(temp_raw_data, lookback=5).unsqueeze(0)
        temp_data = torch.cat((temp_data, ema), dim=0)
        if i == 0:
            columns.append("EMA_5")

        ppo = (temp_data[35, :] - temp_data[26, :]) / temp_data[26, :]
        temp_data = torch.cat((temp_data, ppo.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("PPO_5_10_5")

        macd = temp_data[35, :] - temp_data[26, :]
        temp_data = torch.cat((temp_data, macd.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("MACD_5_10")

        ppo = (temp_data[35, :] - temp_data[27, :]) / temp_data[27, :]
        temp_data = torch.cat((temp_data, ppo.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("PPO_5_20_15")

        macd = temp_data[35, :] - temp_data[27, :]
        temp_data = torch.cat((temp_data, macd.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("MACD_5_20")

        atr = calculate_average_true_range(temp_data, lookback=5).unsqueeze(0)
        temp_data = torch.cat((temp_data, atr), dim=0)
        if i == 0:
            columns.append("ATR_5")
        # more metrics
        clv = calculate_close_line_values(temp_raw_data).unsqueeze(0)
        temp_data = torch.cat((temp_data, clv), dim=0)
        if i == 0:
            columns.append("CLV")

        ad = calculate_accumulation_distribution_index(
            temp_raw_data, lookback=5
        ).unsqueeze(0)
        temp_data = torch.cat((temp_data, ad), dim=0)
        if i == 0:
            columns.append("AD_5")

        ad = calculate_accumulation_distribution_index(
            temp_raw_data, lookback=10
        ).unsqueeze(0)
        temp_data = torch.cat((temp_data, ad), dim=0)
        if i == 0:
            columns.append("AD_10")

        ad = calculate_accumulation_distribution_index(
            temp_raw_data, lookback=20
        ).unsqueeze(0)
        temp_data = torch.cat((temp_data, ad), dim=0)
        if i == 0:
            columns.append("AD_20")

        vpt = calculate_volume_price_trend(temp_raw_data, lookback=5).unsqueeze(0)
        temp_data = torch.cat((temp_data, vpt), dim=0)
        if i == 0:
            columns.append("VPT_5")

        vpt = calculate_volume_price_trend(temp_raw_data, lookback=10).unsqueeze(0)
        temp_data = torch.cat((temp_data, vpt), dim=0)
        if i == 0:
            columns.append("VPT_10")

        vpt = calculate_volume_price_trend(temp_raw_data, lookback=20).unsqueeze(0)
        temp_data = torch.cat((temp_data, vpt), dim=0)
        if i == 0:
            columns.append("VPT_20")

        chaikin = calculate_ema(
            calculate_accumulation_distribution_index(temp_raw_data, lookback=1),
            lookback=3,
        ) - calculate_ema(
            calculate_accumulation_distribution_index(temp_raw_data, lookback=1),
            lookback=10,
        )
        temp_data = torch.cat((temp_data, chaikin.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("Chaikin_3_10_7")

        chaikin = calculate_ema(
            calculate_accumulation_distribution_index(temp_raw_data, lookback=1),
            lookback=5,
        ) - calculate_ema(
            calculate_accumulation_distribution_index(temp_raw_data, lookback=1),
            lookback=10,
        )
        temp_data = torch.cat((temp_data, chaikin.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("Chaikin_5_10_5")

        chaikin = calculate_ema(
            calculate_accumulation_distribution_index(temp_raw_data, lookback=1),
            lookback=5,
        ) - calculate_ema(
            calculate_accumulation_distribution_index(temp_raw_data, lookback=1),
            lookback=20,
        )
        temp_data = torch.cat((temp_data, chaikin.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("Chaikin_5_20_15")

        chaikin = calculate_ema(
            calculate_accumulation_distribution_index(temp_raw_data, lookback=1),
            lookback=10,
        ) - calculate_ema(
            calculate_accumulation_distribution_index(temp_raw_data, lookback=1),
            lookback=20,
        )
        temp_data = torch.cat((temp_data, chaikin.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("Chaikin_10_20_10")

        mfi = calculate_money_flow_index(temp_raw_data, lookback=5).unsqueeze(0)
        temp_data = torch.cat((temp_data, mfi), dim=0)
        if i == 0:
            columns.append("MFI_5")

        mfi = calculate_money_flow_index(temp_raw_data, lookback=10).unsqueeze(0)
        temp_data = torch.cat((temp_data, mfi), dim=0)
        if i == 0:
            columns.append("MFI_10")

        mfi = calculate_money_flow_index(temp_raw_data, lookback=20).unsqueeze(0)
        temp_data = torch.cat((temp_data, mfi), dim=0)
        if i == 0:
            columns.append("MFI_20")

        prices = temp_raw_data
        temp_data = torch.cat((temp_data, prices), dim=0)
        if i == 0:
            columns.append("close")
            columns.append("high")
            columns.append("low")
            columns.append("open")
            columns.append("volume")

        macd_signal = calculate_ema(temp_data[30, :], lookback=9)
        temp_data = torch.cat((temp_data, macd_signal.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("macd_signal_10_20_9")

        macd_histogram = temp_data[30, :] - macd_signal
        temp_data = torch.cat((temp_data, macd_histogram.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("macd_histogram_10_20_9")

        macd_signal = calculate_ema(temp_data[37, :], lookback=5)
        temp_data = torch.cat((temp_data, macd_signal.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("macd_signal_5_10_5")

        macd_histogram = temp_data[37, :] - macd_signal
        temp_data = torch.cat((temp_data, macd_histogram.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("macd_histogram_5_10_5")

        macd_signal = calculate_ema(temp_data[39, :], lookback=9)
        temp_data = torch.cat((temp_data, macd_signal.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("macd_signal_5_20_9")

        macd_histogram = temp_data[39, :] - macd_signal
        temp_data = torch.cat((temp_data, macd_histogram.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("macd_histogram_5_20_9")

        rsi = calculate_rsi(temp_raw_data[0, :], lookback=7)
        temp_data = torch.cat((temp_data, rsi.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("rsi_7")

        rsi = calculate_rsi(temp_raw_data[0, :], lookback=14)
        temp_data = torch.cat((temp_data, rsi.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("rsi_14")

        rsi = calculate_rsi(temp_raw_data[0, :], lookback=21)
        temp_data = torch.cat((temp_data, rsi.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("rsi_21")

        k_percent, d_percent = calculate_stochastic_oscillator(
            temp_raw_data[1, :],
            temp_raw_data[2, :],
            temp_raw_data[0, :],
            k_lookback=14,
            d_lookback=3,
        )
        temp_data = torch.cat((temp_data, k_percent.unsqueeze(0)), dim=0)
        temp_data = torch.cat((temp_data, d_percent.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("k_percent_14")
            columns.append("d_percent_14_3")

        data.append(
            temp_data[:, 20:].unsqueeze(-1)
        )  # getting rid of some trashy-ish data points

    data = torch.cat(data, dim=-1)  # (target_dates+features, time series, tickers)
    data_length = data.shape[1]

    # adding more complex relationships

    # adding VIX data
    vix_data = yf.download(
        "^VIX", start=start_date, end=end_date, progress=False, auto_adjust=False
    )
    aligned_vix_data = pd.DataFrame(columns=vix_data.columns)
    for column in vix_data.columns.levels[0]:
        aligned_vix_data[column, "^VIX"] = align_financial_dataframes(
            {column: vix_data},
            target_column=column,
            fill_method="ffill",
            min_date=start_date,
            max_date=end_date,
        )
    vix_data = aligned_vix_data
    vix_data = vix_data["Close"].to_numpy()
    vix_data = torch.tensor(vix_data, dtype=torch.float32)
    vix_wider_data, vix_wider_columns = calculate_wider_economics_indicators(
        vix_data, "vix"
    )
    vix_wider_data = vix_wider_data[:, : -max(target_dates)]
    vix_wider_data = vix_wider_data[:, 20:]
    # vix_wider_data = vix_wider_data.unsqueeze(-1)
    vix_wider_data = vix_wider_data.tile(1, 1, len(tickers))
    data = torch.cat((data, vix_wider_data), dim=0)
    columns = columns + vix_wider_columns

    # TODO find 2 year yields
    US_treasury_yields = yf.download(
        "^TNX", start=start_date, end=end_date, progress=False
    )
    aligned_US_treasury_yields = pd.DataFrame(
        columns=US_treasury_yields.columns,
    )
    for column in US_treasury_yields.columns.levels[0]:
        aligned_US_treasury_yields[column, "^TNX"] = align_financial_dataframes(
            {column: US_treasury_yields},
            target_column=column,
            fill_method="ffill",
            min_date=start_date,
            max_date=end_date,
        )
    US_treasury_yields = aligned_US_treasury_yields
    US_treasury_yields = US_treasury_yields["Close"].to_numpy()
    wider_data = torch.tensor(US_treasury_yields, dtype=torch.float32).unsqueeze(0)
    wider_columns = ["10_year_treasury_yield_close"]

    US_treasury_yields = yf.download(
        "^FVX", start=start_date, end=end_date, progress=False
    )
    aligned_US_treasury_yields = pd.DataFrame(
        columns=US_treasury_yields.columns,
    )
    for column in US_treasury_yields.columns.levels[0]:
        aligned_US_treasury_yields[column, "^FVX"] = align_financial_dataframes(
            {column: US_treasury_yields},
            target_column=column,
            fill_method="ffill",
            min_date=start_date,
            max_date=end_date,
        )
    US_treasury_yields = aligned_US_treasury_yields
    US_treasury_yields = US_treasury_yields["Close"].to_numpy()
    wider_data = torch.cat(
        (
            wider_data,
            torch.tensor(US_treasury_yields, dtype=torch.float32).unsqueeze(0),
        ),
        dim=0,
    )
    wider_columns.append("5_year_treasury_yield_close")

    US_treasury_yields = yf.download(
        "^IRX", start=start_date, end=end_date, progress=False
    )
    aligned_US_treasury_yields = pd.DataFrame(
        columns=US_treasury_yields.columns,
    )
    for column in US_treasury_yields.columns.levels[0]:
        aligned_US_treasury_yields[column, "^IRX"] = align_financial_dataframes(
            {column: US_treasury_yields},
            target_column=column,
            fill_method="ffill",
            min_date=start_date,
            max_date=end_date,
        )
    US_treasury_yields = aligned_US_treasury_yields
    US_treasury_yields = US_treasury_yields["Close"].to_numpy()
    wider_data = torch.cat(
        (
            wider_data,
            torch.tensor(US_treasury_yields, dtype=torch.float32).unsqueeze(0),
        ),
        dim=0,
    )
    wider_columns.append("3_month_treasury_yield_close")

    spread = wider_data[0, :] - wider_data[1, :]
    wider_data = torch.cat((wider_data, spread.unsqueeze(0)), dim=0)
    wider_columns.append("yield_spread_10y_5y")

    spread = wider_data[0, :] - wider_data[2, :]
    wider_data = torch.cat((wider_data, spread.unsqueeze(0)), dim=0)
    wider_columns.append("yield_spread_10y_3m")

    spread = wider_data[1, :] - wider_data[2, :]
    wider_data = torch.cat((wider_data, spread.unsqueeze(0)), dim=0)
    wider_columns.append("yield_spread_5y_3m")

    change = wider_data[0, 1:] - wider_data[0, :-1]
    change = torch.cat((change[:1, :], change), dim=0)
    wider_data = torch.cat((wider_data, change.unsqueeze(0)), dim=0)
    wider_columns.append("yield_change_10y_abs")

    change = wider_data[1, 1:] - wider_data[1, :-1]
    change = torch.cat((change[:1, :], change), dim=0)
    wider_data = torch.cat((wider_data, change.unsqueeze(0)), dim=0)
    wider_columns.append("yield_change_5y_abs")

    change = wider_data[2, 1:] - wider_data[2, :-1]
    change = torch.cat((change[:1, :], change), dim=0)
    wider_data = torch.cat((wider_data, change.unsqueeze(0)), dim=0)
    wider_columns.append("yield_change_3m_abs")

    sma = calculate_moving_average(wider_data, lookback=20, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("yield_10y_sma_20")

    sma = calculate_moving_average(wider_data, lookback=20, dim=1)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("yield_5y_sma_20")

    sma = calculate_moving_average(wider_data, lookback=20, dim=2)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("yield_3m_sma_20")

    sma = calculate_moving_average(wider_data, lookback=20, dim=3)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("yield_spread_10y_5y_sma_20")

    sma = calculate_moving_average(wider_data, lookback=20, dim=4)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("yield_spread_10y_3m_sma_20")

    sma = calculate_moving_average(wider_data, lookback=20, dim=5)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("yield_spread_5y_3m_sma_20")

    sma = calculate_moving_average(wider_data, lookback=20, dim=6)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("yield_change_10y_abs_sma_20")

    sma = calculate_moving_average(wider_data, lookback=20, dim=7)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("yield_change_5y_abs_sma_20")

    sma = calculate_moving_average(wider_data, lookback=20, dim=8)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("yield_change_3m_abs_sma_20")

    # ema = calculate_ema()
    # can add more things (like change in yields, ema, etc.)

    gold = yf.download(
        "GC=F",
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=False,
        back_adjust=False,
    )
    aligned_gold = pd.DataFrame(columns=gold.columns)
    for column in gold.columns.levels[0]:
        aligned_gold[column, "GC=F"] = align_financial_dataframes(
            {column: gold},
            target_column=column,
            fill_method="ffill",
            min_date=start_date,
            max_date=end_date,
        )
    gold = aligned_gold
    gold = gold["Close"].to_numpy()
    gold = torch.tensor(gold, dtype=torch.float32)
    gold_wider_data, gold_wider_columns = calculate_wider_economics_indicators(
        gold, "gold"
    )
    wider_data = torch.cat((wider_data, gold_wider_data), dim=0)
    wider_columns = wider_columns + gold_wider_columns

    crude_oil = yf.download(
        "CL=F",
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=False,
        back_adjust=False,
    )
    aligned_crude_oil = pd.DataFrame(columns=crude_oil.columns)
    for column in crude_oil.columns.levels[0]:
        aligned_crude_oil[column, "CL=F"] = align_financial_dataframes(
            {column: crude_oil},
            target_column=column,
            fill_method="ffill",
            min_date=start_date,
            max_date=end_date,
        )
    crude_oil = aligned_crude_oil
    crude_oil = crude_oil["Close"].to_numpy()
    crude_oil = torch.tensor(crude_oil, dtype=torch.float32)
    crude_oil_wider_data, crude_oil_wider_columns = (
        calculate_wider_economics_indicators(crude_oil, "crude_oil")
    )
    wider_data = torch.cat((wider_data, crude_oil_wider_data), dim=0)
    wider_columns = wider_columns + crude_oil_wider_columns

    copper = yf.download(
        "HG=F",
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=False,
        back_adjust=False,
    )
    aligned_copper = pd.DataFrame(columns=copper.columns)
    for column in copper.columns.levels[0]:
        aligned_copper[column, "HG=F"] = align_financial_dataframes(
            {column: copper},
            target_column=column,
            fill_method="ffill",
            min_date=start_date,
            max_date=end_date,
        )
    copper = aligned_copper
    copper = copper["Close"].to_numpy()
    copper = torch.tensor(copper, dtype=torch.float32)
    copper_wider_data, copper_wider_columns = calculate_wider_economics_indicators(
        copper, "copper"
    )
    wider_data = torch.cat((wider_data, copper_wider_data), dim=0)
    wider_columns = wider_columns + copper_wider_columns

    silver = yf.download(
        "SI=F",
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=False,
        back_adjust=False,
    )
    aligned_silver = pd.DataFrame(columns=silver.columns)
    for column in silver.columns.levels[0]:
        aligned_silver[column, "SI=F"] = align_financial_dataframes(
            {column: silver},
            target_column=column,
            fill_method="ffill",
            min_date=start_date,
            max_date=end_date,
        )
    silver = aligned_silver
    silver = silver["Close"].to_numpy()
    silver = torch.tensor(silver, dtype=torch.float32)
    silver_wider_data, silver_wider_columns = calculate_wider_economics_indicators(
        silver, "silver"
    )
    wider_data = torch.cat((wider_data, silver_wider_data), dim=0)
    wider_columns = wider_columns + silver_wider_columns

    usd_index = yf.download(
        "DX-Y.NYB",
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=False,
        back_adjust=False,
    )
    aligned_usd_index = pd.DataFrame(columns=usd_index.columns)
    for column in usd_index.columns.levels[0]:
        aligned_usd_index[column, "DX-Y.NYB"] = align_financial_dataframes(
            {column: usd_index},
            target_column=column,
            fill_method="ffill",
            min_date=start_date,
            max_date=end_date,
        )
    usd_index = aligned_usd_index
    usd_index = usd_index["Close"].to_numpy()
    usd_index = torch.tensor(usd_index, dtype=torch.float32)
    usd_index_wider_data, usd_index_columns = calculate_wider_economics_indicators(
        usd_index, "usd_index"
    )
    wider_data = torch.cat((wider_data, usd_index_wider_data))
    wider_columns = wider_columns + usd_index_columns

    wider_data = wider_data[:, : -max(target_dates)]
    wider_data = wider_data.tile(1, 1, len(tickers))
    wider_data = wider_data[:, 20:, :]

    data = torch.cat((data, wider_data), dim=0)
    columns = columns + wider_columns

    # alpha calculation
    snp_500 = yf.download(
        "^GSPC",
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=False,
        back_adjust=False,
    )
    aligned_snp_500 = pd.DataFrame(columns=snp_500.columns)
    for column in snp_500.columns.levels[0]:
        aligned_snp_500[column, "^GSPC"] = align_financial_dataframes(
            {column: snp_500},
            target_column=column,
            fill_method="ffill",
            min_date=start_date,
            max_date=end_date,
        )
    snp_500 = aligned_snp_500
    snp_500 = snp_500["Close"].to_numpy()
    snp_500 = torch.tensor(snp_500, dtype=torch.float32)
    snp_500 = snp_500[: -max(target_dates)]
    snp_500_returns = (snp_500[1:] - snp_500[:-1]) / snp_500[:-1]
    snp_500_returns = torch.cat((snp_500[0:1], snp_500_returns), dim=0)
    snp_500_returns = snp_500_returns[20:]
    alphas = data[0, :, :] - snp_500_returns.repeat(1, len(tickers))
    data = torch.cat((data, alphas.unsqueeze(0)), dim=0)
    columns.append("alpha_snp_500_returns")

    temp = []
    for i in range(len(tickers)):
        ema = calculate_ema(alphas[:, i], lookback=5)
        temp.append(ema)
    data = torch.cat((data, torch.stack(temp, dim=1).unsqueeze(0)), dim=0)
    columns.append("alpha_snp_500_returns_ema_5")

    temp = []
    for i in range(len(tickers)):
        ema = calculate_ema(alphas[:, i], lookback=10)
        temp.append(ema)
    data = torch.cat((data, torch.stack(temp, dim=1).unsqueeze(0)), dim=0)
    columns.append("alpha_snp_500_returns_ema_10")

    temp = []
    for i in range(len(tickers)):
        ema = calculate_ema(alphas[:, i], lookback=20)
        temp.append(ema)
    data = torch.cat((data, torch.stack(temp, dim=1).unsqueeze(0)), dim=0)
    columns.append("alpha_snp_500_returns_ema_20")

    # relative volatility
    temp = []
    for i in range(len(tickers)):
        ...

    # time data
    time_columns = []
    day_of_week = torch.tensor(indexes.dayofweek, dtype=torch.float32)
    sin_dow, cos_dow = cyclical_encode(day_of_week, 7)
    time_data = sin_dow.unsqueeze(0)
    time_data = torch.cat((time_data, cos_dow.unsqueeze(0)), dim=0)
    time_columns.append("sin_day_of_week")
    time_columns.append("cos_day_of_week")

    day_of_month = torch.tensor(indexes.day, dtype=torch.float32)
    sin_dom, cos_dom = cyclical_encode(day_of_month, 31)
    time_data = torch.cat((time_data, sin_dom.unsqueeze(0)), dim=0)
    time_data = torch.cat((time_data, cos_dom.unsqueeze(0)), dim=0)
    time_columns.append("sin_day_of_month")
    time_columns.append("cos_day_of_month")

    day_of_year = torch.tensor(indexes.dayofyear, dtype=torch.float32)
    sin_doy, cos_doy = cyclical_encode(day_of_year, 366)
    time_data = torch.cat((time_data, sin_doy.unsqueeze(0)), dim=0)
    time_data = torch.cat((time_data, cos_doy.unsqueeze(0)), dim=0)
    time_columns.append("sin_day_of_year")
    time_columns.append("cos_day_of_year")

    month = torch.tensor(indexes.month, dtype=torch.float32)
    sin_month, cos_month = cyclical_encode(month, 12)
    time_data = torch.cat((time_data, sin_month.unsqueeze(0)), dim=0)
    time_data = torch.cat((time_data, cos_month.unsqueeze(0)), dim=0)
    time_columns.append("sin_month")
    time_columns.append("cos_month")

    quarter = torch.tensor(indexes.quarter, dtype=torch.float32)
    sin_quarter, cos_quarter = cyclical_encode(quarter, 4)
    time_data = torch.cat((time_data, sin_quarter.unsqueeze(0)), dim=0)
    time_data = torch.cat((time_data, cos_quarter.unsqueeze(0)), dim=0)
    time_columns.append("sin_quarter")
    time_columns.append("cos_quarter")

    is_leap_year = torch.tensor(indexes.is_leap_year, dtype=torch.float32)
    time_data = torch.cat((time_data, is_leap_year.unsqueeze(0)), dim=0)
    time_columns.append("is_leap_year")

    is_month_start = torch.tensor(indexes.is_month_start, dtype=torch.float32)
    time_data = torch.cat((time_data, is_month_start.unsqueeze(0)), dim=0)
    time_columns.append("is_month_start")

    is_month_end = torch.tensor(indexes.is_month_end, dtype=torch.float32)
    time_data = torch.cat((time_data, is_month_end.unsqueeze(0)), dim=0)
    time_columns.append("is_month_end")

    is_quarter_start = torch.tensor(indexes.is_quarter_start, dtype=torch.float32)
    time_data = torch.cat((time_data, is_quarter_start.unsqueeze(0)), dim=0)
    time_columns.append("is_quarter_start")

    is_quarter_end = torch.tensor(indexes.is_quarter_end, dtype=torch.float32)
    time_data = torch.cat((time_data, is_quarter_end.unsqueeze(0)), dim=0)
    time_columns.append("is_quarter_end")

    time_data = time_data[:, : -max(target_dates)]
    time_data = time_data[:, 20:].unsqueeze(-1)
    time_data = time_data.tile(1, 1, len(tickers))
    data = torch.cat((data, time_data), dim=0)
    columns = columns + time_columns

    # adding targets
    target_data = []
    for i in range(len(tickers)):
        temp_raw_data = raw_data[:, :, i]
        reference_period = temp_raw_data[0, : -max(target_dates)]
        temp_data = (
            (temp_raw_data[0, 1 : -(max(target_dates) - 1)] - reference_period)
            / reference_period
        ).unsqueeze(0)
        for j in range(1, len(target_dates)):
            if target_dates[j] == max(target_dates):
                temp_data = torch.cat(
                    (
                        temp_data,
                        (
                            (temp_raw_data[0, max(target_dates) :] - reference_period)
                            / reference_period
                        ).unsqueeze(0),
                    ),
                    dim=0,
                )
            else:
                temp_data = torch.cat(
                    (
                        temp_data,
                        (
                            (
                                temp_raw_data[
                                    0,
                                    target_dates[j] : -(
                                        max(target_dates) - target_dates[j]
                                    ),
                                ]
                                - reference_period
                            )
                            / reference_period
                        ).unsqueeze(0),
                    ),
                    dim=0,
                )

        target_data.append(temp_data.unsqueeze(-1))

    target_data = torch.cat(target_data, dim=-1)
    target_data = target_data[:, 20:]
    data = torch.cat((target_data, data), dim=0)

    column_to_id = {column: i for i, column in enumerate(columns)}

    data[7, 5909] = data[
        7, 5908
    ]  # ffil quickfix for now, (no clue why inf all of a sudden)
    if (torch.isnan(data)).any() or (torch.isinf(data)).any():
        print("Data contains NaN or Inf values.")

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
    ADRr = data[8, :]
    ATRr = []
    for i in range(data.shape[1]):
        ATRr.append(ADRr[max(i - lookback + 1, 0) : i + 1].sum() / min(lookback, i + 1))
    return torch.tensor(ATRr)


def calculate_moving_average(
    data: torch.Tensor, lookback: int = 10, dim: int = 0
) -> torch.Tensor:
    price = data[dim, :]
    SMAr = torch.zeros_like(price)
    for i in range(data.shape[1]):
        SMAr[i] = price[max(i - lookback + 1, 0) : i + 1].mean()
    return SMAr


def calculate_moving_average_returns(
    data: torch.Tensor, lookback: int = 10, dim: int = 0
) -> torch.Tensor:
    price = data[dim, :]
    SMAR = (price[lookback:] - price[:-lookback]) / lookback
    temp = torch.zeros(lookback)
    for i in range(1, lookback):
        temp[i] = (price[i] - price[0]) / i
    temp[0] = temp[1]
    return torch.cat((temp, SMAR))


def calculate_volatility_returns(
    data: torch.Tensor, lookback: int = 30, dim: int = 0
) -> torch.Tensor:
    returns = (data[dim, 1:] - data[dim, :-1]) / data[
        dim, :-1
    ]  # from adj. Close for now
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


def calculate_close_line_values(data: torch.Tensor) -> torch.Tensor:
    close = data[0, :]
    high = data[1, :]
    low = data[2, :]
    clv = (2 * close - high - low) / (high - low)
    return clv


def calculate_accumulation_distribution_index(data: torch.Tensor, lookback: int = 10):
    volume = data[4, :]
    clv = calculate_close_line_values(data)
    ad = torch.zeros(data.shape[1])
    for i in range(data.shape[1]):
        ad[i] = (
            volume[max(i - lookback + 1, 0) : i + 1]
            * clv[max(i - lookback + 1, 0) : i + 1]
        ).sum()
    return ad


def calculate_volume_price_trend(data: torch.Tensor, lookback: int = 10):
    volume = data[4, :]
    price = data[0, :]
    returns = torch.zeros(data.shape[1])
    returns[1:] = (price[1:] - price[:-1]) / price[:-1]
    returns[0] = returns[1]
    vpt = torch.zeros(data.shape[1])
    for i in range(data.shape[1]):
        vpt[i] = (
            volume[max(i - lookback + 1, 0) : i + 1]
            * returns[max(i - lookback + 1, 0) : i + 1]
        ).sum()
    return vpt


def calculate_ema(
    daily_values: torch.Tensor, lookback: int = 10
) -> torch.Tensor:  # to calculate ema for anything
    ema = torch.zeros(daily_values.shape[0])
    multiplier = 2 / (lookback + 1)
    for i in range(daily_values.shape[0]):
        if i == 0:
            ema[i] = daily_values[i] * multiplier
        else:
            ema[i] = daily_values[i] * multiplier + ema[i - 1] * (1 - multiplier)
    return ema


def calculate_money_flow_index(data: torch.Tensor, lookback: int = 10):
    volume = data[4, :]
    close = data[0, :]
    high = data[1, :]
    low = data[2, :]
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    signs = torch.zeros(data.shape[1])
    signs[1:] = torch.sign(typical_price[1:] - typical_price[:-1])
    mfi = torch.zeros(data.shape[1])
    for i in range(data.shape[1]):
        pos_flow = (
            money_flow[max(i - lookback + 1, 0) : i + 1]
            * (signs[max(i - lookback + 1, 0) : i + 1] == 1)
        ).sum()
        mfi[i] = pos_flow / (money_flow[max(i - lookback + 1, 0) : i + 1]).sum()
    return mfi


def calculate_rsi(prices: torch.Tensor, lookback: int = 14):
    deltas = prices[1:] - prices[:-1]
    gains = torch.zeros_like(deltas)
    losses = torch.zeros_like(deltas)
    gains[deltas > 0] = deltas[deltas > 0]
    losses[deltas < 0] = -deltas[deltas < 0]  # Losses are positive values

    avg_gain = torch.zeros_like(prices)
    avg_loss = torch.zeros_like(prices)

    if len(deltas) >= lookback:
        avg_gain[lookback] = gains[:lookback].mean()
        avg_loss[lookback] = losses[:lookback].mean()
        for k_idx in range(lookback + 1, len(prices)):
            avg_gain[k_idx] = (
                avg_gain[k_idx - 1] * (lookback - 1) + gains[k_idx - 1]
            ) / lookback
            avg_loss[k_idx] = (
                avg_loss[k_idx - 1] * (lookback - 1) + losses[k_idx - 1]
            ) / lookback

    rs = torch.zeros_like(avg_gain)
    valid_loss_mask = avg_loss > 1e-12  # Avoid division by zero
    rs[valid_loss_mask] = avg_gain[valid_loss_mask] / avg_loss[valid_loss_mask]

    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[~valid_loss_mask & (avg_gain > 1e-9)] = (
        100.0  # If loss is zero and gain is positive
    )
    rsi[~valid_loss_mask & (avg_gain <= 1e-9)] = (
        50.0  # If both gain and loss are zero (or neg gain)
    )

    rsi[:lookback] = 50.0  # Fill initial period with neutral value
    return rsi


def calculate_stochastic_oscillator(
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    k_lookback: int = 14,
    d_lookback: int = 3,
):
    percent_k = torch.zeros_like(close)
    for k_idx in range(len(close)):
        if k_idx < k_lookback - 1:
            percent_k[k_idx] = 50.0
            continue
        period_low = low[max(0, k_idx - k_lookback + 1) : k_idx + 1].min()
        period_high = high[max(0, k_idx - k_lookback + 1) : k_idx + 1].max()
        if period_high == period_low:
            percent_k[k_idx] = percent_k[k_idx - 1] if k_idx > 0 else 50.0
        else:
            percent_k[k_idx] = (
                100 * (close[k_idx] - period_low) / (period_high - period_low)
            )

    percent_d = calculate_moving_average(
        percent_k.unsqueeze(0), lookback=d_lookback, dim=0
    )  # Use your existing SMA for %D
    return percent_k, percent_d


def cyclical_encode(data, max_val):
    sin_feat = torch.sin(2 * torch.pi * data / max_val)
    cos_feat = torch.cos(2 * torch.pi * data / max_val)
    return sin_feat, cos_feat


def align_financial_dataframes(
    dataframes_dict: Dict[str, pd.DataFrame],
    target_column: str = "Close",
    fill_method: str = "ffill",
    min_date: str = "2000-01-01",
    max_date: str = "2025-01-01",
) -> pd.DataFrame:
    """
    Aligns multiple financial DataFrames (e.g., from yfinance) to a common
    DateTimeIndex and forward-fills missing values.

    Args:
        dataframes_dict (Dict[str, pd.DataFrame]):
            A dictionary where keys are asset names/symbols and values are
            pandas DataFrames. Each DataFrame must have a DateTimeIndex.
        target_column (str):
            The column to primarily use from each DataFrame (e.g., 'Close', 'Adj Close').
            Other columns might also be present and will be aligned.
        fill_method (str):
            Method to fill NaNs after reindexing. 'ffill' (forward-fill) is common.
            'bfill' (backward-fill) is another option. Set to None to skip filling.

    Returns:
        pd.DataFrame: A single DataFrame with all assets aligned to a common
                      DateTimeIndex, containing the target_column for each asset.
                      Column names will be the keys from dataframes_dict.
    """
    if not dataframes_dict:
        return pd.DataFrame()

    processed_series_list = []
    asset_names = []

    # Determine the overall date range
    min_date = None
    max_date = None

    for asset_name, df in dataframes_dict.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"DataFrame for {asset_name} does not have a DateTimeIndex."
            )

        if df.empty or target_column not in df.columns:
            print(
                f"Warning: DataFrame for {asset_name} is empty or missing target column '{target_column}'. Skipping."
            )
            continue

        # Ensure index is timezone-naive for consistent comparison (yfinance can sometimes add tz)
        df.index = df.index.tz_localize(None)

        if min_date is None or df.index.min() < min_date:
            min_date = df.index.min()
        if max_date is None or df.index.max() > max_date:
            max_date = df.index.max()

        # Keep all columns initially, we'll select target_column later if needed
        # or rename it to be asset-specific if keeping multiple columns
        asset_series = df[[target_column]].copy()  # Select the target column
        asset_series.rename(columns={target_column: asset_name}, inplace=True)
        processed_series_list.append(asset_series)
        asset_names.append(asset_name)

    if not processed_series_list:
        print("Warning: No valid DataFrames to align.")
        return pd.DataFrame()

    # Create a continuous date range for all business days (or all days if preferred)
    # Using all days and then ffill is robust to different holiday schedules
    all_dates_index = pd.date_range(
        start=min_date, end=max_date, freq="B"
    )  # 'B' for business days
    # If you want ALL calendar days (then ffill will carry over weekends):
    # all_dates_index = pd.date_range(start=min_date, end=max_date, freq='D')

    # Concatenate all series. This will align existing dates and introduce NaNs elsewhere.
    # Using outer join to keep all dates from all series initially, then reindex.
    if len(processed_series_list) > 1:
        aligned_df = pd.concat(processed_series_list, axis=1, join="outer")
    else:
        aligned_df = processed_series_list[0]

    # Reindex to the common, continuous date range
    aligned_df = aligned_df.reindex(all_dates_index)

    # Fill missing values (typically forward-fill for financial series)
    if fill_method:
        # aligned_df.fillna(method=fill_method, inplace=True)
        aligned_df = aligned_df.ffill()

    # Optional: remove rows where ALL values are still NaN (e.g. very early dates before any asset started)
    aligned_df.dropna(axis=0, how="all", inplace=True)

    return aligned_df


def calculate_wider_economics_indicators(indicator: torch.Tensor, indicator_name: str):
    change = (indicator[1:] - indicator[:-1]) / indicator[:-1]
    change = torch.cat((change[:1, :], change), dim=0)
    indicator_data = torch.cat((indicator.unsqueeze(0), change.unsqueeze(0)), dim=0)
    indicator_columns = [indicator_name + "_close", indicator_name + "_change"]

    # removed because duplicates????
    # sma = calculate_moving_average(indicator.unsqueeze(0), lookback=5, dim=0)
    # indicator_data = torch.cat((indicator_data, sma.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_sma_5")

    # sma = calculate_moving_average(indicator.unsqueeze(0), lookback=10, dim=0)
    # indicator_data = torch.cat((indicator_data, sma.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_sma_10")

    # sma = calculate_moving_average(indicator.unsqueeze(0), lookback=20, dim=0)
    # indicator_data = torch.cat((indicator_data, sma.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_sma_20")

    ema = calculate_ema(indicator, lookback=5)
    indicator_data = torch.cat((indicator_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    indicator_columns.append(indicator_name + "_ema_5")

    ema = calculate_ema(indicator, lookback=10)
    indicator_data = torch.cat((indicator_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    indicator_columns.append(indicator_name + "_ema_10")

    ema = calculate_ema(indicator, lookback=20)
    indicator_data = torch.cat((indicator_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    indicator_columns.append(indicator_name + "_ema_20")

    # sma = calculate_moving_average(change.unsqueeze(0), lookback=5, dim=0)
    # indicator_data = torch.cat((indicator_data, sma.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_change_sma_5")

    # sma = calculate_moving_average(change.unsqueeze(0), lookback=10, dim=0)
    # indicator_data = torch.cat((indicator_data, sma.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_change_sma_10")

    # sma = calculate_moving_average(change.unsqueeze(0), lookback=20, dim=0)
    # indicator_data = torch.cat((indicator_data, sma.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_change_sma_20")

    ema = calculate_ema(change, lookback=5)
    indicator_data = torch.cat((indicator_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    indicator_columns.append(indicator_name + "_change_ema_5")

    ema = calculate_ema(change, lookback=10)
    indicator_data = torch.cat((indicator_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    indicator_columns.append(indicator_name + "_change_ema_10")

    ema = calculate_ema(change, lookback=20)
    indicator_data = torch.cat((indicator_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    indicator_columns.append(indicator_name + "_change_ema_20")

    return indicator_data, indicator_columns


# D. Broader Market / Economic Indicators (using yfinance):
# You're already using VIX. Consider adding:
# Interest Rates:
# 10-Year Treasury Yield (^TNX)
# 2-Year Treasury Yield (^FVX)
# Federal Funds Rate (might need a different source or find a proxy ETF if daily updates are needed, e.g., FEDFUNDS from FRED, but yfinance might not have it directly).
# Processing: Download their historical series. Calculate returns/changes, SMAs, EMAs just like you did for VIX. Concatenate these to your data tensor, repeating for each stock (similar to VIX).
# Commodity Prices:
# Crude Oil (CL=F)
# Gold (GC=F)
# Process these like VIX/Interest Rates.
# US Dollar Index (DX-Y.NYB):
# Can indicate risk-on/risk-off sentiment or international capital flows.
# E. Fundamental Data (More Involved - yf.Ticker().info, .financials, etc.):
# Examples: P/E Ratio, EPS, P/S Ratio, Dividend Yield, Market Cap, Beta.
# Challenge: This data is usually reported quarterly or annually.
# Strategy:
# Fetch for each ticker (e.g., stock_info = yf.Ticker(ticker_symbol).info).
# Extract relevant metrics.
# Create a time series by forward-filling these values. For example, if Q1 EPS is reported, use that EPS value for all trading days in Q1 until Q2 EPS is reported.
# Dynamic Ratios: For ratios like P/E, you'd use the daily closing price and the forward-filled EPS: P_daily / EPS_forward_filled.
# Merge these daily-aligned fundamental features with your existing temp_data.
# Normalization: Fundamental ratios can have very different scales, so normalization will be key.
# This is a significant addition and requires careful data handling and alignment.
# F. Feature Interactions & Relative Strength:
# Stock Return vs. Market Return (Alpha Component):
# You're already downloading ^GSPC (S&P 500) as one of your tickers. If ^GSPC is at index j in your tickers list:
# # Assuming 'returns' is at index 0 of your features in temp_data
# # market_returns = data[0, :, :, j] # S&P 500 returns
# # for stock_idx in range(len(tickers)):
# #     if stock_idx == j: continue # Skip for the market index itself
# #     stock_specific_returns = data[0, :, :, stock_idx]
# #     alpha_component = stock_specific_returns - market_returns
# #     # This alpha_component needs to be added as a new feature for stock_idx
# #     # This requires careful restructuring of how features are assembled or added post-assembly.
# #     # One way: calculate all base features first, then iterate to add relative ones.
# This is a powerful concept but requires careful thought about how to structure it in your (features, time, targets, tickers) tensor. You might compute all base features, then in a second pass, compute relative features and append them.
# Volatility Relative to Market:
# stock_volatility_feature - vix_feature (after both are on comparable scales/forms, e.g., both as % or normalized).


# Okay, it sounds like you've diligently expanded your feature set with solid technical and time-based indicators! It's common for individual feature additions to yield incremental rather than breakthrough improvements, especially if the model was already reasonably good or if the new features have some overlapping information with existing ones.
# Now that you have a rich set of intra-stock and basic market context (VIX) features, the next logical steps involve:
# Deeper Contextual Understanding (More Market & Economic Data): The market doesn't operate in a vacuum.
# Capturing Cross-Asset Relationships (If Applicable): How stocks influence each other.
# Fundamental Data (A Bigger Step): Company-specific financial health.
# Advanced Feature Engineering & Interaction Terms: Creating more sophisticated signals.
# Model Architecture and Training Refinements: Ensuring your model can effectively use the features.
# Let's focus on a few high-impact areas from these.
# I. Deeper Contextual Understanding (More Market & Economic Data):
# You've got VIX. Expand this to other macro factors. This is often where significant, less correlated signals can be found.
# 1. Interest Rates (Crucial):
# Why: Rates affect discount rates, company borrowing costs, economic growth expectations, and sector rotations.
# Data:
# US Treasury Yields: Fetch via yfinance:
# 10-Year Treasury Yield: ^TNX
# 5-Year Treasury Yield: ^FVX
# 2-Year Treasury Yield: ^IRX (often represents short-term rate expectations)
# (Possibly) 3-Month Treasury Bill: ^BIL (or similar ETF)
# Yield Curve Spreads: Calculate spreads between these, e.g.:
# Spread_10Y_2Y = ^TNX - ^IRX (A common recession indicator when it inverts)
# Spread_10Y_3M
# Processing:
# Download their historical series.
# Calculate changes or returns in these rates/spreads (rates are already percentages, so a simple difference rate_t - rate_t-1 is often used).
# Also, consider their SMAs, EMAs.
# Concatenate these derived rate features to your data tensor, repeating for each stock (just like you did for VIX).
# Column Names: Add appropriate names like vix_Close, vix_Returns, tnx_Close, tnx_Change, spread_10y_2y_Value.
# 2. Commodity Prices:
# Why: Can indicate inflation, industrial demand, and global economic health.
# Data:
# Crude Oil Futures: CL=F
# Gold Futures: GC=F
# (Optional) Copper Futures: HG=F (often seen as an economic bellwether, "Dr. Copper")
# Processing:
# Calculate returns or log returns.
# SMAs, EMAs of these returns or prices.
# Add to your global context features, repeated per stock.
# 3. US Dollar Index (DXY):
# Why: Affects multinational company earnings, commodity prices, and capital flows.
# Data: DX-Y.NYB (ICE US Dollar Index Futures)
# Processing: Returns, SMAs, EMAs.
# II. Advanced Feature Engineering & Interaction Terms (using existing & new features):
# Once you have more contextual features, you can create interaction terms or relative strength indicators.
# 1. Stock Return vs. Market Return (Relative Strength / "Alpha"):
# You have individual stock returns and (if ^GSPC is one of your tickers) S&P 500 returns.
# Feature: Stock_Return_t - Market_Return_t
# This explicitly tells the model how much a stock is outperforming or underperforming the market on a given day. This can be a powerful feature.
# Implementation:
# Calculate this after you have the base return for each stock and the market index.
# This feature will be specific to each stock (except for the market index itself, where it would be 0).
# You'll need to add this as a new feature row in your data tensor. If data is (features, time, target_dim, tickers), you'd compute this and then torch.cat it along the features dimension. Ensure the new column names are added.
# 2. Volatility Spreads / Ratios:
# Stock HV vs. VIX: Normalized_Stock_HV_t - Normalized_VIX_t (or ratio). Is the stock more or less volatile than its typical relationship with market volatility?
# Short-term HV vs. Long-term HV for the same stock: HV_10_day / HV_50_day. Can indicate changing volatility regimes for the stock.
# 3. Distance from Moving Averages (Normalized):
# You have SMAs. A useful feature is how far the current price (or return) is from its moving average, normalized by volatility or the MA itself.
# Distance_SMA20 = (Close_Price_t - SMA20_t) / SMA20_t
# Or Distance_SMA20_Vol_Normalized = (Close_Price_t - SMA20_t) / ATR_t
# This quantifies "overbought/oversold" relative to a trend.
# III. Model Architecture & Training Considerations (Briefly):
# Capacity: With more features, ensure your d_model is sufficiently large to create rich embeddings. If you have ~100 features, a d_model of 128-256 might be reasonable.
# Normalization within Model: nn.RMSNorm is good. Ensure it's applied appropriately.
# Attention Mechanism: Money_former_DINT suggests you're using a custom attention. Ensure it can effectively route and weigh information from these diverse feature types.
# Learning Rate & Schedule: Adding many new features might sometimes require a bit more warmup or a slightly different learning rate initially as the model learns to incorporate new information streams.
# Implementation Steps in stocks_time_series_2.py:
# Add New Tickers for Macro Data:
# In download_numerical_financial_data, modify the initial yf.download to include ^TNX, ^FVX, ^IRX, CL=F, GC=F, DX-Y.NYB alongside your existing stock tickers and ^VIX.
# yfinance will return a MultiIndex DataFrame. You'll need to carefully extract the 'Close' price (or relevant price) for these new instruments.
# Process Macro Data:
# After downloading, loop through these new macro symbols.
# Calculate their changes/returns, SMAs, EMAs.
# Store these as separate tensors (e.g., tnx_features (num_tnx_derived_features, time), oil_features, etc.).
# For each individual stock in your main loop (for i in range(len(tickers)):), you will torch.cat these macro feature tensors (repeated or aligned) to that stock's temp_data.
# This means if a stock's data starts later than the macro data, you'll need to align them by date index first. This is the trickiest part – ensuring all time series are correctly aligned before concatenation.
# Alternatively, process all stocks first to get your data (features, time, target_dim, tickers) tensor. Then, process macro features separately to get macro_data (macro_features, time, target_dim, 1). Then, tile macro_data along the tickers dimension and torch.cat it to data along the features dimension. This requires careful date alignment between data and macro_data before the cat.
# Calculate Relative Strength/Interaction Features:
# This is best done after all base individual stock features and global macro features have been assembled into a primary data tensor.
# Example for stock return vs market return:
# # Assuming 'data' is (features, time, target_dim_in_feat, tickers)
# # And you know the feature index for 'returns' (e.g., RETURN_FEAT_IDX)
# # And you know the ticker index for S&P 500 (e.g., GSPC_TICKER_IDX)

# stock_returns = data[RETURN_FEAT_IDX, :, 0, :] # Assuming target_dim_in_feat=0 for base returns
# market_returns = data[RETURN_FEAT_IDX, :, 0, GSPC_TICKER_IDX].unsqueeze(-1) # Keep as (time, 1)

# alpha_features = stock_returns - market_returns.repeat(1, data.shape[3]) # (time, tickers)
# alpha_features[:, GSPC_TICKER_IDX] = 0 # Alpha of market against itself is 0

# # Reshape alpha_features to (1, time, 1, tickers) to cat with data
# alpha_features = alpha_features.unsqueeze(0).unsqueeze(2)
# data = torch.cat((data, alpha_features), dim=0)
# columns.append("Alpha_vs_Market")
# Prioritize:
# Interest Rates: These are often very influential. Start with 10Y and 2Y yields and their spread.
# Key Commodities: Oil and Gold.
# Relative Strength (Stock vs. Market): This is a direct measure of individual performance.
# Adding these will significantly enrich the context your model has, potentially allowing it to uncover more nuanced predictive relationships. Remember that data alignment (dates) is the most critical and often trickiest part when combining different series. Your existing structure where VIX is repeated for each stock is a good pattern to follow for other global macro features.
