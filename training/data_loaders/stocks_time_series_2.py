# TODO potentially use yahoo_fin instead (or RapidAPI) (or polygon.io for proffesional)
import os
import json
import numpy as np
import yfinance as yf
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import List, Dict, Tuple, Optional, Any


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
            return self.sequences_data.size(1) - self.seq_len - 10
        else:  # TODO?
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.preload:
            input_sequence = self.sequences_data[:, idx : idx + self.seq_len, 0, :]
            target_sequence = self.sequences_data[
                0, idx + 1 : idx + self.seq_len + 1, :, :
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
        self._load_metadata()

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
            num_workers=self.num_workers,
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
    start_date: str = "1990-01-01",  # check if all stocks have data from this date
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
    columns = list(raw_data.columns.levels[0])

    # chech which dims are open and which is close (change calcs accordingly, because rn might be using wrong dims in calculations)
    raw_data = torch.tensor(raw_data.values, dtype=torch.float32).reshape(
        -1, len(columns), len(tickers)
    )  # (Time, Features, tickers)
    raw_data = raw_data.transpose(0, 1)  # (Features, Time series, tickers)

    # TODO write somewhere else, maybe in another function
    # removing adj close
    raw_data = raw_data[1:, :, :]
    columns = columns[1:]

    data = []

    for i in range(len(tickers)):
        temp_raw_data = raw_data[:, :, i]
        returns = (temp_raw_data[:, 1:] - temp_raw_data[:, :-1]) / temp_raw_data[:, :-1]
        returns = torch.cat((torch.zeros(temp_raw_data.shape[0], 1), returns), dim=1)
        temp_data = returns

        volatility = calculate_volatility_log_ret(temp_raw_data, lookback=10).unsqueeze(
            0
        )
        temp_data = torch.cat((returns, volatility), dim=0)
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
        # adding this, because maybe this is more informative for the data format im using, as both are from returns, but before was from log returns
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
            columns.append("MACD_10_20_10")

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
            columns.append("MACD_5_10_5")

        ppo = (temp_data[35, :] - temp_data[27, :]) / temp_data[27, :]
        temp_data = torch.cat((temp_data, ppo.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("PPO_5_20_15")

        macd = temp_data[35, :] - temp_data[27, :]
        temp_data = torch.cat((temp_data, macd.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("MACD_5_20_15")

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

        # time_related_data = np.array([indexes.day_of_week, indexes.day, indexes.day_of_year, indexes.month, indexes.quarter, indexes.is_leap_year, indexes.is_month_start, indexes.is_month_end, indexes.is_quarter_start, indexes.is_quarter_end], dtype=np.float32)
        # time_related_data = torch.tensor(time_related_data, dtype=torch.float32)
        # data = torch.cat((time_related_data, data), dim=0)
        # columns = ["day_of_week", "day", "day_of_year", "month", "quarter", "is_leap_year", "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end"] + columns

        # getting returns for multiple horizons
        temp_data = temp_data[
            :, 1:
        ]  # for ease and compatibility with getting multiple targets
        if max(target_dates) != 1:
            temp_temp_data = temp_data[
                : temp_raw_data.shape[0], : -(max(target_dates) - 1)
            ].unsqueeze(
                -1
            )  # (Features, Time series, target dates)
            temp_indicators = temp_data[
                temp_raw_data.shape[0] :, : -(max(target_dates) - 1)
            ].unsqueeze(
                -1
            )  # (Features, Time series, target dates)
            for i in range(1, len(target_dates)):
                if target_dates[i] == max(target_dates):
                    temp_returns = (
                        temp_raw_data[: temp_raw_data.shape[0], target_dates[i]:]
                        - temp_raw_data[
                            : temp_raw_data.shape[0], : -(max(target_dates))
                        ]
                    ) / temp_raw_data[: temp_raw_data.shape[0], : -(max(target_dates))]
                    temp_indicators = torch.cat(
                        (
                            temp_indicators,
                            temp_data[
                                temp_raw_data.shape[0] :, target_dates[i] - 1 :
                            ].unsqueeze(-1),
                        ),
                        dim=-1,
                    )
                else:
                    temp_returns = (
                        temp_raw_data[
                            : temp_raw_data.shape[0],
                            target_dates[i] : -(max(target_dates) - target_dates[i]),
                        ]
                        - temp_raw_data[
                            : temp_raw_data.shape[0], : -(max(target_dates))
                        ]
                    ) / temp_raw_data[: temp_raw_data.shape[0], : -(max(target_dates))]
                    temp_indicators = torch.cat(
                        (
                            temp_indicators,
                            temp_data[
                                temp_raw_data.shape[0] :,
                                target_dates[i]
                                - 1 : -(max(target_dates) - target_dates[i]),
                            ].unsqueeze(-1),
                        ),
                        dim=-1,
                    )
                temp_temp_data = torch.cat(
                    (temp_temp_data, temp_returns.unsqueeze(-1)), dim=-1
                )
            temp_data = torch.cat((temp_temp_data, temp_indicators), dim=0)

        data.append(
            temp_data[:, 20:, :].unsqueeze(-1)
        )  # getting rid of some trashy-ish data points


    data = torch.cat(data, dim=-1) # (features, time series, target dates, tickers)
    data_length = data.shape[1]


    # adding VIX data
    raw_vix_data = yf.download(
        "^VIX", start=start_date, end=end_date, progress=False, auto_adjust=False
    )
    vix_columns = [raw_vix_data.columns.levels[0][1]]
    raw_vix_data = torch.tensor(raw_vix_data[vix_columns].values, dtype=torch.float32).transpose(0,1)
    # TODO if the start date changes to > 1997 ish, you can add a lot more features, because vix changes from daily to hourly (probably)
    for i in range(len(vix_columns)):
        vix_columns[i] = "vix_" + vix_columns[i]

    vix_returns = (raw_vix_data[:,1:] - raw_vix_data[:, :-1]) / raw_vix_data[:, :-1]
    vix_returns = torch.cat((vix_returns[:, :1], vix_returns), dim=1)
    vix_data = torch.cat((raw_vix_data, vix_returns), dim=0)
    vix_columns.append("vix_returns")

    sma_vix = calculate_moving_average(vix_data, lookback=5, dim=0)
    vix_data = torch.cat((vix_data, sma_vix.unsqueeze(0)), dim=0)
    vix_columns.append("vix_sma_5")

    sma_vix = calculate_moving_average(vix_data, lookback=10, dim=0)
    vix_data = torch.cat((vix_data, sma_vix.unsqueeze(0)), dim=0)
    vix_columns.append("vix_sma_10")

    sma_vix = calculate_moving_average(vix_data, lookback=20, dim=0)
    vix_data = torch.cat((vix_data, sma_vix.unsqueeze(0)), dim=0)
    vix_columns.append("vix_sma_20")

    ema_vix = calculate_ema(raw_vix_data[0,:], lookback=5)
    vix_data = torch.cat((vix_data, ema_vix.unsqueeze(0)), dim=0)
    vix_columns.append("vix_ema_5")

    ema_vix = calculate_ema(raw_vix_data[0,:], lookback=10)
    vix_data = torch.cat((vix_data, ema_vix.unsqueeze(0)), dim=0)
    vix_columns.append("vix_ema_10")

    ema_vix = calculate_ema(raw_vix_data[0,:], lookback=20)
    vix_data = torch.cat((vix_data, ema_vix.unsqueeze(0)), dim=0)
    vix_columns.append("vix_ema_20")


    # just a copy of whats above but for vix
    vix_data = vix_data[:, 1:]

    if max(target_dates) != 1:
        times_vix_data = vix_data[
            : raw_vix_data.shape[0], : -(max(target_dates) - 1)
        ].unsqueeze(
            -1
        )
        temp_vix_indicators = vix_data[
            raw_vix_data.shape[0] :, : -(max(target_dates) - 1)
        ].unsqueeze(
            -1
        )
        for i in range(1, len(target_dates)):
            if target_dates[i] == max(target_dates):
                temp_returns = (
                    raw_vix_data[: raw_vix_data.shape[0], target_dates[i] :]
                    - raw_vix_data[
                        : raw_vix_data.shape[0], : -(max(target_dates))
                    ]
                ) / raw_vix_data[: raw_vix_data.shape[0], : -(max(target_dates))]
                temp_vix_indicators = torch.cat(
                    (
                        temp_vix_indicators,
                        vix_data[
                            raw_vix_data.shape[0] :, target_dates[i] - 1 :
                        ].unsqueeze(-1),
                    ),
                    dim=-1,
                )
            else:
                temp_returns = (
                    raw_vix_data[
                        : raw_vix_data.shape[0],
                        target_dates[i] : -(max(target_dates) - target_dates[i]),
                    ]
                    - raw_vix_data[
                        : raw_vix_data.shape[0], : -(max(target_dates))
                    ]
                ) / raw_vix_data[: raw_vix_data.shape[0], : -(max(target_dates))]
                temp_vix_indicators = torch.cat(
                    (
                        temp_vix_indicators,
                        vix_data[
                            raw_vix_data.shape[0] :,
                            target_dates[i]
                            - 1 : -(max(target_dates) - target_dates[i]),
                        ].unsqueeze(-1),
                    ),
                    dim=-1,
                )
            times_vix_data = torch.cat(
                (times_vix_data, temp_returns.unsqueeze(-1)), dim=-1
            )
        vix_data = torch.cat((times_vix_data, temp_vix_indicators), dim=0)
    vix_data = vix_data[:, 20:, :].unsqueeze(-1)

    data = torch.cat((data, vix_data.repeat(1, 1, 1, len(tickers))), dim=0)
    columns = columns + vix_columns

    column_to_id = {column: i for i, column in enumerate(columns)}

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
    ADRr = data[8, :]
    ATRr = []
    for i in range(data.shape[1]):
        ATRr.append(ADRr[max(i - lookback + 1, 0) : i + 1].sum() / min(lookback, i + 1))
    return torch.tensor(ATRr)


def calculate_moving_average(
    data: torch.Tensor, lookback: int = 10, dim: int = 0
) -> torch.Tensor:
    price = data[dim, :]
    SMAr = []
    for i in range(data.shape[1]):
        SMAr.append(price[max(i - lookback + 1, 0) : i + 1].mean())
    return torch.tensor(SMAr)


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
