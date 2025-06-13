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

# TODO noise
# TODO potentially different shape for storage (already in sequence chunks) (precompute z-norm and such?)

# TODO save means and stds for inference
# TODO comparison metrics for later


# mfi was actually pos mfr
# ad index is supposed to be infinite lookback
# ^^ vpt too
# avg true range was just sma of daily range
# adr was wrong


class FinancialNumericalDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        targets_file_path: str,
        seq_len: Optional[int] = None,
        preload: bool = True,
        num_targets: int = 3,
    ):
        self.seq_len = seq_len
        self.preload = preload
        self.file_path = file_path
        self.num_targets = num_targets
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if preload:
            self.sequences_data = torch.load(file_path)
            self.targets_data = torch.load(targets_file_path)
        else:
            self.sequences_data = None

    def __len__(self) -> int:
        if self.preload:
            return self.sequences_data.size(2) - self.seq_len
        else:  # TODO?
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.preload:
            input_sequence = self.sequences_data[
                :, :, idx : idx + self.seq_len, :
            ].to(self.device)
            # target_sequence = self.sequences_data[
            #     : self.num_targets, idx : idx + self.seq_len, :
            # ]
            target_sequence = self.targets_data[:,:, idx : idx + self.seq_len, :].to(self.device)
            return input_sequence, target_sequence
        else:  # TODO?
            raise NotImplementedError


class FinancialNumericalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        train_targets_file: str,
        val_file: str,
        val_targets_file: str,
        test_file: str,
        test_targets_file: str,
        metadata_file: str,
        batch_size: int,
        num_workers: int = 0,
        seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.train_file = train_file
        self.train_targets_file = train_targets_file
        self.val_file = val_file
        self.val_targets_file = val_targets_file
        self.test_file = test_file
        self.test_targets_file = test_targets_file
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
                self.train_targets_file,
                self.seq_len,
                num_targets=len(self._metadata["target_dates"]),
            )
            self.val_dataset = FinancialNumericalDataset(
                self.val_file,
                self.val_targets_file,
                self.seq_len,
                num_targets=len(self._metadata["target_dates"]),
            )
        if stage == "test" or stage is None:
            self.test_dataset = FinancialNumericalDataset(
                self.test_file,
                self.test_targets_file,
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

    # TODO rewrite without for loop over tickers (adapt calculate functions for multiple tickers at once)
    raw_data = raw_data[1:, :, :]
    columns = columns[1:]

    # full_data = []

    # for i in range(len(tickers)):

    #     # adr_data, adr_columns = feature_adr_old(temp_raw_data[0], temp_raw_data[1], temp_raw_data[2])
    #     # temp_data = torch.cat((temp_data, adr_data), dim=0)
    #     # if i == 0:
    #     #     columns.extend(adr_columns)

    #     # price_sma_data, price_sma_columns = feature_price_sma(temp_raw_data[0])
    #     # temp_data = torch.cat((temp_data, price_sma_data), dim=0)
    #     # if i == 0:
    #     #     columns.extend(price_sma_columns)


    #     # column_to_id = {col: idx for idx, col in enumerate(columns)}

    #     # bollinger_data, bollinger_columns = feature_bollinger_bands(temp_data, column_to_id)
    #     # temp_data = torch.cat((temp_data, bollinger_data), dim=0)
    #     # if i == 0:
    #     #     columns.extend(bollinger_columns)

    #     # price_ema_data, price_ema_columns = feature_ema(temp_raw_data[0])
    #     # temp_data = torch.cat((temp_data, price_ema_data), dim=0)
    #     # if i == 0:
    #     #     columns.extend(price_ema_columns)

    #     # column_to_id = {col: idx for idx, col in enumerate(columns)}
        
    #     # macd_data, macd_columns = feature_macd(temp_data, column_to_id)
    #     # temp_data = torch.cat((temp_data, macd_data), dim=0)
    #     # if i == 0:
    #     #     columns.extend(macd_columns)

    #     # ad_data, ad_columns = feature_ad_sma(clv, temp_raw_data[4])
    #     # temp_data = torch.cat((temp_data, ad_data), dim=0)
    #     # if i == 0:
    #     #     columns.extend(ad_columns)

    #     # vpt_data, vpt_columns = feature_vpt_old(temp_raw_data[0], temp_raw_data[4])
    #     # temp_data = torch.cat((temp_data, vpt_data), dim=0)
    #     # if i == 0:
    #     #     columns.extend(vpt_columns)

    #     # chaikin_data, chaikin_columns = feature_chaikin_old(
    #     #     temp_raw_data[4], clv
    #     # )
    #     # temp_data = torch.cat((temp_data, chaikin_data), dim=0)
    #     # if i == 0:
    #     #     columns.extend(chaikin_columns)

    #     # pmfr_data, pmfr_columns = feature_pmfr(
    #     #     close=temp_raw_data[0],
    #     #     high=temp_raw_data[1],
    #     #     low=temp_raw_data[2],
    #     #     volume=temp_raw_data[4],
    #     # )
    #     # temp_data = torch.cat((temp_data, pmfr_data), dim=0)
    #     # if i == 0:
    #     #     columns.extend(pmfr_columns)

    #     # prices = temp_raw_data
    #     # temp_data = torch.cat((temp_data, prices), dim=0)
    #     # if i == 0:
    #     #     columns.append("close")
    #     #     columns.append("high")
    #     #     columns.append("low")
    #     #     columns.append("open")
    #     #     columns.append("volume")

    #     # k_percent, d_percent = calculate_stochastic_oscillator(
    #     #     temp_raw_data[1, :],
    #     #     temp_raw_data[2, :],
    #     #     temp_raw_data[0, :],
    #     #     k_lookback=14,
    #     #     d_lookback=3,
    #     # )
    #     # temp_data = torch.cat((temp_data, k_percent.unsqueeze(0)), dim=0)
    #     # temp_data = torch.cat((temp_data, d_percent.unsqueeze(0)), dim=0)
    #     # if i == 0:
    #     #     columns.append("k_percent_14")
    #     #     columns.append("d_percent_14_3")
    #     # new shit (+ relative vol as well)
        
    #     # standard_ad = calculate_accumulation_distribution_index_standard(
    #     #     temp_raw_data[4], clv
    #     # )
    #     # temp_data = torch.cat((temp_data, standard_ad.unsqueeze(0)), dim=0)
    #     # if i == 0:
    #     #     columns.append("standard_ad")
        
    #     # chaikin_data, chaikin_columns = feature_chaikin_standard(
    #     #     temp_raw_data[4], clv
    #     # )
    #     # temp_data = torch.cat((temp_data, chaikin_data), dim=0)
    #     # if i == 0:
    #     #     columns.extend(chaikin_columns)
        

    #     full_data.append(
    #         temp_data[:, 20:].unsqueeze(-1)
    #     )  # getting rid of some trashy-ish data points

    # TODO maybe norm ema and such using same values as returns and such
    # TODO revisit vpt with sma/ema of itself
    full_data = (raw_data[:,1:,:] - raw_data[:,:-1,:]) / raw_data[:,:-1,:]  # (features, time series, tickers)
    full_data = torch.cat((torch.zeros_like(full_data[:,0:1,:]), full_data), dim=1)
    full_data[4, 5929] = full_data[4, 5928] # ffil fix for inf

    vol_data, vol_columns = feature_volatility_ret(returns=full_data[0:1], prefix="close_returns_")
    full_data = torch.cat((full_data, vol_data), dim=0)
    columns.extend(vol_columns)

    # sma_data, sma_columns = feature_returns_sma(returns=full_data[0:1], prefix="close_returns_")
    # full_data = torch.cat((full_data, sma_data), dim=0)
    # columns.extend(sma_columns)

    # sma_data, sma_columns = feature_returns_sma(returns=full_data[1:2], prefix="high_returns_")
    # full_data = torch.cat((full_data, sma_data), dim=0)
    # columns.extend(sma_columns)

    # sma_data, sma_columns = feature_returns_sma(returns=full_data[2:3], prefix="low_returns_")
    # full_data = torch.cat((full_data, sma_data), dim=0)
    # columns.extend(sma_columns)

    # sma_data, sma_columns = feature_returns_sma(returns=full_data[3:4], prefix="open_returns_")
    # full_data = torch.cat((full_data, sma_data), dim=0)
    # columns.extend(sma_columns)

    # sma_data, sma_columns = feature_returns_sma(returns=full_data[4:5], prefix="volume_returns_")
    # full_data = torch.cat((full_data, sma_data), dim=0)
    # columns.extend(sma_columns)

    full_ema = []
    for i in range(len(tickers)):
        ema_data, ema_columns = feature_ema(full_data[0, :, i], prefix="close_returns_")
        full_ema.append(ema_data.unsqueeze(-1))
    full_data = torch.cat((full_data, torch.cat(full_ema, dim=-1)), dim=0)
    columns.extend(ema_columns)

    full_ema = []
    for i in range(len(tickers)):
        ema_data, ema_columns = feature_ema(full_data[1, :, i], prefix="high_returns_")
        full_ema.append(ema_data.unsqueeze(-1))
    full_data = torch.cat((full_data, torch.cat(full_ema, dim=-1)), dim=0)
    columns.extend(ema_columns)

    full_ema = []
    for i in range(len(tickers)):
        ema_data, ema_columns = feature_ema(full_data[2, :, i], prefix="low_returns_")
        full_ema.append(ema_data.unsqueeze(-1))
    full_data = torch.cat((full_data, torch.cat(full_ema, dim=-1)), dim=0)
    columns.extend(ema_columns)

    full_ema = []
    for i in range(len(tickers)):
        ema_data, ema_columns = feature_ema(full_data[3, :, i], prefix="open_returns_")
        full_ema.append(ema_data.unsqueeze(-1))
    full_data = torch.cat((full_data, torch.cat(full_ema, dim=-1)), dim=0)
    columns.extend(ema_columns)

    full_ema = []
    for i in range(len(tickers)):
        ema_data, ema_columns = feature_ema(full_data[4, :, i], prefix="volume_returns_")
        full_ema.append(ema_data.unsqueeze(-1))
    full_data = torch.cat((full_data, torch.cat(full_ema, dim=-1)), dim=0)
    columns.extend(ema_columns)

    # full_vpt = []
    # vpt_data = calculate_volume_price_trend_standard(raw_data[:4], raw_data[4:])
    # full_vpt.append(vpt_data)
    # full_vpt.append(calculate_sma(vpt_data, lookback=5))
    # full_vpt.append(calculate_sma(vpt_data, lookback=10))
    # full_vpt.append(calculate_sma(vpt_data, lookback=20))
    # full_vpt.append(calculate_sma(vpt_data, lookback=50))
    # # vpt_data[:, 1:] = (vpt_data[:, 1:] - vpt_data[:, :-1])/(vpt_data[:, :-1] + 1e-6)
    # # vpt_data[:, :1] = vpt_data[:, 1:2]
    # full_data = torch.cat((full_data, torch.cat(full_vpt, dim=0)), dim=0)
    # columns.extend(["vpt_close", "vpt_high", "vpt_low", "vpt_open"])
    # columns.extend(["vpt_close_sma_5", "vpt_high_sma_5", "vpt_low_sma_5", "vpt_open_sma_5"])
    # columns.extend(["vpt_close_sma_10", "vpt_high_sma_10", "vpt_low_sma_10", "vpt_open_sma_10"])
    # columns.extend(["vpt_close_sma_20", "vpt_high_sma_20", "vpt_low_sma_20", "vpt_open_sma_20"])
    # columns.extend(["vpt_close_sma_50", "vpt_high_sma_50", "vpt_low_sma_50", "vpt_open_sma_50"])

    # mfi_data, mfi_columns = feature_mfi(raw_data[0], raw_data[1], raw_data[2], raw_data[4]) # meh global norm
    # mfi_data[:, 1:] = (mfi_data[:, 1:] - mfi_data[:, :-1])/(mfi_data[:, :-1] + 1e-6) # change
    # mfi_data[:, :1] = mfi_data[:, 1:2]
    # full_data = torch.cat((full_data, mfi_data), dim=0)
    # columns.extend(mfi_columns)

    # full_atr = []
    # for i in range(len(tickers)):
    #     atr_data, atr_columns = feature_atr(raw_data[0, :, i], raw_data[1, :, i], raw_data[2, :, i])
    #     full_atr.append(atr_data)
    # full_data = torch.cat((full_data, torch.stack(full_atr, dim=-1)), dim=0)
    # columns.extend(atr_columns)

    # full_ppo = []
    # for i in range(len(tickers)):
    #     ppo_data, ppo_columns = feature_ppo(raw_data[0, :, i], prefix="close_")
    #     full_ppo.append(ppo_data)
    # full_data = torch.cat((full_data, torch.stack(full_ppo, dim=-1)), dim=0)
    # columns.extend(ppo_columns)

    # full_ppo = []
    # for i in range(len(tickers)):
    #     ppo_data, ppo_columns = feature_ppo(raw_data[1, :, i], prefix="high_")
    #     full_ppo.append(ppo_data)
    # full_data = torch.cat((full_data, torch.stack(full_ppo, dim=-1)), dim=0)
    # columns.extend(ppo_columns)

    # full_ppo = []
    # for i in range(len(tickers)):
    #     ppo_data, ppo_columns = feature_ppo(raw_data[2, :, i], prefix="low_")
    #     full_ppo.append(ppo_data)
    # full_data = torch.cat((full_data, torch.stack(full_ppo, dim=-1)), dim=0)
    # columns.extend(ppo_columns)

    # full_ppo = []
    # for i in range(len(tickers)):
    #     ppo_data, ppo_columns = feature_ppo(raw_data[3, :, i], prefix="open_")
    #     full_ppo.append(ppo_data)
    # full_data = torch.cat((full_data, torch.stack(full_ppo, dim=-1)), dim=0)
    # columns.extend(ppo_columns)

    # full_ppo = []
    # for i in range(len(tickers)):
    #     ppo_data, ppo_columns = feature_ppo(raw_data[4, :, i], prefix="volume_")
    #     full_ppo.append(ppo_data)
    # full_data = torch.cat((full_data, torch.stack(full_ppo, dim=-1)), dim=0)
    # columns.extend(ppo_columns)

    # full_rsi = []
    # for i in range(len(tickers)):
    #     rsi_data, rsi_columns = feature_rsi(raw_data[0, :, i], prefix="close_") 
    #     full_rsi.append(rsi_data)
    # full_data = torch.cat((full_data, torch.stack(full_rsi, dim=-1)), dim=0)
    # columns.extend(rsi_columns)

    # full_rsi = []
    # for i in range(len(tickers)):
    #     rsi_data, rsi_columns = feature_rsi(raw_data[1, :, i], prefix="high_") 
    #     full_rsi.append(rsi_data)
    # full_data = torch.cat((full_data, torch.stack(full_rsi, dim=-1)), dim=0)
    # columns.extend(rsi_columns)

    # full_rsi = []
    # for i in range(len(tickers)):
    #     rsi_data, rsi_columns = feature_rsi(raw_data[2, :, i], prefix="low_") 
    #     full_rsi.append(rsi_data)
    # full_data = torch.cat((full_data, torch.stack(full_rsi, dim=-1)), dim=0)
    # columns.extend(rsi_columns)

    # full_rsi = []
    # for i in range(len(tickers)):
    #     rsi_data, rsi_columns = feature_rsi(raw_data[3, :, i], prefix="open_") 
    #     full_rsi.append(rsi_data)
    # full_data = torch.cat((full_data, torch.stack(full_rsi, dim=-1)), dim=0)
    # columns.extend(rsi_columns)

    # clv_data = calculate_close_line_values(raw_data[0], raw_data[1], raw_data[2]).unsqueeze(0)
    # full_data = torch.cat((full_data, clv_data), dim=0)
    # columns.extend(["clv"])

    # vix_data = yf.download(
    #     "^VIX", start=start_date, end=end_date, progress=False, auto_adjust=False
    # )
    # aligned_vix_data = pd.DataFrame(columns=vix_data.columns)
    # for column in vix_data.columns.levels[0]:
    #     aligned_vix_data[column, "^VIX"] = align_financial_dataframes(
    #         {column: vix_data},
    #         target_column=column,
    #         fill_method="ffill",
    #         min_date=start_date,
    #         max_date=end_date,
    #     )
    # vix_data = aligned_vix_data
    # vix_data = vix_data.to_numpy()[:, 1:-1]
    # vix_data = torch.tensor(vix_data, dtype=torch.float32)
    # vix_data = vix_data.transpose(0, 1)
    # full_vix = []
    # vix_wider_data, vix_wider_columns = calculate_wider_economics_indicators(
    #     vix_data[0, :], indicator_name="vix_close"
    # )
    # full_vix.append(vix_wider_data)
    # columns.extend(vix_wider_columns)
    # vix_wider_data, vix_wider_columns = calculate_wider_economics_indicators(
    #     vix_data[1, :], indicator_name="vix_high"
    # )
    # full_vix.append(vix_wider_data)
    # columns.extend(vix_wider_columns)
    # vix_wider_data, vix_wider_columns = calculate_wider_economics_indicators(
    #     vix_data[2, :], indicator_name="vix_low"
    # )
    # full_vix.append(vix_wider_data)
    # columns.extend(vix_wider_columns)
    # vix_wider_data, vix_wider_columns = calculate_wider_economics_indicators(
    #     vix_data[3, :], indicator_name="vix_open"
    # )
    # full_vix.append(vix_wider_data)
    # columns.extend(vix_wider_columns)

    # full_vix = torch.cat(full_vix, dim=0).unsqueeze(-1)
    # full_vix = full_vix.expand(full_vix.shape[0], full_vix.shape[1], len(tickers))
    # full_data = torch.cat((full_data, full_vix), dim=0)

    # vix_data = vix_data.unsqueeze(-1)
    # vix_data = vix_data.expand(vix_data.shape[0], vix_data.shape[1], len(tickers))
    # full_data = torch.cat((full_data, vix_data), dim=0)
    # columns.extend(["vix_close", "vix_high", "vix_low", "vix_open"])

    gold_data = yf.download(
        "GC=F", start=start_date, end=end_date, progress=False, auto_adjust=False
    )
    aligned_gold_data = pd.DataFrame(columns=gold_data.columns)
    for column in gold_data.columns.levels[0]:
        aligned_gold_data[column, "GC=F"] = align_financial_dataframes(
            {column: gold_data},
            target_column=column,
            fill_method="ffill",
            min_date=start_date,
            max_date=end_date,
        )
    gold_data = aligned_gold_data
    gold_data = gold_data.to_numpy()[:, 1:2]
    gold_data = torch.tensor(gold_data, dtype=torch.float32)
    gold_data = gold_data.transpose(0, 1)

    gold_change = (gold_data[:, :1] - gold_data[:, :-1])/gold_data[:, :-1]
    gold_change = torch.cat((gold_change[:, 0:1], gold_change), dim=1)
    gold_data = torch.cat((gold_data, gold_change), dim=0)
    
    gold_data = gold_data.unsqueeze(-1)
    gold_data = gold_data.expand(gold_data.shape[0], gold_data.shape[1], len(tickers))
    full_data = torch.cat((full_data, gold_data), dim=0)
    # columns.extend(["gold_close_ch", "gold_high_ch", "gold_low_ch", "gold_open_ch", "gold_volume_ch"])
    columns.extend(["gold_close", "gold_close_ch"])



    data = torch.empty(full_data.shape[0], max(target_dates), full_data.shape[1]-max(target_dates), full_data.shape[2], dtype=torch.float32)
    for i in range(max(target_dates)):
        data[:,i,:,:] = full_data[:,i:-(max(target_dates)-i),:]
    data = data[:, :, 20:, :]  # (features, target_inputs, time series, tickers)

    # adding more complex relationships

    # adding VIX data
    # vix_data = yf.download(
    #     "^VIX", start=start_date, end=end_date, progress=False, auto_adjust=False
    # )
    # aligned_vix_data = pd.DataFrame(columns=vix_data.columns)
    # for column in vix_data.columns.levels[0]:
    #     aligned_vix_data[column, "^VIX"] = align_financial_dataframes(
    #         {column: vix_data},
    #         target_column=column,
    #         fill_method="ffill",
    #         min_date=start_date,
    #         max_date=end_date,
    #     )
    # vix_data = aligned_vix_data
    # vix_data = vix_data["Close"].to_numpy()
    # vix_data = torch.tensor(vix_data, dtype=torch.float32).squeeze(-1)
    # vix_wider_data, vix_wider_columns = calculate_wider_economics_indicators(
    #     vix_data, "vix"
    # )
    # vix_wider_data = vix_wider_data[:, : -max(target_dates)]
    # vix_wider_data = vix_wider_data[:, 20:]
    # vix_wider_data = vix_wider_data.unsqueeze(-1)
    # vix_wider_data = vix_wider_data.tile(1, 1, len(tickers))
    # data = torch.cat((data, vix_wider_data), dim=0)
    # columns = columns + vix_wider_columns

    # # TODO find 2 year yields
    # US_treasury_yields = yf.download(
    #     "^TNX", start=start_date, end=end_date, progress=False
    # )
    # aligned_US_treasury_yields = pd.DataFrame(
    #     columns=US_treasury_yields.columns,
    # )
    # for column in US_treasury_yields.columns.levels[0]:
    #     aligned_US_treasury_yields[column, "^TNX"] = align_financial_dataframes(
    #         {column: US_treasury_yields},
    #         target_column=column,
    #         fill_method="ffill",
    #         min_date=start_date,
    #         max_date=end_date,
    #     )
    # US_treasury_yields = aligned_US_treasury_yields
    # US_treasury_yields = US_treasury_yields["Close"].to_numpy()
    # wider_data = torch.tensor(US_treasury_yields, dtype=torch.float32).unsqueeze(0).squeeze(-1)
    # wider_columns = ["10_year_treasury_yield_close"]

    # US_treasury_yields = yf.download(
    #     "^FVX", start=start_date, end=end_date, progress=False
    # )
    # aligned_US_treasury_yields = pd.DataFrame(
    #     columns=US_treasury_yields.columns,
    # )
    # for column in US_treasury_yields.columns.levels[0]:
    #     aligned_US_treasury_yields[column, "^FVX"] = align_financial_dataframes(
    #         {column: US_treasury_yields},
    #         target_column=column,
    #         fill_method="ffill",
    #         min_date=start_date,
    #         max_date=end_date,
    #     )
    # US_treasury_yields = aligned_US_treasury_yields
    # US_treasury_yields = US_treasury_yields["Close"].to_numpy()
    # wider_data = torch.cat(
    #     (
    #         wider_data,
    #         torch.tensor(US_treasury_yields, dtype=torch.float32).unsqueeze(0).squeeze(-1),
    #     ),
    #     dim=0,
    # )
    # wider_columns.append("5_year_treasury_yield_close")

    # US_treasury_yields = yf.download(
    #     "^IRX", start=start_date, end=end_date, progress=False
    # )
    # aligned_US_treasury_yields = pd.DataFrame(
    #     columns=US_treasury_yields.columns,
    # )
    # for column in US_treasury_yields.columns.levels[0]:
    #     aligned_US_treasury_yields[column, "^IRX"] = align_financial_dataframes(
    #         {column: US_treasury_yields},
    #         target_column=column,
    #         fill_method="ffill",
    #         min_date=start_date,
    #         max_date=end_date,
    #     )
    # US_treasury_yields = aligned_US_treasury_yields
    # US_treasury_yields = US_treasury_yields["Close"].to_numpy()
    # wider_data = torch.cat(
    #     (
    #         wider_data,
    #         torch.tensor(US_treasury_yields, dtype=torch.float32).unsqueeze(0).squeeze(-1),
    #     ),
    #     dim=0,
    # )
    # wider_columns.append("3_month_treasury_yield_close")

    # spread = wider_data[0, :] - wider_data[1, :]
    # wider_data = torch.cat((wider_data, spread.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_spread_10y_5y")

    # spread = wider_data[0, :] - wider_data[2, :]
    # wider_data = torch.cat((wider_data, spread.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_spread_10y_3m")

    # spread = wider_data[1, :] - wider_data[2, :]
    # wider_data = torch.cat((wider_data, spread.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_spread_5y_3m")

    # change = wider_data[0, 1:] - wider_data[0, :-1]
    # change = torch.cat((change[:1], change), dim=0)
    # wider_data = torch.cat((wider_data, change.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_change_10y_abs")

    # change = wider_data[1, 1:] - wider_data[1, :-1]
    # change = torch.cat((change[:1], change), dim=0)
    # wider_data = torch.cat((wider_data, change.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_change_5y_abs")

    # change = wider_data[2, 1:] - wider_data[2, :-1]
    # change = torch.cat((change[:1], change), dim=0)
    # wider_data = torch.cat((wider_data, change.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_change_3m_abs")

    # sma = calculate_sma(wider_data[0], lookback=20)
    # wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_10y_sma_20")

    # sma = calculate_sma(wider_data[1], lookback=20)
    # wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_5y_sma_20")

    # sma = calculate_sma(wider_data[2], lookback=20)
    # wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_3m_sma_20")

    # sma = calculate_sma(wider_data[3], lookback=20)
    # wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_spread_10y_5y_sma_20")

    # sma = calculate_sma(wider_data[4], lookback=20)
    # wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_spread_10y_3m_sma_20")

    # sma = calculate_sma(wider_data[5], lookback=20)
    # wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_spread_5y_3m_sma_20")

    # sma = calculate_sma(wider_data[6], lookback=20)
    # wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_change_10y_abs_sma_20")

    # sma = calculate_sma(wider_data[7], lookback=20)
    # wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_change_5y_abs_sma_20")

    # sma = calculate_sma(wider_data[8], lookback=20)
    # wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    # wider_columns.append("yield_change_3m_abs_sma_20")

    # # ema = calculate_ema_pandas()
    # # can add more things (like change in yields, ema, etc.)

    # gold = yf.download(
    #     "GC=F",
    #     start=start_date,
    #     end=end_date,
    #     progress=True,
    #     auto_adjust=False,
    #     back_adjust=False,
    # )
    # aligned_gold = pd.DataFrame(columns=gold.columns)
    # for column in gold.columns.levels[0]:
    #     aligned_gold[column, "GC=F"] = align_financial_dataframes(
    #         {column: gold},
    #         target_column=column,
    #         fill_method="ffill",
    #         min_date=start_date,
    #         max_date=end_date,
    #     )
    # gold = aligned_gold
    # gold = gold["Close"].to_numpy()
    # gold = torch.tensor(gold, dtype=torch.float32).squeeze(-1)
    # gold_wider_data, gold_wider_columns = calculate_wider_economics_indicators(
    #     gold, "gold"
    # )
    # wider_data = torch.cat((wider_data, gold_wider_data), dim=0)
    # wider_columns = wider_columns + gold_wider_columns

    # crude_oil = yf.download(
    #     "CL=F",
    #     start=start_date,
    #     end=end_date,
    #     progress=True,
    #     auto_adjust=False,
    #     back_adjust=False,
    # )
    # aligned_crude_oil = pd.DataFrame(columns=crude_oil.columns)
    # for column in crude_oil.columns.levels[0]:
    #     aligned_crude_oil[column, "CL=F"] = align_financial_dataframes(
    #         {column: crude_oil},
    #         target_column=column,
    #         fill_method="ffill",
    #         min_date=start_date,
    #         max_date=end_date,
    #     )
    # crude_oil = aligned_crude_oil
    # crude_oil = crude_oil["Close"].to_numpy()
    # crude_oil = torch.tensor(crude_oil, dtype=torch.float32).squeeze(-1)
    # crude_oil_wider_data, crude_oil_wider_columns = (
    #     calculate_wider_economics_indicators(crude_oil, "crude_oil")
    # )
    # wider_data = torch.cat((wider_data, crude_oil_wider_data), dim=0)
    # wider_columns = wider_columns + crude_oil_wider_columns

    # copper = yf.download(
    #     "HG=F",
    #     start=start_date,
    #     end=end_date,
    #     progress=True,
    #     auto_adjust=False,
    #     back_adjust=False,
    # )
    # aligned_copper = pd.DataFrame(columns=copper.columns)
    # for column in copper.columns.levels[0]:
    #     aligned_copper[column, "HG=F"] = align_financial_dataframes(
    #         {column: copper},
    #         target_column=column,
    #         fill_method="ffill",
    #         min_date=start_date,
    #         max_date=end_date,
    #     )
    # copper = aligned_copper
    # copper = copper["Close"].to_numpy()
    # copper = torch.tensor(copper, dtype=torch.float32).squeeze(-1)
    # copper_wider_data, copper_wider_columns = calculate_wider_economics_indicators(
    #     copper, "copper"
    # )
    # wider_data = torch.cat((wider_data, copper_wider_data), dim=0)
    # wider_columns = wider_columns + copper_wider_columns

    # silver = yf.download(
    #     "SI=F",
    #     start=start_date,
    #     end=end_date,
    #     progress=True,
    #     auto_adjust=False,
    #     back_adjust=False,
    # )
    # aligned_silver = pd.DataFrame(columns=silver.columns)
    # for column in silver.columns.levels[0]:
    #     aligned_silver[column, "SI=F"] = align_financial_dataframes(
    #         {column: silver},
    #         target_column=column,
    #         fill_method="ffill",
    #         min_date=start_date,
    #         max_date=end_date,
    #     )
    # silver = aligned_silver
    # silver = silver["Close"].to_numpy()
    # silver = torch.tensor(silver, dtype=torch.float32).squeeze(-1)
    # silver_wider_data, silver_wider_columns = calculate_wider_economics_indicators(
    #     silver, "silver"
    # )
    # wider_data = torch.cat((wider_data, silver_wider_data), dim=0)
    # wider_columns = wider_columns + silver_wider_columns

    # usd_index = yf.download(
    #     "DX-Y.NYB",
    #     start=start_date,
    #     end=end_date,
    #     progress=True,
    #     auto_adjust=False,
    #     back_adjust=False,
    # )
    # aligned_usd_index = pd.DataFrame(columns=usd_index.columns)
    # for column in usd_index.columns.levels[0]:
    #     aligned_usd_index[column, "DX-Y.NYB"] = align_financial_dataframes(
    #         {column: usd_index},
    #         target_column=column,
    #         fill_method="ffill",
    #         min_date=start_date,
    #         max_date=end_date,
    #     )
    # usd_index = aligned_usd_index
    # usd_index = usd_index["Close"].to_numpy()
    # usd_index = torch.tensor(usd_index, dtype=torch.float32).squeeze(-1)
    # usd_index_wider_data, usd_index_columns = calculate_wider_economics_indicators(
    #     usd_index, "usd_index"
    # )
    # wider_data = torch.cat((wider_data, usd_index_wider_data))
    # wider_columns = wider_columns + usd_index_columns

    # wider_data = wider_data[:, : -max(target_dates)]
    # wider_data = wider_data.unsqueeze(-1)
    # wider_data = wider_data.tile(1, 1, len(tickers))
    # wider_data = wider_data[:, 20:, :]

    # data = torch.cat((data, wider_data), dim=0)
    # columns = columns + wider_columns

    # # alpha calculation
    # snp_500 = yf.download(
    #     "^GSPC",
    #     start=start_date,
    #     end=end_date,
    #     progress=True,
    #     auto_adjust=False,
    #     back_adjust=False,
    # )
    # aligned_snp_500 = pd.DataFrame(columns=snp_500.columns)
    # for column in snp_500.columns.levels[0]:
    #     aligned_snp_500[column, "^GSPC"] = align_financial_dataframes(
    #         {column: snp_500},
    #         target_column=column,
    #         fill_method="ffill",
    #         min_date=start_date,
    #         max_date=end_date,
    #     )
    # snp_500 = aligned_snp_500
    # snp_500 = snp_500["Close"].to_numpy()
    # snp_500 = torch.tensor(snp_500, dtype=torch.float32)
    # snp_500 = snp_500[: -max(target_dates)]
    # snp_500_returns = (snp_500[1:] - snp_500[:-1]) / snp_500[:-1]
    # snp_500_returns = torch.cat((snp_500[0:1], snp_500_returns), dim=0)
    # snp_500_returns = snp_500_returns[20:]
    # alphas = data[0, :, :] - snp_500_returns.repeat(1, len(tickers))
    # data = torch.cat((data, alphas.unsqueeze(0)), dim=0)
    # columns.append("alpha_snp_500_returns")

    # temp = []
    # for i in range(len(tickers)):
    #     ema = calculate_ema_pandas(alphas[:, i], lookback=5)
    #     temp.append(ema)
    # data = torch.cat((data, torch.stack(temp, dim=1).unsqueeze(0)), dim=0)
    # columns.append("alpha_snp_500_returns_ema_5")

    # temp = []
    # for i in range(len(tickers)):
    #     ema = calculate_ema_pandas(alphas[:, i], lookback=10)
    #     temp.append(ema)
    # data = torch.cat((data, torch.stack(temp, dim=1).unsqueeze(0)), dim=0)
    # columns.append("alpha_snp_500_returns_ema_10")

    # temp = []
    # for i in range(len(tickers)):
    #     ema = calculate_ema_pandas(alphas[:, i], lookback=20)
    #     temp.append(ema)
    # data = torch.cat((data, torch.stack(temp, dim=1).unsqueeze(0)), dim=0)
    # columns.append("alpha_snp_500_returns_ema_20")

    # relative volatility
    # rel_volatility = data[column_to_id["returns_volatility_5"]] / data[column_to_id["returns_volatility_10"]]
    # data = torch.cat((data, rel_volatility.unsqueeze(0)), dim=0)
    # columns.append("relative_volatility_5_10")
    # rel_volatility = data[column_to_id["returns_volatility_10"]] / data[column_to_id["returns_volatility_20"]]
    # data = torch.cat((data, rel_volatility.unsqueeze(0)), dim=0)
    # columns.append("relative_volatility_10_20")
    # rel_volatility = data[column_to_id["returns_volatility_5"]] / data[column_to_id["returns_volatility_20"]]
    # data = torch.cat((data, rel_volatility.unsqueeze(0)), dim=0)
    # columns.append("relative_volatility_5_20")
    # rel_volatility = data[column_to_id["returns_volatility_20"]] / data[column_to_id["returns_volatility_50"]]
    # data = torch.cat((data, rel_volatility.unsqueeze(0)), dim=0)
    # columns.append("relative_volatility_20_50")
    # rel_volatility = data[column_to_id["returns_volatility_10"]] / data[column_to_id["returns_volatility_50"]]
    # data = torch.cat((data, rel_volatility.unsqueeze(0)), dim=0)
    # columns.append("relative_volatility_10_50")
    # rel_volatility = data[column_to_id["returns_volatility_5"]] / data[column_to_id["returns_volatility_50"]]
    # data = torch.cat((data, rel_volatility.unsqueeze(0)), dim=0)
    # columns.append("relative_volatility_5_50")

    # time data
    time_data, time_columns = feature_time_data(indexes, target_dates, tickers)
    data = torch.cat((data, time_data), dim=0)
    columns.extend(time_columns)


    MTP_targets = torch.empty(
        (5, max(target_dates), data.shape[2]+20, len(tickers)), dtype=torch.float32
    ) # (chlov, target_dates, time series, tickers)
    MTP_full = (raw_data[:, 1:, :] - raw_data[:, :-1, :]) / raw_data[:, :-1, :]
    MTP_full[4, 5928] = MTP_full[4, 5927] # ffil fix for inf
    for i in range(max(target_dates)):
        if i == max(target_dates) - 1:
            MTP_targets[:, i, :, :] = MTP_full[:, i:, :]
        else:
            MTP_targets[:, i, :, :] = MTP_full[:, i:-(max(target_dates) - i - 1), :]
    MTP_targets = MTP_targets[:, :, 20:, :]

    column_to_id = {column: i for i, column in enumerate(columns)}

    if (torch.isnan(data)).any() or (torch.isinf(data)).any():
        print("Data contains NaN or Inf values.")

    if (torch.isnan(MTP_targets)).any() or (torch.isinf(MTP_targets)).any():
        print("MTP_targets contains NaN or Inf values.")
    
    data_length = data.shape[2]

    train_data_length = int(data_length * (1 - val_split_ratio - test_split_ratio))
    val_data_length = int(data_length * val_split_ratio)
    test_data_length = data_length - train_data_length - val_data_length
    train_data, val_data, test_data = torch.split(
        data, [train_data_length, val_data_length, test_data_length], dim=2
    )
    train_indexes, val_indexes, test_indexes = (
        indexes[:train_data_length],
        indexes[train_data_length : train_data_length + val_data_length],
        indexes[train_data_length + val_data_length :],
    )

    # global z-normalization
    means = train_data[: -15].mean(dim=2, keepdim=True)
    stds = train_data[: -15].std(dim=2, keepdim=True)
    train_data[: -15] = (train_data[: -15] - means) / stds
    val_data[: -15] = (val_data[: -15] - means) / stds
    test_data[: -15] = (test_data[: -15] - means) / stds
    # means_close_ret = means[:1]
    # stds_close_ret = stds[:1]
    # train_data[:len(target_dates)] = (train_data[:len(target_dates)] - means_close_ret) / stds_close_ret
    # val_data[:len(target_dates)] = (val_data[:len(target_dates)] - means_close_ret) / stds_close_ret
    # test_data[:len(target_dates)] = (test_data[:len(target_dates)] - means_close_ret) / stds_close_ret

    MTP_targets = (MTP_targets - means[:5]) / stds[:5]

    train_MTP_targets, val_MTP_targets, test_MTP_targets = torch.split(
        MTP_targets, [train_data_length, val_data_length, test_data_length], dim=2
    )
    # train_MTP_targets = (train_MTP_targets - means.unsqueeze(0)) / stds.unsqueeze(0)
    # val_MTP_targets = (val_MTP_targets - means.unsqueeze(0)) / stds.unsqueeze(0)
    # test_MTP_targets = (test_MTP_targets - means.unsqueeze(0)) / stds.unsqueeze(0) 

    # train_data = torch.cat([train_data, means_close_ret.repeat(1, train_data.shape[1], 1), stds_close_ret.repeat(1, train_data.shape[1], 1)], dim=0)
    # val_data = torch.cat([val_data, means_close_ret.repeat(1, val_data.shape[1], 1), stds_close_ret.repeat(1, val_data.shape[1], 1)], dim=0)
    # test_data = torch.cat([test_data, means_close_ret.repeat(1, test_data.shape[1], 1), stds_close_ret.repeat(1, test_data.shape[1], 1)], dim=0)

    save_indexes_to_csv(train_indexes, os.path.join(output_dir, "train.csv"))
    save_indexes_to_csv(val_indexes, os.path.join(output_dir, "val.csv"))
    save_indexes_to_csv(test_indexes, os.path.join(output_dir, "test.csv"))

    torch.save(train_data, os.path.join(output_dir, "train.pt"))
    torch.save(val_data, os.path.join(output_dir, "val.pt"))
    torch.save(test_data, os.path.join(output_dir, "test.pt"))

    torch.save(train_MTP_targets, os.path.join(output_dir, "train_MTP_targets.pt"))
    torch.save(val_MTP_targets, os.path.join(output_dir, "val_MTP_targets.pt"))
    torch.save(test_MTP_targets, os.path.join(output_dir, "test_MTP_targets.pt"))

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
    tickers = ["AAPL", "^GSPC"]  # Reduced for faster testing
    output_dir = "time_series_data"
    seq_len = 64

    download_numerical_financial_data(
        tickers=tickers,
        seq_len=seq_len,
        output_dir=output_dir,
        check_if_already_downloaded=False,
    )

def feature_volatility_log_ret(data: torch.Tensor, prefix: str) -> torch.Tensor:
    vol_columns = []

    volatility = calculate_volatility_log_ret(data, lookback=10)
    vol_data = volatility.unsqueeze(0)
    vol_columns.append(prefix+"volatility_10")

    volatility = calculate_volatility_log_ret(data, lookback=20)
    vol_data = torch.cat((vol_data, volatility.unsqueeze(0)), dim=0)
    vol_columns.append(prefix+"volatility_20")

    volatility = calculate_volatility_log_ret(data, lookback=50)
    vol_data = torch.cat((vol_data, volatility.unsqueeze(0)), dim=0)
    vol_columns.append(prefix+"volatility_50")

    return vol_data, vol_columns

def feature_adr_old(close: torch.Tensor, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    adr_columns = []
    adr = (high - low) / close
    adr_columns.append("ADR")

    atr = calculate_sma(adr, lookback=5)
    adr_data = torch.stack((adr, atr), dim=0)
    adr_columns.append("ADR_sma_5")

    atr = calculate_sma(adr, lookback=10)
    adr_data = torch.cat((adr_data, atr.unsqueeze(0)), dim=0)
    adr_columns.append("ADR_sma_10")

    atr = calculate_sma(adr, lookback=20)
    adr_data = torch.cat((adr_data, atr.unsqueeze(0)), dim=0)
    adr_columns.append("ADR_sma_20")

    return adr_data, adr_columns

def feature_price_sma(
    price: torch.Tensor, prefix: str = "price_") -> torch.Tensor:
    sma_columns = []

    sma = calculate_sma(price, lookback=10)
    sma_data = sma.unsqueeze(0)
    sma_columns.append(prefix + "sma_10")

    sma = calculate_sma(price, lookback=20)
    sma_data = torch.cat((sma_data, sma.unsqueeze(0)), dim=0)
    sma_columns.append(prefix + "sma_20")

    sma = calculate_sma(price, lookback=50)
    sma_data = torch.cat((sma_data, sma.unsqueeze(0)), dim=0)
    sma_columns.append(prefix + "sma_50")

    return sma_data, sma_columns

def feature_returns_sma(returns: torch.Tensor, prefix: str = "returns_") -> torch.Tensor:
    sma_columns = []

    sma = calculate_sma(returns, lookback=5)
    sma_data = sma
    sma_columns.append(prefix + "sma_5")

    sma = calculate_sma(returns, lookback=10)
    sma_data = torch.cat((sma_data, sma), dim=0)
    sma_columns.append(prefix + "sma_10")

    sma = calculate_sma(returns, lookback=20)
    sma_data = torch.cat((sma_data, sma), dim=0)
    sma_columns.append(prefix + "sma_20")

    sma = calculate_sma(returns, lookback=50)
    sma_data = torch.cat((sma_data, sma), dim=0)
    sma_columns.append(prefix + "sma_50")

    return sma_data, sma_columns

def feature_volatility_ret(returns: torch.Tensor, prefix: str = "returns_") -> torch.Tensor:
    vol_columns = []
    vol = calculate_volatility_returns(returns, lookback=5)
    vol_data = vol.unsqueeze(0)
    vol_columns.append(prefix + "volatility_5")

    vol = calculate_volatility_returns(returns, lookback=10)
    vol_data = torch.cat((vol_data, vol.unsqueeze(0)), dim=0)
    vol_columns.append(prefix + "volatility_10")

    vol = calculate_volatility_returns(returns, lookback=20)
    vol_data = torch.cat((vol_data, vol.unsqueeze(0)), dim=0)
    vol_columns.append(prefix + "volatility_20")

    vol = calculate_volatility_returns(returns, lookback=50)
    vol_data = torch.cat((vol_data, vol.unsqueeze(0)), dim=0)
    vol_columns.append(prefix + "volatility_50")

    return vol_data, vol_columns

def feature_bollinger_bands(data: torch.Tensor, column_ids: Dict[str, int]) -> torch.Tensor:
    bb_columns = []
    bb = data[column_ids["returns_sma_5"]] + 1 * data[column_ids["returns_volatility_5"]]
    bb_data = bb.unsqueeze(0)
    bb_columns.append("bollinger_5_up")

    bb = data[column_ids["returns_sma_5"]] - 1 * data[column_ids["returns_volatility_5"]]
    bb_data = torch.cat((bb_data, bb.unsqueeze(0)), dim=0)
    bb_columns.append("bollinger_5_down")

    bb = data[column_ids["returns_sma_10"]] + 1.5 * data[column_ids["returns_volatility_10"]]
    bb_data = torch.cat((bb_data, bb.unsqueeze(0)), dim=0)
    bb_columns.append("bollinger_10_up")

    bb = data[column_ids["returns_sma_10"]] - 1.5 * data[column_ids["returns_volatility_10"]]
    bb_data = torch.cat((bb_data, bb.unsqueeze(0)), dim=0)
    bb_columns.append("bollinger_10_down")

    bb = data[column_ids["returns_sma_20"]] + 2 * data[column_ids["returns_volatility_20"]]
    bb_data = torch.cat((bb_data, bb.unsqueeze(0)), dim=0)
    bb_columns.append("bollinger_20_up")

    bb = data[column_ids["returns_sma_20"]] - 2 * data[column_ids["returns_volatility_20"]]
    bb_data = torch.cat((bb_data, bb.unsqueeze(0)), dim=0)
    bb_columns.append("bollinger_20_down")

    bb = data[column_ids["returns_sma_50"]] + 2.5 * data[column_ids["returns_volatility_50"]]
    bb_data = torch.cat((bb_data, bb.unsqueeze(0)), dim=0)
    bb_columns.append("bollinger_50_up")

    bb = data[column_ids["returns_sma_50"]] - 2.5 * data[column_ids["returns_volatility_50"]]
    bb_data = torch.cat((bb_data, bb.unsqueeze(0)), dim=0)
    bb_columns.append("bollinger_50_down")

    return bb_data, bb_columns

def feature_ema(
    price: torch.Tensor, prefix: str = "price_") -> torch.Tensor:
    ema_columns = []
    ema = calculate_ema_pandas(price, lookback=5)
    ema_data = ema.unsqueeze(0)
    ema_columns.append(prefix + "ema_5")

    ema = calculate_ema_pandas(price, lookback=10)
    ema_data = torch.cat((ema_data, ema.unsqueeze(0)), dim=0)
    ema_columns.append(prefix + "ema_10")

    ema = calculate_ema_pandas(price, lookback=20)
    ema_data = torch.cat((ema_data, ema.unsqueeze(0)), dim=0)
    ema_columns.append(prefix + "ema_20")

    ema = calculate_ema_pandas(price, lookback=50)
    ema_data = torch.cat((ema_data, ema.unsqueeze(0)), dim=0)
    ema_columns.append(prefix + "ema_50")

    return ema_data, ema_columns

def feature_ppo(price: torch.Tensor, prefix: str = "price_") -> torch.Tensor:
    ppo_columns = []
    ema_5 = calculate_ema_pandas(price, lookback=5)
    ema_10 = calculate_ema_pandas(price, lookback=10)
    ema_20 = calculate_ema_pandas(price, lookback=20)

    ppo = (ema_5 - ema_10) / ema_10
    ppo_data = ppo.unsqueeze(0)
    ppo_columns.append(prefix+"ppo_5_10")

    ppo = (ema_10 - ema_20) / ema_20
    ppo_data = torch.cat((ppo_data, ppo.unsqueeze(0)), dim=0)
    ppo_columns.append(prefix+"ppo_10_20")

    ppo = (ema_5 - ema_20) / ema_20
    ppo_data = torch.cat((ppo_data, ppo.unsqueeze(0)), dim=0)
    ppo_columns.append(prefix+"ppo_5_20")

    return ppo_data, ppo_columns

def feature_macd(data: torch.Tensor, column_ids: Dict[str, int]) -> torch.Tensor:
    macd_columns = []
    macd = data[column_ids["price_ema_5"]] - data[column_ids["price_ema_10"]]
    macd_data = macd.unsqueeze(0)
    macd_columns.append("macd_5_10")
    macd_signal = calculate_ema_pandas(macd, lookback=5)
    macd_data = torch.cat((macd_data, macd_signal.unsqueeze(0)), dim=0)
    macd_columns.append("macd_signal_5_10_5")
    macd_histogram = macd - macd_signal
    macd_data = torch.cat((macd_data, macd_histogram.unsqueeze(0)), dim=0)
    macd_columns.append("macd_histogram_5_10_5")

    macd = data[column_ids["price_ema_10"]] - data[column_ids["price_ema_20"]]
    macd_data = torch.cat((macd_data, macd.unsqueeze(0)), dim=0)
    macd_columns.append("macd_10_20")
    macd_signal = calculate_ema_pandas(macd, lookback=10)
    macd_data = torch.cat((macd_data, macd_signal.unsqueeze(0)), dim=0)
    macd_columns.append("macd_signal_10_20_10")
    macd_histogram = macd - macd_signal
    macd_data = torch.cat((macd_data, macd_histogram.unsqueeze(0)), dim=0)
    macd_columns.append("macd_histogram_10_20_10")

    macd = data[column_ids["price_ema_5"]] - data[column_ids["price_ema_20"]]
    macd_data = torch.cat((macd_data, macd.unsqueeze(0)), dim=0)
    macd_columns.append("macd_5_20")
    macd_signal = calculate_ema_pandas(macd, lookback=10)
    macd_data = torch.cat((macd_data, macd_signal.unsqueeze(0)), dim=0)
    macd_columns.append("macd_signal_5_20_10")
    macd_histogram = macd - macd_signal
    macd_data = torch.cat((macd_data, macd_histogram.unsqueeze(0)), dim=0)
    macd_columns.append("macd_histogram_5_20_10")

    return macd_data, macd_columns

def feature_ad_sma(clv: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    ad_columns = []
    ad = calculate_accumulation_distribution_index(volume, clv, lookback=5)
    ad_data = ad.unsqueeze(0)
    ad_columns.append("ad_sma_5")

    ad = calculate_accumulation_distribution_index(volume, clv, lookback=10)
    ad_data = torch.cat((ad_data, ad.unsqueeze(0)), dim=0)
    ad_columns.append("ad_sma_10")

    ad = calculate_accumulation_distribution_index(volume, clv, lookback=20)
    ad_data = torch.cat((ad_data, ad.unsqueeze(0)), dim=0)
    ad_columns.append("ad_sma_20")

    return ad_data, ad_columns

def feature_vpt_old(
    price: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    vpt_columns = []
    vpt = calculate_volume_price_trend(price, volume, lookback=5)
    vpt_data = vpt.unsqueeze(0)
    vpt_columns.append("vpt_sma_5")

    vpt = calculate_volume_price_trend(price, volume, lookback=10)
    vpt_data = torch.cat((vpt_data, vpt.unsqueeze(0)), dim=0)
    vpt_columns.append("vpt_sma_10")

    vpt = calculate_volume_price_trend(price, volume, lookback=20)
    vpt_data = torch.cat((vpt_data, vpt.unsqueeze(0)), dim=0)
    vpt_columns.append("vpt_sma_20")

    return vpt_data, vpt_columns

def feature_chaikin_old(volume: torch.Tensor, clv: torch.Tensor) -> torch.Tensor:
    chaikin_columns = []
    ad_1d = calculate_accumulation_distribution_index(volume, clv, lookback=1)
    ad_ema_3 = calculate_ema_pandas(ad_1d, lookback=3)
    ad_ema_5 = calculate_ema_pandas(ad_1d, lookback=5)
    ad_ema_10 = calculate_ema_pandas(ad_1d, lookback=10)
    ad_ema_20 = calculate_ema_pandas(ad_1d, lookback=20)

    chaikin = ad_ema_3 - ad_ema_10
    chaikin_data = chaikin.unsqueeze(0)
    chaikin_columns.append("chaikin_3_10")

    chaikin = ad_ema_5 - ad_ema_10
    chaikin_data = torch.cat((chaikin_data, chaikin.unsqueeze(0)), dim=0)
    chaikin_columns.append("chaikin_5_10")

    chaikin = ad_ema_10 - ad_ema_20
    chaikin_data = torch.cat((chaikin_data, chaikin.unsqueeze(0)), dim=0)
    chaikin_columns.append("chaikin_10_20")

    chaikin = ad_ema_5 - ad_ema_20
    chaikin_data = torch.cat((chaikin_data, chaikin.unsqueeze(0)), dim=0)
    chaikin_columns.append("chaikin_5_20")

    return chaikin_data, chaikin_columns

def feature_chaikin_standard(volume: torch.Tensor, clv: torch.Tensor) -> torch.Tensor: # how it should be calculated
    chaikin_columns = []
    ad = calculate_accumulation_distribution_index_standard(volume, clv)
    ad_ema_3 = calculate_ema_pandas(ad, lookback=3)
    ad_ema_5 = calculate_ema_pandas(ad, lookback=5)
    ad_ema_10 = calculate_ema_pandas(ad, lookback=10)
    ad_ema_20 = calculate_ema_pandas(ad, lookback=20)

    chaikin = ad_ema_3 - ad_ema_10
    chaikin_data = chaikin.unsqueeze(0)
    chaikin_columns.append("standard_chaikin_3_10")

    chaikin = ad_ema_5 - ad_ema_10
    chaikin_data = torch.cat((chaikin_data, chaikin.unsqueeze(0)), dim=0)
    chaikin_columns.append("standard_chaikin_5_10")

    chaikin = ad_ema_10 - ad_ema_20
    chaikin_data = torch.cat((chaikin_data, chaikin.unsqueeze(0)), dim=0)
    chaikin_columns.append("standard_chaikin_10_20")

    chaikin = ad_ema_5 - ad_ema_20
    chaikin_data = torch.cat((chaikin_data, chaikin.unsqueeze(0)), dim=0)
    chaikin_columns.append("standard_chaikin_5_20")

    return chaikin_data, chaikin_columns

def feature_pmfr(
    close: torch.Tensor, high: torch.Tensor, low: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    pmfr_columns = []
    pmfr = calculate_positive_mfr(close, high, low, volume, lookback=5)
    pmfr_data = pmfr.unsqueeze(0)
    pmfr_columns.append("pmfr_5")

    pmfr = calculate_positive_mfr(close, high, low, volume, lookback=10)
    pmfr_data = torch.cat((pmfr_data, pmfr.unsqueeze(0)), dim=0)
    pmfr_columns.append("pmfr_10")

    pmfr = calculate_positive_mfr(close, high, low, volume, lookback=20)
    pmfr_data = torch.cat((pmfr_data, pmfr.unsqueeze(0)), dim=0)
    pmfr_columns.append("pmfr_20")

    return pmfr_data, pmfr_columns

def feature_rsi(
    close: torch.Tensor, prefix: str = "") -> torch.Tensor:
    rsi_columns = []
    rsi = calculate_rsi(close, lookback=7)
    rsi_data = rsi.unsqueeze(0)
    rsi_columns.append(prefix+"rsi_7")

    rsi = calculate_rsi(close, lookback=14)
    rsi_data = torch.cat((rsi_data, rsi.unsqueeze(0)), dim=0)
    rsi_columns.append(prefix+"rsi_14")

    rsi = calculate_rsi(close, lookback=21)
    rsi_data = torch.cat((rsi_data, rsi.unsqueeze(0)), dim=0)
    rsi_columns.append(prefix+"rsi_21")

    rsi = calculate_rsi(close, lookback=50)
    rsi_data = torch.cat((rsi_data, rsi.unsqueeze(0)), dim=0)
    rsi_columns.append(prefix+"rsi_50")

    return rsi_data, rsi_columns

def feature_mfi(
    close: torch.Tensor, high: torch.Tensor, low: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    mfi_columns = []
    mfi = calculate_mfi(close, high, low, volume, lookback=5)
    mfi_data = mfi.unsqueeze(0)
    mfi_columns.append("mfi_5")

    mfi = calculate_mfi(close, high, low, volume, lookback=10)
    mfi_data = torch.cat((mfi_data, mfi.unsqueeze(0)), dim=0)
    mfi_columns.append("mfi_10")

    mfi = calculate_mfi(close, high, low, volume, lookback=20)
    mfi_data = torch.cat((mfi_data, mfi.unsqueeze(0)), dim=0)
    mfi_columns.append("mfi_20")

    return mfi_data, mfi_columns

def feature_atr(
    close: torch.Tensor, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    atr_columns = []
    tr = calculate_true_range(close, high, low) / close
    atr_data = tr.unsqueeze(0)
    atr_columns.append("tr")

    atr = calculate_ema_pandas(tr, lookback=5*2-1)  # Wilder's smoothing
    atr_data = torch.cat((atr_data, atr.unsqueeze(0)), dim=0)
    atr_columns.append("atr_5")

    atr = calculate_ema_pandas(tr, lookback=10*2-1)  # Wilder's smoothing
    atr_data = torch.cat((atr_data, atr.unsqueeze(0)), dim=0)
    atr_columns.append("atr_10")

    atr = calculate_ema_pandas(tr, lookback=20*2-1)  # Wilder's smoothing
    atr_data = torch.cat((atr_data, atr.unsqueeze(0)), dim=0)
    atr_columns.append("atr_20")

    atr = calculate_ema_pandas(tr, lookback=50*2-1)  # Wilder's smoothing
    atr_data = torch.cat((atr_data, atr.unsqueeze(0)), dim=0)
    atr_columns.append("atr_50")

    return atr_data, atr_columns

def feature_time_data(
    indexes: pd.DatetimeIndex, target_dates: List[int], tickers: List[str]
) -> torch.Tensor:
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

    # time_data = time_data[:, : -max(target_dates)]
    # time_data = time_data[:, 20:].unsqueeze(-1)
    # time_data = time_data.tile(1, 1, len(tickers))
    time_data_target = torch.empty((time_data.shape[0], max(target_dates), time_data.shape[1] - max(target_dates)), dtype=torch.float32)
    for i in range(max(target_dates)):
        time_data_target[:, i, :] = time_data[:, i:-(max(target_dates) - i)]
    time_data_target = time_data_target.unsqueeze(-1)
    time_data = time_data_target[:,:,20:,:].expand(-1, -1, -1, len(tickers))
    return time_data, time_columns

def calculate_volatility_log_ret(
    data: torch.Tensor, lookback: int = 30
) -> torch.Tensor:
    log_returns = torch.log(data[1:] / data[:-1])  # from adj. Close for now
    volatility = [0, 0]
    for i in range(2, data.shape[0]):
        if i < lookback:
            volatility.append(log_returns[:i].std())
        else:
            volatility.append(log_returns[i - lookback : i].std())
    volatility[0] = volatility[2]  # maybe wrong, but its just two datapoints
    volatility[1] = volatility[2]
    return torch.tensor(volatility)



def calculate_true_range(
    close: torch.Tensor, high: torch.Tensor, low: torch.Tensor
) -> torch.Tensor:
    tr1 = high[1:] - low[1:]
    tr2 = torch.abs(high[1:] - close[:-1])
    tr3 = torch.abs(low[1:] - close[:-1])
    trs = torch.stack((tr1, tr2, tr3), dim=0)
    trs = torch.cat((trs[:, 0:1], trs), dim=1)  # add first value
    return torch.max(trs, dim=0).values

def calculate_sma(series: torch.Tensor, lookback: int = 10) -> torch.Tensor:
    sma = torch.empty_like(series)
    cumsum = torch.cumsum(series, dim=1)
    sma[:, :lookback] = cumsum[:, :lookback] / torch.arange(1, lookback + 1).view(1,-1,1)
    sma[:, lookback:] = (cumsum[:, lookback:] - cumsum[:, :-lookback]) / lookback
    return sma


def calculate_moving_average_returns(
    price: torch.Tensor, lookback: int = 10
) -> torch.Tensor:
    SMAR = (price[lookback:] - price[:-lookback]) / lookback
    temp = torch.zeros(lookback)
    for i in range(1, lookback):
        temp[i] = (price[i] - price[0]) / i
    temp[0] = temp[1]
    return torch.cat((temp, SMAR))


def calculate_volatility_returns(
    returns: torch.Tensor, lookback: int = 30
) -> torch.Tensor:
    volatility = torch.zeros(returns.shape[1], returns.shape[2])
    for i in range(2, returns.shape[1]):
        if i < lookback:
            volatility[i] = returns[:, :i].std(dim=(0,1))
        else:
            volatility[i] = returns[:, i - lookback : i].std(dim=(0,1))
    volatility[0] = volatility[2]  # maybe wrong, but its just two datapoints
    volatility[1] = volatility[2]
    return volatility



def calculate_close_line_values(
    close: torch.Tensor, high: torch.Tensor, low: torch.Tensor
) -> torch.Tensor:
    clv = (2 * close - high - low) / (high - low)
    return clv


def calculate_accumulation_distribution_index(
    volume: torch.Tensor, clv: torch.Tensor, lookback: int = 10
):
    ad = torch.empty_like(volume)
    mfv = clv * volume
    mfv_cumsum = torch.cumsum(mfv, dim=0)
    ad[:lookback] = mfv_cumsum[:lookback]
    ad[lookback:] = mfv_cumsum[lookback:] - mfv_cumsum[:-lookback]
    return ad


def calculate_accumulation_distribution_index_standard(
    volume: torch.Tensor, clv: torch.Tensor
):
    return torch.cumsum(clv * volume, dim=0)


def calculate_volume_price_trend(
    price: torch.Tensor, volume: torch.Tensor, lookback: int = 10
):
    returns = torch.zeros(price.shape[0])
    returns[1:] = (price[1:] - price[:-1]) / (price[:-1] + 1e-6)
    returns[0] = returns[1]

    vtr = volume * returns
    rolling_sum_vtr = torch.cumsum(vtr, dim=0)

    vpt = torch.empty_like(price)
    vpt[:lookback] = rolling_sum_vtr[:lookback]
    vpt[lookback:] = rolling_sum_vtr[lookback:] - rolling_sum_vtr[:-lookback]
    return vpt


def calculate_volume_price_trend_standard(price: torch.Tensor, volume: torch.Tensor):
    returns = torch.empty_like(price)
    returns[:, 1:] = (price[:, 1:] - price[:, :-1]) / (price[:, :-1] + 1e-6)
    returns[:, 0] = returns[:, 1]

    vtr = volume * returns
    return torch.cumsum(vtr, dim=1)


def calculate_ema_pandas(
    daily_values: torch.Tensor, lookback: int = 10
) -> torch.Tensor:  # to calculate ema for anything
    original_dtype = daily_values.dtype

    series_pd = pd.Series(daily_values.cpu().numpy())
    ema = series_pd.ewm(span=lookback, adjust=False, min_periods=1).mean()
    return torch.from_numpy(ema.values).to(original_dtype)


def calculate_mfi(
    close: torch.Tensor,
    high: torch.Tensor,
    low: torch.Tensor,
    volume: torch.Tensor,
    lookback: int = 10,
):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    signs = torch.zeros_like(close)
    signs[1:] = torch.sign(typical_price[1:] - typical_price[:-1])

    positive_money_flow = torch.where(
        signs > 0, money_flow, torch.zeros_like(money_flow)
    )
    negative_money_flow = torch.where(
        signs < 0, money_flow, torch.zeros_like(money_flow)
    )

    cumsum_pos_mf = torch.cumsum(positive_money_flow, dim=0)
    rolling_pos_mf = torch.empty_like(positive_money_flow)
    rolling_pos_mf[:lookback] = cumsum_pos_mf[:lookback]
    rolling_pos_mf[lookback:] = cumsum_pos_mf[lookback:] - cumsum_pos_mf[:-lookback]

    cumsum_neg_mf = torch.cumsum(negative_money_flow, dim=0)
    rolling_neg_mf = torch.empty_like(negative_money_flow)
    rolling_neg_mf[:lookback] = cumsum_neg_mf[:lookback]
    rolling_neg_mf[lookback:] = cumsum_neg_mf[lookback:] - cumsum_neg_mf[:-lookback]

    mfi = rolling_pos_mf / (rolling_neg_mf + 1e-6)
    return mfi


def calculate_positive_mfr(
    close: torch.Tensor,
    high: torch.Tensor,
    low: torch.Tensor,
    volume: torch.Tensor,
    lookback: int = 10,
):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    signs = torch.zeros_like(close)
    signs[1:] = torch.sign(typical_price[1:] - typical_price[:-1])

    positive_money_flow = torch.where(
        signs > 0, money_flow, torch.zeros_like(money_flow)
    )

    cumsum_pos_mf = torch.cumsum(positive_money_flow, dim=0)
    rolling_pos_mf = torch.empty_like(positive_money_flow)
    rolling_pos_mf[:lookback] = cumsum_pos_mf[:lookback]
    rolling_pos_mf[lookback:] = cumsum_pos_mf[lookback:] - cumsum_pos_mf[:-lookback]

    cumsum_total_mf = torch.cumsum(money_flow, dim=0)
    rolling_total_mf = torch.empty_like(money_flow)
    rolling_total_mf[:lookback] = cumsum_total_mf[:lookback]
    rolling_total_mf[lookback:] = (
        cumsum_total_mf[lookback:] - cumsum_total_mf[:-lookback]
    )

    mfr = rolling_pos_mf / rolling_total_mf
    return mfr


def calculate_negative_mfr(
    close: torch.Tensor,
    high: torch.Tensor,
    low: torch.Tensor,
    volume: torch.Tensor,
    lookback: int = 10,
):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    signs = torch.zeros_like(close)
    signs[1:] = torch.sign(typical_price[1:] - typical_price[:-1])

    negative_money_flow = torch.where(
        signs < 0, money_flow, torch.zeros_like(money_flow)
    )

    cumsum_neg_mf = torch.cumsum(negative_money_flow, dim=0)
    rolling_neg_mf = torch.empty_like(negative_money_flow)
    rolling_neg_mf[:lookback] = cumsum_neg_mf[:lookback]
    rolling_neg_mf[lookback:] = cumsum_neg_mf[lookback:] - cumsum_neg_mf[:-lookback]

    cumsum_total_mf = torch.cumsum(money_flow, dim=0)
    rolling_total_mf = torch.empty_like(money_flow)
    rolling_total_mf[:lookback] = cumsum_total_mf[:lookback]
    rolling_total_mf[lookback:] = (
        cumsum_total_mf[lookback:] - cumsum_total_mf[:-lookback]
    )

    mfr = rolling_neg_mf / rolling_total_mf
    return mfr

def calculate_rsi(
    prices: torch.Tensor, lookback: int = 14
):  # explore sma smoothing version
    deltas = prices[1:] - prices[:-1]
    gains = torch.zeros_like(deltas)
    losses = torch.zeros_like(deltas)
    gains[deltas > 0] = deltas[deltas > 0]
    losses[deltas < 0] = -deltas[deltas < 0]  # Losses are positive values

    avg_gain = torch.zeros_like(prices)
    avg_loss = torch.zeros_like(prices)

    avg_gain[1:] = calculate_ema_pandas(gains, 2 * lookback - 1)
    avg_loss[1:] = calculate_ema_pandas(losses, 2 * lookback - 1)

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
    percent_k = torch.full_like(close, 50.0)

    low_windows = low.unfold(0, k_lookback, 1)
    high_windows = high.unfold(0, k_lookback, 1)

    period_low_values = torch.min(low_windows, dim=1)[0]
    period_high_values = torch.max(high_windows, dim=1)[0]

    close_for_k = close[k_lookback - 1 :]

    numerator = close_for_k - period_low_values
    denominator = period_high_values - period_low_values
    percent_k_calculated = torch.full_like(numerator, 50.0)

    valid_denominator_mask = denominator > 1e-12

    percent_k_calculated[valid_denominator_mask] = (
        100 * numerator[valid_denominator_mask] / denominator[valid_denominator_mask]
    )

    percent_k[k_lookback - 1 :] = percent_k_calculated

    percent_d = calculate_sma(percent_k, lookback=d_lookback)

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
    change = torch.cat((change[:1], change), dim=0)
    indicator_data = torch.cat((indicator.unsqueeze(0), change.unsqueeze(0)), dim=0)
    indicator_columns = [indicator_name, indicator_name + "_change"]

    # removed because duplicates????
    # sma = calculate_sma(indicator.unsqueeze(0), lookback=5, dim=0)
    # indicator_data = torch.cat((indicator_data, sma.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_sma_5")

    # sma = calculate_sma(indicator.unsqueeze(0), lookback=10, dim=0)
    # indicator_data = torch.cat((indicator_data, sma.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_sma_10")

    # sma = calculate_sma(indicator.unsqueeze(0), lookback=20, dim=0)
    # indicator_data = torch.cat((indicator_data, sma.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_sma_20")

    # ema = calculate_ema_pandas(indicator, lookback=5)
    # indicator_data = torch.cat((indicator_data, ema.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_ema_5")

    # ema = calculate_ema_pandas(indicator, lookback=10)
    # indicator_data = torch.cat((indicator_data, ema.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_ema_10")

    # ema = calculate_ema_pandas(indicator, lookback=20)
    # indicator_data = torch.cat((indicator_data, ema.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_ema_20")

    # sma = calculate_sma(change.unsqueeze(0), lookback=5, dim=0)
    # indicator_data = torch.cat((indicator_data, sma.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_change_sma_5")

    # sma = calculate_sma(change.unsqueeze(0), lookback=10, dim=0)
    # indicator_data = torch.cat((indicator_data, sma.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_change_sma_10")

    # sma = calculate_sma(change.unsqueeze(0), lookback=20, dim=0)
    # indicator_data = torch.cat((indicator_data, sma.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_change_sma_20")

    # ema = calculate_ema_pandas(change, lookback=5)
    # indicator_data = torch.cat((indicator_data, ema.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_change_ema_5")

    # ema = calculate_ema_pandas(change, lookback=10)
    # indicator_data = torch.cat((indicator_data, ema.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_change_ema_10")

    # ema = calculate_ema_pandas(change, lookback=20)
    # indicator_data = torch.cat((indicator_data, ema.unsqueeze(0)), dim=0)
    # indicator_columns.append(indicator_name + "_change_ema_20")

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
