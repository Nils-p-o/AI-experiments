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
import time

import requests
import argparse
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.preprocessing import PowerTransformer, RobustScaler

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


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
            self.sequences_data = torch.load(file_path).to(self.device)
            self.targets_data = torch.load(targets_file_path).to(self.device)
        else:
            self.sequences_data = None

    def __len__(self) -> int:
        if self.preload:
            return self.sequences_data.size(2)
        else:  # TODO?
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.preload:
            input_sequence = self.sequences_data[:, :, idx, :, :]
            target_sequence = self.targets_data[:, :, idx, :, :]
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
        self.persistent_workers = num_workers > 0
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
            persistent_workers=self.persistent_workers,
            drop_last=True,
            # prefetch_factor=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=True,
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


def download_with_retry(tickers, max_retries=5, delay_seconds=3, **kwargs):
    """
    Downloads data from Yahoo Finance with a robust retry mechanism that handles
    partial failures while preserving the default column structure (OHLCV, Ticker).

    Args:
        tickers (str or list): A single ticker string or a list of ticker strings.
        max_retries (int): The maximum number of download attempts.
        delay_seconds (int): The number of seconds to wait between retries.
        **kwargs: Additional keyword arguments to pass to yf.download().

    Returns:
        pd.DataFrame: A DataFrame containing the data. For multiple tickers,
                      columns are a MultiIndex with OHLCV at level 0 and tickers at level 1.

    Raises:
        Exception: Re-raises the last caught exception if all retries fail.
    """
    requested_tickers = [tickers] if isinstance(tickers, str) else tickers
    last_exception = None

    if "progress" not in kwargs:
        kwargs["progress"] = False

    for attempt in range(max_retries):
        try:
            print(
                f"Attempt {attempt + 1}/{max_retries} to download data for: {requested_tickers}..."
            )
            # DO NOT use group_by='ticker' to preserve the desired (OHLCV, Ticker) structure
            data = yf.download(tickers, **kwargs)

            # --- ROBUST VALIDATION STEP ---

            # 1. Check for a completely empty DataFrame (total failure)
            if data.empty:
                raise ValueError("Downloaded data is empty.")

            # 2. Check for partial failures by validating each ticker's data
            failed_tickers = []
            if len(requested_tickers) > 1:
                # Multi-ticker case: DataFrame has MultiIndex columns
                downloaded_tickers = data.columns.get_level_values(1)
                for ticker in requested_tickers:
                    if ticker not in downloaded_tickers:
                        # Ticker is completely missing from the result
                        failed_tickers.append(ticker)
                    else:
                        # Ticker is present, check if its data is all NaN
                        # Use xs() to select the cross-section for this ticker
                        ticker_data = data.xs(ticker, level=1, axis=1)
                        if ticker_data.isnull().all().all():
                            failed_tickers.append(ticker)
            else:
                # Single-ticker case: DataFrame has simple columns
                if data.isnull().all().all():
                    failed_tickers.append(requested_tickers[0])

            if failed_tickers:
                raise ValueError(
                    f"Data for tickers failed (all NaN or missing): {failed_tickers}"
                )

            # If we reach here, all checks passed.
            print("Download successful for all requested tickers.")
            return data

        except Exception as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            last_exception = e
            if attempt < max_retries - 1:
                print(f"Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                print(f"Failed to download data after {max_retries} attempts.")
                if last_exception:
                    raise last_exception
                raise Exception("All download retries failed.")

    return pd.DataFrame()


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
    config_args: argparse.Namespace = None,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if check_if_already_downloaded and is_already_downloaded(tickers, output_dir):
        print("Data already downloaded.")
        return

    raw_data = download_with_retry(
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

    # daily_index = raw_data.index
    # daily_close_prices = raw_data['Close']

    # aligned_fundamentals_df, fundamental_col_names = fetch_and_align_fundamental_data(
    #     tickers, daily_index, daily_close_prices
    # )

    # # Convert the multi-level column dataframe to a tensor
    # # Shape: (time, features, tickers) -> transpose to (features, time, tickers)
    # if not aligned_fundamentals_df.empty:
    #     # Filter to only include columns we successfully generated
    #     aligned_fundamentals_df = aligned_fundamentals_df.reindex(columns=fundamental_col_names, level=0)

    #     fundamental_data_tensor = torch.tensor(aligned_fundamentals_df.values, dtype=torch.float32)
    #     num_fundamental_features = fundamental_data_tensor.shape[1] // len(tickers)
    #     fundamental_data_tensor = fundamental_data_tensor.reshape(len(daily_index), num_fundamental_features, len(tickers))
    #     fundamental_data_tensor = fundamental_data_tensor.permute(1, 0, 2) # (features, time, tickers)
    # else:
    #     fundamental_data_tensor = torch.empty(0)
    #     fundamental_col_names = []

    unique_tickers = sorted(list(set(tickers)))
    ticker_to_id = {ticker: i for i, ticker in enumerate(unique_tickers)}

    indexes = raw_data.index
    columns = list(raw_data.columns.levels[0])

    raw_data = torch.tensor(raw_data.values, dtype=torch.float32).reshape(
        -1, len(columns), len(tickers)
    )  # (Time, Features, tickers)
    raw_data = raw_data.transpose(0, 1)  # (Features, Time series, tickers)

    raw_data = raw_data[1:, :, :]
    full_data, columns, local_columns, time_columns = calculate_features(raw_data, tickers, indexes, config_args)
    full_data[:len(columns)+len(local_columns)] = data_fix_ffill(full_data[:len(columns)+len(local_columns)])

    # global_data, local_data = torch.split(full_data, [len(columns), len(local_columns)], dim=0)
    # # EPS and other fundamentals
    # if not aligned_fundamentals_df.empty:
    #     global_data = torch.cat((global_data, fundamental_data_tensor), dim=0)
    #     local_data = torch.cat((local_data, fundamental_data_tensor), dim=0)
    #     columns.extend(fundamental_col_names)
    #     local_columns.extend(["local_"+col for col in fundamental_col_names])
    #     full_data = torch.cat((global_data, local_data), dim=0)

    columns.extend(local_columns)
    columns.extend(time_columns)

    data = torch.empty(
        full_data.shape[0],
        max(target_dates),
        full_data.shape[1] - max(target_dates),
        full_data.shape[2],
        dtype=torch.float32,
    )
    for i in range(max(target_dates)):
        data[:, i, :, :] = full_data[:, i : -(max(target_dates) - i), :]
    data = data[:, :, 20:, :]  # (features, target_inputs, time series, tickers)

    if config_args.include_sep_in_loss:
        MTP_targets = torch.empty(
            (5, max(target_dates), data.shape[2] + 21, len(tickers)), dtype=torch.float32
        )
    else:
        MTP_targets = torch.empty(
            (5, max(target_dates), data.shape[2] + 20, len(tickers)), dtype=torch.float32
        )  # (chlov, target_dates, time series, tickers)
    MTP_full = (raw_data[:, 1:, :] - raw_data[:, :-1, :]) / raw_data[:, :-1, :]
    if config_args.include_sep_in_loss:
        MTP_full = torch.cat((torch.zeros_like(MTP_full[:, :1, :]), MTP_full), dim=1)
    MTP_full = data_fix_ffill(MTP_full)
    
    # TODO only supposed to be train data
    MTP_targets_classes = torch.full_like(MTP_full, 1)
    MTP_targets_classes[MTP_full < -config_args.classification_threshold] = 0
    MTP_targets_classes[MTP_full > config_args.classification_threshold] = 2
    MTP_targets_classes = MTP_targets_classes.long()

    for i in range(max(target_dates)):
        if i == max(target_dates) - 1:
            MTP_targets[:, i, :, :] = MTP_full[:, i:, :]
        else:
            MTP_targets[:, i, :, :] = MTP_full[:, i : -(max(target_dates) - i - 1), :]
    MTP_targets = MTP_targets[:, :, 20:, :]

    column_to_id = {column: i for i, column in enumerate(columns)}

    if (torch.isnan(data)).any() or (torch.isinf(data)).any():
        print("Data contains NaN or Inf values.")

    if (torch.isnan(MTP_targets)).any() or (torch.isinf(MTP_targets)).any():
        print("MTP_targets contains NaN or Inf values.")

    data_length = data.shape[2]

    train_data_length = int(data_length * (1 - val_split_ratio - test_split_ratio))
    if train_data_length == 0:
        val_data_length = data_length - train_data_length
        test_data_length = 0
    else:
        val_data_length = int(data_length * val_split_ratio)
        test_data_length = data_length - train_data_length - val_data_length
    # train_data, val_data, test_data = torch.split(
    #     data, [train_data_length, val_data_length, test_data_length], dim=2
    # )
    train_indexes, val_indexes, test_indexes = (
        indexes[: train_data_length + seq_len - 1],
        indexes[train_data_length : train_data_length + val_data_length + seq_len - 1],
        indexes[train_data_length + val_data_length :],
    )

    # global z-normalization

    num_of_non_global_norm_feats = len(local_columns) + len(time_columns)
    data, fitted_processors = auto_correct_feature_skew_pre_znorm(data, column_to_id, train_data_length+seq_len-1, num_of_non_global_norm_feats, len(time_columns), seq_len)

    means = data[:-num_of_non_global_norm_feats, :, :train_data_length, :].mean(
        dim=2, keepdim=True
    )
    stds = data[:-num_of_non_global_norm_feats, :, :train_data_length, :].std(
        dim=2, keepdim=True
    )
    data[:-num_of_non_global_norm_feats] = (
        data[:-num_of_non_global_norm_feats] - means
    ) / (stds + 1e-8)

    if config_args.prediction_type != "classification":
        MTP_targets = (MTP_targets - means[:5]) / stds[:5]

    # new reshape (into seq_len chunks)
    data = data.unfold(
        2, seq_len, 1
    ).contiguous()  # (features, target_inputs, time series, tickers, seq_len)

    if config_args.include_sep_in_loss:
        MTP_targets = MTP_targets.unfold(
            2, seq_len + 1, 1
        ).contiguous()
    else:
        MTP_targets = MTP_targets.unfold(
            2, seq_len, 1
        ).contiguous()  # (chlov, target_inputs, time series, tickers, seq_len)

    data = data.permute(0, 1, 2, 4, 3)
    MTP_targets = MTP_targets.permute(0, 1, 2, 4, 3)

    # local znorm
    if num_of_non_global_norm_feats != len(time_columns):
        local_means = data[-num_of_non_global_norm_feats : -len(time_columns)].mean(
            dim=3, keepdim=True
        )
        local_stds = data[-num_of_non_global_norm_feats : -len(time_columns)].std(
            dim=3, keepdim=True
        )
        data[-num_of_non_global_norm_feats : -len(time_columns)] = (
            data[-num_of_non_global_norm_feats : -len(time_columns)] - local_means
        ) / (local_stds + 1e-8)
    
    data = auto_correct_feature_kurtosis_post_znorm(data, column_to_id, train_data_length)
    
    train_data, val_data, test_data = torch.split(
        data,
        [train_data_length, val_data_length - seq_len + 1, test_data_length],
        dim=2,
    )
    train_MTP_targets, val_MTP_targets, test_MTP_targets = torch.split(
        MTP_targets,
        [train_data_length, val_data_length - seq_len + 1, test_data_length],
        dim=2,
    )
    if config_args.n_noisy_copies > 0:
        noisy_copies = generate_noisy_copies(raw_data, tickers, target_dates, indexes, train_data_length, 20, config_args.n_noisy_copies, config_args.noise_factor)
        processed_copies = []
        for copy in noisy_copies:
            copy = apply_processors_to_noisy_copy(copy, fitted_processors)
            copy[:-num_of_non_global_norm_feats] = (copy[:-num_of_non_global_norm_feats] - means) / (stds + 1e-8)

            copy = copy.unfold(2, seq_len, 1).contiguous()
            copy = copy.permute(0, 1, 2, 4, 3)
            copy = copy[:,:,:train_data_length]
            if num_of_non_global_norm_feats != len(time_columns):
                noisy_local_means = copy[-num_of_non_global_norm_feats : -len(time_columns)].mean(
                    dim=3, keepdim=True
                )
                noisy_local_stds = copy[-num_of_non_global_norm_feats : -len(time_columns)].std(
                    dim=3, keepdim=True
                )
                copy[-num_of_non_global_norm_feats : -len(time_columns)] = (
                    copy[-num_of_non_global_norm_feats : -len(time_columns)] - noisy_local_means
                ) / (noisy_local_stds + 1e-8)

            processed_copies.append(copy)

        train_MTP_targets = train_MTP_targets.repeat(1, 1, config_args.n_noisy_copies+1, 1, 1)
        noisy_tensor = torch.cat(processed_copies, dim=2)
        train_data = torch.cat((train_data, noisy_tensor), dim=2)


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

    # (chlov, time, ticker)
    # (chlov, ticker, 3)
    class_counts = torch.empty((MTP_targets_classes.shape[0], MTP_targets_classes.shape[2], 3))
    for i in range(3):
        class_counts[:, :, i] = (MTP_targets_classes == i).sum(dim=1)
    
    class_weights = torch.empty_like(class_counts)
    class_weights = class_counts.sum(dim=-1, keepdim=True) / (3*class_counts)

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
        "class_weights": class_weights.tolist(),
        "train_means": means[:, 0, 0, :].tolist(),
        "train_stds": stds[:, 0, 0, :].tolist(),
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    return


def calculate_features(
    data: torch.Tensor, tickers: List[str], indexes, config_args
) -> Tuple[torch.Tensor, List[str], List[str], List[str]]:
    raw_data = data.clone()

    returns_columns = [
        "close_returns",
        "high_returns",
        "low_returns",
        "open_returns",
        "volume_returns",
    ]
    price_columns = ["close", "high", "low", "open", "volume"]
    columns = []
    columns.extend(returns_columns)

    # TODO maybe norm ema and such using same values as returns and such
    # TODO revisit vpt with sma/ema of itself
    full_data = (raw_data[:, 1:, :] - raw_data[:, :-1, :]) / raw_data[
        :, :-1, :
    ]  # (features, time series, tickers)
    full_data = torch.cat((torch.zeros_like(full_data[:, 0:1, :]), full_data), dim=1)
    # full_data[4, 5929] = full_data[4, 5928] # ffil fix for inf
    full_data = data_fix_ffill(full_data)

    if not config_args.global_exclude_close_vol:
        vol_data, vol_columns = feature_volatility_ret(returns=full_data[0:1], prefix="close_returns_")
        full_data = torch.cat((full_data, vol_data), dim=0)
        columns.extend(vol_columns)

    if not config_args.global_exclude_volume_vol:
        vol_data, vol_columns = feature_volatility_ret(returns=full_data[4:5], prefix="volume_returns_")
        full_data = torch.cat((full_data, vol_data), dim=0)
        columns.extend(vol_columns)

    if not config_args.global_exclude_ema:
        full_ema = []
        full_ema_columns = []
        for i in range(5):
            temp_ema = []
            for j in range(len(tickers)):
                ema_data, ema_columns = feature_ema(full_data[i, :, j], columns[i] + "_")
                temp_ema.append(ema_data.unsqueeze(-1))
            full_ema.append(torch.cat(temp_ema, dim=-1))
            full_ema_columns.extend(ema_columns)
        full_data = torch.cat((full_data, torch.cat(full_ema, dim=0)), dim=0)
        columns.extend(full_ema_columns)

    full_vpt = []
    vpt_data = calculate_volume_price_trend_standard(full_data[:4], raw_data[4:])
    full_vpt.append(vpt_data)
    if not config_args.global_exclude_vpt:
        full_data = torch.cat((full_data, torch.cat(full_vpt, dim=0)), dim=0)
        columns.extend(["vpt_close", "vpt_high", "vpt_low", "vpt_open"])

    if not config_args.global_exclude_ppo:
        full_ppo = []
        full_ppo_columns = []
        for i in range(len(price_columns)):
            temp_ppo = []
            for j in range(len(tickers)):
                ppo_data, ppo_columns = feature_ppo(raw_data[i, :, j], prefix=price_columns[i] + "_")
                temp_ppo.append(ppo_data.unsqueeze(-1))
            full_ppo.append(torch.cat(temp_ppo, dim=-1))
            full_ppo_columns.extend(ppo_columns)
        full_data = torch.cat((full_data, torch.cat(full_ppo, dim=0)), dim=0)
        columns.extend(full_ppo_columns)

    clv_data = calculate_close_line_values(raw_data[0], raw_data[1], raw_data[2]).unsqueeze(0)
    if not config_args.global_exclude_clv:
        full_data = torch.cat((full_data, clv_data), dim=0)
        columns.extend(["clv"])

    prices = raw_data
    local_price_columns = ["local_" + s for s in price_columns]
    local_returns_columns = ["local_" + s for s in returns_columns]
    local_columns = []
    if not config_args.local_exclude_price:
        full_data = torch.cat((full_data, prices), dim=0)
        local_columns.extend(local_price_columns)

    if not config_args.local_exclude_price_ema:
        full_ema = []
        full_ema_columns = []
        for i in range(len(local_price_columns)):
            temp_ema = []
            for j in range(len(tickers)):
                ema_data, ema_columns = feature_ema(prices[i, :, j], prefix=local_price_columns[i] + "_")
                temp_ema.append(ema_data.unsqueeze(-1))
            full_ema.append(torch.cat(temp_ema, dim=-1))
            full_ema_columns.extend(ema_columns)
        full_data = torch.cat((full_data, torch.cat(full_ema, dim=0)), dim=0)
        local_columns.extend(full_ema_columns)

    if not config_args.local_exclude_price_vol:
        full_price_vol = []
        full_price_vol_columns = []
        for i in range(len(local_price_columns)):
            price_vol, vol_columns = feature_volatility_ret(prices[i:i+1], prefix=local_price_columns[i] + "_")
            full_price_vol.append(price_vol)
            full_price_vol_columns.extend(vol_columns)
        full_data = torch.cat((full_data, torch.cat(full_price_vol, dim=0)), dim=0)
        local_columns.extend(full_price_vol_columns)

    if not config_args.local_exclude_vpt:
        full_data = torch.cat((full_data, torch.cat(full_vpt, dim=0)), dim=0)
        local_columns.extend(["local_vpt_close", "local_vpt_high", "local_vpt_low", "local_vpt_open"])

    if not config_args.local_exclude_clv:
        full_data = torch.cat((full_data, clv_data), dim=0)
        local_columns.extend(["local_clv"])

    if not config_args.local_exclude_ppo:
        full_ppo = []
        full_ppo_columns = []
        for i in range(len(local_price_columns)):
            temp_ppo = []
            for j in range(len(tickers)):
                ppo_data, ppo_columns = feature_ppo(prices[i, :, j], prefix=local_price_columns[i] + "_")
                temp_ppo.append(ppo_data.unsqueeze(-1))
            full_ppo.append(torch.cat(temp_ppo, dim=-1))
            full_ppo_columns.extend(ppo_columns)
        full_data = torch.cat((full_data, torch.cat(full_ppo, dim=0)), dim=0)
        local_columns.extend(full_ppo_columns)

    returns = full_data[:5, :, :]
    if not config_args.local_exclude_ret:
        full_data = torch.cat((full_data, returns), dim=0)
        local_columns.extend(local_returns_columns)

    if not config_args.local_exclude_ret_ema:
        full_ema = []
        full_ema_columns = []
        for i in range(len(local_returns_columns)):
            temp_ema = []
            for j in range(len(tickers)):
                ema_data, ema_columns = feature_ema(returns[i, :, j], prefix=local_returns_columns[i] + "_")
                temp_ema.append(ema_data.unsqueeze(-1))
            full_ema.append(torch.cat(temp_ema, dim=-1))
            full_ema_columns.extend(ema_columns)
        full_data = torch.cat((full_data, torch.cat(full_ema, dim=0)), dim=0)
        local_columns.extend(full_ema_columns)

    if not config_args.local_exclude_ret_vol:
        full_ret_vol = []
        full_ret_vol_columns = []
        for i in range(len(local_returns_columns)):
            ret_vol, vol_columns = feature_volatility_ret(returns[i:i+1], prefix=local_returns_columns[i] + "_")
            full_ret_vol.append(ret_vol)
            full_ret_vol_columns.extend(vol_columns)
        full_data = torch.cat((full_data, torch.cat(full_ret_vol, dim=0)), dim=0)
        local_columns.extend(full_ret_vol_columns)

    if not config_args.local_exclude_price_bb:
        bb_data, bb_columns = feature_bollinger_bands_price_histogram(prices[0:1], prefix=local_price_columns[0] + "_")
        full_data = torch.cat((full_data, bb_data), dim=0)
        local_columns.extend(bb_columns)

    if not config_args.local_exclude_ret_bb:
        bb_data, bb_columns = feature_bollinger_bands_price_histogram(returns[0:1], prefix=local_returns_columns[0] + "_")
        full_data = torch.cat((full_data, bb_data), dim=0)
        local_columns.extend(bb_columns)

    time_data, time_columns = feature_time_data(indexes, tickers)
    full_data = torch.cat((full_data, time_data), dim=0)

    if config_args.global_exclude_ret:
        full_data = full_data[5:, :, :]
        columns = columns[5:]

    return full_data, columns, local_columns, time_columns

def data_fix_ffill(data: torch.Tensor) -> torch.Tensor:
    """
    Fills NaN values in a tensor using forward fill.
    This is a robust method to ensure no NaNs remain in the data.

    Args:
        data (torch.Tensor): Input tensor with potential NaN values.

    Returns:
        torch.Tensor: Tensor with NaN values filled.
    """
    # Forward fill
    is_bad = torch.isnan(data) | torch.isinf(data)
    good_values = data.clone()
    good_values[is_bad] = 0  # Temporarily set bad values to 0

    last_good_idx = torch.cummax(
        torch.logical_not(is_bad).int()
        * torch.arange(data.shape[1], device=data.device).view(1, -1, 1),
        dim=1,
    )[0]
    filled_indices = torch.gather(
        last_good_idx,
        1,
        torch.arange(data.shape[1], device=data.device)
        .view(1, -1, 1)
        .expand_as(last_good_idx),
    )
    filled_data = torch.gather(good_values, 1, filled_indices)

    return filled_data

    # flipped_data = torch.flip(good_values, dims=[1])
    # flipped_is_bad = torch.flip(is_bad, dims=[1])

    # last_good_idx = torch.cummax(torch.logical_not(flipped_is_bad).int() * torch.arange(data.shape[1], device=data.device).view(1, -1, 1), dim=1)[0]
    # filled_indices = torch.gather(last_good_idx, 1, torch.arange(data.shape[1], device=data.device).view(1, -1, 1).expand_as(last_good_idx))
    # bfilled_data = torch.gather(flipped_data, 1, filled_indices)

    # # Forward fill
    # bfilled_data = torch.flip(bfilled_data, dims=[1])

    # is_bad = torch.isnan(bfilled_data) | torch.isinf(bfilled_data)
    # last_good_idx = torch.cummax(torch.logical_not(is_bad).int() * torch.arange(bfilled_data.shape[1], device=bfilled_data.device).view(1, -1, 1), dim=1)[0]
    # filled_indices = torch.gather(last_good_idx, 1, torch.arange(bfilled_data.shape[1], device=bfilled_data.device).view(1, -1, 1).expand_as(last_good_idx))

    # return torch.gather(bfilled_data, 1, filled_indices)


def generate_noisy_copies(raw_prices: torch.Tensor, tickers: List[str], target_dates: List[int], indexes, data_length: int, cutoff: int, n_copies: int, noise_factor: float) -> List[torch.Tensor]:
    # relevant_data = raw_prices[:, :data_length + cutoff, :]
    # relevant_indexes = indexes[:data_length + cutoff]

    relevant_data = raw_prices
    relevant_indexes = indexes

    relevant_data_returns = (relevant_data[:, 1:] - relevant_data[:, :-1]) / relevant_data[:, :-1]
    relevant_data_returns = torch.cat([torch.zeros_like(relevant_data_returns[:, :1]), relevant_data_returns], dim=1)

    relevant_vol = []
    for i in range(relevant_data_returns.shape[0]):
        temp_vol = []
        for j in range(relevant_data_returns.shape[2]):
            temp_vol.append(calculate_ema_vol_pandas(relevant_data_returns[i, :, j], lookback=22))
        relevant_vol.append(torch.stack(temp_vol, dim=-1))
    relevant_vol = torch.stack(relevant_vol, dim=0)

    relevant_vol = torch.cat([relevant_vol[:,:1],relevant_vol[:,:-1]], dim=1)

    noisy_copies = []
    for i in range(n_copies):
        noisy_relevant_data = relevant_data * (1 + torch.randn_like(relevant_data) * relevant_vol * noise_factor)
        full_data, columns, local_columns, time_columns = calculate_features(noisy_relevant_data, tickers, relevant_indexes)
        full_data[:len(columns)+len(local_columns)] = data_fix_ffill(full_data[:len(columns)+len(local_columns)])

        data = torch.empty(
            full_data.shape[0],
            max(target_dates),
            full_data.shape[1] - max(target_dates),
            full_data.shape[2],
            dtype=torch.float32,
        )
        for i in range(max(target_dates)):
            data[:, i, :, :] = full_data[:, i : -(max(target_dates) - i), :]
        data = data[:, :, cutoff:, :]  # (features, target_inputs, time series, tickers)

        noisy_copies.append(data.clone())
    
    return noisy_copies

def auto_correct_feature_skew_pre_znorm(data: torch.Tensor, column_to_idx: dict, data_length: int, non_global_feats: int, time_feats: int, seq_len: int, skew_threshold: float=1.0) -> torch.Tensor:
    original_dtype = data.dtype
    fitted_processors = {}
    for idx in range(data.shape[0]-time_feats):
        skew = stats.skew(data[idx, :, :data_length-seq_len+1].flatten().numpy().astype(np.float64))
        if abs(skew) > skew_threshold:
            # signed_log = torch.sign(data[idx]) * torch.log1p(torch.abs(data[idx]))
            # data[idx] = signed_log
            scaler = RobustScaler()
            transformer = PowerTransformer(method='yeo-johnson', standardize=False)

            flat_train_data = data[idx, :, :data_length-seq_len+1].numpy().flatten().reshape(-1, 1).astype(np.float64)

            scaler.fit(flat_train_data)
            flat_train_scaled = scaler.transform(flat_train_data)

            transformer.fit(flat_train_scaled)

            original_shape = data[idx].shape
            data_flat = data[idx].numpy().flatten().reshape(-1, 1).astype(np.float64)
            data_scaled = scaler.transform(data_flat)

            transformed_data = transformer.transform(data_scaled)
            transformed_data = transformed_data.reshape(original_shape)

            data[idx] = torch.from_numpy(transformed_data).to(dtype=original_dtype)

            fitted_processors[idx] = {"scaler": scaler, "transformer": transformer}

    return data, fitted_processors

def auto_correct_feature_kurtosis_post_znorm(data: torch.Tensor, column_to_idx: dict, data_length: int) -> torch.Tensor:
    for idx in range(data.shape[0]):
        lower = torch.quantile(data[idx, :, :data_length], 0.0025)
        upper = torch.quantile(data[idx, :, :data_length], 0.9975)
        data[idx] = torch.clip(data[idx], lower, upper)

    return data

def apply_processors_to_noisy_copy(noisy_copy: torch.Tensor, fitted_processors: dict) -> torch.Tensor:
    original_dtype = noisy_copy.dtype
    for idx in fitted_processors.keys():
        scaler = fitted_processors[idx]["scaler"]
        transformer = fitted_processors[idx]["transformer"]
        
        original_shape = noisy_copy[idx].shape
        data_flat = noisy_copy[idx].numpy().flatten().reshape(-1, 1).astype(np.float64)
        data_scaled = scaler.transform(data_flat)

        transformed_data = transformer.transform(data_scaled)
        transformed_data = transformed_data.reshape(original_shape)

        noisy_copy[idx] = torch.from_numpy(transformed_data).to(dtype=original_dtype)
    return noisy_copy

def draw_graph(original_data, noisy_data):
    plt.plot(original_data, label='Original')
    plt.plot(noisy_data, label='Noisy')
    plt.legend()
    plt.show()
    return

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

# def feature_bollinger_bands_price_histogram_list(price: torch.Tensor, prefix: str = "price_", lookbacks: list = [3, [5]], up: bool=True) -> torch.Tensor:
#     bb_columns = []
#     bb_data = []

#     for lookback in lookbacks:
#         sma = calculate_sma(price, lookback[0])
#         vol = calculate_volatility_returns(price, lookback[0]).unsqueeze(0)

#         normed_signal = (price - sma) / (vol + 1e-5)

#         for mult in lookback[1]:
#             bb_signal = normed_signal - mult/10 if up else normed_signal + mult/10
#             bb_data.append(bb_signal)
#             bb_columns.append(prefix + "bollinger_" + ("up" if up else "down") + "_signal_" + str(mult) + "_" + str(lookback[0]))

#     bb_data = torch.cat(bb_data, dim=0)
#     return bb_data, bb_columns

# def feature_ppo_list(price: torch.Tensor, prefix: str = "price_", lookbacks: list = [5, [10, 20]]) -> torch.Tensor:
#     ppo_columns = []
#     emas = {}

#     all_lookbacks = [n[:1]+n[1] for n in lookbacks]
#     all_lookbacks = set(m for n in all_lookbacks for m in n)
#     for i in all_lookbacks:
#         ema = calculate_ema_pandas(price, i)
#         emas[i] = ema
#     ppo_data = []
#     for i in lookbacks:
#         for j in i[1]:
#             ppo = (emas[i[0]] - emas[j])/emas[j]
#             ppo_data.append(ppo)
#             ppo_columns.append(prefix + "ppo_"+str(i[0])+"_"+str(j))
    
#     ppo_data = torch.stack(ppo_data, dim=0)
#     return ppo_data, ppo_columns

# def feature_ema(
#     price: torch.Tensor, prefix: str = "price_", lookbacks: list = [5, 10, 20, 50]) -> torch.Tensor:
#     ema_columns = []
#     ema = calculate_ema_pandas(price, lookback=lookbacks[0])
#     ema_data = ema.unsqueeze(0)
#     ema_columns.append(prefix + "ema_" + str(lookbacks[0]))

#     for lookback in lookbacks[1:]:
#         ema = calculate_ema_pandas(price, lookback=lookback)
#         ema_data = torch.cat((ema_data, ema.unsqueeze(0)), dim=0)
#         ema_columns.append(prefix + "ema_" + str(lookback))

#     return ema_data, ema_columns

# def feature_volatility_ret(returns: torch.Tensor, prefix: str = "returns_", lookbacks: list = [3, 5, 7, 10, 20]) -> torch.Tensor:
#     vol_columns = []

#     vol = calculate_volatility_returns(returns, lookback=lookbacks[0])
#     vol_data = vol.unsqueeze(0)
#     vol_columns.append(prefix + "volatility_" + str(lookbacks[0]))

#     for i in range(1, len(lookbacks)):
#         vol = calculate_volatility_returns(returns, lookback=lookbacks[i])
#         vol_data = torch.cat((vol_data, vol.unsqueeze(0)), dim=0)
#         vol_columns.append(prefix + "volatility_" + str(lookbacks[i]))

#     return vol_data, vol_columns

def feature_volatility_ret(
    returns: torch.Tensor, prefix: str = "returns_"
) -> torch.Tensor:
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


def feature_bollinger_bands_returns(
    returns: torch.Tensor, prefix: str = "returns_"
) -> torch.Tensor:
    bb_columns = []
    sma_5 = calculate_sma(returns, lookback=5)
    vol_5 = calculate_volatility_returns(returns, lookback=5).unsqueeze(0)
    bb = sma_5 + 0.5 * vol_5
    bb_data = bb
    bb_columns.append(prefix + "bollinger_5_up")
    bb = sma_5 - 0.5 * vol_5
    bb_data = torch.cat((bb_data, bb), dim=0)
    bb_columns.append(prefix + "bollinger_5_down")

    sma_10 = calculate_sma(returns, lookback=10)
    vol_10 = calculate_volatility_returns(returns, lookback=10).unsqueeze(0)
    bb = sma_10 + 0.5 * vol_10
    bb_data = torch.cat((bb_data, bb), dim=0)
    bb_columns.append(prefix + "bollinger_10_up")
    bb = sma_10 - 0.5 * vol_10
    bb_data = torch.cat((bb_data, bb), dim=0)
    bb_columns.append(prefix + "bollinger_10_down")

    sma_20 = calculate_sma(returns, lookback=20)
    vol_20 = calculate_volatility_returns(returns, lookback=20).unsqueeze(0)
    bb = sma_20 + 0.5 * vol_20
    bb_data = torch.cat((bb_data, bb), dim=0)
    bb_columns.append(prefix + "bollinger_20_up")
    bb = sma_20 - 0.5 * vol_20
    bb_data = torch.cat((bb_data, bb), dim=0)
    bb_columns.append(prefix + "bollinger_20_down")

    sma_50 = calculate_sma(returns, lookback=50)
    vol_50 = calculate_volatility_returns(returns, lookback=50).unsqueeze(0)
    bb = sma_50 + 0.5 * vol_50
    bb_data = torch.cat((bb_data, bb), dim=0)
    bb_columns.append(prefix + "bollinger_50_up")
    bb = sma_50 - 0.5 * vol_50
    bb_data = torch.cat((bb_data, bb), dim=0)
    bb_columns.append(prefix + "bollinger_50_down")

    return bb_data, bb_columns


def feature_bollinger_bands_price_histogram(
    price: torch.Tensor, prefix: str = "price_"
) -> torch.Tensor:
    bb_columns = []
    sma_5 = calculate_sma(price, lookback=5)
    vol_5 = calculate_volatility_returns(price, lookback=5).unsqueeze(0)
    bb = sma_5 + 1 * vol_5
    bb_signal = price - bb
    bb_data = bb_signal
    bb_columns.append(prefix + "bollinger_5_up_signal")
    bb = sma_5 - 1 * vol_5
    bb_signal = price - bb
    bb_data = torch.cat((bb_data, bb_signal), dim=0)
    bb_columns.append(prefix + "bollinger_5_down_signal")

    sma_10 = calculate_sma(price, lookback=10)
    vol_10 = calculate_volatility_returns(price, lookback=10).unsqueeze(0)
    bb = sma_10 + 1.5 * vol_10
    bb_signal = price - bb
    bb_data = torch.cat((bb_data, bb_signal), dim=0)
    bb_columns.append(prefix + "bollinger_10_up_signal")
    bb = sma_10 - 1.5 * vol_10
    bb_signal = price - bb
    bb_data = torch.cat((bb_data, bb_signal), dim=0)
    bb_columns.append(prefix + "bollinger_10_down_signal")

    sma_20 = calculate_sma(price, lookback=20)
    vol_20 = calculate_volatility_returns(price, lookback=20).unsqueeze(0)
    bb = sma_20 + 2 * vol_20
    bb_signal = price - bb
    bb_data = torch.cat((bb_data, bb_signal), dim=0)
    bb_columns.append(prefix + "bollinger_20_up_signal")
    bb = sma_20 - 2 * vol_20
    bb_signal = price - bb
    bb_data = torch.cat((bb_data, bb_signal), dim=0)
    bb_columns.append(prefix + "bollinger_20_down_signal")

    sma_50 = calculate_sma(price, lookback=50)
    vol_50 = calculate_volatility_returns(price, lookback=50).unsqueeze(0)
    bb = sma_50 + 2.5 * vol_50
    bb_signal = price - bb
    bb_data = torch.cat((bb_data, bb_signal), dim=0)
    bb_columns.append(prefix + "bollinger_50_up_signal")
    bb = sma_50 - 2.5 * vol_50
    bb_signal = price - bb
    bb_data = torch.cat((bb_data, bb_signal), dim=0)
    bb_columns.append(prefix + "bollinger_50_down_signal")

    return bb_data, bb_columns


def feature_ema(price: torch.Tensor, prefix: str = "price_") -> torch.Tensor:
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
    ppo_columns.append(prefix + "ppo_5_10")

    ppo = (ema_10 - ema_20) / ema_20
    ppo_data = torch.cat((ppo_data, ppo.unsqueeze(0)), dim=0)
    ppo_columns.append(prefix + "ppo_10_20")

    ppo = (ema_5 - ema_20) / ema_20
    ppo_data = torch.cat((ppo_data, ppo.unsqueeze(0)), dim=0)
    ppo_columns.append(prefix + "ppo_5_20")

    return ppo_data, ppo_columns


def feature_macd(price: torch.Tensor, prefix: str = "price_") -> torch.Tensor:
    macd_columns = []
    ema_5 = calculate_ema_pandas(price, lookback=5)
    ema_10 = calculate_ema_pandas(price, lookback=10)
    ema_20 = calculate_ema_pandas(price, lookback=20)

    macd = ema_5 - ema_10
    macd_data = macd.unsqueeze(0)
    macd_columns.append(prefix + "macd_5_10")
    macd_signal = calculate_ema_pandas(macd, lookback=5)
    macd_data = torch.cat((macd_data, macd_signal.unsqueeze(0)), dim=0)
    macd_columns.append(prefix + "macd_signal_5_10_5")
    macd_histogram = macd - macd_signal
    macd_data = torch.cat((macd_data, macd_histogram.unsqueeze(0)), dim=0)
    macd_columns.append(prefix + "macd_histogram_5_10_5")

    macd = ema_10 - ema_20
    macd_data = torch.cat((macd_data, macd.unsqueeze(0)), dim=0)
    macd_columns.append(prefix + "macd_10_20")
    macd_signal = calculate_ema_pandas(macd, lookback=10)
    macd_data = torch.cat((macd_data, macd_signal.unsqueeze(0)), dim=0)
    macd_columns.append(prefix + "macd_signal_10_20_10")
    macd_histogram = macd - macd_signal
    macd_data = torch.cat((macd_data, macd_histogram.unsqueeze(0)), dim=0)
    macd_columns.append(prefix + "macd_histogram_10_20_10")

    macd = ema_5 - ema_20
    macd_data = torch.cat((macd_data, macd.unsqueeze(0)), dim=0)
    macd_columns.append(prefix + "macd_5_20")
    macd_signal = calculate_ema_pandas(macd, lookback=10)
    macd_data = torch.cat((macd_data, macd_signal.unsqueeze(0)), dim=0)
    macd_columns.append(prefix + "macd_signal_5_20_10")
    macd_histogram = macd - macd_signal
    macd_data = torch.cat((macd_data, macd_histogram.unsqueeze(0)), dim=0)
    macd_columns.append(prefix + "macd_histogram_5_20_10")

    return macd_data, macd_columns


def feature_macd_histogram(price: torch.Tensor, prefix: str = "close_") -> torch.Tensor:
    macd_histogram_columns = []
    ema_5 = calculate_ema_pandas(price, lookback=5)
    ema_10 = calculate_ema_pandas(price, lookback=10)
    ema_20 = calculate_ema_pandas(price, lookback=20)

    macd = ema_5 - ema_10
    macd_signal = calculate_ema_pandas(macd, lookback=5)
    macd_histogram = macd - macd_signal
    macd_histogram_data = macd_histogram.unsqueeze(0)
    macd_histogram_columns.append(prefix + "macd_histogram_5_10_5")

    macd = ema_10 - ema_20
    macd_signal = calculate_ema_pandas(macd, lookback=10)
    macd_histogram = macd - macd_signal
    macd_histogram_data = torch.cat(
        (macd_histogram_data, macd_histogram.unsqueeze(0)), dim=0
    )
    macd_histogram_columns.append(prefix + "macd_histogram_10_20_10")

    macd = ema_5 - ema_20
    macd_signal = calculate_ema_pandas(macd, lookback=10)
    macd_histogram = macd - macd_signal
    macd_histogram_data = torch.cat(
        (macd_histogram_data, macd_histogram.unsqueeze(0)), dim=0
    )
    macd_histogram_columns.append(prefix + "macd_histogram_5_20_10")

    return macd_histogram_data, macd_histogram_columns


def feature_time_data(
    indexes: pd.DatetimeIndex, tickers: List[str]
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

    # is_leap_year = torch.tensor(indexes.is_leap_year, dtype=torch.float32)
    # time_data = torch.cat((time_data, is_leap_year.unsqueeze(0)), dim=0)
    # time_columns.append("is_leap_year")

    # is_month_start = torch.tensor(indexes.is_month_start, dtype=torch.float32)
    # time_data = torch.cat((time_data, is_month_start.unsqueeze(0)), dim=0)
    # time_columns.append("is_month_start")

    # is_month_end = torch.tensor(indexes.is_month_end, dtype=torch.float32)
    # time_data = torch.cat((time_data, is_month_end.unsqueeze(0)), dim=0)
    # time_columns.append("is_month_end")

    # is_quarter_start = torch.tensor(indexes.is_quarter_start, dtype=torch.float32)
    # time_data = torch.cat((time_data, is_quarter_start.unsqueeze(0)), dim=0)
    # time_columns.append("is_quarter_start")

    # is_quarter_end = torch.tensor(indexes.is_quarter_end, dtype=torch.float32)
    # time_data = torch.cat((time_data, is_quarter_end.unsqueeze(0)), dim=0)
    # time_columns.append("is_quarter_end")

    time_data = time_data.unsqueeze(-1)
    time_data = time_data.expand(-1, -1, len(tickers))
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
    sma[:, :lookback] = cumsum[:, :lookback] / torch.arange(1, lookback + 1).view(
        1, -1, 1
    )
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
    for i in range(2, lookback):
        volatility[i] = returns[:, :i].std(dim=(0,1))
    windowed = returns.unfold(1, lookback, 1).contiguous()
    volatility[lookback-1:] = windowed.std(dim=(0,-1))
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
    mfv_cumsum = torch.cumsum(mfv, dim=1)
    ad[:, :lookback] = mfv_cumsum[:, :lookback]
    ad[:, lookback:] = mfv_cumsum[:, lookback:] - mfv_cumsum[:, :-lookback]
    return ad


def calculate_accumulation_distribution_index_standard(
    volume: torch.Tensor, clv: torch.Tensor
):
    return torch.cumsum(clv * volume, dim=1)


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


def calculate_volume_price_trend_standard(returns: torch.Tensor, volume: torch.Tensor):
    vtr = volume * returns
    return torch.cumsum(vtr, dim=1)


def calculate_ema_pandas(
    daily_values: torch.Tensor, lookback: int = 10
) -> torch.Tensor:  # to calculate ema for anything
    original_dtype = daily_values.dtype

    series_pd = pd.Series(daily_values.cpu().numpy())
    ema = series_pd.ewm(span=lookback, adjust=False, min_periods=1).mean()
    return torch.from_numpy(ema.values).to(original_dtype)

def calculate_ema_vol_pandas(
    returns: torch.Tensor, lookback: int = 22
) -> torch.Tensor: # to calculate ema of volatility
    original_dtype = returns.dtype

    series_pd = pd.Series(returns.cpu().numpy())
    emv = series_pd.ewm(span=lookback, adjust=False, min_periods=1).std()
    emv.ffill(inplace=True)
    emv.bfill(inplace=True) # idk if this is valid for feature, but ir wrks for noise
    return torch.from_numpy(emv.values).to(original_dtype)

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

    low_windows = low.unfold(1, k_lookback, 1)
    high_windows = high.unfold(1, k_lookback, 1)

    period_low_values = torch.min(low_windows, dim=-1)[0]
    period_high_values = torch.max(high_windows, dim=-1)[0]

    close_for_k = close[:, k_lookback - 1 :]

    numerator = close_for_k - period_low_values
    denominator = period_high_values - period_low_values
    percent_k_calculated = torch.full_like(numerator, 50.0)

    valid_denominator_mask = denominator > 1e-12

    percent_k_calculated[valid_denominator_mask] = (
        100 * numerator[valid_denominator_mask] / denominator[valid_denominator_mask]
    )

    percent_k[:, k_lookback - 1 :] = percent_k_calculated

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


METRIC_TAG_MAP = {
    "revenue": (
        [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
        ],
        "USD",
    ),
    "net_income": (["NetIncomeLoss", "ProfitLoss"], "USD"),
    "eps": (["EarningsPerShareDiluted", "EarningsPerShareBasic"], "USD/shares"),
    "shares_outstanding": (
        [
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingBasic",
        ],
        "shares",
    ),
    "total_assets": (["Assets"], "USD"),
    "total_liabilities": (["Liabilities"], "USD"),
    "current_assets": (["AssetsCurrent"], "USD"),
    "current_liabilities": (["LiabilitiesCurrent"], "USD"),
    "equity": (["StockholdersEquity"], "USD"),
    "operating_cash_flow": (["NetCashProvidedByUsedInOperatingActivities"], "USD"),
    "capex": (
        [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
        ],
        "USD",
    ),
}


def _get_sec_data_series(
    company_facts: dict, tag_list: List[str], unit: str
) -> Optional[pd.Series]:
    """
    Helper to find the first available data for a list of possible XBRL tags
    and return it as a pandas Series indexed by the filing end date.
    """
    for standard in company_facts.get("facts", {}):
        for tag in tag_list:
            if (
                tag in company_facts["facts"][standard]
                and unit in company_facts["facts"][standard][tag]["units"]
            ):
                data = company_facts["facts"][standard][tag]["units"][unit]
                if data:
                    df = pd.DataFrame(data)
                    # Filter for annual (10-K) and quarterly (10-Q) filings
                    df = df[df["frame"].notna()]
                    if not df.empty:
                        df["end"] = pd.to_datetime(df["end"])
                        # Use the most recent value for any given date
                        series = df.drop_duplicates(
                            subset="end", keep="last"
                        ).set_index("end")["val"]
                        return series.sort_index()
    return None


def fetch_and_align_fundamental_data(
    tickers: List[str], daily_index: pd.DatetimeIndex, daily_close_prices: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetches a wide range of fundamental data, calculates key ratios,
    and aligns them to a daily time series using a robust build-then-populate method.
    """
    print("Fetching full historical fundamental data from SEC EDGAR API...")
    headers = {"User-Agent": "YourName YourEmail@example.com"}

    try:
        company_tickers_response = requests.get(
            "https://www.sec.gov/files/company_tickers.json", headers=headers
        )
        company_tickers_response.raise_for_status()
        ticker_to_cik = {
            item["ticker"]: str(item["cik_str"]).zfill(10)
            for item in company_tickers_response.json().values()
        }
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Could not download ticker-to-CIK mapping. Error: {e}")
        return pd.DataFrame(), []

    # --- Step 1: Fetch and calculate all raw/ratio series for each ticker individually ---
    all_tickers_features = {}
    for ticker in tickers:
        if ticker.startswith("^"):
            continue
        cik = ticker_to_cik.get(ticker.upper())
        if not cik:
            continue

        print(f"  - Processing fundamentals for {ticker} (CIK: {cik})")
        try:
            facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            facts_response = requests.get(facts_url, headers=headers)
            facts_response.raise_for_status()
            facts_data = facts_response.json()

            raw_components = {
                metric: _get_sec_data_series(facts_data, tags, unit)
                for metric, (tags, unit) in METRIC_TAG_MAP.items()
            }

            calculated_ratios = {}

            def safe_divide(num, den):
                if num is None or den is None:
                    return None
                num_aligned, den_aligned = num.align(den, method="ffill", join="outer")
                return num_aligned / den_aligned.where(den_aligned != 0, np.nan)

            calculated_ratios["roe"] = safe_divide(
                raw_components["net_income"], raw_components["equity"]
            )
            calculated_ratios["roa"] = safe_divide(
                raw_components["net_income"], raw_components["total_assets"]
            )
            calculated_ratios["net_profit_margin"] = safe_divide(
                raw_components["net_income"], raw_components["revenue"]
            )
            calculated_ratios["debt_to_equity"] = safe_divide(
                raw_components["total_liabilities"], raw_components["equity"]
            )
            calculated_ratios["current_ratio"] = safe_divide(
                raw_components["current_assets"], raw_components["current_liabilities"]
            )
            calculated_ratios["asset_turnover"] = safe_divide(
                raw_components["revenue"], raw_components["total_assets"]
            )

            if raw_components["operating_cash_flow"] is not None:
                capex = raw_components.get(
                    "capex",
                    pd.Series(0, index=raw_components["operating_cash_flow"].index),
                ).abs()
                fcf, _ = raw_components["operating_cash_flow"].align(
                    capex, method="ffill", join="outer"
                )
                fcf = fcf.fillna(0) - capex.reindex(fcf.index, method="ffill").fillna(0)
                calculated_ratios["fcf_per_share"] = safe_divide(
                    fcf, raw_components["shares_outstanding"]
                )

            ticker_features = {
                k: v for k, v in calculated_ratios.items() if v is not None
            }
            # Add raw fundamentals needed for later valuation calcs
            if raw_components["eps"] is not None:
                ticker_features["eps"] = raw_components["eps"]
            if raw_components["shares_outstanding"] is not None:
                ticker_features["shares_outstanding"] = raw_components[
                    "shares_outstanding"
                ]
            if raw_components["revenue"] is not None:
                ticker_features["total_revenue"] = raw_components["revenue"]
            if raw_components["equity"] is not None:
                ticker_features["total_equity"] = raw_components["equity"]

            all_tickers_features[ticker] = ticker_features
            time.sleep(0.15)
        except Exception as e:
            print(f"    - An unexpected error occurred for {ticker}: {e}")

    # --- Step 2: Create the final, perfectly structured container DataFrame ---
    base_feature_names = sorted(
        list(set(k for d in all_tickers_features.values() for k in d.keys()))
    )
    valuation_feature_names = ["pe_ratio", "ps_ratio", "pb_ratio"]
    all_feature_names = base_feature_names + valuation_feature_names

    multi_index_columns = pd.MultiIndex.from_product(
        [all_feature_names, tickers], names=["feature", "ticker"]
    )
    final_df = pd.DataFrame(index=daily_index, columns=multi_index_columns)

    # --- Step 3: Populate the container with the calculated data ---
    for ticker, features_dict in all_tickers_features.items():
        for feature_name, raw_series in features_dict.items():
            if raw_series is None or raw_series.empty:
                continue

            # Align the quarterly/annual series to the daily index
            combined_index = daily_index.union(raw_series.index).sort_values()
            aligned_series = (
                raw_series.reindex(combined_index).ffill().reindex(daily_index).bfill()
            )

            # Place the aligned data into the correct column of our final DataFrame
            final_df.loc[:, (feature_name, ticker)] = aligned_series

    # --- Step 4: Calculate valuation ratios directly on the aligned DataFrame ---
    market_cap = daily_close_prices * final_df["shares_outstanding"]

    if "eps" in final_df:
        pe_ratio = daily_close_prices / final_df["eps"]
        pe_ratio[final_df["eps"] <= 0] = np.nan  # Use NaN for meaningless P/E
        final_df["pe_ratio"] = pe_ratio

    if "total_revenue" in final_df:
        ps_ratio = market_cap / final_df["total_revenue"]
        final_df["ps_ratio"] = ps_ratio

    if "total_equity" in final_df:
        pb_ratio = market_cap / final_df["total_equity"]
        final_df["pb_ratio"] = pb_ratio

    # --- Step 5: Final Cleanup ---
    # Ensure column order is consistent
    final_df = final_df.reindex(columns=all_feature_names, level=0)

    # Fill any remaining NaNs (e.g., for ^GSPC or from calcs) with 0 and clean infs
    final_df.fillna(0, inplace=True)
    final_df.replace([np.inf, -np.inf], 0, inplace=True)

    return final_df, all_feature_names


# E. Fundamental Data (More Involved - yf.Ticker().info, .financials, etc.):
# Examples: P/E Ratio, EPS, P/S Ratio, Dividend Yield, Market Cap, Beta.
# Challenge: This data is usually reported quarterly or annually.


# F. Feature Interactions & Relative Strength:
# Stock Return vs. Market Return (Alpha Component):
# Volatility Relative to Market:
# stock_volatility_feature - vix_feature (after both are on comparable scales/forms, e.g., both as % or normalized).


# Deeper Contextual Understanding (More Market & Economic Data): The market doesn't operate in a vacuum.
# Fundamental Data (A Bigger Step): Company-specific financial health.
# 1. Interest Rates (Crucial):
# Why: Rates affect discount rates, company borrowing costs, economic growth expectations, and sector rotations.

# 2. Volatility Spreads / Ratios:
# Stock HV vs. VIX: Normalized_Stock_HV_t - Normalized_VIX_t (or ratio). Is the stock more or less volatile than its typical relationship with market volatility?
# Short-term HV vs. Long-term HV for the same stock: HV_10_day / HV_50_day. Can indicate changing volatility regimes for the stock.
