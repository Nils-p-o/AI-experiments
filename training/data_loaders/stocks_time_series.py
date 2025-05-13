# TODO rewrite this to be what i want
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
# TODO and TODO
class FinancialNumericalDataset(Dataset):
    def __init__(self, file_path: str, seq_len: Optional[int] = None, preload: bool = True):
        self.seq_len = seq_len
        self.preload = preload
        self.file_path = file_path
        if preload:
            self.sequences_data = torch.load(file_path)
        else:
            self.sequences_data = None

    def __len__(self) -> int: # TODO simulaneous length of time series
        if self.preload:
            return self.sequences_data.size(-1) - self.seq_len
        else: # TODO?
            raise NotImplementedError
    def __getitem__(self, idx):
        if self.preload:
            input_sequence = self.sequences_data[:,idx : idx + self.seq_len]
            target_sequence = self.sequences_data[:,idx + 1 : idx + self.seq_len + 1]
            return input_sequence, target_sequence
        else: # TODO?
            raise NotImplementedError

class FinancialNumericalDataModule(pl.LightningDataModule):
    # TODO add info from dates (.csv files)
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
            self.train_dataset = FinancialNumericalDataset(self.train_file, self.seq_len)
            self.val_dataset = FinancialNumericalDataset(self.val_file, self.seq_len)
        if stage == "test" or stage is None:
            self.test_dataset = FinancialNumericalDataset(self.test_file, self.seq_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers-3,
            persistent_workers=True
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
        for i in range(len(indexes)):
            f.write(f"{indexes[i]}, {indexes.day_of_week[i]}, {indexes.day_of_year[i]}\n")
            # could include is quarter start/end and year start/end
        

def download_numerical_financial_data(
    tickers: List[str],
    seq_len: int = 64,
    output_dir: str = "time_series_data",
    start_date: str = "1900-01-01",
    end_date: str = "2025-01-01",
    val_split_ratio: float = 0.15, 
    test_split_ratio: float = 0.1,
    check_if_already_downloaded: bool = True
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if check_if_already_downloaded and is_already_downloaded(tickers, output_dir):
        print("Data already downloaded.")
        return
    #download all available data by default
    raw_data = yf.download(tickers, start=start_date, end=end_date, progress=True, auto_adjust=False, back_adjust=False)
    if raw_data.empty:
        print("No data downloaded.")
        return

    unique_tickers = sorted(list(set(tickers)))
    ticker_to_id = {ticker: i for i, ticker in enumerate(unique_tickers)}
    
    # TODO future removing tickers without data (maybe replace with missing data tokens?)
    # TODO add column_to_id to metadata file
    indexes = raw_data.index
    # can get day of week, month, year, etc.
    columns = raw_data.columns

    # names and tickers
    data_length = len(indexes)
    raw_data = torch.tensor(raw_data.values, dtype=torch.float32) # (Time, Features)
    raw_data = raw_data.transpose(0, 1) # (Features, Time)

    train_data_length = int(data_length * (1 - val_split_ratio - test_split_ratio))
    val_data_length = int(data_length * val_split_ratio)
    test_data_length = data_length - train_data_length - val_data_length
    train_data_raw, val_data_raw, test_data_raw = torch.split(raw_data, [train_data_length, val_data_length, test_data_length], dim=1)
    train_indexes, val_indexes, test_indexes = indexes[:train_data_length], indexes[train_data_length:train_data_length+val_data_length], indexes[train_data_length+val_data_length:]
    
    # no normalization (will do that on the fly, per sequence)
    
    save_indexes_to_csv(train_indexes, os.path.join(output_dir, "train.csv"))
    save_indexes_to_csv(val_indexes, os.path.join(output_dir, "val.csv"))
    save_indexes_to_csv(test_indexes, os.path.join(output_dir, "test.csv"))
    
    torch.save(train_data_raw, os.path.join(output_dir, "train.pt"))
    torch.save(val_data_raw, os.path.join(output_dir, "val.pt"))
    torch.save(test_data_raw, os.path.join(output_dir, "test.pt"))
    
    collected_data_start_date = str(train_indexes[0])[:10]
    collected_data_end_date = str(test_indexes[-1])[:10]

    metadata = {
        "tickers": unique_tickers,
        "ticker_to_id": ticker_to_id,
        "start_date": collected_data_start_date,
        "end_date": collected_data_end_date,
        "val_split_ratio": val_split_ratio,
        "test_split_ratio": test_split_ratio,
        "columns": list(columns.get_level_values(0)),
        "indexes": len(indexes)
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    tickers = ["AAPL"] # Reduced for faster testing
    output_dir = "time_series_data"
    seq_len = 64

    download_numerical_financial_data(
        tickers=tickers,
        seq_len=seq_len,
        output_dir=output_dir,
    )
    # ... (rest of the __main__ test block, ensure it uses the correct paths and checks new target shape)
    if os.path.exists(os.path.join(output_dir, "metadata.json")):
        print("\n--- Testing FinancialNumericalDataModule (refined) ---")
        dm = FinancialNumericalDataModule(
            train_file=os.path.join(output_dir, "train.pt"),
            val_file=os.path.join(output_dir, "val.pt"),
            test_file=os.path.join(output_dir, "test.pt"),
            metadata_file=os.path.join(output_dir, "metadata.json"),
            batch_size=2, seq_len=seq_len
        )
        dm.prepare_data()
        dm.setup('fit')
        print(f"Input feat dim: {dm.get_input_feature_dim()}, Target feat dim (data): {dm.get_target_feature_dim()}, Num tickers: {dm.get_num_tickers()}")
        
        if dm.train_dataset and len(dm.train_dataset) > 0:
            for batch in dm.train_dataloader():
                inputs, targets, ticker_ids = batch
                print("Sample batch shapes:")
                print(f"  Inputs: {inputs.shape}, Targets: {targets.shape}, Ticker IDs: {ticker_ids.shape}")
                # Expected: Inputs: (B, S, F_in), Targets: (B, S, 1), Ticker IDs: (B, 1)
                break
        else:
            print("Train dataset is empty.")