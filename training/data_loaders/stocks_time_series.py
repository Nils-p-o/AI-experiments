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

# class FinancialNumericalDataset(Dataset):
#     def __init__(self, file_path: str):
#         try:
#             self.sequences_data = torch.load(file_path)
#             if not isinstance(self.sequences_data, list) or \
#                (len(self.sequences_data) > 0 and (
#                    not isinstance(self.sequences_data[0], dict) or
#                    not all(k in self.sequences_data[0] for k in ['input', 'target', 'ticker_id'])
#                )):
#                 raise ValueError(f"Data in {file_path} is not in the expected format (list of dicts with 'input', 'target', 'ticker_id').")
#         except Exception as e:
#             print(f"Error loading or validating data from {file_path}: {e}")
#             self.sequences_data = []

#     def __len__(self) -> int:
#         return len(self.sequences_data)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         if not self.sequences_data:
#             raise IndexError("Dataset is empty or failed to load.")
#         item = self.sequences_data[idx]
#         return item['input'], item['target'], item['ticker_id']

# class FinancialNumericalDataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         train_file: str,
#         val_file: str,
#         test_file: str,
#         metadata_file: str,
#         batch_size: int,
#         num_workers: int = 4,
#         seq_len: Optional[int] = None,
#     ):
#         super().__init__()
#         self.train_file = train_file
#         self.val_file = val_file
#         self.test_file = test_file
#         self.metadata_file = metadata_file
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self._seq_len_from_metadata = seq_len 

#         self.train_dataset: Optional[FinancialNumericalDataset] = None
#         self.val_dataset: Optional[FinancialNumericalDataset] = None
#         self.test_dataset: Optional[FinancialNumericalDataset] = None
        
#         self._metadata: Optional[Dict[str, Any]] = None
#         self._load_metadata() 

#     def _load_metadata(self):
#         if self._metadata is None:
#             if not os.path.exists(self.metadata_file):
#                 # This can be an issue if prepare_data hasn't run, handle gracefully or ensure it's called
#                 print(f"Warning: Metadata file {self.metadata_file} not found during init.")
#                 return
#             try:
#                 with open(self.metadata_file, 'r') as f:
#                     self._metadata = json.load(f)
#                 if self._seq_len_from_metadata is None: 
#                      self._seq_len_from_metadata = self._metadata.get('seq_len')
#                 elif self._seq_len_from_metadata != self._metadata.get('seq_len'):
#                     print(f"Warning: seq_len provided to DataModule ({self._seq_len_from_metadata}) "
#                           f"differs from seq_len in metadata ({self._metadata.get('seq_len')}). "
#                           "Ensure model compatibility.")
#             except Exception as e:
#                 print(f"Warning: Could not load or parse metadata file {self.metadata_file} during init: {e}")

#     def prepare_data(self):
#         if not all(os.path.exists(f) for f in [self.train_file, self.val_file, self.test_file, self.metadata_file]):
#             raise FileNotFoundError(
#                 "Data files or metadata file not found. "
#                 "Run data generation script (e.g., in money_train.py) first."
#             )
#         if self._metadata is None: 
#             self._load_metadata()
#             if self._metadata is None:
#                  raise RuntimeError(f"Critical: Metadata file {self.metadata_file} could not be loaded.")

#     def setup(self, stage: Optional[str] = None):
#         if self._metadata is None:
#             self._load_metadata() 
#             if self._metadata is None:
#                 raise RuntimeError("Metadata not loaded, cannot setup datasets.")

#         if stage == "fit" or stage is None:
#             self.train_dataset = FinancialNumericalDataset(self.train_file)
#             self.val_dataset = FinancialNumericalDataset(self.val_file)
#         if stage == "test" or stage is None:
#             self.test_dataset = FinancialNumericalDataset(self.test_file)

#     def train_dataloader(self) -> DataLoader:
#         if not self.train_dataset:
#             raise ValueError("Train dataset not initialized.")
#         return DataLoader(
#             self.train_dataset, batch_size=self.batch_size, shuffle=True,
#             num_workers=self.num_workers, pin_memory=True,
#             persistent_workers=(self.num_workers > 0)
#         )

#     def val_dataloader(self) -> DataLoader:
#         if not self.val_dataset:
#             raise ValueError("Validation dataset not initialized.")
#         return DataLoader(
#             self.val_dataset, batch_size=self.batch_size, shuffle=False,
#             num_workers=self.num_workers, pin_memory=True,
#             persistent_workers=(self.num_workers > 0)
#         )

#     def test_dataloader(self) -> DataLoader:
#         if not self.test_dataset:
#             raise ValueError("Test dataset not initialized.")
#         return DataLoader(
#             self.test_dataset, batch_size=self.batch_size, shuffle=False,
#             num_workers=self.num_workers, pin_memory=True,
#             persistent_workers=(self.num_workers > 0)
#         )

#     def get_input_feature_dim(self) -> int:
#         if self._metadata:
#             return self._metadata.get('num_input_features', 1)
#         raise RuntimeError("Metadata not loaded.")

#     def get_target_feature_dim(self) -> int: # This is the dimension of one element in the target sequence
#         if self._metadata:
#             return self._metadata.get('num_target_features', 1) 
#         raise RuntimeError("Metadata not loaded.")

#     def get_num_tickers(self) -> int:
#         if self._metadata:
#             return self._metadata.get('num_tickers', 0)
#         raise RuntimeError("Metadata not loaded.")
        
#     def get_seq_len(self) -> Optional[int]:
#         return self._seq_len_from_metadata

#     def get_normalization_stats(self, feature_name: str) -> Optional[Dict[str, float]]:
#         if self._metadata:
#             return self._metadata.get('normalization_stats', {}).get(feature_name)
#         raise RuntimeError("Metadata not loaded.")

# def download_and_process_numerical_financial_data(
#     tickers: List[str],
#     start_date: str,
#     end_date: str,
#     features_to_use: List[str] = ['Close'],
#     target_feature: str = 'Close',
#     target_is_pct_change: bool = True,
#     seq_len: int = 64,
#     output_dir: str = "financial_numerical_data",
#     val_split_date: Optional[str] = None,
#     test_split_date: Optional[str] = None,
#     val_split_ratio: float = 0.15, # Used if dates are None
#     test_split_ratio: float = 0.15, # Used if dates are None
# ):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     print(f"Downloading data for tickers: {tickers} from {start_date} to {end_date}...")
#     raw_data = yf.download(tickers, start=start_date, end=end_date, progress=True, auto_adjust=False, back_adjust=False)
#     if raw_data.empty:
#         print("No data downloaded.")
#         return

#     unique_tickers = sorted(list(set(tickers)))
#     ticker_to_id = {ticker: i for i, ticker in enumerate(unique_tickers)}
    
#     all_ticker_data_pct_change = {}

#     print("Calculating percentage changes...")
#     for ticker in unique_tickers:
#         ticker_specific_df = pd.DataFrame()
#         # Construct a temporary df for the current ticker with all required features
#         temp_df_for_ticker = pd.DataFrame(index=raw_data.index)
#         for feature_col_base in list(set(features_to_use + [target_feature])): # Unique features needed
#             if len(unique_tickers) > 1:
#                 # Multi-ticker: columns are (Feature, Ticker)
#                 if (feature_col_base, ticker) in raw_data.columns:
#                     temp_df_for_ticker[feature_col_base] = raw_data[(feature_col_base, ticker)]
#                 elif feature_col_base in raw_data.columns.levels[0]: # If feature exists but not for this ticker (e.g. from .JO)
#                      temp_df_for_ticker[feature_col_base] = np.nan
#                 else: # Should not happen if yf.download worked for this ticker
#                     print(f"Warning: Feature {feature_col_base} for ticker {ticker} not found in multi-download.")
#                     temp_df_for_ticker[feature_col_base] = np.nan
#             else: # Single-ticker: columns are just Feature
#                 if feature_col_base in raw_data.columns:
#                     temp_df_for_ticker[feature_col_base] = raw_data[feature_col_base]
#                 else:
#                     print(f"Warning: Feature {feature_col_base} for ticker {ticker} not found in single-download.")
#                     temp_df_for_ticker[feature_col_base] = np.nan
        
#         temp_df_for_ticker.dropna(how='all', inplace=True) # Drop rows where ALL selected features are NaN
#         if temp_df_for_ticker.empty:
#             print(f"No valid data for ticker {ticker} after initial feature selection.")
#             continue

#         for feature_col_base in list(set(features_to_use + [target_feature])):
#             if feature_col_base in temp_df_for_ticker.columns:
#                 pct_change_col_name = f"{feature_col_base}_pct_change"
#                 # Fill NaNs before pct_change to avoid large initial changes if data starts with NaN
#                 ticker_specific_df[pct_change_col_name] = temp_df_for_ticker[feature_col_base].fillna(method='bfill').fillna(method='ffill').pct_change().fillna(0.0)
#             else: # Should not happen if previous checks passed
#                 print(f"Unexpected: {feature_col_base} not in temp_df_for_ticker for {ticker} during pct_change calc.")
        
#         if not ticker_specific_df.empty:
#             all_ticker_data_pct_change[ticker] = ticker_specific_df.dropna(how='all') # Remove rows that might become all NaN after pct_change

#     # --- Step 2: Split data ---
#     train_data_raw_pct, val_data_raw_pct, test_data_raw_pct = defaultdict(pd.DataFrame), defaultdict(pd.DataFrame), defaultdict(pd.DataFrame)
#     for ticker, df in all_ticker_data_pct_change.items():
#         if df.empty: continue
#         df.index = pd.to_datetime(df.index) # Ensure datetime index

#         if val_split_date and test_split_date:
#             val_dt, test_dt = pd.to_datetime(val_split_date), pd.to_datetime(test_split_date)
#             train_df = df[df.index < val_dt]
#             val_df = df[(df.index >= val_dt) & (df.index < test_dt)]
#             test_df = df[df.index >= test_dt]
#         else:
#             # Ensure indices are unique before attempting train_test_split if shuffle=False isn't enough
#             df = df[~df.index.duplicated(keep='first')]
#             if len(df) < 3: # Not enough data to split
#                 train_df, val_df, test_df = df, pd.DataFrame(), pd.DataFrame()
#             else:
#                 tt_split_point = int(len(df) * (1 - test_split_ratio))
#                 tv_split_point = int(tt_split_point * (1 - (val_split_ratio / (1-test_split_ratio if (1-test_split_ratio) > 0 else 1) )))
                
#                 train_df = df.iloc[:tv_split_point]
#                 val_df = df.iloc[tv_split_point:tt_split_point]
#                 test_df = df.iloc[tt_split_point:]
        
#         if not train_df.empty: train_data_raw_pct[ticker] = train_df
#         if not val_df.empty: val_data_raw_pct[ticker] = val_df
#         if not test_df.empty: test_data_raw_pct[ticker] = test_df
        
#     # --- Step 3: Calculate Normalization Statistics from TRAINING data ONLY ---
#     normalization_stats = {}
#     input_feature_cols_pct = [f"{f}_pct_change" for f in features_to_use]
#     target_col_name_pct = f"{target_feature}_pct_change"
#     all_features_for_norm = list(set(input_feature_cols_pct + ([target_col_name_pct] if target_is_pct_change else [])))

#     for pct_col in all_features_for_norm:
#         all_train_values = []
#         for ticker_train_df in train_data_raw_pct.values():
#             if pct_col in ticker_train_df.columns:
#                 all_train_values.extend(ticker_train_df[pct_col].dropna().tolist())
        
#         if all_train_values:
#             mean_val, std_val = np.mean(all_train_values), np.std(all_train_values)
#             normalization_stats[pct_col] = {'mean': mean_val, 'std': std_val if std_val > 1e-7 else 1.0}
#         else:
#             normalization_stats[pct_col] = {'mean': 0.0, 'std': 1.0} # Fallback

#     # --- Step 4: Normalize data and Create Sequences ---
#     final_sequences = {'train': [], 'val': [], 'test': []}
#     num_input_features = len(input_feature_cols_pct)
#     # Ground truth target will have 1 feature (the predicted pct_change value)
#     num_target_features_data = 1 

#     for split_name, split_data_map in [('train', train_data_raw_pct), ('val', val_data_raw_pct), ('test', test_data_raw_pct)]:
#         for ticker, df_raw_pct_ticker in split_data_map.items():
#             if df_raw_pct_ticker.empty or len(df_raw_pct_ticker) <= seq_len: # Need seq_len + 1 for shifted target
#                 continue

#             df_normalized = pd.DataFrame(index=df_raw_pct_ticker.index)
#             for col in input_feature_cols_pct:
#                 stats = normalization_stats.get(col, {'mean': 0.0, 'std': 1.0})
#                 df_normalized[col] = (df_raw_pct_ticker[col] - stats['mean']) / stats['std']
            
#             if target_is_pct_change:
#                 stats_target = normalization_stats.get(target_col_name_pct, {'mean': 0.0, 'std': 1.0})
#                 target_values_normalized_ticker = (df_raw_pct_ticker[target_col_name_pct] - stats_target['mean']) / stats_target['std']
#             else:
#                 raise NotImplementedError("Raw price targets need specific normalization and handling.")

#             input_data_np = df_normalized[input_feature_cols_pct].values
#             target_data_np = target_values_normalized_ticker.values # This is a 1D array for the ticker

#             current_ticker_id = ticker_to_id[ticker]
#             # Ensure enough data for input sequence AND corresponding target sequence
#             for i in range(len(input_data_np) - seq_len): 
#                 input_s = input_data_np[i : i + seq_len]
#                 # Target sequence is the next value for each input step, matching input_s length
#                 target_s = target_data_np[i + 1 : i + seq_len + 1] 
                
#                 if len(input_s) == seq_len and len(target_s) == seq_len:
#                     final_sequences[split_name].append({
#                         'input': torch.tensor(input_s, dtype=torch.float32),
#                         'target': torch.tensor(target_s.reshape(-1, num_target_features_data), dtype=torch.float32),
#                         'ticker_id': torch.tensor([current_ticker_id], dtype=torch.long)
#                     })
    
#     # --- Step 5: Save data and metadata ---
#     for split_name, data_list in final_sequences.items():
#         if data_list:
#             torch.save(data_list, os.path.join(output_dir, f"{split_name}.pt"))
    
#     metadata = {
#         "tickers": unique_tickers, "ticker_to_id": ticker_to_id, "num_tickers": len(unique_tickers),
#         "start_date_data": start_date, "end_date_data": end_date,
#         "features_used_for_input": features_to_use, "input_feature_columns_pct": input_feature_cols_pct,
#         "target_feature_base_name": target_feature, "target_is_pct_change": target_is_pct_change,
#         "num_input_features": num_input_features, "num_target_features": num_target_features_data,
#         "seq_len": seq_len, "normalization_stats": normalization_stats,
#         "val_split_date_used": val_split_date, "test_split_date_used": test_split_date,
#         "train_sequences_count": len(final_sequences['train']),
#         "val_sequences_count": len(final_sequences['val']),
#         "test_sequences_count": len(final_sequences['test']),
#     }
#     with open(os.path.join(output_dir, "metadata.json"), "w") as f:
#         json.dump(metadata, f, indent=4)
#     print(f"Numerical financial data processing complete. Files saved in {output_dir}")

# # ... (if __name__ == "__main__": block for testing, update paths and feature names)
# if __name__ == "__main__":
#     example_tickers = ["AAPL", "MSFT"] # Reduced for faster testing
#     example_start_date = "1900-01-01" # Shorter period
#     example_end_date = "2023-12-31"
#     example_output_dir = "my_financial_numerical_data_refined"
#     example_seq_len = 32
    
#     val_split_dt = "2022-06-01"
#     test_split_dt = "2023-01-01"

#     print("--- Running download_and_process_numerical_financial_data (refined) ---")
#     download_and_process_numerical_financial_data(
#         tickers=example_tickers,
#         start_date=example_start_date,
#         end_date=example_end_date,
#         features_to_use=['Close', 'Volume'], 
#         target_feature='Close',
#         seq_len=example_seq_len,
#         output_dir=example_output_dir,
#         val_split_date=val_split_dt,
#         test_split_date=test_split_dt
#     )
#     # ... (rest of the __main__ test block, ensure it uses the correct paths and checks new target shape)
#     if os.path.exists(os.path.join(example_output_dir, "metadata.json")):
#         print("\n--- Testing FinancialNumericalDataModule (refined) ---")
#         dm = FinancialNumericalDataModule(
#             train_file=os.path.join(example_output_dir, "train.pt"),
#             val_file=os.path.join(example_output_dir, "val.pt"),
#             test_file=os.path.join(example_output_dir, "test.pt"),
#             metadata_file=os.path.join(example_output_dir, "metadata.json"),
#             batch_size=2, seq_len=example_seq_len
#         )
#         dm.prepare_data()
#         dm.setup('fit')
#         print(f"Input feat dim: {dm.get_input_feature_dim()}, Target feat dim (data): {dm.get_target_feature_dim()}, Num tickers: {dm.get_num_tickers()}")
        
#         if dm.train_dataset and len(dm.train_dataset) > 0:
#             for batch in dm.train_dataloader():
#                 inputs, targets, ticker_ids = batch
#                 print("Sample batch shapes:")
#                 print(f"  Inputs: {inputs.shape}, Targets: {targets.shape}, Ticker IDs: {ticker_ids.shape}")
#                 # Expected: Inputs: (B, S, F_in), Targets: (B, S, 1), Ticker IDs: (B, 1)
#                 break
#         else:
#             print("Train dataset is empty.")


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
        self._load_metadata()

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
    with open(os.path.join(output_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    downloaded = True
    for ticker in tickers:
        if ticker not in metadata["tickers"]:
            downloaded = False
            break
    return downloaded
    

def save_indexes_to_csv(indexes, output_file):
    with open(output_file, "w") as f:
        for i in range(len(indexes)):
            f.write(f"{indexes[i]}, {indexes.day_of_week[i]}, {indexes.day_of_year[i]}\n")
            # could include is quarter start/end and year start/end
        

def download_numerical_financial_data(
    tickers: List[str],
    seq_len: int = 64,
    output_dir: str = "financial_numerical_data",
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