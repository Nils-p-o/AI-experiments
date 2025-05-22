import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
import yfinance as yf
import argparse
import pandas as pd

from transformer_arch.money.money_former_DINT_2 import Money_former_DINT
from training.data_loaders.stocks_time_series_2 import (
    calculate_accumulation_distribution_index,
    calculate_average_true_range,
    calculate_close_line_values,
    calculate_ema,
    calculate_ema_returns,
    calculate_money_flow_index,
    calculate_moving_average,
    calculate_moving_average_returns,
    calculate_volatility_log_ret,
    calculate_volatility_returns,
    calculate_volume_price_trend,
    calculate_rsi,
    calculate_stochastic_oscillator,
    cyclical_encode)


def load_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    args = argparse.Namespace(**metadata)
    return args

def load_model(model_path, args):
    match args.architecture:
        case "Money_former_DINT":
            model = Money_former_DINT(args)
        case _:
            raise ValueError(f"Unsupported architecture: {args.architecture}")

    model.load_state_dict(
        torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        ).state_dict()
    )  # TODO load to gpu if available
    model.eval()
    return model

def download_and_process_input_data(args):
    target_dates = args.indices_to_predict
    tickers = args.tickers
    start_date = "2020-01-01"
    raw_data = yf.download(
        tickers, start=start_date, interval="1d", auto_adjust=False, back_adjust=False
    )

    if raw_data.empty:
        print("No data downloaded.")
        return

    # TODO future removing tickers without data (maybe replace with missing data tokens?)
    indexes = raw_data.index
    # can get day of week, month, year, etc.
    columns = list(raw_data.columns.levels[0])

    # names and tickers
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

        macd_signal = calculate_ema(temp_data[30,:], lookback=9)
        temp_data = torch.cat((temp_data, macd_signal.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("macd_signal_10_20_9")
        
        macd_histogram = temp_data[30,:] - macd_signal
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
        
        rsi = calculate_rsi(temp_raw_data[0,:], lookback=7)
        temp_data = torch.cat((temp_data, rsi.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("rsi_7")
        
        rsi = calculate_rsi(temp_raw_data[0,:], lookback=14)
        temp_data = torch.cat((temp_data, rsi.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("rsi_14")
        
        rsi = calculate_rsi(temp_raw_data[0,:], lookback=21)
        temp_data = torch.cat((temp_data, rsi.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("rsi_21")
        
        k_percent, d_percent = calculate_stochastic_oscillator(temp_raw_data[1,:], temp_raw_data[2,:], temp_raw_data[0, :], k_lookback=14, d_lookback=3)
        temp_data = torch.cat((temp_data, k_percent.unsqueeze(0)), dim=0)
        temp_data = torch.cat((temp_data, d_percent.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("k_percent_14")
            columns.append("d_percent_14_3")
        
        day_of_week = torch.tensor(indexes.dayofweek, dtype=torch.float32)
        sin_dow, cos_dow = cyclical_encode(day_of_week, 7)
        temp_data = torch.cat((temp_data, sin_dow.unsqueeze(0)), dim=0)
        temp_data = torch.cat((temp_data, cos_dow.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("sin_day_of_week")
            columns.append("cos_day_of_week")
        
        day_of_month = torch.tensor(indexes.day, dtype=torch.float32)
        sin_dom, cos_dom = cyclical_encode(day_of_month, 31)
        temp_data = torch.cat((temp_data, sin_dom.unsqueeze(0)), dim=0)
        temp_data = torch.cat((temp_data, cos_dom.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("sin_day_of_month")
            columns.append("cos_day_of_month")
        
        day_of_year = torch.tensor(indexes.dayofyear, dtype=torch.float32)
        sin_doy, cos_doy = cyclical_encode(day_of_year, 366)
        temp_data = torch.cat((temp_data, sin_doy.unsqueeze(0)), dim=0)
        temp_data = torch.cat((temp_data, cos_doy.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("sin_day_of_year")
            columns.append("cos_day_of_year")
        
        month = torch.tensor(indexes.month, dtype=torch.float32)
        sin_month, cos_month = cyclical_encode(month, 12)
        temp_data = torch.cat((temp_data, sin_month.unsqueeze(0)), dim=0)
        temp_data = torch.cat((temp_data, cos_month.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("sin_month")
            columns.append("cos_month")
        
        quarter = torch.tensor(indexes.quarter, dtype=torch.float32)
        sin_quarter, cos_quarter = cyclical_encode(quarter, 4)
        temp_data = torch.cat((temp_data, sin_quarter.unsqueeze(0)), dim=0)
        temp_data = torch.cat((temp_data, cos_quarter.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("sin_quarter")
            columns.append("cos_quarter")
        
        is_leap_year = torch.tensor(indexes.is_leap_year, dtype=torch.float32)
        temp_data = torch.cat((temp_data, is_leap_year.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("is_leap_year")
        
        is_month_start = torch.tensor(indexes.is_month_start, dtype=torch.float32)
        temp_data = torch.cat((temp_data, is_month_start.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("is_month_start")
        
        is_month_end = torch.tensor(indexes.is_month_end, dtype=torch.float32)
        temp_data = torch.cat((temp_data, is_month_end.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("is_month_end")
        
        is_quarter_start = torch.tensor(indexes.is_quarter_start, dtype=torch.float32)
        temp_data = torch.cat((temp_data, is_quarter_start.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("is_quarter_start")
        
        is_quarter_end = torch.tensor(indexes.is_quarter_end, dtype=torch.float32)
        temp_data = torch.cat((temp_data, is_quarter_end.unsqueeze(0)), dim=0)
        if i == 0:
            columns.append("is_quarter_end")

        # TODO write somewhere else, maybe in another function

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
        "^VIX", start=start_date, interval="1d", progress=False, auto_adjust=False
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
    true_prices = raw_data[0, -data_length:, :].unsqueeze(0)  # (1, seq_len, num_sequences)

    if (torch.isnan(data)).any() or (torch.isinf(data)).any():
        print("Data contains NaN or Inf values.")

    return data, true_prices

def get_predictions(model, inputs, args):
    # inputs = inputs.unsqueeze(0)
    batch_size, num_features, seq_len, num_sequences = inputs.shape
        
    known_inputs = inputs[:,0, :].unsqueeze(2) # (batch_size, seq_len, 1, num_sequences)

    target_input_means = known_inputs.mean(dim=1, keepdim=True).tile(1, known_inputs.shape[1],1,1)
    target_input_stds = known_inputs.std(dim=1, keepdim=True).tile(1, known_inputs.shape[1],1,1)

    norm_inputs = (known_inputs - target_input_means) / target_input_stds

    additional_inputs = torch.cat((inputs[:, 1:71, :, :], inputs[:, 86:, :, :]), dim=1) # (batch_size, features, seq_len, num_sequences)
    additional_inputs = additional_inputs.transpose(1, 2) # (batch_size, seq_len, features, num_sequences)
    means_of_additonal_inputs = additional_inputs.mean(dim=1, keepdim=True).tile(1, additional_inputs.shape[1],1,1)
    stds_of_additonal_inputs = additional_inputs.std(dim=1, keepdim=True).tile(1, additional_inputs.shape[1],1,1)
    full_inputs = torch.cat([norm_inputs, (additional_inputs-means_of_additonal_inputs)/stds_of_additonal_inputs], dim=2)
    # full_inputs = torch.cat([full_inputs, target_input_means, target_input_stds], dim=2)
    full_inputs = torch.cat([full_inputs, inputs[:,71:86,:,:].transpose(1,2)], dim=2) # adding time based data (need to be careful with this)
    full_inputs = full_inputs.transpose(2, 3)

    seperator = torch.zeros((batch_size, 1),dtype=torch.int, device=inputs.device) # (batch_size, 1)
    tickers = torch.arange(len(args.tickers), device=inputs.device)
    tickers = tickers.unsqueeze(0).repeat(batch_size, 1) # (batch_size, num_sequences)

    outputs = model(full_inputs, seperator, tickers).view(batch_size, -1, len(args.indices_to_predict), len(args.tickers)) #.transpose(-1, -2) # (batch_size, seq_len, targets*num_sequences) 
    outputs = outputs[:,1:,:,:]

    # for debugging reasons
    if torch.isnan(outputs).any():
        print("NAN IN OUTPUTS")

    if isinstance(outputs, torch.Tensor):
        raw_logits = outputs
    else:
        raw_logits = outputs.logits
    
    preds = raw_logits * target_input_stds.tile(1, 1, len(args.indices_to_predict),1) + target_input_means.tile(1, 1, len(args.indices_to_predict), 1)

    return preds.detach()

def adapt_longer_preds(long_preds, true_returns, days_predicted_ahead, days_to_adapt_to, checked_days):
    days = days_predicted_ahead
    long_preds = long_preds[-(checked_days+days-1):-(days-1), :] # (seq_len, num_sequences)
    long_preds = long_preds + 1.0
    for i in range(days-days_to_adapt_to):
        long_preds = long_preds / (true_returns[:,-(checked_days+days-1-i):-(days-1-i), :] + 1.0)
    long_preds = long_preds - 1.0
    return long_preds


def run_trading_strategy(predictions_1d_ahead, actual_1d_returns, long_trade_threshold=0.005, short_trade_threshold=0.005, initial_capital=100000.0, transaction_cost_pct=0.0005):
    """
    Simulates a simple trading strategy.

    Args:
        predictions_1d_ahead (torch.Tensor): Model's predicted 1-day returns.
                                            Shape: (num_days_in_backtest, num_sequences_or_tickers)
        actual_1d_returns (torch.Tensor): Actual 1-day returns.
                                          Shape: (num_days_in_backtest, num_sequences_or_tickers)
        trade_threshold (float): Minimum predicted return to trigger a trade.
        initial_capital (float): Starting capital for the simulation.
        transaction_cost_pct (float): Percentage transaction cost per trade (e.g., 0.0005 for 0.05%).

    Returns:
        pd.DataFrame: Daily portfolio values and returns.
        dict: Summary statistics of the strategy.
    """
    num_days, num_tickers = predictions_1d_ahead.shape
    portfolio_values = []
    daily_portfolio_returns = []
    current_capital = initial_capital
    
    # For simplicity, assume equal allocation if multiple signals on the same day
    # and we trade all signals that meet the threshold.
    # More complex strategies would involve position sizing, risk management, etc.

    print(f"\n--- Running Trading Strategy ---")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Long Trade Threshold: {long_trade_threshold*100:.2f}%")
    print(f"Short Trade Threshold: {short_trade_threshold*100:.2f}%")
    print(f"Transaction Cost (per trade): {transaction_cost_pct*100:.3f}%")

    for day_idx in range(num_days):
        capital_at_start_of_day = current_capital
        daily_pnl = 0.0
        num_trades_today = 0

        # Decide trades for today based on yesterday's predictions (or predictions for today)
        # Assuming predictions_1d_ahead[day_idx] are predictions for the returns that will occur on day_idx
        
        signals = predictions_1d_ahead[day_idx]  # Predictions for today's returns for all tickers
        realized_returns = actual_1d_returns[day_idx] # Actual returns for today for all tickers

        active_signals = []
        for ticker_idx in range(num_tickers):
            if signals[ticker_idx] > long_trade_threshold:
                active_signals.append({'action': 'long', 'ticker_idx': ticker_idx})
            elif signals[ticker_idx] < -short_trade_threshold:
                active_signals.append({'action': 'short', 'ticker_idx': ticker_idx})
        
        if not active_signals:
            # No trades, portfolio return is 0 (assuming capital is in cash)
            daily_portfolio_returns.append(0.0)
            portfolio_values.append(current_capital)
            if day_idx < 5 or day_idx == num_days -1 : # Print first few and last
                 print(f"Day {day_idx+1}: No trades. Capital: ${current_capital:,.2f}")
            continue

        capital_per_trade = capital_at_start_of_day / len(active_signals) # Equal allocation

        for trade in active_signals:
            ticker_idx = trade['ticker_idx']
            invested_amount = capital_per_trade
            
            # Apply transaction costs for entry
            trade_pnl = 0
            cost_entry = invested_amount * transaction_cost_pct
            
            if trade['action'] == 'long':
                trade_pnl = invested_amount * realized_returns[ticker_idx]
            elif trade['action'] == 'short':
                trade_pnl = invested_amount * (-realized_returns[ticker_idx]) # Profit if actual return is negative
            
            # Apply transaction costs for exit (assuming exit at end of period)
            cost_exit = (invested_amount + trade_pnl) * transaction_cost_pct # Cost on the closing value
            
            daily_pnl += (trade_pnl - cost_entry - cost_exit)
            num_trades_today +=1

        current_capital += daily_pnl
        if capital_at_start_of_day > 0 :
            day_return_pct = daily_pnl / capital_at_start_of_day
        else:
            day_return_pct = 0.0
            
        daily_portfolio_returns.append(day_return_pct)
        portfolio_values.append(current_capital)
        
        if day_idx < 5 or day_idx == num_days -1 or num_trades_today > 0 and day_idx % (num_days//10 if num_days > 20 else 1) ==0 :
            print(f"Day {day_idx+1}: Trades: {num_trades_today}. Day P&L: ${daily_pnl:,.2f}. Day Return: {day_return_pct*100:.2f}%. Capital: ${current_capital:,.2f}")


    portfolio_df = pd.DataFrame({'portfolio_value': portfolio_values, 'daily_return': daily_portfolio_returns})
    
    # Calculate summary statistics
    total_return = (current_capital / initial_capital) - 1
    # Assuming 252 trading days in a year
    annualized_return = ( (1 + total_return) ** (252 / num_days) ) - 1 if num_days > 0 else 0.0
    annualized_volatility = np.std(daily_portfolio_returns) * np.sqrt(252) if num_days > 0 else 0.0
    sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility != 0 else 0.0 # Assuming risk-free rate = 0

    # Max Drawdown
    cumulative_returns = (1 + portfolio_df['daily_return']).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    num_winning_days = (portfolio_df['daily_return'] > 0).sum()
    num_losing_days = (portfolio_df['daily_return'] < 0).sum()
    win_loss_ratio = num_winning_days / num_losing_days if num_losing_days > 0 else float('inf')


    stats = {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Number of Trading Days": num_days,
        "Number of Winning Days": num_winning_days,
        "Number of Losing Days": num_losing_days,
        "Win/Loss Day Ratio": win_loss_ratio,
        "Final Capital": current_capital
    }
    print("\n--- Strategy Summary ---")
    for key, value in stats.items():
        if isinstance(value, float) and key not in ["Sharpe Ratio", "Win/Loss Day Ratio"]:
            print(f"{key}: {value*100:.2f}%" if "Return" in key or "Drawdown" in key or "Volatility" in key else f"{key}: {value:.2f}")
        elif isinstance(value, float) and key in ["Sharpe Ratio", "Win/Loss Day Ratio"]:
             print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
            
    return portfolio_df, stats

def run_trading_strategy_Nday_signal(
    predictions_Nday_ahead,  # Model's predicted N-day CUMULATIVE returns
    actual_1d_returns,      # Actual 1-day returns for P&L calculation
    trade_threshold=0.01,   # Minimum N-day predicted return to trigger a trade (e.g., 1% for 5-day)
    initial_capital=100000.0,
    transaction_cost_pct=0.0005,
    signal_horizon_name="N-day" # For logging
):
    """
    Simulates a trading strategy using N-day predictions, but executes trades
    evaluated on the next 1-day actual return.

    Args:
        predictions_Nday_ahead (torch.Tensor): Model's predicted N-day cumulative returns.
                                            Shape: (num_days_in_backtest, num_sequences_or_tickers)
        actual_1d_returns (torch.Tensor): Actual 1-day returns for daily P&L.
                                          Shape: (num_days_in_backtest, num_sequences_or_tickers)
        trade_threshold (float): Minimum predicted N-day return to trigger a trade.
        initial_capital (float): Starting capital.
        transaction_cost_pct (float): Transaction cost.
        signal_horizon_name (str): Name for the signal horizon (e.g., "5-day", "10-day")

    Returns:
        pd.DataFrame: Daily portfolio values and returns.
        dict: Summary statistics of the strategy.
    """
    num_days, num_tickers = predictions_Nday_ahead.shape
    if num_days != actual_1d_returns.shape[0]:
        raise ValueError("Predictions and actual 1-day returns must have the same number of days.")
        
    portfolio_values = []
    daily_portfolio_returns = []
    current_capital = initial_capital
    
    print(f"\n--- Running Trading Strategy (Signal: {signal_horizon_name}) ---")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Trade Threshold (for {signal_horizon_name} signal): {trade_threshold*100:.2f}%")
    print(f"Transaction Cost (per trade): {transaction_cost_pct*100:.3f}%")

    for day_idx in range(num_days):
        capital_at_start_of_day = current_capital
        daily_pnl = 0.0
        num_trades_today = 0
        
        # Predictions_Nday_ahead[day_idx] are N-day cumulative return predictions made based on data BEFORE today.
        # We use these to decide trades for today, and P&L is based on actual_1d_returns[day_idx].
        signals_Nday = predictions_Nday_ahead[day_idx]
        realized_1d_returns_today = actual_1d_returns[day_idx]

        active_signals = []
        for ticker_idx in range(num_tickers):
            if signals_Nday[ticker_idx] > trade_threshold:
                active_signals.append({'action': 'long', 'ticker_idx': ticker_idx})
            elif signals_Nday[ticker_idx] < -trade_threshold: # Assuming symmetrical threshold
                active_signals.append({'action': 'short', 'ticker_idx': ticker_idx})
        
        if not active_signals:
            daily_portfolio_returns.append(0.0)
            portfolio_values.append(current_capital)
            if day_idx < 5 or day_idx == num_days -1:
                 print(f"Day {day_idx+1}: No trades. Capital: ${current_capital:,.2f}")
            continue

        capital_per_trade = capital_at_start_of_day / len(active_signals)

        for trade in active_signals:
            ticker_idx = trade['ticker_idx']
            invested_amount = capital_per_trade
            
            cost_entry = invested_amount * transaction_cost_pct
            trade_pnl_pre_exit_cost = 0
            
            if trade['action'] == 'long':
                trade_pnl_pre_exit_cost = invested_amount * realized_1d_returns_today[ticker_idx]
            elif trade['action'] == 'short':
                trade_pnl_pre_exit_cost = invested_amount * (-realized_1d_returns_today[ticker_idx])
            
            cost_exit = (invested_amount + trade_pnl_pre_exit_cost) * transaction_cost_pct
            
            daily_pnl += (trade_pnl_pre_exit_cost - cost_entry - cost_exit)
            num_trades_today +=1

        current_capital += daily_pnl
        day_return_pct = daily_pnl / capital_at_start_of_day if capital_at_start_of_day > 1e-6 else 0.0 # Avoid div by zero if capital wiped
            
        daily_portfolio_returns.append(day_return_pct)
        portfolio_values.append(current_capital)
        
        if day_idx < 5 or day_idx == num_days -1 or (num_trades_today > 0 and day_idx % (num_days//10 if num_days > 20 else 1) ==0):
            print(f"Day {day_idx+1}: Trades: {num_trades_today}. Day P&L: ${daily_pnl:,.2f}. Day Return: {day_return_pct*100:.2f}%. Capital: ${current_capital:,.2f}")

    portfolio_df = pd.DataFrame({'portfolio_value': portfolio_values, 'daily_return': daily_portfolio_returns})
    
    total_return = (current_capital / initial_capital) - 1
    annualized_return = ((1 + total_return) ** (252 / num_days)) - 1 if num_days > 0 else 0.0
    annualized_volatility = np.std(daily_portfolio_returns) * np.sqrt(252) if num_days > 0 else 0.0
    sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility > 1e-9 else 0.0 # Avoid div by zero

    cumulative_returns = (1 + pd.Series(portfolio_df['daily_return'])).cumprod() # Use pandas Series for cumprod
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0
    
    num_winning_days = (portfolio_df['daily_return'] > 0).sum()
    num_losing_days = (portfolio_df['daily_return'] < 0).sum()
    win_loss_ratio = num_winning_days / num_losing_days if num_losing_days > 0 else float('inf')

    stats = {
        "Signal Horizon Used": signal_horizon_name,
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Number of Trading Days": num_days,
        "Number of Winning Days": num_winning_days,
        "Number of Losing Days": num_losing_days,
        "Win/Loss Day Ratio": win_loss_ratio,
        "Final Capital": current_capital
    }
    print(f"\n--- Strategy Summary (Signal: {signal_horizon_name}) ---")
    for key, value in stats.items():
        if isinstance(value, float) and key not in ["Sharpe Ratio", "Win/Loss Day Ratio", "Signal Horizon Used"]:
            print(f"{key}: {value*100:.2f}%" if "Return" in key or "Drawdown" in key or "Volatility" in key else f"{key}: {value:.2f}")
        elif isinstance(value, float) and key in ["Sharpe Ratio", "Win/Loss Day Ratio"]:
             print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
            
    return portfolio_df, stats


if __name__ == "__main__":
    metadata_path = "good_models/model_metadata.json"
    model_path = "good_models\Money_former_DINT_test_Money_former_DINT_16_128_512_4_4_24.pth"
    nr_of_days_to_check = 1000
    args = load_metadata(metadata_path)
    num_tickers = len(args.indices_to_predict)
    graph = False

    if graph == True:
        data, true_prices = download_and_process_input_data(args) # (features, seq_len, targets, num_sequences)
        data = data[:, -(args.seq_len+nr_of_days_to_check+max(args.indices_to_predict)):-1, 0, :] 
        true_prices = true_prices[:, -(args.seq_len+nr_of_days_to_check+max(args.indices_to_predict)):, :]
        true_returns = (true_prices[:,1:,:] - true_prices[:,:-1,:])/true_prices[:,:-1,:]
        true_returns_5 = (true_prices[:,5:,:] - true_prices[:,:-5,:])/true_prices[:,:-5,:]
        true_returns_10 = (true_prices[:,10:,:] - true_prices[:,:-10,:])/true_prices[:,:-10,:] 

        model = load_model(model_path, args)
        predictions = []
        for i in range(nr_of_days_to_check+max(args.indices_to_predict)-1):
            predictions.append(get_predictions(model, data[:, i:i+args.seq_len, :], args)[:,-1,:,:]) # (seq_len, targets,num_sequences)
        predictions = torch.cat(predictions, dim=0) # (seq_len, targets,num_sequences)

        full_seq_preds_1d = get_predictions(model, data[:,-args.seq_len:, :], args)[:,:,0,:].unsqueeze(0) # (seq_len, num_sequences)
        full_seq_preds_5d = get_predictions(model, data[:,-args.seq_len-4:-4, :], args)[:,:,1,:].unsqueeze(0) # (seq_len, num_sequences)
        full_seq_preds_10d = get_predictions(model, data[:,-args.seq_len-9:-9, :], args)[:,:,2,:].unsqueeze(0) # (seq_len, num_sequences)

        d1_predictions = predictions[-nr_of_days_to_check:, 0, :].unsqueeze(0) # (seq_len, num_sequences)
        days = args.indices_to_predict[1]
        adapted_predictions = adapt_longer_preds(predictions[:,1,:], true_returns, days, 1, nr_of_days_to_check)
        d1_predictions = torch.cat([d1_predictions, adapted_predictions], dim=0) # (adapted_days, seq_len, num_sequences)

        days = args.indices_to_predict[2]
        adapted_predictions = adapt_longer_preds(predictions[:,2,:], true_returns, days, 1, nr_of_days_to_check)
        d1_predictions = torch.cat([d1_predictions, adapted_predictions], dim=0) # (adapted_days, seq_len, num_sequences)

        d5_predictions = predictions[-nr_of_days_to_check-4:-4, 1, :].unsqueeze(0) # (seq_len, num_sequences)
        days = args.indices_to_predict[2]
        adapted_predictions = adapt_longer_preds(predictions[:,2,:], true_returns, days, 5, nr_of_days_to_check)
        d5_predictions = torch.cat([d5_predictions, adapted_predictions], dim=0) # (adapted_days, seq_len, num_sequences)

        d1_predictions = d1_predictions.transpose(0,1)
        d5_predictions = d5_predictions.transpose(0,1)
        # TODO compare with actual prices (because less head hurt)
        plt.plot(true_returns[0, -nr_of_days_to_check:, 0])
        plt.plot(d1_predictions[:, :, 0])
        plt.plot(full_seq_preds_1d[0,:,0])
        plt.legend([f"true 1day {args.tickers[0]}", f"predictions 1day {args.tickers[0]}", f"predictions 5day {args.tickers[0]}", f"predictions 10day {args.tickers[0]}", f"full_seq_preds_1d {args.tickers[0]}"])
        plt.show()

        plt.plot(true_returns_5[0, -nr_of_days_to_check:, 0])
        plt.plot(d5_predictions[:, :, 0])
        plt.plot(full_seq_preds_5d[0,:,0])
        plt.legend([f"true 5 days {args.tickers[0]}", f"predictions 5day {args.tickers[0]}", f"predictions 10day {args.tickers[0]}", f"full_seq_preds_5d {args.tickers[0]}"])
        plt.show()

        plt.plot(true_returns_10[0, -nr_of_days_to_check:, 0])
        plt.plot(predictions[:-9, 2, 0])
        plt.plot(full_seq_preds_10d[0,:,0])
        plt.legend([f"true 10 days {args.tickers[0]}", f"predictions 10day {args.tickers[0]}", f"full_seq_preds_10d {args.tickers[0]}"])
        plt.show()

        plt.plot(true_returns[0, -nr_of_days_to_check:, 1])
        plt.plot(d1_predictions[:, :, 1])
        plt.plot(full_seq_preds_1d[0,:,1])
        plt.legend([f"true 1day {args.tickers[1]}", f"predictions 1day {args.tickers[1]}", f"predictions 5day {args.tickers[1]}", f"predictions 10day {args.tickers[1]}", f"full_seq_preds_1d {args.tickers[1]}"])
        plt.show()

        plt.plot(true_returns_5[0, -nr_of_days_to_check:, 1])
        plt.plot(d5_predictions[:, :, 1])
        plt.plot(full_seq_preds_5d[0,:,1])
        plt.legend([f"true 5 days {args.tickers[1]}", f"predictions 5day {args.tickers[1]}", f"predictions 10day {args.tickers[1]}", f"full_seq_preds_5d {args.tickers[1]}"])
        plt.show()

        plt.plot(true_returns_10[0, -nr_of_days_to_check:, 1])
        plt.plot(predictions[:-9, 2, 1])
        plt.plot(full_seq_preds_10d[0,:,1])
        plt.legend([f"true 10 days {args.tickers[1]}", f"predictions 10day {args.tickers[1]}", f"full_seq_preds_10d {args.tickers[1]}"])
        plt.show()

        MAE_fn = nn.L1Loss()
        print(f"naive 1d:{MAE_fn(torch.zeros_like(true_returns[0, -nr_of_days_to_check:, :]), true_returns[0, -nr_of_days_to_check:, :])}")
        print(f"naive 5d:{MAE_fn(torch.zeros_like(true_returns_5[0, -nr_of_days_to_check:, :]), true_returns_5[0, -nr_of_days_to_check:, :])}")
        print(f"naive 10d:{MAE_fn(torch.zeros_like(true_returns_10[0, -nr_of_days_to_check:, :]), true_returns_10[0, -nr_of_days_to_check:, :])}")

        print(f"1d 1d:{MAE_fn(d1_predictions[:,0], true_returns[0, -nr_of_days_to_check:, :])}")
        print(f"1d 5d:{MAE_fn(d1_predictions[:,1], true_returns[0, -nr_of_days_to_check:, :])}")
        print(f"1d 10d:{MAE_fn(d1_predictions[:,2], true_returns[0, -nr_of_days_to_check:, :])}")
        print(f"5d 5d:{MAE_fn(d5_predictions[:,0], true_returns_5[0, -nr_of_days_to_check:, :])}")
        print(f"5d 10d:{MAE_fn(d5_predictions[:,1], true_returns_5[0, -nr_of_days_to_check:, :])}")
        print(f"10d 10d:{MAE_fn(predictions[:-9, 2, :], true_returns_10[0, -nr_of_days_to_check:, :])}")

    all_processed_data, all_true_prices = download_and_process_input_data(args)
    total_available_timesteps = all_processed_data.shape[1]

    required_timesteps_for_rolling_preds = args.seq_len + nr_of_days_to_check -1
    if total_available_timesteps < required_timesteps_for_rolling_preds:
        raise ValueError(f"Not enough data ({total_available_timesteps} available) for seq_len ({args.seq_len}) and nr_of_days_to_check ({nr_of_days_to_check}). Need {required_timesteps_for_rolling_preds}.")
    
    data_for_backtest_input_generation = all_processed_data[:, -required_timesteps_for_rolling_preds:, 0, :]
    true_prices_for_backtest_period = all_true_prices[0, -(nr_of_days_to_check + 1):, :]
    actual_1d_returns_for_backtest = (true_prices_for_backtest_period[1:, :] - true_prices_for_backtest_period[:-1, :]) / (true_prices_for_backtest_period[:-1, :] + 1e-8)

    model = load_model(model_path, args)

    # model_predictions_1d_list = []
    # for i in range(nr_of_days_to_check):
    #     current_input_seq = data_for_backtest_input_generation[:, i : i + args.seq_len, :].unsqueeze(0)
    #     preds_all_horizons_for_last_step = get_predictions(model, current_input_seq, args)[0, -1, :, :]
    #     pred_1d = preds_all_horizons_for_last_step[0, :] # Shape: (num_tickers)
    #     model_predictions_1d_list.append(pred_1d)

    # model_predictions_1d_tensor = torch.stack(model_predictions_1d_list, dim=0)

    all_horizons_predictions_list = []
    for i in range(nr_of_days_to_check):
        current_input_seq = data_for_backtest_input_generation[:, i : i + args.seq_len, :].unsqueeze(0)
        # Get predictions for all horizons from the model's last output step
        preds_all_horizons_for_last_step = get_predictions(model, current_input_seq, args)[0, -1, :, :]
        # Shape: (num_prediction_horizons, num_tickers)
        all_horizons_predictions_list.append(preds_all_horizons_for_last_step)

    all_horizons_predictions_tensor = torch.stack(all_horizons_predictions_list, dim=0)


    initial_capital_main = 1000
    if len(args.indices_to_predict) > 1 and args.indices_to_predict[1] == 5:
        predictions_5day = all_horizons_predictions_tensor[:, 1, :] # (nr_of_days_to_check, num_tickers)
        portfolio_df_5d, stats_5d = run_trading_strategy_Nday_signal(
            predictions_5day.cpu(),
            actual_1d_returns_for_backtest.cpu(),
            trade_threshold=0.008, # e.g., predict 1% move over 5 days to trade
            initial_capital=initial_capital_main,
            transaction_cost_pct=0.0005,
            signal_horizon_name="5-day"
        )
        # if portfolio_df_5d is not None and not portfolio_df_5d.empty:
        #     plt.figure(figsize=(12,6))
        #     cumulative_portfolio_returns_plot_5d = (1 + pd.Series(portfolio_df_5d['daily_return'])).cumprod()
        #     plt.plot(cumulative_portfolio_returns_plot_5d, label="Strategy (5-day signal)")
        #     plt.title(f"Strategy (5-day signal) Portfolio Value (Tickers: {', '.join(args.tickers)})")
        #     plt.xlabel("Trading Day")
        #     plt.ylabel("Cumulative Return (Normalized)")
        #     plt.legend()
        #     plt.grid(True)
        #     plt.savefig(f"strategy_portfolio_5day_signal_{'_'.join(args.tickers)}.png")
        #     plt.show()

    # --- Run Strategy for 10-day Predictions ---
    # Assuming 10-day predictions are at index 2
    if len(args.indices_to_predict) > 2 and args.indices_to_predict[2] == 10:
        predictions_10day = all_horizons_predictions_tensor[:, 2, :] # (nr_of_days_to_check, num_tickers)
        portfolio_df_10d, stats_10d = run_trading_strategy_Nday_signal(
            predictions_10day.cpu(),
            actual_1d_returns_for_backtest.cpu(),
            trade_threshold=0.01, # e.g., predict 1.5% move over 10 days to trade
            initial_capital=initial_capital_main,
            transaction_cost_pct=0.0005,
            signal_horizon_name="10-day"
        )
        # if portfolio_df_10d is not None and not portfolio_df_10d.empty:
        #     plt.figure(figsize=(12,6))
        #     cumulative_portfolio_returns_plot_10d = (1 + pd.Series(portfolio_df_10d['daily_return'])).cumprod()
        #     plt.plot(cumulative_portfolio_returns_plot_10d, label="Strategy (10-day signal)")
        #     plt.title(f"Strategy (10-day signal) Portfolio Value (Tickers: {', '.join(args.tickers)})")
        #     plt.xlabel("Trading Day")
        #     plt.ylabel("Cumulative Return (Normalized)")
        #     plt.legend()
        #     plt.grid(True)
        #     plt.savefig(f"strategy_portfolio_10day_signal_{'_'.join(args.tickers)}.png")
        #     plt.show()

    # --- You can also run the original 1-day signal strategy for comparison ---
    if len(args.indices_to_predict) > 0 and args.indices_to_predict[0] == 1:
        predictions_1day = all_horizons_predictions_tensor[:, 0, :] # (nr_of_days_to_check, num_tickers)
        # Assuming your original run_trading_strategy is still available or this one is adapted
        portfolio_df_1d, stats_1d = run_trading_strategy_Nday_signal( # Can reuse Nday_signal for 1-day
            predictions_1day.cpu(),
            actual_1d_returns_for_backtest.cpu(),
            trade_threshold=0.002, # Your original 1-day threshold
            initial_capital=initial_capital_main,
            transaction_cost_pct=0.0005,
            signal_horizon_name="1-day"
        )
        # if portfolio_df_1d is not None and not portfolio_df_1d.empty:
        #     plt.figure(figsize=(12,6))
        #     cumulative_portfolio_returns_plot_1d = (1 + pd.Series(portfolio_df_1d['daily_return'])).cumprod()
        #     plt.plot(cumulative_portfolio_returns_plot_1d, label="Strategy (1-day signal)")
        #     plt.title(f"Strategy (1-day signal) Portfolio Value (Tickers: {', '.join(args.tickers)})")
        #     plt.xlabel("Trading Day")
        #     plt.ylabel("Cumulative Return (Normalized)")
        #     plt.legend()
        #     plt.grid(True)
        #     plt.savefig(f"strategy_portfolio_1day_signal_{'_'.join(args.tickers)}.png")
        #     plt.show()
    
    if len(args.indices_to_predict) > 1:
        prediction_time_weights = torch.tensor([[1.0],[4.0],[12.0]], dtype=torch.float32)
        averaged_predictions = (all_horizons_predictions_tensor / prediction_time_weights).mean(dim=1)

        portfolio_df_combined, stats_combined = run_trading_strategy_Nday_signal(
            averaged_predictions.cpu(),
            actual_1d_returns_for_backtest.cpu(),
            trade_threshold=0.0020, # e.g., predict 1% move over 5 days to trade
            initial_capital=initial_capital_main,
            transaction_cost_pct=0.0005,
            signal_horizon_name="Combined"
        )
        if portfolio_df_combined is not None and not portfolio_df_combined.empty:
            plt.figure(figsize=(12,6))
            cumulative_portfolio_returns_plot_combined = (1 + pd.Series(portfolio_df_combined['daily_return'])).cumprod()
            plt.plot(cumulative_portfolio_returns_plot_combined, label="Strategy (Combined signal)")
            plt.title(f"Strategy (Combined signal) Portfolio Value (Tickers: {', '.join(args.tickers)})")
            plt.xlabel("Trading Day")
            plt.ylabel("Cumulative Return (Normalized)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"strategy_portfolio_combined_signal_{'_'.join(args.tickers)}.png")
            plt.show()

    # --- Comparison Plot (if you ran multiple strategies) ---
    plt.figure(figsize=(14, 7))
    if 'portfolio_df_1d' in locals() and portfolio_df_1d is not None and not portfolio_df_1d.empty:
        plt.plot((1 + pd.Series(portfolio_df_1d['daily_return'])).cumprod(), label="Strategy (1-day Signal)")
    if 'portfolio_df_5d' in locals() and portfolio_df_5d is not None and not portfolio_df_5d.empty:
        plt.plot((1 + pd.Series(portfolio_df_5d['daily_return'])).cumprod(), label="Strategy (5-day Signal)")
    if 'portfolio_df_10d' in locals() and portfolio_df_10d is not None and not portfolio_df_10d.empty:
        plt.plot((1 + pd.Series(portfolio_df_10d['daily_return'])).cumprod(), label="Strategy (10-day Signal)")
    if 'portfolio_df_combined' in locals() and portfolio_df_combined is not None and not portfolio_df_combined.empty:
        plt.plot((1 + pd.Series(portfolio_df_combined['daily_return'])).cumprod(), label="Strategy (Combined Signal)")
    plt.title(f"Strategy Comparison (Tickers: {', '.join(args.tickers)})")
    
    # Add Buy & Hold for comparison
    if len(args.tickers) > 0:
        # buy_and_hold_returns_first_ticker = actual_1d_returns_for_backtest[:, 0].cpu()
        # buy_and_hold_equity_curve = (1 + buy_and_hold_returns_first_ticker).cumprod(dim=0)
        # plt.plot(buy_and_hold_equity_curve.numpy(), label=f"Buy & Hold {args.tickers[0]}", linestyle='--')

        if len(args.tickers) > 1:
            buy_and_hold_returns_avg_tickers = actual_1d_returns_for_backtest.cpu().mean(dim=1)
            buy_and_hold_equity_curve_avg = (1 + buy_and_hold_returns_avg_tickers).cumprod(dim=0)
            plt.plot(buy_and_hold_equity_curve_avg.numpy(), label=f"Buy & Hold Avg of {len(args.tickers)} Tickers", linestyle=':')
            
    plt.title("Comparison of Strategies vs. Buy & Hold")
    plt.xlabel("Trading Day in Backtest Period")
    plt.ylabel("Cumulative Return (Normalized to 1)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"strategy_comparison_{'_'.join(args.tickers)}.png")
    plt.show()

    # MAE_fn = nn.L1Loss()
    # print(f"Naive 1d MAE (predict zero return): {MAE_fn(torch.zeros_like(true_returns_for_mae), true_returns_for_mae):.6f}")
    # print(f"Model 1d MAE: {MAE_fn(preds_for_mae, true_returns_for_mae):.6f}")

    # --- Plotting for the first ticker and 1-day predictions ---
    # if len(args.tickers) > 0:
    #     ticker_to_plot_idx = 0
    #     ticker_name = args.tickers[ticker_to_plot_idx]
    #     plt.figure(figsize=(12, 6))
    #     # plt.plot(true_returns_for_mae[:, ticker_to_plot_idx].cpu().numpy(), label=f"True 1-day Returns ({ticker_name})")
    #     # plt.plot(preds_for_mae[:, ticker_to_plot_idx].cpu().numpy(), label=f"Predicted 1-day Returns ({ticker_name})", alpha=0.7)
    #     plt.title(f"1-Day Return Prediction vs Actual for {ticker_name}")
    #     plt.xlabel("Trading Day in Backtest Period")
    #     plt.ylabel("Return")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(f"predictions_vs_actual_1d_{ticker_name}.png")
    #     plt.show()