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
    cyclical_encode,
    align_financial_dataframes,
)

from bayes_opt import BayesianOptimization
# from bayes_opt.util import UtilityFunction


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
    end_date = "2025-05-25"
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

    # TODO future removing tickers without data (maybe replace with missing data tokens?)
    indexes = raw_data.index
    # full_indexes = raw_data.index
    # can get day of week, month, year, etc.
    columns = list(raw_data.columns.levels[0])

    # chech which dims are open and which is close (change calcs accordingly, because rn might be using wrong dims in calculations)
    raw_data = torch.tensor(raw_data.values, dtype=torch.float32).reshape(
        -1, len(columns), len(tickers)
    )  # (Time, Features, tickers)
    raw_data = raw_data.transpose(0, 1)  # (Features, Time series, tickers)

    # TODO write somewhere else, maybe in another function
    # TODO reorganize, because there is no need to calculate anything else for different time points other than close (target value)
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
                        temp_raw_data[: temp_raw_data.shape[0], target_dates[i] :]
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

    data = torch.cat(data, dim=-1)  # (features, time series, target dates, tickers)
    data_length = data.shape[1]

    # adding more complex relationships

    # adding VIX data
    raw_vix_data = yf.download(
        "^VIX", start=start_date, end=end_date, progress=False, auto_adjust=False
    )
    aligned_vix_data = pd.DataFrame(columns=raw_vix_data.columns)
    for column in raw_vix_data.columns.levels[0]:
        aligned_vix_data[column, "^VIX"] = align_financial_dataframes(
            {column: raw_vix_data},
            target_column=column,
            fill_method="ffill",
            min_date=start_date,
            max_date=end_date,
        )
    raw_vix_data = aligned_vix_data
    vix_columns = [raw_vix_data.columns.levels[0][1]]
    raw_vix_data = torch.tensor(
        raw_vix_data[vix_columns].values, dtype=torch.float32
    ).transpose(0, 1)
    # TODO if the start date changes to > 1997 ish, you can add a lot more features, because vix changes from daily to hourly (probably)
    for i in range(len(vix_columns)):
        vix_columns[i] = "vix_" + vix_columns[i]

    vix_returns = (raw_vix_data[:, 1:] - raw_vix_data[:, :-1]) / raw_vix_data[:, :-1]
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

    ema_vix = calculate_ema(raw_vix_data[0, :], lookback=5)
    vix_data = torch.cat((vix_data, ema_vix.unsqueeze(0)), dim=0)
    vix_columns.append("vix_ema_5")

    ema_vix = calculate_ema(raw_vix_data[0, :], lookback=10)
    vix_data = torch.cat((vix_data, ema_vix.unsqueeze(0)), dim=0)
    vix_columns.append("vix_ema_10")

    ema_vix = calculate_ema(raw_vix_data[0, :], lookback=20)
    vix_data = torch.cat((vix_data, ema_vix.unsqueeze(0)), dim=0)
    vix_columns.append("vix_ema_20")

    # just a copy of whats above but for vix
    vix_data = vix_data[:, 1:]

    vix_data = vix_data[:, : -(max(target_dates) - 1)]
    vix_data = vix_data.unsqueeze(-1).tile(
        1, 1, len(target_dates)
    )  # correct way to handle for now
    vix_data = vix_data[:, 20:, :].unsqueeze(-1)

    data = torch.cat((data, vix_data.repeat(1, 1, 1, len(tickers))), dim=0)
    columns = columns + vix_columns

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
    wider_data = torch.cat((wider_data, gold.unsqueeze(0)), dim=0)
    wider_columns.append("gold_close")

    change = (gold[1:] - gold[:-1]) / gold[:-1]
    change = torch.cat((change[:1, :], change), dim=0)
    wider_data = torch.cat((wider_data, change.unsqueeze(0)), dim=0)
    wider_columns.append("gold_change")

    sma = calculate_moving_average(gold.unsqueeze(0), lookback=5, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("gold_sma_5")

    sma = calculate_moving_average(gold.unsqueeze(0), lookback=10, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("gold_sma_10")

    sma = calculate_moving_average(gold.unsqueeze(0), lookback=20, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("gold_sma_20")

    ema = calculate_ema(gold, lookback=5)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("gold_ema_5")

    ema = calculate_ema(gold, lookback=10)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("gold_ema_10")

    ema = calculate_ema(gold, lookback=20)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("gold_ema_20")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=5, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("gold_change_sma_5")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=10, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("gold_change_sma_10")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=20, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("gold_change_sma_20")

    ema = calculate_ema(change, lookback=5)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("gold_change_ema_5")

    ema = calculate_ema(change, lookback=10)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("gold_change_ema_10")

    ema = calculate_ema(change, lookback=20)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("gold_change_ema_20")

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
    wider_data = torch.cat((wider_data, crude_oil.unsqueeze(0)), dim=0)
    wider_columns.append("crude_oil_close")

    change = (crude_oil[1:] - crude_oil[:-1]) / crude_oil[:-1]
    change = torch.cat((change[:1, :], change), dim=0)
    wider_data = torch.cat((wider_data, change.unsqueeze(0)), dim=0)
    wider_columns.append("crude_oil_change")

    sma = calculate_moving_average(crude_oil.unsqueeze(0), lookback=5, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("crude_oil_sma_5")

    sma = calculate_moving_average(crude_oil.unsqueeze(0), lookback=10, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("crude_oil_sma_10")

    sma = calculate_moving_average(crude_oil.unsqueeze(0), lookback=20, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("crude_oil_sma_20")

    ema = calculate_ema(crude_oil, lookback=5)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("crude_oil_ema_5")

    ema = calculate_ema(crude_oil, lookback=10)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("crude_oil_ema_10")

    ema = calculate_ema(crude_oil, lookback=20)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("crude_oil_ema_20")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=5, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("crude_oil_change_sma_5")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=10, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("crude_oil_change_sma_10")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=20, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("crude_oil_change_sma_20")

    ema = calculate_ema(change, lookback=5)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("crude_oil_change_ema_5")

    ema = calculate_ema(change, lookback=10)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("crude_oil_change_ema_10")

    ema = calculate_ema(change, lookback=20)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("crude_oil_change_ema_20")

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
    wider_data = torch.cat((wider_data, copper.unsqueeze(0)), dim=0)
    wider_columns.append("copper_close")

    change = (copper[1:] - copper[:-1]) / copper[:-1]
    change = torch.cat((change[:1, :], change), dim=0)
    wider_data = torch.cat((wider_data, change.unsqueeze(0)), dim=0)
    wider_columns.append("copper_change")

    sma = calculate_moving_average(copper.unsqueeze(0), lookback=5, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("copper_sma_5")

    sma = calculate_moving_average(copper.unsqueeze(0), lookback=10, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("copper_sma_10")

    sma = calculate_moving_average(copper.unsqueeze(0), lookback=20, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("copper_sma_20")

    ema = calculate_ema(copper, lookback=5)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("copper_ema_5")

    ema = calculate_ema(copper, lookback=10)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("copper_ema_10")

    ema = calculate_ema(copper, lookback=20)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("copper_ema_20")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=5, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("copper_change_sma_5")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=10, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("copper_change_sma_10")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=20, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("copper_change_sma_20")

    ema = calculate_ema(change, lookback=5)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("copper_change_ema_5")

    ema = calculate_ema(change, lookback=10)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("copper_change_ema_10")

    ema = calculate_ema(change, lookback=20)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("copper_change_ema_20")

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
    wider_data = torch.cat((wider_data, silver.unsqueeze(0)), dim=0)
    wider_columns.append("silver_close")

    change = (silver[1:] - silver[:-1]) / silver[:-1]
    change = torch.cat((change[:1, :], change), dim=0)
    wider_data = torch.cat((wider_data, change.unsqueeze(0)), dim=0)
    wider_columns.append("silver_change")

    sma = calculate_moving_average(silver.unsqueeze(0), lookback=5, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("silver_sma_5")

    sma = calculate_moving_average(silver.unsqueeze(0), lookback=10, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("silver_sma_10")

    sma = calculate_moving_average(silver.unsqueeze(0), lookback=20, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("silver_sma_20")

    ema = calculate_ema(silver, lookback=5)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("silver_ema_5")

    ema = calculate_ema(silver, lookback=10)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("silver_ema_10")

    ema = calculate_ema(silver, lookback=20)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("silver_ema_20")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=5, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("silver_change_sma_5")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=10, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("silver_change_sma_10")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=20, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("silver_change_sma_20")

    ema = calculate_ema(change, lookback=5)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("silver_change_ema_5")

    ema = calculate_ema(change, lookback=10)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("silver_change_ema_10")

    ema = calculate_ema(change, lookback=20)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("silver_change_ema_20")

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
    wider_data = torch.cat((wider_data, usd_index.unsqueeze(0)), dim=0)
    wider_columns.append("usd_index_close")

    change = (usd_index[1:] - usd_index[:-1]) / usd_index[:-1]
    change = torch.cat((change[:1, :], change), dim=0)
    wider_data = torch.cat((wider_data, change.unsqueeze(0)), dim=0)
    wider_columns.append("usd_index_change")

    sma = calculate_moving_average(usd_index.unsqueeze(0), lookback=5, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("usd_index_sma_5")

    sma = calculate_moving_average(usd_index.unsqueeze(0), lookback=10, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("usd_index_sma_10")

    sma = calculate_moving_average(usd_index.unsqueeze(0), lookback=20, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("usd_index_sma_20")

    ema = calculate_ema(usd_index, lookback=5)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("usd_index_ema_5")

    ema = calculate_ema(usd_index, lookback=10)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("usd_index_ema_10")

    ema = calculate_ema(usd_index, lookback=20)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("usd_index_ema_20")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=5, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("usd_index_change_sma_5")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=10, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("usd_index_change_sma_10")

    sma = calculate_moving_average(change.unsqueeze(0), lookback=20, dim=0)
    wider_data = torch.cat((wider_data, sma.unsqueeze(0)), dim=0)
    wider_columns.append("usd_index_change_sma_20")

    ema = calculate_ema(change, lookback=5)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("usd_index_change_ema_5")

    ema = calculate_ema(change, lookback=10)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("usd_index_change_ema_10")

    ema = calculate_ema(change, lookback=20)
    wider_data = torch.cat((wider_data, ema.unsqueeze(0).unsqueeze(-1)), dim=0)
    wider_columns.append("usd_index_change_ema_20")

    wider_data = wider_data[:, 1:]
    wider_data = wider_data[:, : -(max(target_dates) - 1)]
    wider_data = wider_data.tile(1, 1, len(target_dates))
    wider_data = wider_data[:, 20:, :].unsqueeze(-1).tile(1, 1, 1, len(tickers))

    data = torch.cat((data, wider_data), dim=0)
    true_prices = raw_data[:, : -(max(target_dates) - 2), :]  # maybe to align with data
    true_prices = true_prices[0, -data_length:, :].unsqueeze(0)

    data[4,864,:,:] = data[4,863,:,:] # ffil quickfix, no clue why there is suddenly infinity there
    if (torch.isnan(data)).any() or (torch.isinf(data)).any():
        print("Data contains NaN or Inf values.")

    return data, true_prices


def get_predictions(model, inputs, args):
    # inputs = inputs.unsqueeze(0)
    batch_size, num_features, seq_len, num_sequences = inputs.shape

    known_inputs = inputs[:, 0, :].unsqueeze(
        2
    )  # (batch_size, seq_len, 1, num_sequences)

    target_input_means = known_inputs.mean(dim=1, keepdim=True).tile(
        1, known_inputs.shape[1], 1, 1
    )
    target_input_stds = known_inputs.std(dim=1, keepdim=True).tile(
        1, known_inputs.shape[1], 1, 1
    )

    norm_inputs = (known_inputs - target_input_means) / target_input_stds

    additional_inputs = torch.cat(
        (inputs[:, 1:71, :, :], inputs[:, 86:, :, :]), dim=1
    )  # (batch_size, features, seq_len, num_sequences)
    additional_inputs = additional_inputs.transpose(
        1, 2
    )  # (batch_size, seq_len, features, num_sequences)
    means_of_additonal_inputs = additional_inputs.mean(dim=1, keepdim=True).tile(
        1, additional_inputs.shape[1], 1, 1
    )
    stds_of_additonal_inputs = additional_inputs.std(dim=1, keepdim=True).tile(
        1, additional_inputs.shape[1], 1, 1
    )
    full_inputs = torch.cat(
        [
            norm_inputs,
            (additional_inputs - means_of_additonal_inputs) / stds_of_additonal_inputs,
        ],
        dim=2,
    )
    # full_inputs = torch.cat([full_inputs, target_input_means, target_input_stds], dim=2)
    full_inputs = torch.cat(
        [full_inputs, inputs[:, 71:86, :, :].transpose(1, 2)], dim=2
    )  # adding time based data (need to be careful with this)
    full_inputs = full_inputs.transpose(2, 3)

    seperator = torch.zeros(
        (batch_size, 1), dtype=torch.int, device=inputs.device
    )  # (batch_size, 1)
    tickers = torch.arange(len(args.tickers), device=inputs.device)
    tickers = tickers.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, num_sequences)

    outputs = model(full_inputs, seperator, tickers).view(
        batch_size, -1, len(args.indices_to_predict), len(args.tickers)
    )  # .transpose(-1, -2) # (batch_size, seq_len, targets*num_sequences)
    outputs = outputs[:, 1:, :, :]

    # for debugging reasons
    if torch.isnan(outputs).any():
        print("NAN IN OUTPUTS")

    if isinstance(outputs, torch.Tensor):
        raw_logits = outputs
    else:
        raw_logits = outputs.logits

    preds = raw_logits * target_input_stds.tile(
        1, 1, len(args.indices_to_predict), 1
    ) + target_input_means.tile(1, 1, len(args.indices_to_predict), 1)

    return preds.detach()


def adapt_longer_preds(
    long_preds, true_returns, days_predicted_ahead, days_to_adapt_to, checked_days
):
    days = days_predicted_ahead
    long_preds = long_preds[
        -(checked_days + days - 1) : -(days - 1), :
    ]  # (seq_len, num_sequences)
    long_preds = long_preds + 1.0
    for i in range(days - days_to_adapt_to):
        long_preds = long_preds / (
            true_returns[:, -(checked_days + days - 1 - i) : -(days - 1 - i), :] + 1.0
        )
    long_preds = long_preds - 1.0
    return long_preds


def run_trading_strategy_Nday_signal( # TODO add verbose option
    predictions_Nday_ahead,  # Model's predicted N-day CUMULATIVE returns
    actual_1d_returns,  # Actual 1-day returns for P&L calculation
    trade_threshold_up=0.01,  # Minimum N-day predicted return to trigger a trade (e.g., 1% for 5-day)
    trade_threshold_down=0.01, 
    initial_capital=100000.0,
    transaction_cost_pct=0.0005,
    signal_horizon_name="N-day",  # For logging
    verbose = 0
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
        raise ValueError(
            "Predictions and actual 1-day returns must have the same number of days."
        )

    portfolio_values = []
    daily_portfolio_returns = []
    current_capital = initial_capital

    # print(f"\n--- Running Trading Strategy (Signal: {signal_horizon_name}) ---")
    # print(f"Initial Capital: ${initial_capital:,.2f}")
    # print(
    #     f"Trade Threshold up (for {signal_horizon_name} signal): {trade_threshold_up*100:.2f}%"
    # )
    # print(
    #     f"Trade Threshold down (for {signal_horizon_name} signal): {trade_threshold_down*100:.2f}%"
    # )
    # print(f"Transaction Cost (per trade): {transaction_cost_pct*100:.3f}%")

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
            if signals_Nday[ticker_idx] > trade_threshold_up:
                active_signals.append({"action": "long", "ticker_idx": ticker_idx})
            elif (
                signals_Nday[ticker_idx] < -trade_threshold_down
            ):  # Assuming symmetrical threshold
                active_signals.append({"action": "short", "ticker_idx": ticker_idx})

        if not active_signals:
            daily_portfolio_returns.append(0.0)
            portfolio_values.append(current_capital)
            # if day_idx < 5 or day_idx == num_days - 1:
            #     print(f"Day {day_idx+1}: No trades. Capital: ${current_capital:,.2f}")
            continue

        capital_per_trade = capital_at_start_of_day / len(active_signals)

        for trade in active_signals:
            ticker_idx = trade["ticker_idx"]
            invested_amount = capital_per_trade

            cost_entry = invested_amount * transaction_cost_pct
            trade_pnl_pre_exit_cost = 0

            if trade["action"] == "long":
                trade_pnl_pre_exit_cost = (
                    invested_amount * realized_1d_returns_today[ticker_idx]
                )
            elif trade["action"] == "short":
                trade_pnl_pre_exit_cost = invested_amount * (
                    -realized_1d_returns_today[ticker_idx]
                )

            cost_exit = (
                invested_amount + trade_pnl_pre_exit_cost
            ) * transaction_cost_pct

            daily_pnl += trade_pnl_pre_exit_cost - cost_entry - cost_exit
            num_trades_today += 1

        current_capital += daily_pnl
        day_return_pct = (
            daily_pnl / capital_at_start_of_day
            if capital_at_start_of_day > 1e-6
            else 0.0
        )  # Avoid div by zero if capital wiped

        daily_portfolio_returns.append(day_return_pct)
        portfolio_values.append(current_capital)

        # if (
        #     day_idx < 5
        #     or day_idx == num_days - 1
        #     or (
        #         num_trades_today > 0
        #         and day_idx % (num_days // 10 if num_days > 20 else 1) == 0
        #     )
        # ):
        #     print(
        #         f"Day {day_idx+1}: Trades: {num_trades_today}. Day P&L: ${daily_pnl:,.2f}. Day Return: {day_return_pct*100:.2f}%. Capital: ${current_capital:,.2f}"
        #     )

    portfolio_df = pd.DataFrame(
        {"portfolio_value": portfolio_values, "daily_return": daily_portfolio_returns}
    )

    total_return = (current_capital / initial_capital) - 1
    annualized_return = (
        ((1 + total_return) ** (252 / num_days)) - 1 if num_days > 0 else 0.0
    )
    annualized_volatility = (
        np.std(daily_portfolio_returns) * np.sqrt(252) if num_days > 0 else 0.0
    )
    sharpe_ratio = (
        (annualized_return / annualized_volatility)
        if annualized_volatility > 1e-9
        else 0.0
    )  # Avoid div by zero

    cumulative_returns = (
        1 + pd.Series(portfolio_df["daily_return"])
    ).cumprod()  # Use pandas Series for cumprod
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0

    num_winning_days = (portfolio_df["daily_return"] > 0).sum()
    num_losing_days = (portfolio_df["daily_return"] < 0).sum()
    win_loss_ratio = (
        num_winning_days / num_losing_days if num_losing_days > 0 else float("inf")
    )

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
        "Final Capital": current_capital,
    }
    if verbose > 0:
        print(f"\n--- Strategy Summary (Signal: {signal_horizon_name}) ---")
        for key, value in stats.items():
            if isinstance(value, float) and key not in [
                "Sharpe Ratio",
                "Win/Loss Day Ratio",
                "Signal Horizon Used",
            ]:
                print(
                    f"{key}: {value*100:.2f}%"
                    if "Return" in key or "Drawdown" in key or "Volatility" in key
                    else f"{key}: {value:.2f}"
                )
            elif isinstance(value, float) and key in ["Sharpe Ratio", "Win/Loss Day Ratio"]:
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

    return portfolio_df, stats

def objective_function_asymmetric_thresholds(long_threshold_raw, short_threshold_raw, min_step=0.001):
    """
    Objective function for Bayesian Optimization when tuning asymmetric long and short thresholds.
    """

    long_threshold = round(long_threshold_raw / min_step) * min_step
    # long_threshold = max(lower_bound_thresh, min(upper_bound_thresh, long_threshold))

    short_threshold = round(short_threshold_raw / min_step) * min_step
    # short_threshold = max(lower_bound_thresh, min(upper_bound_thresh, short_threshold))

    print(f"  Testing long_thresh: {long_threshold:.4f} (raw: {long_threshold_raw:.4f}), short_thresh_abs: {short_threshold:.4f} (raw: {short_threshold_raw:.4f})")

    # Modify your strategy function or pass these as separate args if it supports it
    # For now, assuming run_trading_strategy_Nday_signal is modified or we make a wrapper
    # Let's assume a modified run_trading_strategy_Nday_signal that takes long_thresh and short_thresh_abs
    
    # If your strategy function doesn't take separate long/short thresholds, you'd need to adapt it
    # or create a version that does. For simplicity, if it only takes one `trade_threshold`,
    # this asymmetric optimization won't directly fit.
    # Let's assume you have a version like this:
    # def run_trading_strategy_Nday_signal_asymmetric(..., long_threshold, short_threshold_abs, ...):

    # For now, I'll reuse the single threshold version and show how to set up the optimizer.
    # If you want asymmetric, you'll need to adapt your strategy function.
    # For demonstration with the current function, let's just average them or pick one.
    # THIS IS NOT IDEAL FOR ASYMMETRIC, but shows the setup.
    # A BETTER WAY: Modify run_trading_strategy_Nday_signal to accept long_threshold and short_threshold_abs
    
    # Assuming run_trading_strategy_Nday_signal is adapted:
    _, strategy_stats = run_trading_strategy_Nday_signal(
        predictions_Nday_ahead=OPTIMIZATION_PREDS_NDAY,
        actual_1d_returns=OPTIMIZATION_ACTUAL_1D_RETURNS,
        trade_threshold_up=long_threshold, # Pass the tuned long threshold
        trade_threshold_down=short_threshold, # Pass the tuned short threshold
        initial_capital=OPTIMIZATION_INITIAL_CAPITAL,
        transaction_cost_pct=OPTIMIZATION_TRANSACTION_COST,
        signal_horizon_name=OPTIMIZATION_SIGNAL_HORIZON_NAME
    )

    # TEMPORARY for demonstration if strategy function is not changed:
    # Use the average or just one for now to show optimizer setup
    # effective_threshold = (long_threshold + short_threshold) / 2.0 
    # print(f"    (Using effective_threshold: {effective_threshold:.4f} for single-threshold strategy function)")
    # _, strategy_stats = run_trading_strategy_Nday_signal(
    #     predictions_Nday_ahead=OPTIMIZATION_PREDS_NDAY,
    #     actual_1d_returns=OPTIMIZATION_ACTUAL_1D_RETURNS,
    #     trade_threshold=effective_threshold, # Using the single symmetrical threshold
    #     initial_capital=OPTIMIZATION_INITIAL_CAPITAL,
    #     transaction_cost_pct=OPTIMIZATION_TRANSACTION_COST,
    #     signal_horizon_name=OPTIMIZATION_SIGNAL_HORIZON_NAME
    # )


    sharpe = strategy_stats.get("Sharpe Ratio", 0.0)
    if not np.isfinite(sharpe) or sharpe < -5:
        return -10.0
    return sharpe

def objective_function_asymmetric_thresholds_combined_signal(long_threshold_raw, short_threshold_raw, weight1, weight5, weight10, min_step=0.001, min_weight_step = 0.1):
    """
    Objective function for Bayesian Optimization when tuning asymmetric long and short thresholds.
    """

    long_threshold = round(long_threshold_raw / min_step) * min_step

    short_threshold = round(short_threshold_raw / min_step) * min_step

    rounded_weights = [round(weight / min_weight_step) * min_weight_step for weight in [weight1, weight5, weight10]]

    print(f"  Testing long_thresh: {long_threshold:.4f} (raw: {long_threshold_raw:.4f}), short_thresh_abs: {short_threshold:.4f} (raw: {short_threshold_raw:.4f}) weight1: {rounded_weights[0]}, weight5: {rounded_weights[1]}, weight10: {rounded_weights[2]}")

   
    weights = torch.tensor(rounded_weights).view(1, -1, 1)
    optimization_preds_nday_weighted = (OPTIMIZATION_PREDS_NDAY / weights).mean(dim=1)

    _, strategy_stats = run_trading_strategy_Nday_signal(
        predictions_Nday_ahead=optimization_preds_nday_weighted,
        actual_1d_returns=OPTIMIZATION_ACTUAL_1D_RETURNS,
        trade_threshold_up=long_threshold,
        trade_threshold_down=short_threshold,
        initial_capital=OPTIMIZATION_INITIAL_CAPITAL,
        transaction_cost_pct=OPTIMIZATION_TRANSACTION_COST,
        signal_horizon_name=OPTIMIZATION_SIGNAL_HORIZON_NAME
    )

    sharpe = strategy_stats.get("Sharpe Ratio", 0.0)
    if not np.isfinite(sharpe) or sharpe < -5:
        return -10.0
    return sharpe


if __name__ == "__main__":
    # metadata_path = "good_models/model_metadata.json"
    # model_path = "good_models\Money_former_DINT_test_Money_former_DINT_16_128_512_4_4_24.pth"
    metadata_path = (
        "good_models\Money_former_DINT_test_Money_former_DINT_16_128_128_6_32_32.json"
    )
    model_path = (
        "good_models\Money_former_DINT_test_Money_former_DINT_16_128_128_6_32_32.pth"
    )
    nr_of_days_to_check = 1000
    args = load_metadata(metadata_path)
    num_tickers = len(args.indices_to_predict)
    graph = False
    pre_optim = False

    if graph == True:
        data, true_prices = download_and_process_input_data(
            args
        )  # (features, seq_len, targets, num_sequences)
        data = data[
            :,
            -(args.seq_len + nr_of_days_to_check + max(args.indices_to_predict)) : -1,
            0,
            :,
        ]
        true_prices = true_prices[
            :, -(args.seq_len + nr_of_days_to_check + max(args.indices_to_predict)) :, :
        ]
        true_returns = (true_prices[:, 1:, :] - true_prices[:, :-1, :]) / true_prices[
            :, :-1, :
        ]
        true_returns_5 = (true_prices[:, 5:, :] - true_prices[:, :-5, :]) / true_prices[
            :, :-5, :
        ]
        true_returns_10 = (
            true_prices[:, 10:, :] - true_prices[:, :-10, :]
        ) / true_prices[:, :-10, :]

        model = load_model(model_path, args)
        predictions = []
        for i in range(nr_of_days_to_check + max(args.indices_to_predict) - 1):
            predictions.append(
                get_predictions(model, data[:, i : i + args.seq_len, :], args)[
                    :, -1, :, :
                ]
            )  # (seq_len, targets,num_sequences)
        predictions = torch.cat(predictions, dim=0)  # (seq_len, targets,num_sequences)

        full_seq_preds_1d = get_predictions(model, data[:, -args.seq_len :, :], args)[
            :, :, 0, :
        ].unsqueeze(
            0
        )  # (seq_len, num_sequences)
        full_seq_preds_5d = get_predictions(
            model, data[:, -args.seq_len - 4 : -4, :], args
        )[:, :, 1, :].unsqueeze(
            0
        )  # (seq_len, num_sequences)
        full_seq_preds_10d = get_predictions(
            model, data[:, -args.seq_len - 9 : -9, :], args
        )[:, :, 2, :].unsqueeze(
            0
        )  # (seq_len, num_sequences)

        d1_predictions = predictions[-nr_of_days_to_check:, 0, :].unsqueeze(
            0
        )  # (seq_len, num_sequences)
        days = args.indices_to_predict[1]
        adapted_predictions = adapt_longer_preds(
            predictions[:, 1, :], true_returns, days, 1, nr_of_days_to_check
        )
        d1_predictions = torch.cat(
            [d1_predictions, adapted_predictions], dim=0
        )  # (adapted_days, seq_len, num_sequences)

        days = args.indices_to_predict[2]
        adapted_predictions = adapt_longer_preds(
            predictions[:, 2, :], true_returns, days, 1, nr_of_days_to_check
        )
        d1_predictions = torch.cat(
            [d1_predictions, adapted_predictions], dim=0
        )  # (adapted_days, seq_len, num_sequences)

        d5_predictions = predictions[-nr_of_days_to_check - 4 : -4, 1, :].unsqueeze(
            0
        )  # (seq_len, num_sequences)
        days = args.indices_to_predict[2]
        adapted_predictions = adapt_longer_preds(
            predictions[:, 2, :], true_returns, days, 5, nr_of_days_to_check
        )
        d5_predictions = torch.cat(
            [d5_predictions, adapted_predictions], dim=0
        )  # (adapted_days, seq_len, num_sequences)

        d1_predictions = d1_predictions.transpose(0, 1)
        d5_predictions = d5_predictions.transpose(0, 1)
        # TODO compare with actual prices (because less head hurt)
        plt.plot(true_returns[0, -nr_of_days_to_check:, 0])
        plt.plot(d1_predictions[:, :, 0])
        plt.plot(full_seq_preds_1d[0, :, 0])
        plt.legend(
            [
                f"true 1day {args.tickers[0]}",
                f"predictions 1day {args.tickers[0]}",
                f"predictions 5day {args.tickers[0]}",
                f"predictions 10day {args.tickers[0]}",
                f"full_seq_preds_1d {args.tickers[0]}",
            ]
        )
        plt.show()

        plt.plot(true_returns_5[0, -nr_of_days_to_check:, 0])
        plt.plot(d5_predictions[:, :, 0])
        plt.plot(full_seq_preds_5d[0, :, 0])
        plt.legend(
            [
                f"true 5 days {args.tickers[0]}",
                f"predictions 5day {args.tickers[0]}",
                f"predictions 10day {args.tickers[0]}",
                f"full_seq_preds_5d {args.tickers[0]}",
            ]
        )
        plt.show()

        plt.plot(true_returns_10[0, -nr_of_days_to_check:, 0])
        plt.plot(predictions[:-9, 2, 0])
        plt.plot(full_seq_preds_10d[0, :, 0])
        plt.legend(
            [
                f"true 10 days {args.tickers[0]}",
                f"predictions 10day {args.tickers[0]}",
                f"full_seq_preds_10d {args.tickers[0]}",
            ]
        )
        plt.show()

        plt.plot(true_returns[0, -nr_of_days_to_check:, 1])
        plt.plot(d1_predictions[:, :, 1])
        plt.plot(full_seq_preds_1d[0, :, 1])
        plt.legend(
            [
                f"true 1day {args.tickers[1]}",
                f"predictions 1day {args.tickers[1]}",
                f"predictions 5day {args.tickers[1]}",
                f"predictions 10day {args.tickers[1]}",
                f"full_seq_preds_1d {args.tickers[1]}",
            ]
        )
        plt.show()

        plt.plot(true_returns_5[0, -nr_of_days_to_check:, 1])
        plt.plot(d5_predictions[:, :, 1])
        plt.plot(full_seq_preds_5d[0, :, 1])
        plt.legend(
            [
                f"true 5 days {args.tickers[1]}",
                f"predictions 5day {args.tickers[1]}",
                f"predictions 10day {args.tickers[1]}",
                f"full_seq_preds_5d {args.tickers[1]}",
            ]
        )
        plt.show()

        plt.plot(true_returns_10[0, -nr_of_days_to_check:, 1])
        plt.plot(predictions[:-9, 2, 1])
        plt.plot(full_seq_preds_10d[0, :, 1])
        plt.legend(
            [
                f"true 10 days {args.tickers[1]}",
                f"predictions 10day {args.tickers[1]}",
                f"full_seq_preds_10d {args.tickers[1]}",
            ]
        )
        plt.show()

        MAE_fn = nn.L1Loss()
        print(
            f"naive 1d:{MAE_fn(torch.zeros_like(true_returns[0, -nr_of_days_to_check:, :]), true_returns[0, -nr_of_days_to_check:, :])}"
        )
        print(
            f"naive 5d:{MAE_fn(torch.zeros_like(true_returns_5[0, -nr_of_days_to_check:, :]), true_returns_5[0, -nr_of_days_to_check:, :])}"
        )
        print(
            f"naive 10d:{MAE_fn(torch.zeros_like(true_returns_10[0, -nr_of_days_to_check:, :]), true_returns_10[0, -nr_of_days_to_check:, :])}"
        )

        print(
            f"1d 1d:{MAE_fn(d1_predictions[:,0], true_returns[0, -nr_of_days_to_check:, :])}"
        )
        print(
            f"1d 5d:{MAE_fn(d1_predictions[:,1], true_returns[0, -nr_of_days_to_check:, :])}"
        )
        print(
            f"1d 10d:{MAE_fn(d1_predictions[:,2], true_returns[0, -nr_of_days_to_check:, :])}"
        )
        print(
            f"5d 5d:{MAE_fn(d5_predictions[:,0], true_returns_5[0, -nr_of_days_to_check:, :])}"
        )
        print(
            f"5d 10d:{MAE_fn(d5_predictions[:,1], true_returns_5[0, -nr_of_days_to_check:, :])}"
        )
        print(
            f"10d 10d:{MAE_fn(predictions[:-9, 2, :], true_returns_10[0, -nr_of_days_to_check:, :])}"
        )

    all_processed_data, all_true_prices = download_and_process_input_data(args)
    total_available_timesteps = all_processed_data.shape[1]

    required_timesteps_for_rolling_preds = args.seq_len + nr_of_days_to_check - 1
    if total_available_timesteps < required_timesteps_for_rolling_preds:
        raise ValueError(
            f"Not enough data ({total_available_timesteps} available) for seq_len ({args.seq_len}) and nr_of_days_to_check ({nr_of_days_to_check}). Need {required_timesteps_for_rolling_preds}."
        )

    data_for_backtest_input_generation = all_processed_data[
        :, -required_timesteps_for_rolling_preds:, 0, :
    ]
    true_prices_for_backtest_period = all_true_prices[
        0, -(nr_of_days_to_check + 1) :, :
    ]
    actual_1d_returns_for_backtest = (
        true_prices_for_backtest_period[1:, :] - true_prices_for_backtest_period[:-1, :]
    ) / (true_prices_for_backtest_period[:-1, :] + 1e-8)

    model = load_model(model_path, args)

    all_horizons_predictions_list = []
    for i in range(nr_of_days_to_check):
        current_input_seq = data_for_backtest_input_generation[
            :, i : i + args.seq_len, :
        ].unsqueeze(0)
        # Get predictions for all horizons from the model's last output step
        preds_all_horizons_for_last_step = get_predictions(
            model, current_input_seq, args
        )[0, -1, :, :]
        # Shape: (num_prediction_horizons, num_tickers)
        all_horizons_predictions_list.append(preds_all_horizons_for_last_step)

    all_horizons_predictions_tensor = torch.stack(all_horizons_predictions_list, dim=0)

    initial_capital_main = 1000

    if pre_optim:
        if len(args.indices_to_predict) > 1 and args.indices_to_predict[1] == 5:
            predictions_5day = all_horizons_predictions_tensor[
                :, 1, :
            ]  # (nr_of_days_to_check, num_tickers)
            portfolio_df_5d, stats_5d = run_trading_strategy_Nday_signal(
                predictions_5day.cpu(),
                actual_1d_returns_for_backtest.cpu(),
                trade_threshold=0.015,  # e.g., predict 1% move over 5 days to trade
                initial_capital=initial_capital_main,
                transaction_cost_pct=0.0005,
                signal_horizon_name="5-day",
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
            predictions_10day = all_horizons_predictions_tensor[
                :, 2, :
            ]  # (nr_of_days_to_check, num_tickers)
            portfolio_df_10d, stats_10d = run_trading_strategy_Nday_signal(
                predictions_10day.cpu(),
                actual_1d_returns_for_backtest.cpu(),
                trade_threshold=0.03,  # e.g., predict 1.5% move over 10 days to trade
                initial_capital=initial_capital_main,
                transaction_cost_pct=0.0005,
                signal_horizon_name="10-day",
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
            predictions_1day = all_horizons_predictions_tensor[
                :, 0, :
            ]  # (nr_of_days_to_check, num_tickers)
            # Assuming your original run_trading_strategy is still available or this one is adapted
            portfolio_df_1d, stats_1d = (
                run_trading_strategy_Nday_signal(  # Can reuse Nday_signal for 1-day
                    predictions_1day.cpu(),
                    actual_1d_returns_for_backtest.cpu(),
                    trade_threshold=0.003,  # Your original 1-day threshold
                    initial_capital=initial_capital_main,
                    transaction_cost_pct=0.0005,
                    signal_horizon_name="1-day",
                )
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
            prediction_time_weights = torch.tensor(
                [[1.0], [5.0], [10.0]], dtype=torch.float32
            )
            averaged_predictions = (
                all_horizons_predictions_tensor / prediction_time_weights
            ).mean(dim=1)

            portfolio_df_combined, stats_combined = run_trading_strategy_Nday_signal(
                averaged_predictions.cpu(),
                actual_1d_returns_for_backtest.cpu(),
                trade_threshold=0.003,  # e.g., predict 1% move over 5 days to trade
                initial_capital=initial_capital_main,
                transaction_cost_pct=0.0005,
                signal_horizon_name="Combined",
            )
            if portfolio_df_combined is not None and not portfolio_df_combined.empty:
                plt.figure(figsize=(12, 6))
                cumulative_portfolio_returns_plot_combined = (
                    1 + pd.Series(portfolio_df_combined["daily_return"])
                ).cumprod()
                plt.plot(
                    cumulative_portfolio_returns_plot_combined,
                    label="Strategy (Combined signal)",
                )
                plt.title(
                    f"Strategy (Combined signal) Portfolio Value (Tickers: {', '.join(args.tickers)})"
                )
                plt.xlabel("Trading Day")
                plt.ylabel("Cumulative Return (Normalized)")
                plt.legend()
                plt.grid(True)
                plt.savefig(
                    f"strategy_portfolio_combined_signal_{'_'.join(args.tickers)}.png"
                )
                plt.show()

        # --- Comparison Plot (if you ran multiple strategies) ---
        plt.figure(figsize=(14, 7))
        if (
            "portfolio_df_1d" in locals()
            and portfolio_df_1d is not None
            and not portfolio_df_1d.empty
        ):
            plt.plot(
                (1 + pd.Series(portfolio_df_1d["daily_return"])).cumprod(),
                label="Strategy (1-day Signal)",
            )
        if (
            "portfolio_df_5d" in locals()
            and portfolio_df_5d is not None
            and not portfolio_df_5d.empty
        ):
            plt.plot(
                (1 + pd.Series(portfolio_df_5d["daily_return"])).cumprod(),
                label="Strategy (5-day Signal)",
            )
        if (
            "portfolio_df_10d" in locals()
            and portfolio_df_10d is not None
            and not portfolio_df_10d.empty
        ):
            plt.plot(
                (1 + pd.Series(portfolio_df_10d["daily_return"])).cumprod(),
                label="Strategy (10-day Signal)",
            )
        if (
            "portfolio_df_combined" in locals()
            and portfolio_df_combined is not None
            and not portfolio_df_combined.empty
        ):
            plt.plot(
                (1 + pd.Series(portfolio_df_combined["daily_return"])).cumprod(),
                label="Strategy (Combined Signal)",
            )
        plt.title(f"Strategy Comparison (Tickers: {', '.join(args.tickers)})")

        if len(args.tickers) > 0:

            if len(args.tickers) > 1:
                buy_and_hold_returns_avg_tickers = (
                    actual_1d_returns_for_backtest.cpu().mean(dim=1)
                )
                buy_and_hold_equity_curve_avg = (
                    1 + buy_and_hold_returns_avg_tickers
                ).cumprod(dim=0)
                plt.plot(
                    buy_and_hold_equity_curve_avg.numpy(),
                    label=f"Buy & Hold Avg of {len(args.tickers)} Tickers",
                    linestyle=":",
                )

        plt.title("Comparison of Strategies vs. Buy & Hold")
        plt.xlabel("Trading Day in Backtest Period")
        plt.ylabel("Cumulative Return (Normalized to 1)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"strategy_comparison_{'_'.join(args.tickers)}.png")
        plt.show()

    
    # TODO maybe try total returns instead of sharpe ratio?


    # ---- preparing data for optimization (10 day preds) ----
    print("\n--- Setting up for Bayesian Optimization of 10-day Signal Thresholds ---")
    OPTIMIZATION_PREDS_NDAY = all_horizons_predictions_tensor[:500, 2, :].cpu()
    OPTIMIZATION_ACTUAL_1D_RETURNS = actual_1d_returns_for_backtest[:500, :].cpu()
    OPTIMIZATION_INITIAL_CAPITAL = initial_capital_main # Use the one defined earlier
    OPTIMIZATION_TRANSACTION_COST = 0.0005 # Or your preferred value
    OPTIMIZATION_SIGNAL_HORIZON_NAME = "10-day (Opt.)"

    min_step = 0.001
    print("\nOptimizing asymmetric thresholds")
    pbounds_asymmetric = {
        'long_threshold_raw': (0.000, 0.10),
        'short_threshold_raw': (0.000, 0.10)
    }

    optimizer_asymmetric = BayesianOptimization(
        f=objective_function_asymmetric_thresholds,
        pbounds=pbounds_asymmetric,
        random_state=1,
        verbose=1
    )
    optimizer_asymmetric.maximize(init_points=5, n_iter=40) # More iterations if more params

    print("\nBest result (asymmetric thresholds):")
    best_params_asymmetric = optimizer_asymmetric.max['params']
    best_sharpe_asymmetric = optimizer_asymmetric.max['target']
    
    optimal_long_thresh = round(best_params_asymmetric['long_threshold_raw'] / min_step) * min_step
    optimal_long_thresh = max(0.0, min(0.05, optimal_long_thresh))
    optimal_short_thresh = round(best_params_asymmetric['short_threshold_raw'] / min_step) * min_step
    optimal_short_thresh = max(0.0, min(0.05, optimal_short_thresh))

    print(f"Optimal discretized long threshold: {optimal_long_thresh:.4f}")
    print(f"Optimal discretized short threshold abs: {optimal_short_thresh:.4f}")
    print(f"Achieved Sharpe Ratio: {best_sharpe_asymmetric:.4f}")

    # add plots for comparison
    if len(args.indices_to_predict) > 2 and args.indices_to_predict[2] == 10:
        predictions_10day = all_horizons_predictions_tensor[
            :, 2, :
        ]  # (nr_of_days_to_check, num_tickers)
        optim_portfolio_df_10d, optim_stats_10d = run_trading_strategy_Nday_signal(
            predictions_10day.cpu(),
            actual_1d_returns_for_backtest.cpu(),
            trade_threshold_up=optimal_long_thresh,
            trade_threshold_down=optimal_short_thresh,  # e.g., predict 1.5% move over 10 days to trade
            initial_capital=initial_capital_main,
            transaction_cost_pct=0.0005,
            signal_horizon_name="10-day",
            verbose=1
        )
    
    # ---- preparing data for optimisation (5 day preds) ----
    print("\n--- Setting up for Bayesian Optimization of 5-day Signal Thresholds ---")
    OPTIMIZATION_PREDS_NDAY = all_horizons_predictions_tensor[:500, 1, :].cpu()
    OPTIMIZATION_ACTUAL_1D_RETURNS = actual_1d_returns_for_backtest[:500, :].cpu()
    OPTIMIZATION_INITIAL_CAPITAL = initial_capital_main # Use the one defined earlier
    OPTIMIZATION_TRANSACTION_COST = 0.0005 # Or your preferred value
    OPTIMIZATION_SIGNAL_HORIZON_NAME = "5-day (Opt.)"

    min_step = 0.001
    # maybe try total returns instead of sharpe ratio?
    print("\nOptimizing asymmetric thresholds")
    pbounds_asymmetric = {
        'long_threshold_raw': (0.000, 0.10),
        'short_threshold_raw': (0.000, 0.10)
    }

    optimizer_asymmetric = BayesianOptimization(
        f=objective_function_asymmetric_thresholds,
        pbounds=pbounds_asymmetric,
        random_state=1,
        verbose=1
    )
    optimizer_asymmetric.maximize(init_points=5, n_iter=40) # More iterations if more params

    print("\nBest result (asymmetric thresholds):")
    best_params_asymmetric = optimizer_asymmetric.max['params']
    best_sharpe_asymmetric = optimizer_asymmetric.max['target']
    
    optimal_long_thresh = round(best_params_asymmetric['long_threshold_raw'] / min_step) * min_step
    optimal_long_thresh = max(0.0, min(0.05, optimal_long_thresh))
    optimal_short_thresh = round(best_params_asymmetric['short_threshold_raw'] / min_step) * min_step
    optimal_short_thresh = max(0.0, min(0.05, optimal_short_thresh))

    print(f"Optimal discretized long threshold: {optimal_long_thresh:.4f}")
    print(f"Optimal discretized short threshold abs: {optimal_short_thresh:.4f}")
    print(f"Achieved Sharpe Ratio: {best_sharpe_asymmetric:.4f}")

    # add plots for comparison
    if len(args.indices_to_predict) > 1 and args.indices_to_predict[1] == 5:
        predictions_5day = all_horizons_predictions_tensor[
            :, 1, :
        ]  # (nr_of_days_to_check, num_tickers)
        optim_portfolio_df_5d, optim_stats_5d = run_trading_strategy_Nday_signal(
            predictions_5day.cpu(),
            actual_1d_returns_for_backtest.cpu(),
            trade_threshold_up=optimal_long_thresh,
            trade_threshold_down=optimal_short_thresh,
            initial_capital=initial_capital_main,
            transaction_cost_pct=0.0005,
            signal_horizon_name="5-day",
            verbose=1
        )
    
    # ---- preparing data for optimisation (1 day preds) ----
    print("\n--- Setting up for Bayesian Optimization of 1-day Signal Thresholds ---")
    OPTIMIZATION_PREDS_NDAY = all_horizons_predictions_tensor[:500, 0, :].cpu()
    OPTIMIZATION_ACTUAL_1D_RETURNS = actual_1d_returns_for_backtest[:500, :].cpu()
    OPTIMIZATION_INITIAL_CAPITAL = initial_capital_main # Use the one defined earlier
    OPTIMIZATION_TRANSACTION_COST = 0.0005 # Or your preferred value
    OPTIMIZATION_SIGNAL_HORIZON_NAME = "1-day (Opt.)"

    min_step = 0.001
    # maybe try total returns instead of sharpe ratio?
    print("\nOptimizing asymmetric thresholds")
    pbounds_asymmetric = {
        'long_threshold_raw': (0.000, 0.10),
        'short_threshold_raw': (0.000, 0.10)
    }

    optimizer_asymmetric = BayesianOptimization(
        f=objective_function_asymmetric_thresholds,
        pbounds=pbounds_asymmetric,
        random_state=1,
        verbose=1
    )
    optimizer_asymmetric.maximize(init_points=5, n_iter=40) # More iterations if more params

    print("\nBest result (asymmetric thresholds):")
    best_params_asymmetric = optimizer_asymmetric.max['params']
    best_sharpe_asymmetric = optimizer_asymmetric.max['target']
    
    optimal_long_thresh = round(best_params_asymmetric['long_threshold_raw'] / min_step) * min_step
    optimal_long_thresh = max(0.0, min(0.05, optimal_long_thresh))
    optimal_short_thresh = round(best_params_asymmetric['short_threshold_raw'] / min_step) * min_step
    optimal_short_thresh = max(0.0, min(0.05, optimal_short_thresh))

    print(f"Optimal discretized long threshold: {optimal_long_thresh:.4f}")
    print(f"Optimal discretized short threshold abs: {optimal_short_thresh:.4f}")
    print(f"Achieved Sharpe Ratio: {best_sharpe_asymmetric:.4f}")

    # add plots for comparison
    if len(args.indices_to_predict) > 1 and args.indices_to_predict[0] == 1:
        predictions_1day = all_horizons_predictions_tensor[
            :, 0, :
        ]  # (nr_of_days_to_check, num_tickers)
        optim_portfolio_df_1d, optim_stats_1d = run_trading_strategy_Nday_signal(
            predictions_1day.cpu(),
            actual_1d_returns_for_backtest.cpu(),
            trade_threshold_up=optimal_long_thresh,
            trade_threshold_down=optimal_short_thresh,
            initial_capital=initial_capital_main,
            transaction_cost_pct=0.0005,
            signal_horizon_name="1-day (Opt.)",
            verbose=1
        )
    

    # optim combined preds
    print("\n--- Setting up for Bayesian Optimization of Combined Signal Thresholds ---")
    OPTIMIZATION_PREDS_NDAY = all_horizons_predictions_tensor[:500, :, :].cpu()
    OPTIMIZATION_ACTUAL_1D_RETURNS = actual_1d_returns_for_backtest[:500, :].cpu()
    OPTIMIZATION_INITIAL_CAPITAL = initial_capital_main # Use the one defined earlier
    OPTIMIZATION_TRANSACTION_COST = 0.0005 # Or your preferred value
    OPTIMIZATION_SIGNAL_HORIZON_NAME = "Combined (Opt.)"

    min_step = 0.001
    min_weight_step = 0.1
    # maybe try total returns instead of sharpe ratio?
    print("\nOptimizing asymmetric thresholds")
    pbounds_asymmetric = {
        'long_threshold_raw': (0.000, 0.08),
        'short_threshold_raw': (0.000, 0.08),
        'weight1': (1.0, 10.0),
        'weight5': (1.0, 10.0),
        'weight10': (1.0, 10.0)
    }

    optimizer_asymmetric = BayesianOptimization(
        f=objective_function_asymmetric_thresholds_combined_signal,
        pbounds=pbounds_asymmetric,
        random_state=1,
        verbose=1
    )
    optimizer_asymmetric.maximize(init_points=5, n_iter=100) # More iterations if more params

    print("\nBest result (asymmetric thresholds):")
    best_params_asymmetric = optimizer_asymmetric.max['params']
    best_sharpe_asymmetric = optimizer_asymmetric.max['target']
    
    optimal_long_thresh = round(best_params_asymmetric['long_threshold_raw'] / min_step) * min_step
    optimal_short_thresh = round(best_params_asymmetric['short_threshold_raw'] / min_step) * min_step

    optimal_rounded_weights = {
        'weight1': round(best_params_asymmetric['weight1'] / min_weight_step) * min_weight_step,
        'weight5': round(best_params_asymmetric['weight5'] / min_weight_step) * min_weight_step,
        'weight10': round(best_params_asymmetric['weight10'] / min_weight_step) * min_weight_step
    }
    rounded_weights = torch.tensor([optimal_rounded_weights['weight1'], optimal_rounded_weights['weight5'], optimal_rounded_weights['weight10']]).view(1, -1, 1)

    print(f"Optimal discretized long threshold: {optimal_long_thresh:.4f}")
    print(f"Optimal discretized short threshold abs: {optimal_short_thresh:.4f}")
    print(f"Optimal discretized weights: {optimal_rounded_weights}")
    print(f"Achieved Sharpe Ratio: {best_sharpe_asymmetric:.4f}")

    # add plots for comparison
    if len(args.indices_to_predict) > 1 and args.indices_to_predict[0] == 1:
        predictions_combined = (all_horizons_predictions_tensor[
            :, :, :
        ]/rounded_weights).mean(dim=1)  # (nr_of_days_to_check, num_tickers)
        optim_portfolio_df_comb, optim_stats_comb = run_trading_strategy_Nday_signal(
            predictions_combined.cpu(),
            actual_1d_returns_for_backtest.cpu(),
            trade_threshold_up=optimal_long_thresh,
            trade_threshold_down=optimal_short_thresh,
            initial_capital=initial_capital_main,
            transaction_cost_pct=0.0005,
            signal_horizon_name="Combined (Opt.)",
            verbose=1
        )


    if (
        "optim_portfolio_df_10d" in locals()
        and optim_portfolio_df_10d is not None
        and not optim_portfolio_df_10d.empty
    ):
        plt.plot(
            (1 + pd.Series(optim_portfolio_df_10d["daily_return"])).cumprod(),
            label="Strategy (10-day Signal)",
        )
    if (
        "optim_portfolio_df_5d" in locals()
        and optim_portfolio_df_5d is not None
        and not optim_portfolio_df_5d.empty
    ):
        plt.plot(
            (1 + pd.Series(optim_portfolio_df_5d["daily_return"])).cumprod(),
            label="Strategy (5-day Signal)",
        )
    if (
        "optim_portfolio_df_1d" in locals()
        and optim_portfolio_df_1d is not None
        and not optim_portfolio_df_1d.empty
    ):
        plt.plot(
            (1 + pd.Series(optim_portfolio_df_1d["daily_return"])).cumprod(),
            label="Strategy (1-day Signal)",
        )
    if (
        "optim_portfolio_df_comb" in locals()
        and optim_portfolio_df_comb is not None
        and not optim_portfolio_df_comb.empty
    ):
        plt.plot(
            (1 + pd.Series(optim_portfolio_df_comb["daily_return"])).cumprod(),
            label="Strategy (Combined Signal)",
        )
    if len(args.tickers) > 1:
        buy_and_hold_returns_avg_tickers = (
            actual_1d_returns_for_backtest.cpu().mean(dim=1)
        )
        buy_and_hold_equity_curve_avg = (
            1 + buy_and_hold_returns_avg_tickers
        ).cumprod(dim=0)
        plt.plot(
            buy_and_hold_equity_curve_avg.numpy(),
            label=f"Buy & Hold Avg of {len(args.tickers)} Tickers",
            linestyle=":",
        )

    plt.title("Comparison of Strategies vs. Buy & Hold")
    plt.xlabel("Trading Day in Backtest Period")
    plt.ylabel("Cumulative Return (Normalized to 1)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"optim_strategy_comparison_10day_{'_'.join(args.tickers)}.png")
    plt.show()