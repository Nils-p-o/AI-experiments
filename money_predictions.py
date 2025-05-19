import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
import yfinance as yf
import argparse

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
    calculate_volume_price_trend)


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
    tickers = args.tickers
    start_date = "2025-01-01"
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

        adr = (temp_raw_data[2, :] - temp_raw_data[3, :]) / temp_raw_data[
            1, :
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
        temp_data = temp_data[:, 1:]  
        # for ease and compatibility with getting multiple targets

        data.append(
            temp_data[:, 20:].unsqueeze(-1)
        )  # getting rid of some trashy-ish data points

    data = torch.cat(data, dim=-1) # (features, seq_len, num_sequences)
    data_length = data.shape[1]
    true_prices = raw_data[0, -data_length:, :].unsqueeze(0)  # (1, seq_len, num_sequences)

    if (torch.isnan(data)).any() or (torch.isinf(data)).any():
        print("Data contains NaN or Inf values.")

    return data, true_prices

def get_predictions(model, inputs, args):
    # inputs (features, seq_len, num_sequences)
    known_inputs = inputs[0, :, :].unsqueeze(0) # (1,seq_len, num_sequences)

    target_input_means = known_inputs.mean(dim=1, keepdim=True).tile(1, known_inputs.shape[1],1)
    target_input_stds = known_inputs.std(dim=1, keepdim=True).tile(1, known_inputs.shape[1],1)
    norm_inputs = (known_inputs - target_input_means) / target_input_stds

    additional_inputs = inputs[1:, :, :] # (features, seq_len, num_sequences)
    means_of_additonal_inputs = additional_inputs.mean(dim=1, keepdim=True).tile(1, additional_inputs.shape[1],1)
    stds_of_additonal_inputs = additional_inputs.std(dim=1, keepdim=True).tile(1, additional_inputs.shape[1],1)
    
    full_inputs = torch.cat([norm_inputs, (additional_inputs-means_of_additonal_inputs)/stds_of_additonal_inputs], dim=0)
    full_inputs = full_inputs.transpose(0,-1)
    full_inputs = full_inputs.transpose(0,1) # (seq_len, num_sequences, features)
    full_inputs = full_inputs.unsqueeze(0) # (1, seq_len, num_sequences, features)

    seperator = torch.zeros((1, 1),dtype=torch.int) # (1, 1)
    tickers = torch.arange(inputs.shape[-1], device=inputs.device)
    tickers = tickers.unsqueeze(0) # (1, num_sequences)

    outputs = model(full_inputs, seperator, tickers).view(-1, len(args.indices_to_predict), inputs.shape[-1]) # (seq_len, targets*num_sequences)
    outputs = outputs[1:,:,:]

    if torch.isnan(outputs).any():
        print("NaN in outputs")
    target_input_means = target_input_means.transpose(0,1)
    target_input_stds = target_input_stds.transpose(0,1)
    preds = outputs * target_input_stds.tile(1, len(args.indices_to_predict), 1) + target_input_means.tile(1, len(args.indices_to_predict), 1)

    return preds.detach()

def adapt_longer_preds(long_preds, true_returns, days_predicted_ahead, days_to_adapt_to, checked_days):
    days = days_predicted_ahead
    long_preds = long_preds[-(checked_days+days-1):-(days-1), :] # (seq_len, num_sequences)
    long_preds = long_preds + 1.0
    for i in range(days-days_to_adapt_to):
        long_preds = long_preds / (true_returns[:,-(checked_days+days-1-i):-(days-1-i), :] + 1.0)
    long_preds = long_preds - 1.0
    return long_preds

if __name__ == "__main__":
    metadata_path = "good_models/model_metadata.json"
    model_path = "good_models/money_Money_former_DINT_Money_optim_Money_former_DINT_16_128_512_6_4_16_for_inference.pth"
    nr_of_days_to_check = 16
    args = load_metadata(metadata_path)

    data, true_prices = download_and_process_input_data(args) # (features, seq_len, num_sequences)
    data = data[:, -(args.seq_len+nr_of_days_to_check+max(args.indices_to_predict)):-1, :] # introducing shift of one day?
    true_prices = true_prices[:, -(args.seq_len+nr_of_days_to_check+max(args.indices_to_predict)):, :]
    true_returns = (true_prices[:,1:,:] - true_prices[:,:-1,:])/true_prices[:,:-1,:]
    true_returns_5 = (true_prices[:,5:,:] - true_prices[:,:-5,:])/true_prices[:,:-5,:]
    true_returns_10 = (true_prices[:,10:,:] - true_prices[:,:-10,:])/true_prices[:,:-10,:] 

    model = load_model(model_path, args)
    predictions = []
    for i in range(nr_of_days_to_check+max(args.indices_to_predict)-1):
        predictions.append(get_predictions(model, data[:, i:i+args.seq_len, :], args)[-1,:,:].unsqueeze(0)) # (seq_len, targets,num_sequences)
    predictions = torch.cat(predictions, dim=0) # (seq_len, targets,num_sequences)

    full_seq_preds_1d = get_predictions(model, data[:,-args.seq_len:, :], args)[:,0,:].unsqueeze(0) # (seq_len, num_sequences)
    full_seq_preds_5d = get_predictions(model, data[:,-args.seq_len-4:-4, :], args)[:,1,:].unsqueeze(0) # (seq_len, num_sequences)
    full_seq_preds_10d = get_predictions(model, data[:,-args.seq_len-9:-9, :], args)[:,2,:].unsqueeze(0) # (seq_len, num_sequences)

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
    # TODO compare taking only last pred, vs taking full sequence of preds (shouodl be the same-ish, otherwise, likely wrong info flow)
    # plt.plot(true_returns[0, -nr_of_days_to_check:, 0])
    # plt.plot(d1_predictions[:, :, 0])
    # plt.plot(full_seq_preds_1d[0,:,0])
    # plt.legend([f"true 1day {args.tickers[0]}", f"predictions 1day {args.tickers[0]}", f"predictions 5day {args.tickers[0]}", f"predictions 10day {args.tickers[0]}", f"full_seq_preds_1d {args.tickers[0]}"])
    # plt.show()

    # plt.plot(true_returns_5[0, -nr_of_days_to_check:, 0])
    # plt.plot(d5_predictions[:, :, 0])
    # plt.plot(full_seq_preds_5d[0,:,0])
    # plt.legend([f"true 5 days {args.tickers[0]}", f"predictions 5day {args.tickers[0]}", f"predictions 10day {args.tickers[0]}", f"full_seq_preds_5d {args.tickers[0]}"])
    # plt.show()

    # plt.plot(true_returns_10[0, -nr_of_days_to_check:, 0])
    # plt.plot(predictions[:-9, 2, 0])
    # plt.plot(full_seq_preds_10d[0,:,0])
    # plt.legend([f"true 10 days {args.tickers[0]}", f"predictions 10day {args.tickers[0]}", f"full_seq_preds_10d {args.tickers[0]}"])
    # plt.show()

    # plt.plot(true_returns[0, -nr_of_days_to_check:, 1])
    # plt.plot(d1_predictions[:, :, 1])
    # plt.plot(full_seq_preds_1d[0,:,1])
    # plt.legend([f"true 1day {args.tickers[1]}", f"predictions 1day {args.tickers[1]}", f"predictions 5day {args.tickers[1]}", f"predictions 10day {args.tickers[1]}", f"full_seq_preds_1d {args.tickers[1]}"])
    # plt.show()

    # plt.plot(true_returns_5[0, -nr_of_days_to_check:, 1])
    # plt.plot(d5_predictions[:, :, 1])
    # plt.plot(full_seq_preds_5d[0,:,1])
    # plt.legend([f"true 5 days {args.tickers[1]}", f"predictions 5day {args.tickers[1]}", f"predictions 10day {args.tickers[1]}", f"full_seq_preds_5d {args.tickers[1]}"])
    # plt.show()

    # plt.plot(true_returns_10[0, -nr_of_days_to_check:, 1])
    # plt.plot(predictions[:-9, 2, 1])
    # plt.plot(full_seq_preds_10d[0,:,1])
    # plt.legend([f"true 10 days {args.tickers[1]}", f"predictions 10day {args.tickers[1]}", f"full_seq_preds_10d {args.tickers[1]}"])
    # plt.show()

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

    # print(f"full seq 1d 1d:{MAE_fn(full_seq_preds_1d, true_returns[0, -nr_of_days_to_check:, :])}")
    # print(f"full seq 5d 5d:{MAE_fn(full_seq_preds_5d, true_returns_5[0, -nr_of_days_to_check:, :])}")
    # print(f"full seq 10d 10d:{MAE_fn(full_seq_preds_10d, true_returns_10[0, -nr_of_days_to_check:, :])}")