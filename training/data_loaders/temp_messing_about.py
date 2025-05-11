import yfinance as yf

ticker = "AAPL"

data = yf.Ticker(ticker=ticker).history(period="max",interval="1d", auto_adjust=True, actions=True)

print(data)
ticker_info = yf.Ticker(ticker=ticker).info
# for key in ticker_info:
#     print(f"{key}:{ticker_info[key]}")

# print(yf.Ticker(ticker=ticker)._price_history)
# print(yf.Ticker(ticker=ticker).get_analyst_price_targets())
# print(yf.Ticker(ticker=ticker).earnings)
# print(yf.Ticker(ticker=ticker).get_earnings_history())
# print(yf.Ticker(ticker=ticker).income_stmt)
