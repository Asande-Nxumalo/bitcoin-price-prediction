
import requests
import pandas as pd
import time

# Defining Binance settings
SYMBOL = "BTCUSDT"
INTERVAL = "4h"
BASE_URL = "https://api.binance.com/api/v3/klines"


# Function to download Bitcoin data
def download_bitcoin_data():

    # Displays a message in the terminal
    print("Downloading Bitcoin data...")

    # Creates an empty Python list
    all_data = []

    # Calculates the starting time (3 years ago) in milliseconds
    start_time = int(
        (pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=3)).timestamp() * 1000
    )

    # Downloading Bitcoin candles
    while True:

        # Dictionary containing Binance request parameters
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": start_time,
            "limit": 1000
        }

        # Sends request to Binance
        response = requests.get(
            BASE_URL,
            params=params,
            timeout=30
        )

        # Checks whether the HTTP request was successful
        response.raise_for_status()

        # Converts Binance response into Python data
        candles = response.json()

        # Stops if Binance returns no more candles
        if not candles:
            break

        # Adds downloaded candles to our list
        all_data.extend(candles)

        # Moves the starting time to the next candle
        start_time = candles[-1][0] + 1

        # Displays the number of downloaded candles
        print(f"Downloaded {len(all_data)} candles")

        # Stops if fewer than 1000 candles were returned
        if len(candles) < 1000:
            break

        # Pauses for 0.2 seconds
        time.sleep(0.2)

    # Names of the Binance data columns
    columns = [
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "number_of_trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]

    # Creates a pandas DataFrame
    df = pd.DataFrame(all_data, columns=columns)

    # Converts prices and volume into numeric values
    numeric_columns = [
        "open", "high", "low", "close", "volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column])

    # Converts timestamps from milliseconds into readable dates
    df["open_time"] = pd.to_datetime(
        df["open_time"], unit="ms", utc=True
    )

    # Removes repeated candles
    df = df.drop_duplicates(subset="open_time")

    # Arranges candles in chronological order
    df = df.sort_values("open_time")

    # Removes unfinished candles
    now = pd.Timestamp.now(tz="UTC")

    df = df[
        df["open_time"] + pd.Timedelta(hours=4) <= now
    ]

    # Saves the dataset
    df.to_csv("data/bitcoin_4h.csv", index=False)

    # Displays completion messages
    print("\nDownload completed!")
    print(f"Total candles: {len(df)}")
    print("Saved to data/bitcoin_4h.csv")


# Runs the function
if __name__ == "__main__":
    download_bitcoin_data()
