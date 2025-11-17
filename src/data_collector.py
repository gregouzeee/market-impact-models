"""Automatization script in order to collect data about CS2 skin market
@Author : pier8o"""

import requests
import json
from datetime import datetime as dt
import os
import time
import pandas as pd
import s3fs

# Creates filesystem object
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
BUCKET_OUT = "pier8o"
FILE_KEY_OUT_S3 = "cs2_market_data/cs2_market_data_ak47.csv"
FILE_PATH_OUT_S3 = BUCKET_OUT + "/" + FILE_KEY_OUT_S3
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

# Sets the frequency of requests and the delay between them to avoid overloading.
NEW_DATA_TIME = 3600
NO_SPAMMING_TIME = 0.1

API_KEY = "Go see if I'm over there!"

# The following list contains the name of the skins we want to obtain the information.
items = [
    # Safari Mesh
    'AK-47 | Safari Mesh (Factory New)',
    'AK-47 | Safari Mesh (Minimal Wear)',
    'AK-47 | Safari Mesh (Field-Tested)',
    'AK-47 | Safari Mesh (Well-Worn)',
    'AK-47 | Safari Mesh (Battle-Scarred)',
    
    # Olive Polycam
    'AK-47 | Olive Polycam (Factory New)',
    'AK-47 | Olive Polycam (Minimal Wear)',
    'AK-47 | Olive Polycam (Field-Tested)',
    'AK-47 | Olive Polycam (Well-Worn)',
    'AK-47 | Olive Polycam (Battle-Scarred)',
    
    # Baroque Purple
    'AK-47 | Baroque Purple (Factory New)',
    'AK-47 | Baroque Purple (Minimal Wear)',
    'AK-47 | Baroque Purple (Field-Tested)',
    'AK-47 | Baroque Purple (Well-Worn)',
    'AK-47 | Baroque Purple (Battle-Scarred)',
    
    # Varicamo Grey
    'AK-47 | Varicamo Grey (Factory New)',
    'AK-47 | Varicamo Grey (Minimal Wear)',
    'AK-47 | Varicamo Grey (Field-Tested)',
    'AK-47 | Varicamo Grey (Well-Worn)',
    'AK-47 | Varicamo Grey (Battle-Scarred)',
    
    # Jungle Spray
    'AK-47 | Jungle Spray (Factory New)',
    'AK-47 | Jungle Spray (Minimal Wear)',
    'AK-47 | Jungle Spray (Field-Tested)',
    'AK-47 | Jungle Spray (Well-Worn)',
    'AK-47 | Jungle Spray (Battle-Scarred)',
    
    # Safety Net
    'AK-47 | Safety Net (Factory New)',
    'AK-47 | Safety Net (Minimal Wear)',
    'AK-47 | Safety Net (Field-Tested)',
    'AK-47 | Safety Net (Well-Worn)',
    'AK-47 | Safety Net (Battle-Scarred)',
    
    # Midnight Laminate
    'AK-47 | Midnight Laminate (Factory New)',
    'AK-47 | Midnight Laminate (Minimal Wear)',
    'AK-47 | Midnight Laminate (Field-Tested)',
    'AK-47 | Midnight Laminate (Well-Worn)',
    'AK-47 | Midnight Laminate (Battle-Scarred)',
    
    # Slate
    'AK-47 | Slate (Factory New)',
    'AK-47 | Slate (Minimal Wear)',
    'AK-47 | Slate (Field-Tested)',
    'AK-47 | Slate (Well-Worn)',
    'AK-47 | Slate (Battle-Scarred)',
    
    # Emerald Pinstripe
    'AK-47 | Emerald Pinstripe (Factory New)',
    'AK-47 | Emerald Pinstripe (Minimal Wear)',
    'AK-47 | Emerald Pinstripe (Field-Tested)',
    'AK-47 | Emerald Pinstripe (Well-Worn)',
    'AK-47 | Emerald Pinstripe (Battle-Scarred)',
    
    # Rat Rod
    'AK-47 | Rat Rod (Factory New)',
    'AK-47 | Rat Rod (Minimal Wear)',
    'AK-47 | Rat Rod (Field-Tested)',
    'AK-47 | Rat Rod (Well-Worn)',
    'AK-47 | Rat Rod (Battle-Scarred)',
    
    # Steel Delta
    'AK-47 | Steel Delta (Factory New)',
    'AK-47 | Steel Delta (Minimal Wear)',
    'AK-47 | Steel Delta (Field-Tested)',
    'AK-47 | Steel Delta (Well-Worn)',
    'AK-47 | Steel Delta (Battle-Scarred)',
    
    # Uncharted
    'AK-47 | Uncharted (Factory New)',
    'AK-47 | Uncharted (Minimal Wear)',
    'AK-47 | Uncharted (Field-Tested)',
    'AK-47 | Uncharted (Well-Worn)',
    'AK-47 | Uncharted (Battle-Scarred)',
    
    # Wintergreen
    'AK-47 | Wintergreen (Factory New)',
    'AK-47 | Wintergreen (Minimal Wear)',
    'AK-47 | Wintergreen (Field-Tested)',
    'AK-47 | Wintergreen (Well-Worn)',
    'AK-47 | Wintergreen (Battle-Scarred)',
    
    # Elite Build
    'AK-47 | Elite Build (Factory New)',
    'AK-47 | Elite Build (Minimal Wear)',
    'AK-47 | Elite Build (Field-Tested)',
    'AK-47 | Elite Build (Well-Worn)',
    'AK-47 | Elite Build (Battle-Scarred)',
    
    # Green Laminate
    'AK-47 | Green Laminate (Factory New)',
    'AK-47 | Green Laminate (Minimal Wear)',
    'AK-47 | Green Laminate (Field-Tested)',
    'AK-47 | Green Laminate (Well-Worn)',
    'AK-47 | Green Laminate (Battle-Scarred)',
    
    # Ice Coaled
    'AK-47 | Ice Coaled (Factory New)',
    'AK-47 | Ice Coaled (Minimal Wear)',
    'AK-47 | Ice Coaled (Field-Tested)',
    'AK-47 | Ice Coaled (Well-Worn)',
    'AK-47 | Ice Coaled (Battle-Scarred)',
    
    # The Outsiders
    'AK-47 | The Outsiders (Factory New)',
    'AK-47 | The Outsiders (Minimal Wear)',
    'AK-47 | The Outsiders (Field-Tested)',
    'AK-47 | The Outsiders (Well-Worn)',
    'AK-47 | The Outsiders (Battle-Scarred)',
    
    # Phantom Disruptor
    'AK-47 | Phantom Disruptor (Factory New)',
    'AK-47 | Phantom Disruptor (Minimal Wear)',
    'AK-47 | Phantom Disruptor (Field-Tested)',
    'AK-47 | Phantom Disruptor (Well-Worn)',
    'AK-47 | Phantom Disruptor (Battle-Scarred)',
    
    # Frontside Misty
    'AK-47 | Frontside Misty (Factory New)',
    'AK-47 | Frontside Misty (Minimal Wear)',
    'AK-47 | Frontside Misty (Field-Tested)',
    'AK-47 | Frontside Misty (Well-Worn)',
    'AK-47 | Frontside Misty (Battle-Scarred)',
    
    # Point Disarray
    'AK-47 | Point Disarray (Factory New)',
    'AK-47 | Point Disarray (Minimal Wear)',
    'AK-47 | Point Disarray (Field-Tested)',
    'AK-47 | Point Disarray (Well-Worn)',
    'AK-47 | Point Disarray (Battle-Scarred)',
    
    # Inheritance
    'AK-47 | Inheritance (Factory New)',
    'AK-47 | Inheritance (Minimal Wear)',
    'AK-47 | Inheritance (Field-Tested)',
    'AK-47 | Inheritance (Well-Worn)',
    'AK-47 | Inheritance (Battle-Scarred)',
    
    # Asiimov
    'AK-47 | Asiimov (Factory New)',
    'AK-47 | Asiimov (Minimal Wear)',
    'AK-47 | Asiimov (Field-Tested)',
    'AK-47 | Asiimov (Well-Worn)',
    'AK-47 | Asiimov (Battle-Scarred)',
    
    # Neon Rider
    'AK-47 | Neon Rider (Factory New)',
    'AK-47 | Neon Rider (Minimal Wear)',
    'AK-47 | Neon Rider (Field-Tested)',
    'AK-47 | Neon Rider (Well-Worn)',
    'AK-47 | Neon Rider (Battle-Scarred)',
    
    # Aquamarine Revenge
    'AK-47 | Aquamarine Revenge (Factory New)',
    'AK-47 | Aquamarine Revenge (Minimal Wear)',
    'AK-47 | Aquamarine Revenge (Field-Tested)',
    'AK-47 | Aquamarine Revenge (Well-Worn)',
    'AK-47 | Aquamarine Revenge (Battle-Scarred)',
    
    # Nightwish
    'AK-47 | Nightwish (Factory New)',
    'AK-47 | Nightwish (Minimal Wear)',
    'AK-47 | Nightwish (Field-Tested)',
    'AK-47 | Nightwish (Well-Worn)',
    'AK-47 | Nightwish (Battle-Scarred)'
]

def prepare_orderbook(raw_api_data, item_name):
    """
    Transforms the brute data of the API into a DataFrame row, in order to create the database.

    raw_api_data:
        The full JSON response of the API.
    item_name:
        The name of the item that we have the information in the JSON.
    """
    time_stamp = dt.now().isoformat()   # Gets the actual time

    histogram = raw_api_data.get('histogram', {}) # Gets the 'histogram' section of the JSON.
    
    # Exctracts arrays
    buy_orders = histogram.get('buy_order_array', [])
    sell_orders = histogram.get('sell_order_array', [])

    try:
        # Gets the bid-ask spray from the JSON. In case of missing data, the minus operation can make the program crash.
        spread = histogram.get('lowest_sell_order', 0) - histogram.get('highest_buy_order', 0)
    except:
        print(f"Failed to get the spread for {item_name} at {time_stamp}")
        spread = None
    
    row = {
        # 'Index' columns
        'timestamp': time_stamp,
        'item_name': item_name,
        
        # Important requests (for faster operations)
        'highest_buy': histogram.get('highest_buy_order'),
        'lowest_sell': histogram.get('lowest_sell_order'),
        'spread': spread,
        'total_buy_quantity': histogram.get('buy_order_summary', {}).get('quantity', 0),
        'total_sell_quantity': histogram.get('sell_order_summary', {}).get('quantity', 0),
        
        # Full order book in JSON. For further analysis.
        'buy_orders_json': json.dumps(buy_orders),
        'sell_orders_json': json.dumps(sell_orders),
    }
    
    return row


def get_from_steam_api(item: str):
    """Does the API request for a selected item."""
    url = "https://api.steamapis.com/market/item/730/" + item + "?api_key=" + API_KEY
    response = requests.get(url)
    data = response.json()
    return data


def collect_and_save():
    rows = []

    for item in items:
        # Gets the API data.
        raw_data = get_from_steam_api(item)  # Votre fonction API

        # Prepares the orderbook.
        row = prepare_orderbook(raw_data, item)
        rows.append(row)

        time.sleep(NO_SPAMMING_TIME)

    df = pd.DataFrame(rows)     # Creates the dataframe.

    # Saves it into a file on the SSPCLoud.
    file_exists = fs.exists(FILE_PATH_OUT_S3)
    mode = 'a' if file_exists else 'w'
    with fs.open(FILE_PATH_OUT_S3, mode) as file_out:
        df.to_csv(file_out, mode='a', header=False, index=False)

    print(f"✓ {len(df)} snapshots sauvegardés")


while True:
    time.sleep(NEW_DATA_TIME)
    collect_and_save()

    pass


"""
# Comment lire et utiliser les données après
def analyze_item_history(item_name, start_date, end_date):
    
    Analyse l'historique d'un item sur une période.
    
    # Lire le Parquet (filtré automatiquement par les partitions date)
    df = pd.read_parquet('market_data/')
    
    # Filtrer par item et période
    mask = (
        (df['item_name'] == item_name) &
        (df['timestamp'] >= start_date) &
        (df['timestamp'] <= end_date)
    )
    item_df = df[mask]
    
    # Analyser les métriques (colonnes normales - très rapide)
    print(f"Spread moyen : {item_df['spread'].mean():.2f}")
    print(f"Prix d'achat max : {item_df['highest_buy'].max():.2f}")
    print(f"Prix de vente min : {item_df['lowest_sell'].min():.2f}")
    
    # Si besoin d'analyser les order books complets
    # (parse le JSON seulement quand nécessaire)
    first_snapshot = item_df.iloc[0]
    buy_orders = json.loads(first_snapshot['buy_orders_json'])
    
    print(f"Premier buy order : {buy_orders[0]}")
    
    return item_df

"""
