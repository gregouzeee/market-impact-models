"""
Script to collect order book data from Binance for market impact calibration.

This script collects order book snapshots at regular intervals to calibrate
the market impact model parameters (k, η, φ, ψ).
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
import os

class BinanceOrderBookCollector:
    """Collect order book data from Binance Public API."""

    def __init__(self, symbol='BTCUSDT', limit=5000):
        """
        Initialize the order book collector.

        Parameters:
        -----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        limit : int
            Number of order book levels to fetch (max 5000)
        """
        self.symbol = symbol
        self.limit = limit
        self.base_url = 'https://api.binance.com/api/v3'

    def get_orderbook_snapshot(self):
        """Fetch a single order book snapshot."""
        endpoint = f'{self.base_url}/depth'
        params = {
            'symbol': self.symbol,
            'limit': self.limit
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'bids': data['bids'],  # [[price, quantity], ...]
                'asks': data['asks'],
                'lastUpdateId': data['lastUpdateId']
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching order book: {e}")
            return None

    def collect_snapshots(self, duration_minutes=60, interval_seconds=10):
        """
        Collect order book snapshots over a period.

        Parameters:
        -----------
        duration_minutes : int
            Total duration to collect data (in minutes)
        interval_seconds : int
            Interval between snapshots (in seconds)

        Returns:
        --------
        list of dict
            List of order book snapshots
        """
        snapshots = []
        num_snapshots = (duration_minutes * 60) // interval_seconds

        print(f"📊 Collecting {num_snapshots} order book snapshots for {self.symbol}")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Interval: {interval_seconds} seconds")
        print(f"   Limit: {self.limit} levels\n")

        for i in range(num_snapshots):
            snapshot = self.get_orderbook_snapshot()

            if snapshot:
                snapshots.append(snapshot)

                if (i + 1) % 10 == 0:
                    print(f"   ✅ Collected {i + 1}/{num_snapshots} snapshots")

            time.sleep(interval_seconds)

        print(f"\n✅ Collection complete: {len(snapshots)} snapshots")
        return snapshots

    def save_snapshots(self, snapshots, output_dir='data/orderbook'):
        """
        Save snapshots to JSON file.

        - On SSP Cloud: saves locally AND uploads to S3
        - Locally: saves to local directory only
        """
        from pathlib import Path
        import sys
        sys.path.append(str(Path(__file__).parent))
        from s3_utils import is_ssp_cloud, get_s3_manager

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{output_dir}/{self.symbol}_orderbook_{timestamp}.json'

        # Save locally first
        with open(filename, 'w') as f:
            json.dump(snapshots, f, indent=2)

        print(f"💾 Saved locally: {filename}")

        # If on SSP Cloud, also upload to S3
        if is_ssp_cloud():
            print(f"☁️  Running on SSP Cloud - uploading to S3...")
            s3 = get_s3_manager()
            if s3:
                s3_key = f"market-impact-data/orderbook/{self.symbol}_orderbook_{timestamp}.json"
                try:
                    s3.upload_file(filename, s3_key)
                except Exception as e:
                    print(f"⚠️  S3 upload failed (continuing): {e}")

        return filename


def calculate_orderbook_metrics(snapshot):
    """Calculate key metrics from an order book snapshot."""

    # Convert to arrays
    bids = np.array(snapshot['bids'], dtype=float)
    asks = np.array(snapshot['asks'], dtype=float)

    # Best bid/ask
    best_bid = bids[0, 0] if len(bids) > 0 else None
    best_ask = asks[0, 0] if len(asks) > 0 else None

    if best_bid is None or best_ask is None:
        return None

    mid_price = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    spread_bps = (spread / mid_price) * 10000

    # Liquidity metrics
    bid_liquidity_10 = bids[:10, 1].sum()  # Top 10 levels
    ask_liquidity_10 = asks[:10, 1].sum()

    bid_liquidity_100 = bids[:100, 1].sum()  # Top 100 levels
    ask_liquidity_100 = asks[:100, 1].sum()

    # Order book imbalance
    total_bid_volume = bids[:, 1].sum()
    total_ask_volume = asks[:, 1].sum()
    imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

    return {
        'timestamp': snapshot['timestamp'],
        'mid_price': mid_price,
        'best_bid': best_bid,
        'best_ask': best_ask,
        'spread': spread,
        'spread_bps': spread_bps,
        'bid_liquidity_10': bid_liquidity_10,
        'ask_liquidity_10': ask_liquidity_10,
        'bid_liquidity_100': bid_liquidity_100,
        'ask_liquidity_100': ask_liquidity_100,
        'total_bid_volume': total_bid_volume,
        'total_ask_volume': total_ask_volume,
        'imbalance': imbalance
    }


def walk_the_book(orderbook_side, target_quantity):
    """
    Simulate market order execution by walking through the order book.

    Parameters:
    -----------
    orderbook_side : np.array
        Order book side (bids or asks) as [[price, quantity], ...]
    target_quantity : float
        Total quantity to execute

    Returns:
    --------
    dict with execution details
    """
    orders = np.array(orderbook_side, dtype=float)

    remaining = target_quantity
    total_cost = 0.0
    levels_used = 0

    for price, quantity in orders:
        if remaining <= 0:
            break

        executed = min(remaining, quantity)
        total_cost += executed * price
        remaining -= executed
        levels_used += 1

    if remaining > 0:
        # Not enough liquidity in the book
        avg_price = None
    else:
        avg_price = total_cost / target_quantity

    return {
        'target_quantity': target_quantity,
        'executed_quantity': target_quantity - remaining,
        'avg_price': avg_price,
        'levels_used': levels_used,
        'sufficient_liquidity': remaining <= 0
    }


if __name__ == '__main__':
    # Example usage
    collector = BinanceOrderBookCollector(symbol='BTCUSDT', limit=1000)

    # Collect snapshots for 1 hour at 10-second intervals
    snapshots = collector.collect_snapshots(duration_minutes=60, interval_seconds=10)

    # Save raw data
    filename = collector.save_snapshots(snapshots)

    # Calculate and display metrics for first snapshot
    if snapshots:
        metrics = calculate_orderbook_metrics(snapshots[0])
        print("\n📊 Sample metrics from first snapshot:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key:20s}: {value:,.4f}")
            else:
                print(f"   {key:20s}: {value}")

        # Simulate market impact for different sizes
        print("\n💥 Simulated market impact (buying):")
        snapshot = snapshots[0]
        asks = np.array(snapshot['asks'], dtype=float)
        mid_price = (float(snapshot['bids'][0][0]) + float(snapshot['asks'][0][0])) / 2

        for size in [1, 5, 10, 50, 100]:
            result = walk_the_book(asks, size)
            if result['avg_price']:
                impact = (result['avg_price'] - mid_price) / mid_price
                print(f"   {size:6.1f} BTC: avg_price=${result['avg_price']:,.2f}, "
                      f"impact={impact*10000:6.2f} bps, {result['levels_used']} levels")
            else:
                print(f"   {size:6.1f} BTC: insufficient liquidity")
