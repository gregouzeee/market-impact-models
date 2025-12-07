"""
Calibrate market impact model parameters from order book data.

This script estimates the parameters (k, η, φ, ψ) of the Almgren-Chriss
power-law model using order book snapshots.

References:
- Almgren et al. (2005): Direct Estimation of Equity Market Impact
- Kyle (1985): Continuous Auctions and Insider Trading
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import seaborn as sns


class MarketImpactCalibrator:
    """Calibrate market impact parameters from order book data."""

    def __init__(self, orderbook_file, daily_volume):
        """
        Initialize calibrator.

        Parameters:
        -----------
        orderbook_file : str
            Path to JSON file with order book snapshots
        daily_volume : float
            Average daily trading volume (for participation rate calculation)
        """
        self.orderbook_file = orderbook_file
        self.daily_volume = daily_volume
        self.snapshots = self._load_snapshots()

    def _load_snapshots(self):
        """Load order book snapshots from JSON."""
        with open(self.orderbook_file, 'r') as f:
            snapshots = json.load(f)
        print(f"✅ Loaded {len(snapshots)} order book snapshots")
        return snapshots

    def estimate_spread_cost(self, reference_size=5.0):
        """
        Estimate proportional costs (ψ) from EFFECTIVE bid-ask spread.

        ψ represents the spread + trading fees.

        Note: We use EFFECTIVE spread (round-trip cost of a reference order)
        instead of quoted spread (best bid - best ask) because:
        - Quoted spread can be at tick minimum (0.01 USD for BTCUSDT)
        - Effective spread captures real execution costs by walking the book

        Parameters:
        -----------
        reference_size : float
            Size of reference order for spread calculation (default: 0.1 BTC)
        """
        quoted_spreads = []
        effective_spreads = []

        for snapshot in self.snapshots:
            bids = np.array(snapshot['bids'], dtype=float)
            asks = np.array(snapshot['asks'], dtype=float)

            if len(bids) == 0 or len(asks) == 0:
                continue

            best_bid = bids[0, 0]
            best_ask = asks[0, 0]
            mid_price = (best_bid + best_ask) / 2

            # Quoted spread (may be too small)
            quoted_spread_prop = (best_ask - best_bid) / mid_price
            quoted_spreads.append(quoted_spread_prop)

            # Effective spread: simulate round-trip order
            # 1. Simulate BUY order (walk up the asks)
            remaining = reference_size
            total_cost = 0.0
            for price, quantity in asks:
                if remaining <= 0:
                    break
                executed = min(remaining, quantity)
                total_cost += executed * price
                remaining -= executed

            if remaining > 0:
                # Not enough liquidity for buy side
                continue

            avg_buy_price = total_cost / reference_size

            # 2. Simulate SELL order (walk down the bids)
            remaining = reference_size
            total_revenue = 0.0
            for price, quantity in bids:
                if remaining <= 0:
                    break
                executed = min(remaining, quantity)
                total_revenue += executed * price
                remaining -= executed

            if remaining > 0:
                # Not enough liquidity for sell side
                continue

            avg_sell_price = total_revenue / reference_size

            # Effective spread = half of round-trip cost
            effective_spread_prop = (avg_buy_price - avg_sell_price) / 2 / mid_price
            effective_spreads.append(effective_spread_prop)

        psi_quoted = np.median(quoted_spreads) if quoted_spreads else 0
        psi_effective = np.median(effective_spreads) if effective_spreads else 0

        print(f"\n📊 Proportional costs (ψ):")
        print(f"   Quoted spread (best bid-ask):    {psi_quoted:.6f} ({psi_quoted*10000:.2f} bps)")
        print(f"   Effective spread ({reference_size} BTC round-trip): {psi_effective:.6f} ({psi_effective*10000:.2f} bps)")
        print(f"   → Using EFFECTIVE spread for calibration")

        return psi_effective

    def estimate_kyle_lambda(self, max_trade_size=100):
        """
        Estimate Kyle's lambda (permanent impact parameter k).

        Kyle's lambda measures price impact per unit volume:
        ΔP = λ × Q

        Parameters:
        -----------
        max_trade_size : float
            Maximum trade size to test (in base currency units)

        Returns:
        --------
        lambda_kyle : float
            Estimated Kyle's lambda (≈ permanent impact k)
        """
        trade_sizes = np.linspace(1, max_trade_size, 20)
        price_impacts = []

        print(f"\n📈 Estimating Kyle's lambda (permanent impact)...")

        for snapshot in self.snapshots[:min(100, len(self.snapshots))]:  # Use first 100 snapshots
            bids = np.array(snapshot['bids'], dtype=float)
            asks = np.array(snapshot['asks'], dtype=float)

            if len(bids) == 0 or len(asks) == 0:
                continue

            mid_price = (bids[0, 0] + asks[0, 0]) / 2

            # Simulate buy orders (walk up the ask side)
            for size in trade_sizes:
                remaining = size
                total_cost = 0.0

                for price, quantity in asks:
                    if remaining <= 0:
                        break
                    executed = min(remaining, quantity)
                    total_cost += executed * price
                    remaining -= executed

                if remaining <= 0:  # Sufficient liquidity
                    avg_price = total_cost / size
                    impact = (avg_price - mid_price) / mid_price
                    price_impacts.append({'size': size, 'impact': impact})

        # Regression: impact = λ × size
        df = pd.DataFrame(price_impacts)
        df_grouped = df.groupby('size')['impact'].mean().reset_index()

        slope, intercept, r_value, p_value, std_err = linregress(
            df_grouped['size'], df_grouped['impact']
        )

        lambda_kyle = slope

        print(f"   Kyle's lambda (λ): {lambda_kyle:.8f}")
        print(f"   R² = {r_value**2:.4f}")
        print(f"   For 100 units: impact ≈ {lambda_kyle * 100 * 10000:.2f} bps")

        return lambda_kyle, df_grouped

    def estimate_temporary_impact(self, test_sizes=None):
        """
        Estimate temporary impact function: L(ρ) = η|ρ|^(1+φ) + ψ|ρ|

        Parameters:
        -----------
        test_sizes : list of float
            Trade sizes to test (default: [1, 2, 5, 10, 20, 50, 100])

        Returns:
        --------
        dict with calibrated parameters (η, φ, ψ)
        """
        if test_sizes is None:
            test_sizes = [1, 2, 5, 10, 20, 50, 100]

        print(f"\n💥 Estimating temporary impact function...")
        print(f"   Testing {len(test_sizes)} different trade sizes")

        impact_data = []

        for snapshot in self.snapshots:
            bids = np.array(snapshot['bids'], dtype=float)
            asks = np.array(snapshot['asks'], dtype=float)

            if len(bids) == 0 or len(asks) == 0:
                continue

            mid_price = (bids[0, 0] + asks[0, 0]) / 2

            for size in test_sizes:
                # Participation rate
                rho = size / self.daily_volume

                # Simulate market buy order
                remaining = size
                total_cost = 0.0

                for price, quantity in asks:
                    if remaining <= 0:
                        break
                    executed = min(remaining, quantity)
                    total_cost += executed * price
                    remaining -= executed

                if remaining <= 0:  # Sufficient liquidity
                    avg_price = total_cost / size
                    # Temporary impact (total slippage)
                    impact = (avg_price - mid_price) / mid_price

                    impact_data.append({
                        'size': size,
                        'rho': rho,
                        'impact': impact
                    })

        df = pd.DataFrame(impact_data)
        df_grouped = df.groupby(['size', 'rho'])['impact'].mean().reset_index()

        print(f"   Collected {len(df)} impact measurements")
        print(f"   Average samples per size: {len(df) / len(test_sizes):.0f}")

        # Non-linear regression: impact = η × ρ^(1+φ) + ψ × ρ
        def power_law_impact(rho, eta, phi, psi):
            """Power-law impact function."""
            return eta * np.abs(rho)**(1 + phi) + psi * np.abs(rho)

        # Initial guess
        p0 = [0.01, 0.5, 0.0001]  # [η, φ, ψ]

        try:
            # Fit the model
            params, covariance = curve_fit(
                power_law_impact,
                df_grouped['rho'],
                df_grouped['impact'],
                p0=p0,
                bounds=([0, 0.1, 0], [1, 1.5, 0.01]),  # Constrain parameters
                maxfev=10000
            )

            eta, phi, psi_fitted = params
            perr = np.sqrt(np.diag(covariance))

            print(f"\n✅ Calibration successful!")
            print(f"   η (execution cost coeff): {eta:.6f} ± {perr[0]:.6f}")
            print(f"   φ (power exponent):       {phi:.4f} ± {perr[1]:.4f}")
            print(f"   ψ (proportional costs):   {psi_fitted:.6f} ± {perr[2]:.6f} ({psi_fitted*10000:.2f} bps)")

            # Goodness of fit
            y_pred = power_law_impact(df_grouped['rho'], *params)
            ss_res = np.sum((df_grouped['impact'] - y_pred)**2)
            ss_tot = np.sum((df_grouped['impact'] - df_grouped['impact'].mean())**2)
            r_squared = 1 - (ss_res / ss_tot)

            print(f"   R² = {r_squared:.4f}")

            return {
                'eta': eta,
                'phi': phi,
                'psi': psi_fitted,
                'eta_std': perr[0],
                'phi_std': perr[1],
                'psi_std': perr[2],
                'r_squared': r_squared,
                'data': df_grouped
            }

        except RuntimeError as e:
            print(f"❌ Calibration failed: {e}")
            return None

    def plot_impact_curve(self, calibration_result):
        """Plot the calibrated impact curve."""
        if calibration_result is None:
            print("No calibration result to plot")
            return

        df = calibration_result['data']
        eta = calibration_result['eta']
        phi = calibration_result['phi']
        psi = calibration_result['psi']

        # Generate smooth curve
        rho_smooth = np.linspace(df['rho'].min(), df['rho'].max(), 100)
        impact_smooth = eta * rho_smooth**(1 + phi) + psi * rho_smooth

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Impact vs participation rate
        ax = axes[0]
        ax.scatter(df['rho'] * 100, df['impact'] * 10000, alpha=0.6, label='Observed')
        ax.plot(rho_smooth * 100, impact_smooth * 10000, 'r-', linewidth=2,
                label=f'Fitted: η={eta:.4f}, φ={phi:.2f}')
        ax.set_xlabel('Participation rate (%)')
        ax.set_ylabel('Market impact (bps)')
        ax.set_title('Market Impact vs Participation Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Log-log plot (to see power-law)
        ax = axes[1]
        ax.scatter(np.log(df['rho']), np.log(df['impact']), alpha=0.6, label='Observed')
        ax.plot(np.log(rho_smooth), np.log(impact_smooth), 'r-', linewidth=2,
                label=f'Power-law: φ={phi:.2f}')
        ax.set_xlabel('log(ρ)')
        ax.set_ylabel('log(impact)')
        ax.set_title('Log-Log Plot (slope ≈ 1+φ)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/impact_calibration.png', dpi=150)
        print(f"\n📊 Plot saved to 'results/impact_calibration.png'")
        plt.show()


def main():
    """Main calibration workflow."""
    import os

    # Check if order book data exists
    orderbook_dir = 'data/orderbook'
    if not os.path.exists(orderbook_dir):
        print(f"❌ No order book data found in {orderbook_dir}")
        print(f"   Run 'python scripts/collect_orderbook.py' first to collect data")
        return

    # Find most recent order book file
    files = [f for f in os.listdir(orderbook_dir) if f.endswith('.json')]
    if not files:
        print(f"❌ No JSON files found in {orderbook_dir}")
        return

    orderbook_file = os.path.join(orderbook_dir, sorted(files)[-1])
    print(f"📂 Using order book data: {orderbook_file}")

    # Load daily volume from historical data
    try:
        df_params = pd.read_parquet('data/processed/market_parameters.parquet')
        params = df_params[df_params['symbol'] == 'BTCUSDT'].iloc[0]
        daily_volume = params['volume_per_day']
        print(f"📊 Daily volume (from historical data): {daily_volume:,.0f} units/day")
    except Exception as e:
        print(f"⚠️ Could not load historical volume: {e}")
        daily_volume = 25000  # Default estimate
        print(f"   Using default: {daily_volume:,.0f} units/day")

    # Create calibrator
    calibrator = MarketImpactCalibrator(orderbook_file, daily_volume)

    # 1. Estimate spread (ψ)
    psi = calibrator.estimate_spread_cost()

    # 2. Estimate Kyle's lambda (k)
    lambda_kyle, kyle_data = calibrator.estimate_kyle_lambda(max_trade_size=100)

    # 3. Estimate temporary impact (η, φ)
    calibration = calibrator.estimate_temporary_impact()

    if calibration:
        # Plot results
        calibrator.plot_impact_curve(calibration)

        # Save calibrated parameters
        results = {
            'psi': psi,
            'k': lambda_kyle,
            'eta': calibration['eta'],
            'phi': calibration['phi'],
            'r_squared': calibration['r_squared'],
            'daily_volume': daily_volume,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        os.makedirs('results', exist_ok=True)
        with open('results/calibrated_parameters.json', 'w') as f:
            json.dump(results, f, indent=2, default=float)

        print(f"\n💾 Calibrated parameters saved to 'results/calibrated_parameters.json'")

        print("\n" + "="*70)
        print("SUMMARY - Calibrated Parameters:")
        print("="*70)
        print(f"  k  (permanent impact)   = {lambda_kyle:.8f}")
        print(f"  η  (execution cost)     = {calibration['eta']:.6f}")
        print(f"  φ  (power exponent)     = {calibration['phi']:.4f}")
        print(f"  ψ  (proportional costs) = {psi:.6f} ({psi*10000:.2f} bps)")
        print(f"  R² (goodness of fit)    = {calibration['r_squared']:.4f}")
        print("="*70)


if __name__ == '__main__':
    main()
