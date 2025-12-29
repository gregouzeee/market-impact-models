# Market Impact Models - Almgren-Chriss

**Authors**: Grégoire Marguier & Pierre Robin-Schnepf

**ENSAE Paris** - Python for Data Science (2025-2026)

---

## Research Question

> **How can we model and optimize transaction costs related to market impact when executing large orders?**

When an institutional investor executes a large order, they face a fundamental dilemma: executing quickly leads to significant market impact, while executing slowly exposes them to volatility risk.

This project implements the **Almgren-Chriss (2001) model** to solve this optimal execution problem, featuring:
- Multi-source data collection (Databento, Binance API)
- Empirical calibration of model parameters
- Execution strategy comparison (TWAP vs Optimal)

---

## Quick Setup (SSPCloud / Onyxia)

```bash
# 1. Clone the repository
git clone https://github.com/gregouzeee/market-impact-models.git
cd market-impact-models

# 2. Install dependencies
pip install -r requirements.txt
```

Notebooks are ready to run.

---

## Project Structure

```
market-impact-models/
├── notebooks/
│   ├── 00_rapport_final.ipynb        # Summary report (FR)
│   ├── 00_final_report.ipynb         # Final report (EN)
│   ├── 01_data_collection.ipynb      # Historical data collection
│   ├── 02_orderbook_collection.ipynb # Real-time orderbook collection
│   ├── 03_calibration.ipynb          # Parameter calibration
│   ├── 04_almgren_chriss_quadratic.ipynb  # AC quadratic model
│   └── 05_almgren_chriss_powerlaw.ipynb   # AC power-law model
├── src/                              # Python modules
├── data/                             # Data (generated)
├── results/                          # Results and figures
└── requirements.txt                  # Dependencies
```

---

## Notebooks

### Final Report
| Notebook | Description |
|----------|-------------|
| `00_final_report.ipynb` | **Summary notebook** - Start here |
| `00_rapport_final.ipynb` | French version |

### Full Pipeline (optional)
| # | Notebook | Description | Duration |
|---|----------|-------------|----------|
| 1 | `01_data_collection.ipynb` | OHLCV data (stocks + crypto) | ~5 min |
| 2 | `02_orderbook_collection.ipynb` | Binance orderbook snapshots | ~60 min* |
| 3 | `03_calibration.ipynb` | Calibrate η, k, ψ | ~2 min |
| 4 | `04_almgren_chriss_quadratic.ipynb` | AC model (analytical solution) | ~1 min |
| 5 | `05_almgren_chriss_powerlaw.ipynb` | AC power-law model | ~1 min |

*Configurable via `DURATION_MINUTES`

---

## Data Sources

| Source | Assets | Frequency | Period |
|--------|--------|-----------|--------|
| Databento (S3) | AAPL, MSFT, GOOG | 1 min | Jan-Jun 2025 |
| Binance API | BTC, ETH, SOL | 1 min | November 2025 |
| Binance API | Order book | Real-time | On demand |

**Note**: Stock data (Databento) is only accessible from SSPCloud. Crypto data works everywhere.

**Language**: Technical notebooks (01-05) are written in English. The summary notebook and README are available in both French and English.

---

## Key Results

- **Calibration**: R² = 0.95 (quadratic model)
- **Optimization savings**: 10-15% vs TWAP in medium urgency regime (κT ~ 1-3)
- **Insight**: On Binance, transaction fees (10 bps) dominate spread (~0 bps)

---

## References

1. Almgren, R., & Chriss, N. (2001). *Optimal execution of portfolio transactions*. Journal of Risk.
2. Kyle, A. S. (1985). *Continuous auctions and insider trading*. Econometrica.
3. Gatheral, J. (2010). *No-dynamic-arbitrage and market impact*. Quantitative Finance.

---

*ENSAE Paris - Python for Data Science (2025-2026)*
