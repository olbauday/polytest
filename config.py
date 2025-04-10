# config.py
import math
import logging

# --- Version ---
# Incremented version for Percentage Profit Taking
VERSION = "4.5"

# --- File Paths ---
SAVE_FILE = f"trading_data_v{VERSION}.json"
LOG_FILE = "trading_bot.log"

# --- Logging Configuration ---
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

# --- Core Risk & Stop-Loss Parameters ---
RISK_PER_TRADE_PCT = 0.04
DIRECTIONAL_STOP_LOSS_PCT = 0.15
ACCUMULATION_STOP_LOSS_PCT = 0.08
HEDGED_STOP_LOSS_PCT_BASIS = 0.05
HEDGED_HOLD_AVG_COST_THRESHOLD = 0.97

# --- Rule Triggers ---
MIN_BUY_PRICE = 0.40
BUY_THRESHOLD = 0.55 # Max Ask price for initial buy

# --- Profit Taking / Scraping (Percentage Based) ---
# PROFIT_TAKE_PRICE_THRESHOLD = 0.85 # --- REMOVED / COMMENTED OUT --- Not used for percentage gain logic
PROFIT_TAKE_PERCENTAGE_GAIN_THRESHOLD = 0.15 # **NEW**: Trigger sell if BID price >= AvgCost * (1 + THIS_VALUE) (e.g., 0.15 = 15% gain)

# --- Settings for *how much* to sell once threshold is met ---
ENABLE_PROFIT_SCRAPE_HEDGE = True   # If True, use scraping logic (PROFIT_SCRAPE_SELL_PCT)
PROFIT_SCRAPE_SELL_PCT = 0.10      # Sell this SMALL percentage when scraping profit
PROFIT_TAKE_SELL_PCT = 0.75       # Sell this percentage if ENABLE_PROFIT_SCRAPE_HEDGE = False (normal profit take)

# --- Other Rule Triggers ---
ACCUMULATION_DROP_THRESHOLD = 0.07
COST_BASIS_ARB_THRESHOLD = 0.99
HEDGE_PRICE_DROP_THRESHOLD = 0.10

# --- Sizing Overlays ---
ADV_ALLOCATION_MAP = { 1000: 0.05, 5000: 0.10, 10000: 0.15, 50000: 0.20, 100000: 0.30, float('inf'): 0.40 }
COST_ARB_ACCUM_SIZE_PCT_OF_BALANCE = 0.05
ARB_BUY_SIZE_PCT_OF_BALANCE = 0.10

# --- Market Price Arbitrage ---
ARB_THRESHOLD = 0.005

# --- Precision & Tolerances ---
SHARE_DECIMALS = 2
SHARE_ROUNDING_FUNC = lambda x: math.floor(x * (10**SHARE_DECIMALS)) / (10**SHARE_DECIMALS)
ZERO_PRICE_THRESHOLD = 0.0001
ZERO_SHARE_THRESHOLD = 0.0001

# --- Position States ---
POSITION_STATES = ['FLAT', 'DIRECTIONAL_YES', 'DIRECTIONAL_NO', 'HEDGED', 'COST_BASIS_ARB']
HEDGE_IMBALANCE_TOLERANCE_SHARES = 1.0

# --- Strategy Feature Flags ---
ENABLE_MARKET_ARBITRAGE = True
ENABLE_COST_BASIS_ARBITRAGE = True
ENABLE_ACCUMULATION = True
ENABLE_HEDGING = True
HEDGE_MATCH_SHARES = False
# ENABLE_PROFIT_SCRAPE_HEDGE defined above