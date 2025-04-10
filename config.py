# config.py
import math
import logging

# --- Version ---
VERSION = "4.3" # Updated version number for MIN_BUY_PRICE change and default alignment

# --- File Paths ---
SAVE_FILE = f"trading_data_v{VERSION}.json" # Filename for saving application state
LOG_FILE = "trading_bot.log"               # Log file name

# --- Logging Configuration ---
LOG_LEVEL = logging.DEBUG # Level of detail: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

# --- Core Risk & Stop-Loss Parameters ---
# --- Defaults aligned with suggested app.py settings (as of last discussion) ---
RISK_PER_TRADE_PCT = 0.04       # Default Risk % (app.py suggested default: 4%)
DIRECTIONAL_STOP_LOSS_PCT = 0.15 # Default Base stop loss % (app.py suggested default: 15%)
ACCUMULATION_STOP_LOSS_PCT = 0.08 # Default Tighter stop loss % for Accumulation (app.py suggested default: 8%)
                                 # Set to None to disable specific accum stop and use DIRECTIONAL_STOP_LOSS_PCT
HEDGED_STOP_LOSS_PCT_BASIS = 0.05 # Default Exit HEDGED/COST_ARB if loss > 5% cost basis (app.py suggested default: 5%)
# --- Conditional Hold Threshold for Hedged Stop Loss ---
HEDGED_HOLD_AVG_COST_THRESHOLD = 0.97 # Default: Apply % stop loss UNLESS avg cost per pair is BELOW this (app.py suggested default: 97c)
                                     # 0.0 = always apply % stop loss; 1.0 = never apply % stop loss

# --- Rule Triggers ---
# --- ADDED MIN_BUY_PRICE ---
MIN_BUY_PRICE = 0.40 # **NEW**: Minimum price (ask) required to initiate a BUY action (YES or NO)
# --- Updated BUY_THRESHOLD default and comment ---
BUY_THRESHOLD = 0.55 # Default *Maximum* price (ask) to initiate a BUY (app.py suggested default: 55c)
                     # Works with MIN_BUY_PRICE. Buy considered if MIN_BUY_PRICE <= ask <= BUY_THRESHOLD.

# --- Profit Taking (Absolute Price) ---
PROFIT_TAKE_PRICE_THRESHOLD = 0.85  # Default: Trigger profit taking SELL if BID price > this (app.py suggested default: 85c)
PROFIT_TAKE_SELL_PCT = 0.75       # Default: Sell this percentage of shares on profit take (app.py suggested default: 75%)

# --- Deprecated Profit Taking (Percentage Gain) ---
# SELL_THRESHOLDS = { ... } # Kept for reference, but marked as not used

# --- Other Rule Triggers ---
ACCUMULATION_DROP_THRESHOLD = 0.07 # Default: Accumulate if ASK drops >= 7% below avg cost (app.py suggested default: 7%)
COST_BASIS_ARB_THRESHOLD = 0.99 # Default: Qualify for COST_BASIS_ARB if AvgCost Sum < 99c (app.py suggested default: 99c)
HEDGE_PRICE_DROP_THRESHOLD = 0.10 # Default: Hedge DIRECTIONAL if BID drops >= 10% against position (app.py suggested default: 10%)

# --- Sizing Overlays ---
ADV_ALLOCATION_MAP = {
    # ADV $: Max % of Balance
    1000: 0.05, 5000: 0.10, 10000: 0.15, 50000: 0.20, 100000: 0.30, float('inf'): 0.40,
}
# --- Specific Sizing Percentages (Defaults aligned with suggested app.py settings) ---
COST_ARB_ACCUM_SIZE_PCT_OF_BALANCE = 0.05 # Default size for Cost Basis Arb accumulation (app.py suggested default: 5%)
ARB_BUY_SIZE_PCT_OF_BALANCE = 0.10      # Default size for Market Price Arb buys (app.py suggested default: 10%)

# --- Market Price Arbitrage ---
ARB_THRESHOLD = 0.005 # Default Min price sum deviation for market arb (app.py suggested default: 0.5%)

# --- Precision & Tolerances ---
SHARE_DECIMALS = 2 # Number of decimal places for share quantities
SHARE_ROUNDING_FUNC = lambda x: math.floor(x * (10**SHARE_DECIMALS)) / (10**SHARE_DECIMALS) # Rounding method (floor)
# Example alternative: round to nearest
# SHARE_ROUNDING_FUNC = lambda x: round(x, SHARE_DECIMALS)
ZERO_PRICE_THRESHOLD = 0.0001  # Threshold below which prices are treated as effectively zero
ZERO_SHARE_THRESHOLD = 0.0001  # Threshold below which share quantities are treated as effectively zero

# --- Position States ---
POSITION_STATES = ['FLAT', 'DIRECTIONAL_YES', 'DIRECTIONAL_NO', 'HEDGED', 'COST_BASIS_ARB']
# Tolerance for state determination (HEDGED/COST_ARB vs DIRECTIONAL)
HEDGE_IMBALANCE_TOLERANCE_SHARES = 1.0 # If abs(yes_shares - no_shares) <= this, consider it potentially HEDGED/COST_ARB

# --- Strategy Feature Flags (Example - Add flags as needed) ---
ENABLE_MARKET_ARBITRAGE = True
ENABLE_COST_BASIS_ARBITRAGE = True
ENABLE_ACCUMULATION = True
ENABLE_HEDGING = True
HEDGE_MATCH_SHARES = False # If True, when hedging, buy opposite shares only up to the amount of the primary holding
# HEDGE_UNWIND_SELL_PCT = 0.50 # Example: % of pairs to unwind in HEDGED state if unwind condition met (add if needed)