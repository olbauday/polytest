# config.py
import math
import logging

# --- Version ---
VERSION = "4.2"    # Version identifier for the application (updated for change)

# --- File Paths ---
SAVE_FILE = f"trading_data_v{VERSION}.json" # Filename for saving application state
LOG_FILE = "trading_bot.log"               # Log file name

# --- Logging Configuration ---
LOG_LEVEL = logging.DEBUG # Level of detail: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

# --- Core Risk & Stop-Loss Parameters ---
RISK_PER_TRADE_PCT = 0.01       # Risk % of global balance on new directional trades
DIRECTIONAL_STOP_LOSS_PCT = 0.15 # Base stop loss % below entry/avg price for directional trades
ACCUMULATION_STOP_LOSS_PCT = 0.10 # Tighter stop loss % used specifically when sizing Accumulation buys
                                 # Set to None to use DIRECTIONAL_STOP_LOSS_PCT
HEDGED_STOP_LOSS_PCT_BASIS = 0.05 # Exit HEDGED/COST_ARB if unrealized loss > 5% of total cost basis...
# --- Conditional Hold Threshold for Hedged Stop Loss ---
HEDGED_HOLD_AVG_COST_THRESHOLD = 0.85 # ... UNLESS the average cost per pair is BELOW this threshold.
                                     # 0.0 = always apply stop loss based on %; 1.0 = never apply % stop loss

# --- Rule Triggers ---
BUY_THRESHOLD = 0.45 # Price threshold for initial BUY entry (<= ASK price)

# --- Profit Taking (Absolute Price) ---
PROFIT_TAKE_PRICE_THRESHOLD = 0.55  # Trigger profit taking SELL if BID price > this value
PROFIT_TAKE_SELL_PCT = 0.50       # Sell this percentage of current shares when threshold is met

# --- Deprecated Profit Taking (Percentage Gain) ---
# SELL_THRESHOLDS = { # Profit Taking Tiers (Price >= Avg Cost + % Increase) -> Sell % of CURRENT holdings
#     0.05: 0.30, # Sell 30% if price up 5% from avg cost
#     0.10: 0.20, # Sell additional 20% (of remaining) if price up 10% from avg cost
#     0.15: 0.50, # Sell remaining 50% (of remaining) if price up 15% from avg cost
# } # NOTE: This is no longer used by the primary profit taking rule (Priority 5)

# --- Other Rule Triggers ---
ACCUMULATION_DROP_THRESHOLD = 0.08 # Accumulate more if ASK price drops >= 8% below avg cost
COST_BASIS_ARB_THRESHOLD = 1.00 # Qualify for COST_BASIS_ARB state if AvgCost_Y + AvgCost_N < this value
HEDGE_PRICE_DROP_THRESHOLD = 0.12 # Hedge DIRECTIONAL if BID price drops >= 12% against the larger position

# --- Sizing Overlays ---
# ADV Cap - Max allocation % of total balance based on market liquidity estimate
ADV_ALLOCATION_MAP = {
    # ADV $: Max % of Balance
    1000: 0.05,     # Up to $1k ADV -> Max 5% of balance
    5000: 0.10,     # Up to $5k ADV -> Max 10% of balance
    10000: 0.15,    # Up to $10k ADV -> Max 15% of balance
    50000: 0.20,    # Up to $50k ADV -> Max 20% of balance
    100000: 0.30,   # Up to $100k ADV -> Max 30% of balance
    float('inf'): 0.40, # Effectively unlimited ADV -> Max 40% of balance
}
# Specific Sizing Percentages (subject to ADV caps and risk rules)
COST_ARB_ACCUM_SIZE_PCT_OF_BALANCE = 0.10 # Target size for Cost Basis Arb accumulation buys
ARB_BUY_SIZE_PCT_OF_BALANCE = 0.10      # Target size for Market Price Arb buys

# --- Market Price Arbitrage ---
ARB_THRESHOLD = 0.005 # Min price sum deviation from 1.00 for market arb action (e.g., > 1.005 or < 0.995)

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