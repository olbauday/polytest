# app.py
import streamlit as st
import pandas as pd
import io
import logging
import copy
import matplotlib.pyplot as plt
from datetime import datetime
import itertools # For Grid Search
import time # To estimate optimization time
import numpy as np # For linspace in grid search

# --- Import your custom modules ---
# Ensure these files are in the same directory
try:
    import strategy_engine
    import config # Import the base config to get defaults
    import utilsbacktest as utils # Use utils alias for clarity
except ImportError as e:
    st.error(f"Error importing local modules (strategy_engine.py, config.py, utilsbacktest.py): {e}")
    st.error("Make sure these files exist in the same directory as app.py")
    st.stop()

# --- Global Variables / Constants ---
INITIAL_BALANCE = 1000.00
SIMULATED_SPREAD_PCT = 0.005 # Example spread % (0.5%)
MARKET_NAME_DEFAULT = "BacktestMarket"
DEFAULT_MARKET_ADV = float('inf') # Assume high liquidity for backtesting unless specified otherwise

# --- Logging Setup ---
log_stream = io.StringIO()
# Get logger specific to this module to avoid interfering with Streamlit's root logger potentially
app_logger = logging.getLogger('backtester_app')
app_logger.handlers.clear()
app_logger.propagate = False # Prevent messages going to root logger setup by Streamlit
# Ensure log level is applied correctly from config
log_level_from_config = getattr(config, 'LOG_LEVEL', logging.INFO) # Default to INFO if not set
log_handler = logging.StreamHandler(log_stream)
log_formatter = logging.Formatter(config.LOG_FORMAT)
log_handler.setFormatter(log_formatter)
app_logger.addHandler(log_handler)
app_logger.setLevel(log_level_from_config)


# --- Batch Profitability Analysis ---
def calculate_batch_profitability(performance_df, batch_duration_days=14):
    """
    Analyzes profitability over fixed time batches.

    Args:
        performance_df (pd.DataFrame): DataFrame with 'Timestamp' (datetime)
                                      and 'Total Value' columns.
        batch_duration_days (int): Duration of each batch in days.

    Returns:
        tuple: (percentage_profitable, num_profitable, total_batches)
               Returns (0.0, 0, 0) if not enough data.
    """
    if performance_df is None or performance_df.empty or 'Timestamp' not in performance_df.columns or 'Total Value' not in performance_df.columns:
        app_logger.warning("Batch analysis skipped: Performance DataFrame is missing or invalid.")
        return 0.0, 0, 0

    # Ensure Timestamp is datetime type and set as index
    df = performance_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        try:
            # Convert Timestamp assuming it might be various formats (unix epoch, string)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df.dropna(subset=['Timestamp'], inplace=True) # Remove rows where conversion failed
            if df.empty:
                 app_logger.warning("Batch analysis skipped: No valid timestamps after conversion.")
                 return 0.0, 0, 0
        except Exception as e:
            app_logger.error(f"Error converting Timestamp column to datetime in batch analysis: {e}")
            return 0.0, 0, 0 # Cannot proceed without datetime index

    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True) # Ensure chronological order

    # Define the resampling frequency (e.g., '14D' for 14 days)
    resample_freq = f'{batch_duration_days}D'

    # Resample to get the first and last value within each period
    try:
        # Use 'Total Value' which reflects the portfolio value at that timestamp
        resampled = df['Total Value'].resample(resample_freq).agg(['first', 'last'])
        app_logger.debug(f"Resampled data for batch analysis:\n{resampled}")
    except ValueError as e:
         # Might happen if duration is invalid or index isn't datetime
         app_logger.error(f"Error resampling data with frequency '{resample_freq}': {e}")
         return 0.0, 0, 0
    except TypeError as e:
         app_logger.error(f"TypeError during resampling, possibly non-datetime index? Error: {e}")
         return 0.0, 0, 0


    # Drop periods where we don't have both a start and end value (e.g., incomplete first/last period if data ends mid-batch)
    resampled.dropna(subset=['first', 'last'], inplace=True)
    app_logger.debug(f"Resampled data after dropna:\n{resampled}")


    if resampled.empty:
        app_logger.warning(f"Batch analysis skipped: No complete {batch_duration_days}-day batches found in the data.")
        return 0.0, 0, 0 # Not enough data for even one complete batch

    # Calculate profit for each batch (End Value - Start Value)
    resampled['profit'] = resampled['last'] - resampled['first']

    # Count profitable batches (profit > tiny threshold to avoid float noise)
    num_profitable = (resampled['profit'] > 1e-6).sum()
    total_batches = len(resampled)

    if total_batches == 0:
        app_logger.warning("Batch analysis: Total complete batches is zero.") # Should be caught by empty check above, but safety
        return 0.0, 0, 0

    percentage_profitable = (num_profitable / total_batches) * 100
    app_logger.info(f"Batch analysis: {num_profitable} profitable batches out of {total_batches} ({percentage_profitable:.1f}%).")

    return percentage_profitable, num_profitable, total_batches


# --- Backtesting Core Logic ---
def run_backtest(df, initial_balance, current_config_dict):
    """
    Runs the backtesting simulation FOR A SINGLE PARAMETER CONFIGURATION.

    Args:
        df (pd.DataFrame): DataFrame with historical price data.
        initial_balance (float): Starting cash balance.
        current_config_dict (dict): Dictionary containing the specific parameter
                                     values for this run.

    Returns:
        tuple: (final_value, num_trades, performance_df, trade_log, final_portfolio, log_output)
               Returns None for final_value if backtest fails critically.
    """
    # --- Clear log buffer for this run ---
    log_stream.seek(0)
    log_stream.truncate(0)
    app_logger.info(f"--- Starting Single Backtest Run ---")
    # Log parameters used (handle potential complex objects like dictionaries)
    log_params = {k: (f"{v:.4f}" if isinstance(v, (float, np.float64)) else str(v)) for k, v in current_config_dict.items()} # Use str() for safety
    app_logger.info(f"Parameters: {log_params}")
    app_logger.info(f"Initial balance: {utils.format_currency(initial_balance)}")

    # --- Create a temporary config module/object for this run (or patch directly) ---
    # We will use the direct patching method below, as originally implemented.

    # --- Portfolio State Initialization ---
    portfolio = {
        'balance': initial_balance,
        'market_data': {'position_state': 'FLAT', 'directional_stop_loss': None},
        'stats': {'yes_shares': 0.0, 'no_shares': 0.0, 'yes_investment': 0.0, 'no_investment': 0.0, 'yes_avg_price': 0.0, 'no_avg_price': 0.0},
        'last_recommendation': None
    }
    performance_log = []
    trade_log = []

    # --- Identify Price Column ---
    # Prioritize columns exactly named 'price', then case-insensitive 'price'
    price_col = None
    if 'price' in df.columns:
        price_col = 'price'
    else:
        price_col = next((col for col in df.columns if 'price' in col.lower()), None)

    if not price_col:
        app_logger.error("Failed to find price column ('price' or contains 'price') in CSV.")
        return None, 0, pd.DataFrame(), [], portfolio, log_stream.getvalue()
    app_logger.info(f"Using price column: '{price_col}'")

    # --- Store original config values FOR strategy_engine.config ---
    # This is crucial for the patching mechanism to work correctly and be reversible
    original_config_values = {}
    for key in current_config_dict.keys():
        if hasattr(strategy_engine.config, key):
             original_config_values[key] = getattr(strategy_engine.config, key)
        # Note: We don't need to patch utilsbacktest.config as it directly uses config's SHARE_DECIMALS

    # --- Main Backtesting Loop ---
    try: # Wrap the main loop for better error catching during restore
        for index, row in df.iterrows():
            timestamp = row['Timestamp (UTC)']
            try:
                # Attempt conversion assuming it's Unix timestamp; handle other formats if necessary
                dt_object = datetime.utcfromtimestamp(float(timestamp))
            except ValueError:
                 app_logger.error(f"Could not parse timestamp: {timestamp} at index {index}. Skipping row.")
                 continue # Skip this row if timestamp is bad

            # --- 1. Simulate Market Prices ---
            # Use the identified price column
            try:
                yes_price_raw = float(row[price_col])
            except (ValueError, TypeError):
                app_logger.warning(f"Invalid price value '{row[price_col]}' at {dt_object}. Using 0.5.")
                yes_price_raw = 0.5 # Default to 0.5 on error? Or skip row?

            # Use current_config_dict values for thresholds during simulation
            zero_thresh = current_config_dict.get('ZERO_PRICE_THRESHOLD', config.ZERO_PRICE_THRESHOLD)
            share_decimals = current_config_dict.get('SHARE_DECIMALS', config.SHARE_DECIMALS)
            # Get rounding function if specified, otherwise use simple round
            share_rounding_func_from_config = current_config_dict.get('SHARE_ROUNDING_FUNC', config.SHARE_ROUNDING_FUNC)
            share_rounding_func = share_rounding_func_from_config if callable(share_rounding_func_from_config) else (lambda x: round(x, int(share_decimals)))


            yes_price = max(zero_thresh, min(1.0 - zero_thresh, yes_price_raw))
            no_price = 1.0 - yes_price

            # Simulate spread based on the *validated* yes_price
            half_spread = yes_price * (SIMULATED_SPREAD_PCT / 2.0)
            yes_ask = min(1.0 - zero_thresh, yes_price + half_spread)
            yes_bid = max(zero_thresh, yes_price - half_spread)

            # Simulate spread for NO side based on its price
            half_spread_no = no_price * (SIMULATED_SPREAD_PCT / 2.0)
            no_ask = min(1.0 - zero_thresh, no_price + half_spread_no)
            no_bid = max(zero_thresh, no_price - half_spread_no)

            # Ensure bid is never higher than ask after spread calculation
            yes_bid = min(yes_bid, yes_ask - zero_thresh)
            no_bid = min(no_bid, no_ask - zero_thresh)
            # Ensure prices didn't become invalid after adjustments
            yes_bid = max(zero_thresh, yes_bid)
            no_bid = max(zero_thresh, no_bid)
            yes_ask = max(zero_thresh, yes_ask)
            no_ask = max(zero_thresh, no_ask)


            market_adv = DEFAULT_MARKET_ADV # Keep it simple for backtesting

            app_logger.debug(f"--- Timestamp: {dt_object} ---")
            app_logger.debug(f"Raw Yes Price: {yes_price_raw:.4f} -> Validated: {yes_price:.4f}")
            app_logger.debug(f"Simulated Y_Bid:{yes_bid:.4f}, Y_Ask:{yes_ask:.4f} | N_Bid:{no_bid:.4f}, N_Ask:{no_ask:.4f}")

            # --- 2. Get Strategy Recommendation ---
            current_market_data = copy.deepcopy(portfolio['market_data'])
            current_stats = copy.deepcopy(portfolio['stats'])
            current_balance = portfolio['balance']

            recommendation = None # Initialize
            try:
                # --- Temporarily patch strategy_engine's config view ---
                for key, value in current_config_dict.items():
                    if hasattr(strategy_engine.config, key):
                        setattr(strategy_engine.config, key, value)

                # --- Call the strategy engine ---
                recommendation, analysis_details = strategy_engine.calculate_strategy_recommendation(
                    market_name=MARKET_NAME_DEFAULT,
                    yes_bid=yes_bid, yes_ask=yes_ask, no_bid=no_bid, no_ask=no_ask,
                    market_adv=market_adv, market_data=current_market_data,
                    stats=current_stats, current_balance=current_balance
                )
                portfolio['last_recommendation'] = recommendation

            except Exception as e:
                app_logger.error(f"Error during calculate_strategy_recommendation at {dt_object}: {e}", exc_info=True)
                # recommendation remains None
            finally:
                # --- Restore original config values ALWAYS ---
                for key, value in original_config_values.items():
                     if hasattr(strategy_engine.config, key):
                          setattr(strategy_engine.config, key, value)

            if recommendation is None: # Skip step if strategy failed
                 app_logger.warning(f"Skipping step at {dt_object} due to strategy calculation error.")
                 continue

            # --- 3. Simulate Trade Execution ---
            action = recommendation['action_type']
            side = recommendation['side']
            # Recommendation should provide rounded shares based on its *own* config view
            shares_rounded = recommendation['shares_rounded']
            price = recommendation['price'] # Ask for BUY, Bid for SELL, Sum for ARB
            executed_trade = False
            trade_details = {}

            # Use config values from the *current run's dictionary* for execution checks
            zero_share_thresh = current_config_dict.get('ZERO_SHARE_THRESHOLD', config.ZERO_SHARE_THRESHOLD)

            # Check if action is valid and shares > 0
            if action != 'HOLD' and shares_rounded > zero_share_thresh:
                app_logger.info(f"Attempting Action: {recommendation['display_text']}")
                cost_proceeds = 0.0
                current_stats_mut = portfolio['stats'] # Get mutable reference

                # --- BUY Simulation ---
                if action == 'BUY':
                    cost = shares_rounded * price # Price is Ask
                    if current_balance >= cost - zero_thresh: # Allow for tiny float errors
                        cost_proceeds = -cost
                        portfolio['balance'] += cost_proceeds
                        app_logger.debug(f"BUY {side}: Cost={cost:.4f}, New Balance={portfolio['balance']:.4f}")

                        # Update stats
                        if side == 'YES':
                            new_total_shares = current_stats_mut['yes_shares'] + shares_rounded
                            current_stats_mut['yes_investment'] += cost
                            current_stats_mut['yes_avg_price'] = current_stats_mut['yes_investment'] / new_total_shares if new_total_shares > zero_share_thresh else 0.0
                            current_stats_mut['yes_shares'] = new_total_shares
                            app_logger.debug(f"Updated YES: Shares={new_total_shares:.{share_decimals}f}, AvgP={current_stats_mut['yes_avg_price']:.4f}")
                            # Update stop loss if calculated by strategy (Entry/Accumulation)
                            if recommendation.get('calculated_stop_level') is not None:
                                 portfolio['market_data']['directional_stop_loss'] = recommendation['calculated_stop_level']
                                 app_logger.info(f"Directional Stop Loss for YES updated to: {portfolio['market_data']['directional_stop_loss']:.4f}")
                        elif side == 'NO':
                            new_total_shares = current_stats_mut['no_shares'] + shares_rounded
                            current_stats_mut['no_investment'] += cost
                            current_stats_mut['no_avg_price'] = current_stats_mut['no_investment'] / new_total_shares if new_total_shares > zero_share_thresh else 0.0
                            current_stats_mut['no_shares'] = new_total_shares
                            app_logger.debug(f"Updated NO: Shares={new_total_shares:.{share_decimals}f}, AvgP={current_stats_mut['no_avg_price']:.4f}")
                             # Update stop loss if calculated by strategy (Entry/Accumulation)
                            if recommendation.get('calculated_stop_level') is not None:
                                 portfolio['market_data']['directional_stop_loss'] = recommendation['calculated_stop_level']
                                 app_logger.info(f"Directional Stop Loss for NO updated to: {portfolio['market_data']['directional_stop_loss']:.4f}")

                        executed_trade = True
                        trade_details = {'Timestamp': dt_object, 'Action': action, 'Side': side, 'Shares': shares_rounded, 'Price': price, 'Cost': cost, 'Proceeds': 0.0}
                    else:
                        app_logger.warning(f"Insufficient balance for BUY {side}: Need {cost:.2f}, Have {current_balance:.2f}")

                # --- SELL / SELL_STOP Simulation ---
                elif action in ['SELL', 'SELL_STOP']:
                     proceeds = 0.0
                     can_sell = False
                     side_sold_log = side # For logging actual execution
                     shares_to_sell = shares_rounded # Use the rounded shares from recommendation
                     cost_basis_sold = 0.0 # Track cost basis sold
                     price_exec = 0.0 # Price at which trade is executed

                     if side == 'YES' and current_stats_mut['yes_shares'] >= shares_to_sell - zero_share_thresh:
                         can_sell = True; price_exec = yes_bid # Use actual YES bid
                         proceeds = shares_to_sell * price_exec
                         cost_basis_sold = shares_to_sell * current_stats_mut.get('yes_avg_price', 0.0)
                         app_logger.debug(f"SELL YES: Reducing shares by {shares_to_sell:.{share_decimals}f} from {current_stats_mut['yes_shares']:.{share_decimals}f} @ {price_exec:.4f}")
                         current_stats_mut['yes_shares'] -= shares_to_sell
                         current_stats_mut['yes_investment'] -= cost_basis_sold

                     elif side == 'NO' and current_stats_mut['no_shares'] >= shares_to_sell - zero_share_thresh:
                         can_sell = True; price_exec = no_bid # Use actual NO bid
                         proceeds = shares_to_sell * price_exec
                         cost_basis_sold = shares_to_sell * current_stats_mut.get('no_avg_price', 0.0)
                         app_logger.debug(f"SELL NO: Reducing shares by {shares_to_sell:.{share_decimals}f} from {current_stats_mut['no_shares']:.{share_decimals}f} @ {price_exec:.4f}")
                         current_stats_mut['no_shares'] -= shares_to_sell
                         current_stats_mut['no_investment'] -= cost_basis_sold

                     elif side == 'ALL_YES' and current_stats_mut['yes_shares'] > zero_share_thresh:
                          # Sell exactly what we have, ignore recommended rounded amount if different (should match ideally)
                          shares_to_sell = current_stats_mut['yes_shares']
                          can_sell = True; price_exec = yes_bid # Use actual YES bid
                          proceeds = shares_to_sell * price_exec
                          cost_basis_sold = current_stats_mut['yes_investment']
                          side_sold_log = 'YES' # Log actual side
                          app_logger.debug(f"SELL STOP ALL_YES: Selling all {shares_to_sell:.{share_decimals}f} shares @ {price_exec:.4f}. Proceeds={proceeds:.4f}")
                          current_stats_mut['yes_investment'] = 0.0 # Selling all investment
                          current_stats_mut['yes_shares'] = 0.0

                     elif side == 'ALL_NO' and current_stats_mut['no_shares'] > zero_share_thresh:
                          # Sell exactly what we have
                          shares_to_sell = current_stats_mut['no_shares']
                          can_sell = True; price_exec = no_bid # Use actual NO bid
                          proceeds = shares_to_sell * price_exec
                          cost_basis_sold = current_stats_mut['no_investment']
                          side_sold_log = 'NO' # Log actual side
                          app_logger.debug(f"SELL STOP ALL_NO: Selling all {shares_to_sell:.{share_decimals}f} shares @ {price_exec:.4f}. Proceeds={proceeds:.4f}")
                          current_stats_mut['no_investment'] = 0.0 # Selling all investment
                          current_stats_mut['no_shares'] = 0.0

                     elif side == 'ALL_PAIRS' and current_stats_mut['yes_shares'] >= shares_to_sell - zero_share_thresh and current_stats_mut['no_shares'] >= shares_to_sell - zero_share_thresh:
                          can_sell = True; # shares_to_sell is number of pairs
                          proceeds_yes = shares_to_sell * yes_bid
                          proceeds_no = shares_to_sell * no_bid
                          proceeds = proceeds_yes + proceeds_no
                          price_exec = price # Use the sum price from rec for logging consistency
                          side_sold_log = 'PAIR'

                          cost_basis_sold_yes = shares_to_sell * current_stats_mut.get('yes_avg_price', 0.0)
                          cost_basis_sold_no = shares_to_sell * current_stats_mut.get('no_avg_price', 0.0)
                          cost_basis_sold = cost_basis_sold_yes + cost_basis_sold_no

                          app_logger.debug(f"SELL STOP ALL_PAIRS: Reducing YES by {shares_to_sell:.{share_decimals}f} from {current_stats_mut['yes_shares']:.{share_decimals}f}")
                          app_logger.debug(f"SELL STOP ALL_PAIRS: Reducing NO by {shares_to_sell:.{share_decimals}f} from {current_stats_mut['no_shares']:.{share_decimals}f}")
                          current_stats_mut['yes_shares'] -= shares_to_sell
                          current_stats_mut['no_shares'] -= shares_to_sell
                          current_stats_mut['yes_investment'] -= cost_basis_sold_yes
                          current_stats_mut['no_investment'] -= cost_basis_sold_no

                     # Update stats and balance if sell occurred
                     if can_sell:
                          cost_proceeds = proceeds # Positive for proceeds
                          portfolio['balance'] += cost_proceeds
                          executed_trade = True
                          trade_details = {'Timestamp': dt_object, 'Action': action, 'Side': side_sold_log, 'Shares': shares_to_sell, 'Price': price_exec, 'Cost': 0.0, 'Proceeds': proceeds}
                          app_logger.info(f"Trade Executed: {action} {side_sold_log} {shares_to_sell:.{share_decimals}f} shares. Proceeds={proceeds:.4f}. New Balance={portfolio['balance']:.4f}")

                          # Recalculate avg prices or zero out if needed for the sold side(s)
                          if side_sold_log in ['YES', 'PAIR', 'ALL_YES']:
                              current_stats_mut['yes_investment'] = max(0.0, current_stats_mut['yes_investment']) # Ensure not negative
                              if current_stats_mut['yes_shares'] < zero_share_thresh or current_stats_mut['yes_investment'] <= zero_thresh:
                                  current_stats_mut['yes_shares'] = 0.0; current_stats_mut['yes_investment'] = 0.0; current_stats_mut['yes_avg_price'] = 0.0
                              else: current_stats_mut['yes_avg_price'] = current_stats_mut['yes_investment'] / current_stats_mut['yes_shares']
                          if side_sold_log in ['NO', 'PAIR', 'ALL_NO']:
                              current_stats_mut['no_investment'] = max(0.0, current_stats_mut['no_investment']) # Ensure not negative
                              if current_stats_mut['no_shares'] < zero_share_thresh or current_stats_mut['no_investment'] <= zero_thresh:
                                  current_stats_mut['no_shares'] = 0.0; current_stats_mut['no_investment'] = 0.0; current_stats_mut['no_avg_price'] = 0.0
                              else: current_stats_mut['no_avg_price'] = current_stats_mut['no_investment'] / current_stats_mut['no_shares']

                     else: # Log why sell failed
                         app_logger.warning(f"Trade NOT Executed: {action} {side} {shares_rounded}. Check share balance.")
                         app_logger.warning(f"Needed: {shares_to_sell:.{share_decimals}f}, Have YES: {current_stats_mut['yes_shares']:.{share_decimals}f}, Have NO: {current_stats_mut['no_shares']:.{share_decimals}f}")

                # --- BUY_ARB Simulation ---
                elif action == 'BUY_ARB' and side == 'PAIR':
                    cost = shares_rounded * price # Price is yes_ask + no_ask (recommendation price)
                    if current_balance >= cost - zero_thresh:
                        cost_proceeds = -cost
                        portfolio['balance'] += cost_proceeds
                        # Use actual asks for investment calculation
                        cost_yes = shares_rounded * yes_ask
                        cost_no = shares_rounded * no_ask
                        app_logger.debug(f"BUY_ARB PAIR: Cost={cost:.4f}, New Balance={portfolio['balance']:.4f}")

                        # Update YES stats
                        new_yes_shares = current_stats_mut['yes_shares'] + shares_rounded
                        current_stats_mut['yes_investment'] += cost_yes
                        current_stats_mut['yes_avg_price'] = current_stats_mut['yes_investment'] / new_yes_shares if new_yes_shares > zero_share_thresh else 0.0
                        current_stats_mut['yes_shares'] = new_yes_shares
                        app_logger.debug(f"Updated YES: Shares={new_yes_shares:.{share_decimals}f}, AvgP={current_stats_mut['yes_avg_price']:.4f}")

                        # Update NO stats
                        new_no_shares = current_stats_mut['no_shares'] + shares_rounded
                        current_stats_mut['no_investment'] += cost_no
                        current_stats_mut['no_avg_price'] = current_stats_mut['no_investment'] / new_no_shares if new_no_shares > zero_share_thresh else 0.0
                        current_stats_mut['no_shares'] = new_no_shares
                        app_logger.debug(f"Updated NO: Shares={new_no_shares:.{share_decimals}f}, AvgP={current_stats_mut['no_avg_price']:.4f}")

                        executed_trade = True
                        trade_details = {'Timestamp': dt_object, 'Action': action, 'Side': side, 'Shares': shares_rounded, 'Price': price, 'Cost': cost, 'Proceeds': 0.0}
                    else:
                         app_logger.warning(f"Insufficient balance for BUY_ARB: Need {cost:.2f}, Have {current_balance:.2f}")


                # --- SELL_ARB Simulation ---
                elif action == 'SELL_ARB' and side == 'PAIR':
                    # Price is yes_bid + no_bid (recommendation price)
                    if current_stats_mut['yes_shares'] >= shares_rounded - zero_share_thresh and current_stats_mut['no_shares'] >= shares_rounded - zero_share_thresh:
                         shares_to_sell = shares_rounded # Number of pairs
                         # Use actual Bids for proceeds calculation
                         proceeds_yes = shares_to_sell * yes_bid
                         proceeds_no = shares_to_sell * no_bid
                         proceeds = proceeds_yes + proceeds_no

                         cost_proceeds = proceeds
                         portfolio['balance'] += cost_proceeds
                         app_logger.debug(f"SELL_ARB PAIR: Proceeds={proceeds:.4f}, New Balance={portfolio['balance']:.4f}")

                         cost_basis_sold_yes = shares_to_sell * current_stats_mut.get('yes_avg_price', 0.0)
                         cost_basis_sold_no = shares_to_sell * current_stats_mut.get('no_avg_price', 0.0)

                         # Update YES stats
                         current_stats_mut['yes_shares'] -= shares_to_sell
                         current_stats_mut['yes_investment'] -= cost_basis_sold_yes
                         current_stats_mut['yes_investment'] = max(0.0, current_stats_mut['yes_investment']) # Ensure not negative
                         if current_stats_mut['yes_shares'] < zero_share_thresh or current_stats_mut['yes_investment'] <= zero_thresh:
                             current_stats_mut['yes_shares'] = 0.0; current_stats_mut['yes_investment'] = 0.0; current_stats_mut['yes_avg_price'] = 0.0
                         else: current_stats_mut['yes_avg_price'] = current_stats_mut['yes_investment'] / current_stats_mut['yes_shares']
                         app_logger.debug(f"Updated YES: Shares={current_stats_mut['yes_shares']:.{share_decimals}f}, AvgP={current_stats_mut['yes_avg_price']:.4f}")

                         # Update NO stats
                         current_stats_mut['no_shares'] -= shares_to_sell
                         current_stats_mut['no_investment'] -= cost_basis_sold_no
                         current_stats_mut['no_investment'] = max(0.0, current_stats_mut['no_investment']) # Ensure not negative
                         if current_stats_mut['no_shares'] < zero_share_thresh or current_stats_mut['no_investment'] <= zero_thresh:
                             current_stats_mut['no_shares'] = 0.0; current_stats_mut['no_investment'] = 0.0; current_stats_mut['no_avg_price'] = 0.0
                         else: current_stats_mut['no_avg_price'] = current_stats_mut['no_investment'] / current_stats_mut['no_shares']
                         app_logger.debug(f"Updated NO: Shares={current_stats_mut['no_shares']:.{share_decimals}f}, AvgP={current_stats_mut['no_avg_price']:.4f}")

                         executed_trade = True
                         trade_details = {'Timestamp': dt_object, 'Action': action, 'Side': side, 'Shares': shares_to_sell, 'Price': price, 'Cost': 0.0, 'Proceeds': proceeds}
                    else:
                        app_logger.warning(f"Insufficient shares for SELL_ARB PAIR: Have Y:{current_stats_mut['yes_shares']:.{share_decimals}f} N:{current_stats_mut['no_shares']:.{share_decimals}f}, Need {shares_rounded:.{share_decimals}f}")

                # Log executed trade
                if executed_trade:
                    # Round values in trade details for logging consistency
                    for key in ['Shares', 'Price', 'Cost', 'Proceeds']:
                        if key in trade_details:
                            trade_details[key] = round(trade_details[key], 4) # Use 4 decimals for logging prices/shares
                    app_logger.info(f"Appending Trade to Log: {trade_details}")
                    trade_log.append(trade_details)

            # --- 4. Update Position State ---
            # This MUST happen *after* trades are settled for the timestamp
            yes_s = portfolio['stats']['yes_shares']
            no_s = portfolio['stats']['no_shares']
            avg_yes_p = portfolio['stats']['yes_avg_price']
            avg_no_p = portfolio['stats']['no_avg_price']
            current_state = portfolio['market_data']['position_state']
            new_state = current_state # Default to no change

            # Use config values from the *current run's dictionary*
            cost_basis_arb_thresh = current_config_dict.get('COST_BASIS_ARB_THRESHOLD', config.COST_BASIS_ARB_THRESHOLD)
            zero_share_thresh_state = current_config_dict.get('ZERO_SHARE_THRESHOLD', config.ZERO_SHARE_THRESHOLD) # Use the correct threshold for state check

            # Determine new state based on current share balances
            is_flat = yes_s < zero_share_thresh_state and no_s < zero_share_thresh_state
            is_dir_yes = yes_s >= zero_share_thresh_state and no_s < zero_share_thresh_state
            is_dir_no = no_s >= zero_share_thresh_state and yes_s < zero_share_thresh_state
            is_holding_both = yes_s >= zero_share_thresh_state and no_s >= zero_share_thresh_state

            if is_flat:
                new_state = 'FLAT'
            elif is_dir_yes:
                new_state = 'DIRECTIONAL_YES'
            elif is_dir_no:
                new_state = 'DIRECTIONAL_NO'
            elif is_holding_both:
                # Check for COST_BASIS_ARB state eligibility
                cost_sum = avg_yes_p + avg_no_p
                # Ensure both avg prices are valid before summing for comparison
                if avg_yes_p > zero_thresh and avg_no_p > zero_thresh and cost_sum < cost_basis_arb_thresh:
                     new_state = 'COST_BASIS_ARB'
                else:
                     new_state = 'HEDGED' # Default if holding both but not meeting cost arb criteria

            # Apply state change and clear directional stop if applicable
            if new_state != current_state:
                app_logger.info(f"State Change: {current_state} -> {new_state}")
                portfolio['market_data']['position_state'] = new_state
                # Clear directional stop if we move away from a directional state or become flat
                if new_state == 'FLAT' or new_state == 'HEDGED' or new_state == 'COST_BASIS_ARB':
                     if portfolio['market_data']['directional_stop_loss'] is not None:
                          app_logger.info(f"Clearing directional stop loss due to state change to {new_state}.")
                          portfolio['market_data']['directional_stop_loss'] = None
            # Log current state even if no change
            app_logger.debug(f"Position State at end of step: {portfolio['market_data']['position_state']}")


            # --- 5. Log Performance ---
            current_value_yes = portfolio['stats']['yes_shares'] * yes_bid # Value based on sell price (Bid)
            current_value_no = portfolio['stats']['no_shares'] * no_bid
            total_portfolio_value = portfolio['balance'] + current_value_yes + current_value_no
            performance_log.append({
                'Timestamp': dt_object,
                'Balance': portfolio['balance'],
                'Yes Shares': portfolio['stats']['yes_shares'],
                'No Shares': portfolio['stats']['no_shares'],
                'Yes Avg Price': portfolio['stats']['yes_avg_price'],
                'No Avg Price': portfolio['stats']['no_avg_price'],
                'Yes Value': current_value_yes,
                'No Value': current_value_no,
                'Total Value': total_portfolio_value,
                'Position State': portfolio['market_data']['position_state'],
                'Stop Level': portfolio['market_data']['directional_stop_loss'] # Log current stop
            })
            app_logger.debug(f"End of Step {dt_object}: Balance={portfolio['balance']:.2f}, Port Value={total_portfolio_value:.2f}, State={portfolio['market_data']['position_state']}")

    except Exception as loop_error:
         app_logger.error(f"Critical error during backtest loop at index {index}, timestamp {timestamp}: {loop_error}", exc_info=True)
         # Fall through to finally block to restore config
    finally:
        # --- Ensure original config values are restored even on error ---
        for key, value in original_config_values.items():
             if hasattr(strategy_engine.config, key):
                  setattr(strategy_engine.config, key, value)

    # --- Backtest End for this config ---
    final_portfolio_value = performance_log[-1]['Total Value'] if performance_log else initial_balance
    num_trades = len(trade_log)
    performance_df = pd.DataFrame(performance_log)
    run_log_output = log_stream.getvalue() # Capture logs for this specific run

    app_logger.info(f"--- Finished Single Backtest Run ---")
    app_logger.info(f"Final Portfolio Value: {utils.format_currency(final_portfolio_value)}")
    app_logger.info(f"Number of Trades: {num_trades}")

    # Ensure final value is returned even if loop errored mid-way but some logs exist
    if not performance_log and final_portfolio_value == initial_balance:
        app_logger.warning("Backtest loop may have encountered an early error. No performance logged.")
        # Optionally return None if you consider this a failure
        # return None, 0, pd.DataFrame(), [], portfolio, run_log_output

    return final_portfolio_value, num_trades, performance_df, trade_log, portfolio, run_log_output


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Strategy Engine Backtester & Optimizer")
st.write(f"Using Strategy Engine V{config.VERSION}")

# --- File Upload (Multiple) ---
uploaded_files = st.file_uploader(
    "Upload Historical Data CSV(s) (Requires 'Timestamp (UTC)' and 'price' [or contains 'price'] columns)",
    type="csv",
    accept_multiple_files=True
)

# --- Combined DataFrame ---
combined_df = None
if uploaded_files:
    all_dfs = []
    st.write("### Uploaded Files:")
    combined_file_names = []
    for uploaded_file in uploaded_files:
        combined_file_names.append(uploaded_file.name)
        st.write(f"- {uploaded_file.name}")
        try:
            # Try standard UTF-8 first, then latin1 as fallback
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0) # Reset file pointer
                df = pd.read_csv(uploaded_file, encoding='latin1')

            # --- Basic Column Validation ---
            if 'Timestamp (UTC)' not in df.columns:
                 st.error(f"File '{uploaded_file.name}' is missing the required 'Timestamp (UTC)' column.")
                 all_dfs = [] # Invalidate run
                 break
            # Check for price column
            price_col = None
            if 'price' in df.columns: price_col = 'price'
            else: price_col = next((col for col in df.columns if 'price' in col.lower()), None)
            if price_col is None:
                st.error(f"File '{uploaded_file.name}' is missing a price column ('price' or contains 'price').")
                all_dfs = [] # Invalidate run
                break

            all_dfs.append(df)
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
            all_dfs = [] # Invalidate run
            break

    if all_dfs: # Only proceed if all files were read and validated successfully
        try:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            # Ensure Timestamp is numeric for sorting AFTER combining
            combined_df['Timestamp (UTC)'] = pd.to_numeric(combined_df['Timestamp (UTC)'], errors='coerce')
            combined_df.dropna(subset=['Timestamp (UTC)'], inplace=True) # Drop rows where timestamp couldn't be converted

            # --- Re-identify price column in combined_df ---
            price_col_combined = None
            if 'price' in combined_df.columns: price_col_combined = 'price'
            else: price_col_combined = next((col for col in combined_df.columns if 'price' in col.lower()), None)

            # Convert price column to numeric AFTER combining, coercing errors
            if price_col_combined:
                combined_df[price_col_combined] = pd.to_numeric(combined_df[price_col_combined], errors='coerce')
                combined_df.dropna(subset=[price_col_combined], inplace=True) # Drop rows where price couldn't be converted
            else:
                # This case should have been caught during individual file validation
                st.error("Price column not found in combined data. This should not happen.")
                combined_df = None

            if combined_df is not None and not combined_df.empty:
                 combined_df.sort_values(by='Timestamp (UTC)', inplace=True)
                 combined_df.reset_index(drop=True, inplace=True)
                 st.write("### Combined Data Preview (Sorted, First 5 Rows):")
                 st.dataframe(combined_df.head())
                 st.success(f"Successfully combined and validated {len(uploaded_files)} file(s) into {len(combined_df)} usable data points.")
            elif combined_df is not None and combined_df.empty:
                 st.warning("No valid data points remaining after cleaning (Timestamp/Price conversion errors).")
                 combined_df = None # Ensure it's None if empty
            # else: combined_df is already None due to earlier error

        except Exception as e:
            st.error(f"Error combining or processing files: {e}")
            combined_df = None

# --- Parameter Configuration & Optimization ---
st.sidebar.header("Strategy Parameters & Optimization")

enable_optimization = st.sidebar.checkbox("Enable Parameter Optimization (Grid Search)")

# --- Define Parameters to Tune (Reflect config.py v4.2) ---
# Format: "CONFIG_VAR_NAME": ("Display Name", min_val, max_val, num_steps_for_opt, default_val_from_config)
# --- Define Parameters to Tune (Reflect config.py v4.2) ---
# Format: "CONFIG_VAR_NAME": ("Display Name", min_val, max_val, num_steps_for_opt, NEW_DEFAULT_VAL)
params_to_tune = {
    "RISK_PER_TRADE_PCT": ("Risk Per Trade %", 0.03, 0.05, 5, 0.04),  # Default: 4% (mid-range)
    "BUY_THRESHOLD": ("Buy Threshold (<= Ask)", 0.40, 0.80, 5, 0.55), # Default: 55% (mid-point of your new range, allows buying up to 55c initially)

    # Stop Losses - Defaults often start mid-range, adjust based on risk tolerance
    "DIRECTIONAL_STOP_LOSS_PCT": ("Directional Stop %", 0.05, 0.25, 5, 0.125), # Default: 15% (allows reasonable price movement against entry)
    "ACCUMULATION_STOP_LOSS_PCT": ("Accumulation Stop %", 0.03, 0.20, 4, 0.08), # Default: 8% (often tighter than directional) - Set to None if you prefer config default logic
    "HEDGED_STOP_LOSS_PCT_BASIS": ("Hedged Stop Loss % Basis", 0.01, 0.10, 4, 0.05), # Default: 5%

    # Thresholds for Action - Defaults often require some noticeable price change
    "ACCUMULATION_DROP_THRESHOLD": ("Accumulation Drop % >=", 0.02, 0.15, 4, 0.07), # Default: 7% drop to consider accumulating
    "HEDGE_PRICE_DROP_THRESHOLD": ("Hedge Drop % >=", 0.03, 0.20, 4, 0.10),  # Default: 10% drop to consider hedging
    "HEDGED_HOLD_AVG_COST_THRESHOLD": ("Hedged Hold Cost Thresh (<)", 0.80, 0.99, 4, 0.97), # Default: Hold hedge if cost < 97c

    # Profit Taking - Default to taking profit when price is quite favourable
    "PROFIT_TAKE_PRICE_THRESHOLD": ("Profit Take Price Thresh (> Bid)", 0.50, 0.95, 6, 0.85), # Default: Consider selling YES if bid > 85c
    "PROFIT_TAKE_SELL_PCT": ("Profit Take Sell %", 0.10, 1.00, 5, 0.75), # Default: Sell 75% of position when taking profit

    # Arbitrage Settings - Defaults often look for smaller, clearer opportunities
    "COST_BASIS_ARB_THRESHOLD": ("Cost Arb Thresh (< AvgSum)", 0.95, 1.05, 3, 0.99), # Default: Look for cost basis < 99c
    "COST_ARB_ACCUM_SIZE_PCT_OF_BALANCE": ("Cost Arb Size % Bal", 0.01, 0.15, 3, 0.05), # Default: Use 5% balance for cost arb
    "ARB_THRESHOLD": ("Market Arb Spread Thresh", 0.001, 0.02, 4, 0.005), # Default: Look for >0.5% market spread for arb
    "ARB_BUY_SIZE_PCT_OF_BALANCE": ("Market Arb Size % Bal", 0.02, 0.20, 3, 0.10), # Default: Use 10% balance for market arb
}

config_overrides = {}

# Build UI for parameters
for key, (name, p_min, p_max, p_steps, p_default) in params_to_tune.items():
    # Handle potential None default for ACCUMULATION_STOP_LOSS_PCT
    actual_default = p_default if p_default is not None else p_min # Use min if default is None for slider

    if enable_optimization:
        st.sidebar.markdown(f"**{name} Range**")
        col1, col2, col3 = st.sidebar.columns(3)
        # Heuristic step calculation, ensuring it's not too small/large
        step_val = max(1e-5, min(0.1, (p_max - p_min) / 10))
        format_str = "%.4f" if step_val < 0.01 else "%.3f"

        val_min = col1.number_input("Min", min_value=float(p_min), max_value=float(p_max), value=float(p_min), step=step_val, key=f"{key}_min", format=format_str)
        val_max = col2.number_input("Max", min_value=float(p_min), max_value=float(p_max), value=float(p_max), step=step_val, key=f"{key}_max", format=format_str)
        n_steps = col3.number_input("Steps", min_value=2, max_value=20, value=int(p_steps), step=1, key=f"{key}_steps")
        # Ensure val_max >= val_min
        val_max = max(val_min, val_max)
        # Special handling for ACCUMULATION_STOP_LOSS_PCT: Decide if None is included
        include_none_option = st.sidebar.checkbox(f"Include 'None' for {name}?", value=(p_default is None), key=f"{key}_inc_none") if key == 'ACCUMULATION_STOP_LOSS_PCT' else False

        param_values = np.linspace(val_min, val_max, int(n_steps))
        if include_none_option:
             # Add None as a distinct value to test
             config_overrides[key] = np.append(param_values, None) # Store range + None
        else:
             config_overrides[key] = param_values # Store just the range

    else:
        # Simple slider for single run
        step_val = max(1e-5, min(0.1, (p_max - p_min) / 20))
        format_str = "%.4f" if step_val < 0.01 else "%.3f"
        # Special handling for ACCUMULATION_STOP_LOSS_PCT which can be None
        if key == 'ACCUMULATION_STOP_LOSS_PCT':
            use_specific_stop = st.sidebar.checkbox(f"Use Specific Stop for {name}?", value=(p_default is not None), key=f"{key}_use_specific")
            if use_specific_stop:
                 # Show slider only if using specific
                 config_overrides[key] = st.sidebar.slider(
                      f"{name} (Specific Value)", float(p_min), float(p_max), float(actual_default), step_val, format=format_str, key=f"{key}_single_specific"
                 )
            else:
                config_overrides[key] = None # Explicitly set to None
        else:
             config_overrides[key] = st.sidebar.slider(
                  name, float(p_min), float(p_max), float(actual_default), step_val, format=format_str, key=f"{key}_single"
             )

st.sidebar.info("Note: Parameters like ADV map, rounding, zero thresholds are taken from config.py and not optimized here.")

# --- Run Button Logic ---
if combined_df is not None and not combined_df.empty:
    if st.button("Run Backtest / Optimization"):
        st.markdown("---") # Separator
        run_start_time = time.time()

        if not enable_optimization:
            # --- Single Run ---
            st.header("Single Backtest Run")
            st.write("Running with specified parameters...")
            # Collect scalar values and handle potential None for ACCUMULATION_STOP_LOSS_PCT
            single_run_params = {}
            for k, v in config_overrides.items():
                 if not isinstance(v, np.ndarray): # Exclude optimization ranges
                      single_run_params[k] = v

            # --- Run the single backtest ---
            final_val, n_trades, perf_df, trade_log_list, final_port, logs = run_backtest(
                combined_df.copy(), INITIAL_BALANCE, single_run_params
            )

            run_end_time = time.time()
            st.write(f"Run completed in {run_end_time - run_start_time:.2f} seconds.")

            # --- Results Display (Single Run) ---
            if final_val is not None and perf_df is not None and not perf_df.empty:
                st.subheader("Results")
                pnl = final_val - INITIAL_BALANCE
                pnl_pct = (pnl / INITIAL_BALANCE) * 100 if INITIAL_BALANCE > 0 else 0
                final_row = perf_df.iloc[-1]
                final_cash = final_row['Balance']
                final_shares_value = final_row['Yes Value'] + final_row['No Value']

                # --- Calculate Batch Profitability ---
                batch_pct, batch_num_profit, batch_total = calculate_batch_profitability(perf_df, batch_duration_days=14)
                # --- END Calculate Batch Profitability ---

                col1, col2, col3 = st.columns(3)
                col1.metric("Final Portfolio Value", utils.format_currency(final_val), f"{utils.format_currency(pnl)} ({pnl_pct:.2f}%)")
                col2.metric("Final Cash Balance", utils.format_currency(final_cash))
                col3.metric("Final Shares Value", utils.format_currency(final_shares_value))
                # Display Batch Results and Trades Count
                col_trades, col_batch = st.columns(2)
                col_trades.metric("Number of Trades", n_trades)
                if batch_total > 0:
                     col_batch.metric("Profitable 2-Week Batches", f"{batch_pct:.1f}%", f"{batch_num_profit} of {batch_total}")
                else:
                     col_batch.metric("Profitable 2-Week Batches", "N/A", "Insufficient Data")


                # Tabs for detailed results
                tab1, tab2, tab3, tab4 = st.tabs(["Equity Curve", "Final State", "Trade Log", "Run Logs"])

                with tab1:
                    st.write("Performance Over Time")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(perf_df['Timestamp'], perf_df['Total Value'], label='Portfolio Value')
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Portfolio Value ($)")
                    ax.set_title("Portfolio Value Over Time")
                    ax.legend()
                    ax.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                with tab2:
                    st.write("Final Portfolio State")
                    # Display formatted final portfolio stats
                    st.text(f"Final Balance: {utils.format_currency(final_port['balance'])}")
                    st.text(f"YES Shares: {utils.format_shares(final_port['stats']['yes_shares'])}")
                    st.text(f"YES Avg Price: {utils.format_price(final_port['stats']['yes_avg_price'])}")
                    st.text(f"NO Shares: {utils.format_shares(final_port['stats']['no_shares'])}")
                    st.text(f"NO Avg Price: {utils.format_price(final_port['stats']['no_avg_price'])}")
                    st.text(f"Position State: {final_port['market_data']['position_state']}")
                    st.text(f"Stop Level: {utils.format_price(final_port['market_data']['directional_stop_loss']) if final_port['market_data']['directional_stop_loss'] is not None else 'None'}")
                    # st.json(final_port) # Keep JSON view optional

                with tab3:
                    st.write("Trade Log")
                    if trade_log_list:
                        trade_df = pd.DataFrame(trade_log_list)
                        # Format columns for display
                        for col in ['Price', 'Cost', 'Proceeds']:
                             if col in trade_df.columns:
                                  trade_df[col] = trade_df[col].apply(lambda x: f"{x:.4f}")
                        share_decimals_disp = single_run_params.get('SHARE_DECIMALS', config.SHARE_DECIMALS)
                        for col in ['Shares']:
                            if col in trade_df.columns:
                                trade_df[col] = trade_df[col].apply(lambda x: f"{x:.{share_decimals_disp}f}")
                        st.dataframe(trade_df)
                    else:
                        st.info("No trades were executed.")

                with tab4:
                    st.write("Strategy Engine Logs")
                    st.text_area("Logs", logs, height=400, key="single_run_logs")
            else:
                 st.error("Backtest failed or produced no results. Check logs.")
                 st.text_area("Run Logs (Error)", logs, height=400, key="single_run_error_logs")

        else:
            # --- Optimization Run ---
            st.header("Parameter Optimization Run")
            st.write("Running Grid Search...")

            # 1. Generate Parameter Combinations
            param_keys = [k for k, v in config_overrides.items() if isinstance(v, np.ndarray)] # Get keys being optimized
            param_ranges = [config_overrides[k] for k in param_keys]

            # Get non-optimizing parameters (single values set in UI)
            non_opt_params = {}
            for k, v in config_overrides.items():
                 if not isinstance(v, np.ndarray):
                      non_opt_params[k] = v

            # Create combinations from optimizable ranges
            combinations = list(itertools.product(*param_ranges))
            total_combinations = len(combinations)
            st.write(f"Total parameter combinations to test: {total_combinations}")

            if total_combinations == 0:
                 st.warning("No parameters selected for optimization ranges.")
                 st.stop()
            if total_combinations > 500:
                 st.warning(f"Warning: {total_combinations} combinations may take a very long time!")

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_placeholder = st.empty() # Placeholder for results table

            # 2. Loop through combinations and run backtest
            for i, combo_values in enumerate(combinations):
                current_run_params = dict(zip(param_keys, combo_values))
                # Add non-tuned params back
                current_run_params.update(non_opt_params)
                # Handle the 'None' value if it came from np.append
                if 'ACCUMULATION_STOP_LOSS_PCT' in current_run_params and current_run_params['ACCUMULATION_STOP_LOSS_PCT'] is None:
                     pass # Already None
                elif 'ACCUMULATION_STOP_LOSS_PCT' in current_run_params and isinstance(current_run_params['ACCUMULATION_STOP_LOSS_PCT'], (float, np.float64, int)):
                      current_run_params['ACCUMULATION_STOP_LOSS_PCT'] = float(current_run_params['ACCUMULATION_STOP_LOSS_PCT']) # Ensure it's float


                status_text.text(f"Running combination {i+1}/{total_combinations}...")

                # Run backtest for this combo - capture essentials
                final_val, n_trades, perf_df, _, final_port, _ = run_backtest(
                    combined_df.copy(), INITIAL_BALANCE, current_run_params
                )

                if final_val is not None and perf_df is not None and not perf_df.empty:
                     # Extract final cash/shares value for summary table
                     final_row = perf_df.iloc[-1]
                     final_cash = final_row['Balance']
                     final_shares_value = final_row['Yes Value'] + final_row['No Value']
                     # Calculate batch profitability for this run
                     batch_pct, batch_num_profit, batch_total = calculate_batch_profitability(perf_df, batch_duration_days=14)

                     result_entry = {
                          "params": current_run_params,
                          "final_value": final_val,
                          "final_cash": final_cash,
                          "final_shares_value": final_shares_value,
                          "num_trades": n_trades,
                          "batch_profit_pct": batch_pct,
                          "profitable_batches": batch_num_profit,
                          "total_batches": batch_total
                          }
                     results.append(result_entry)
                else:
                     # Log combo failure without stopping optimization
                     log_params_fail = {k: (f"{v:.4f}" if isinstance(v, (float, np.float64)) else str(v)) for k, v in current_run_params.items()}
                     app_logger.warning(f"Combination {i+1} failed or produced no results (params: {log_params_fail}). Skipping result.")

                progress_bar.progress((i + 1) / total_combinations)

            status_text.text(f"Optimization finished testing {total_combinations} combinations.")
            run_end_time = time.time()
            st.write(f"Optimization completed in {run_end_time - run_start_time:.2f} seconds.")

            # 3. Process and Display Results
            if not results:
                st.error("No successful backtest runs during optimization.")
            else:
                results_df = pd.DataFrame(results)
                # Format parameters for better display in table (optional)
                def format_params(params_dict):
                     return ', '.join([f"{k}={v:.4f}" if isinstance(v, (float, np.float64)) else f"{k}={v}"
                                        for k, v in params_dict.items() if k in param_keys]) # Show only tuned params
                results_df['tuned_params_str'] = results_df['params'].apply(format_params)
                # Sort by batch profitability first, then final value
                results_df.sort_values(by=["batch_profit_pct", "final_value"], ascending=[False, False], inplace=True)

                st.subheader("Optimization Results (Top 10 by Batch Profitability)")
                # Select and format columns for display
                display_df = results_df.head(10)[[
                     'final_value', 'batch_profit_pct', 'profitable_batches', 'total_batches', 'num_trades', 'tuned_params_str'
                     ]].copy()
                display_df['final_value'] = display_df['final_value'].apply(utils.format_currency)
                display_df['batch_profit_pct'] = display_df['batch_profit_pct'].apply(lambda x: f"{x:.1f}%")
                display_df['batch_info'] = display_df.apply(lambda row: f"{row['profitable_batches']} of {row['total_batches']}" if row['total_batches'] > 0 else "N/A", axis=1)
                results_placeholder.dataframe(display_df[['final_value', 'batch_profit_pct', 'batch_info', 'num_trades', 'tuned_params_str']].set_index('tuned_params_str')) # Show tuned params as index

                st.subheader("Best Parameter Set Found (by Batch Profitability):")
                best_result = results_df.iloc[0]
                col1, col2, col3 = st.columns(3)
                col1.metric("Best Final Value", utils.format_currency(best_result['final_value']))
                col2.metric("Best Batch Profitability", f"{best_result['batch_profit_pct']:.1f}%", f"{best_result['profitable_batches']} of {best_result['total_batches']}")
                col3.metric("Number of Trades (Best)", best_result['num_trades'])

                st.write("Best Parameters (including non-tuned):")
                # Format None nicely for display
                best_params_display = {}
                for k,v in best_result['params'].items():
                    if isinstance(v, (float, np.float64)): best_params_display[k] = f"{v:.4f}"
                    elif v is None: best_params_display[k] = "None"
                    else: best_params_display[k] = str(v)
                st.json(best_params_display)


                # Optional: Rerun with best params
                if st.checkbox("Rerun backtest with BEST parameters found", key="rerun_best_opt"):
                    st.markdown("---")
                    st.header("Detailed Rerun with Best Parameters")

                    # --- Rerun the backtest ---
                    final_val_best, n_trades_best, perf_df_best, trade_log_list_best, final_port_best, logs_best = run_backtest(
                         combined_df.copy(), INITIAL_BALANCE, best_result['params']
                    )

                    # --- Results Display (Best Run Rerun) ---
                    if final_val_best is not None and perf_df_best is not None and not perf_df_best.empty:
                         final_row_best = perf_df_best.iloc[-1]
                         final_cash_best = final_row_best['Balance']
                         final_shares_val_best = final_row_best['Yes Value'] + final_row_best['No Value']

                         # --- Calculate Batch Profitability for Best Run ---
                         batch_pct_best, batch_num_profit_best, batch_total_best = calculate_batch_profitability(perf_df_best, batch_duration_days=14)
                         # --- END Calculate Batch Profitability ---

                         col1, col2, col3 = st.columns(3)
                         col1.metric("Final Value (Rerun)", utils.format_currency(final_val_best))
                         col2.metric("Final Cash (Rerun)", utils.format_currency(final_cash_best))
                         col3.metric("Final Shares Value (Rerun)", utils.format_currency(final_shares_val_best))
                         # Display Batch Results and Trades Count
                         col_trades_best, col_batch_best = st.columns(2)
                         col_trades_best.metric("Trades (Rerun)", n_trades_best)
                         if batch_total_best > 0:
                              col_batch_best.metric("Profitable 2-Week Batches (Best)", f"{batch_pct_best:.1f}%", f"{batch_num_profit_best} of {batch_total_best}")
                         else:
                              col_batch_best.metric("Profitable 2-Week Batches (Best)", "N/A", "Insufficient Data")


                         tab1, tab2, tab3, tab4 = st.tabs(["Equity Curve (Best)", "Final State (Best)", "Trade Log (Best)", "Run Logs (Best)"])
                         with tab1:
                              st.write("Performance Over Time (Best Params)")
                              fig, ax = plt.subplots(figsize=(12, 5))
                              ax.plot(perf_df_best['Timestamp'], perf_df_best['Total Value'], label='Portfolio Value (Best)')
                              ax.set_xlabel("Time"); ax.set_ylabel("Portfolio Value ($)")
                              ax.set_title("Portfolio Value Over Time (Best Parameters)")
                              ax.legend(); ax.grid(True); plt.xticks(rotation=45); plt.tight_layout()
                              st.pyplot(fig)
                         with tab2:
                              st.write("Final Portfolio State (Best)")
                              st.text(f"Final Balance: {utils.format_currency(final_port_best['balance'])}")
                              st.text(f"YES Shares: {utils.format_shares(final_port_best['stats']['yes_shares'])}")
                              st.text(f"YES Avg Price: {utils.format_price(final_port_best['stats']['yes_avg_price'])}")
                              st.text(f"NO Shares: {utils.format_shares(final_port_best['stats']['no_shares'])}")
                              st.text(f"NO Avg Price: {utils.format_price(final_port_best['stats']['no_avg_price'])}")
                              st.text(f"Position State: {final_port_best['market_data']['position_state']}")
                              st.text(f"Stop Level: {utils.format_price(final_port_best['market_data']['directional_stop_loss']) if final_port_best['market_data']['directional_stop_loss'] is not None else 'None'}")
                         with tab3:
                              st.write("Trade Log (Best)")
                              if trade_log_list_best:
                                   trade_df_best = pd.DataFrame(trade_log_list_best)
                                   share_decimals_disp_best = best_result['params'].get('SHARE_DECIMALS', config.SHARE_DECIMALS)
                                   for col in ['Price', 'Cost', 'Proceeds']:
                                       if col in trade_df_best.columns: trade_df_best[col] = trade_df_best[col].apply(lambda x: f"{x:.4f}")
                                   for col in ['Shares']:
                                       if col in trade_df_best.columns: trade_df_best[col] = trade_df_best[col].apply(lambda x: f"{x:.{share_decimals_disp_best}f}")
                                   st.dataframe(trade_df_best)
                              else: st.info("No trades executed in best run.")
                         with tab4:
                              st.write("Strategy Engine Logs (Best)")
                              st.text_area("Logs", logs_best, height=400, key="best_run_logs")
                    else:
                         st.error("Rerun with best parameters failed or produced no results.")

else:
    st.info("Please upload one or more valid CSV files to begin.")