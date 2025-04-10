# backtestapp.py
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
    import config # Import the base config to get defaults (Version 4.5 now)
    import utilsbacktest as utils # Use utils alias for clarity
except ImportError as e:
    st.error(f"Error importing local modules (strategy_engine.py, config.py V4.5, utilsbacktest.py): {e}")
    st.error("Make sure these files exist in the same directory as backtestapp.py")
    st.stop()

# --- Global Variables / Constants ---
INITIAL_BALANCE = 1000.00
SIMULATED_SPREAD_PCT = 0.005 # Example spread % (0.5%)
MARKET_NAME_DEFAULT = "BacktestMarket"
DEFAULT_MARKET_ADV = float('inf') # Assume high liquidity for backtesting unless specified otherwise

# --- Logging Setup ---
log_stream = io.StringIO()
app_logger = logging.getLogger('backtester_app') # Ensure this name matches strategy_engine's
app_logger.handlers.clear()
app_logger.propagate = False
log_level_from_config = getattr(config, 'LOG_LEVEL', logging.INFO)
log_handler = logging.StreamHandler(log_stream)
log_formatter = logging.Formatter(config.LOG_FORMAT)
log_handler.setFormatter(log_formatter)
app_logger.addHandler(log_handler)
app_logger.setLevel(log_level_from_config)


# --- Batch Profitability Analysis ---
def calculate_batch_profitability(performance_df, batch_duration_days=14):
    """ Analyzes profitability over fixed time batches. """
    if performance_df is None or performance_df.empty or 'Timestamp' not in performance_df.columns or 'Total Value' not in performance_df.columns:
        app_logger.warning("Batch analysis skipped: Performance DataFrame is missing or invalid.")
        return 0.0, 0, 0
    df = performance_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df.dropna(subset=['Timestamp'], inplace=True)
            if df.empty: app_logger.warning("Batch analysis skipped: No valid timestamps after conversion."); return 0.0, 0, 0
        except Exception as e: app_logger.error(f"Error converting Timestamp in batch analysis: {e}"); return 0.0, 0, 0
    df.set_index('Timestamp', inplace=True); df.sort_index(inplace=True)
    resample_freq = f'{batch_duration_days}D'
    try:
        resampled = df['Total Value'].resample(resample_freq).agg(['first', 'last'])
        app_logger.debug(f"Resampled data for batch analysis:\n{resampled}")
    except Exception as e: app_logger.error(f"Error resampling with freq '{resample_freq}': {e}"); return 0.0, 0, 0
    resampled.dropna(subset=['first', 'last'], inplace=True)
    app_logger.debug(f"Resampled data after dropna:\n{resampled}")
    if resampled.empty: app_logger.warning(f"Batch analysis skipped: No complete {batch_duration_days}-day batches found."); return 0.0, 0, 0
    resampled['profit'] = resampled['last'] - resampled['first']
    num_profitable = (resampled['profit'] > 1e-6).sum()
    total_batches = len(resampled)
    if total_batches == 0: app_logger.warning("Batch analysis: Total complete batches is zero."); return 0.0, 0, 0
    percentage_profitable = (num_profitable / total_batches) * 100
    app_logger.info(f"Batch analysis: {num_profitable} profitable batches out of {total_batches} ({percentage_profitable:.1f}%).")
    return percentage_profitable, num_profitable, total_batches


# --- Backtesting Core Logic ---
def run_backtest(df, initial_balance, current_config_dict):
    """ Runs the backtesting simulation FOR A SINGLE PARAMETER CONFIGURATION. """
    log_stream.seek(0); log_stream.truncate(0) # Clear log buffer
    app_logger.info(f"--- Starting Single Backtest Run ---")
    log_params = {k: (f"{v:.4f}" if isinstance(v, (float, np.float64)) else str(v)) for k, v in current_config_dict.items()}
    app_logger.info(f"Parameters: {log_params}")
    app_logger.info(f"Initial balance: {utils.format_currency(initial_balance)}")

    portfolio = {
        'balance': initial_balance,
        'market_data': {'position_state': 'FLAT', 'directional_stop_loss': None},
        'stats': {'yes_shares': 0.0, 'no_shares': 0.0, 'yes_investment': 0.0, 'no_investment': 0.0, 'yes_avg_price': 0.0, 'no_avg_price': 0.0},
        'last_recommendation': None
    }
    performance_log = []; trade_log = []

    price_col = 'price' if 'price' in df.columns else next((col for col in df.columns if 'price' in col.lower()), None)
    if not price_col:
        app_logger.error("Failed to find price column ('price' or contains 'price') in CSV.")
        return None, 0, pd.DataFrame(), [], portfolio, log_stream.getvalue()
    app_logger.info(f"Using price column: '{price_col}'")

    original_config_values = {}
    # Store original values before patching
    for key in current_config_dict.keys():
        if hasattr(strategy_engine, 'config') and hasattr(strategy_engine.config, key):
             original_config_values[key] = getattr(strategy_engine.config, key)
        elif hasattr(config, key):
             original_config_values[key] = getattr(config, key) # Backup check

    # --- Main Backtesting Loop ---
    try:
        for index, row in df.iterrows():
            timestamp = row['Timestamp (UTC)']
            try: dt_object = datetime.utcfromtimestamp(float(timestamp))
            except ValueError: app_logger.error(f"Bad timestamp: {timestamp} at index {index}. Skipping."); continue

            # --- 1. Simulate Market Prices ---
            try: yes_price_raw = float(row[price_col])
            except (ValueError, TypeError): app_logger.warning(f"Invalid price '{row[price_col]}' at {dt_object}. Using 0.5."); yes_price_raw = 0.5

            zero_thresh = current_config_dict.get('ZERO_PRICE_THRESHOLD', config.ZERO_PRICE_THRESHOLD)
            share_decimals = int(current_config_dict.get('SHARE_DECIMALS', config.SHARE_DECIMALS))
            share_rounding_func_cfg = current_config_dict.get('SHARE_ROUNDING_FUNC', config.SHARE_ROUNDING_FUNC)
            share_rounding_func = share_rounding_func_cfg if callable(share_rounding_func_cfg) else (lambda x: round(x, share_decimals))

            yes_price = max(zero_thresh, min(1.0 - zero_thresh, yes_price_raw)); no_price = 1.0 - yes_price
            half_spread = yes_price * (SIMULATED_SPREAD_PCT / 2.0); yes_ask = min(1.0 - zero_thresh, yes_price + half_spread); yes_bid = max(zero_thresh, yes_price - half_spread)
            half_spread_no = no_price * (SIMULATED_SPREAD_PCT / 2.0); no_ask = min(1.0 - zero_thresh, no_price + half_spread_no); no_bid = max(zero_thresh, no_price - half_spread_no)
            yes_bid = min(yes_bid, yes_ask - zero_thresh); no_bid = min(no_bid, no_ask - zero_thresh)
            yes_bid = max(zero_thresh, yes_bid); no_bid = max(zero_thresh, no_bid); yes_ask = max(zero_thresh, yes_ask); no_ask = max(zero_thresh, no_ask)
            market_adv = DEFAULT_MARKET_ADV

            app_logger.debug(f"--- Timestamp: {dt_object} ---")
            app_logger.debug(f"Raw Yes Price: {yes_price_raw:.4f} -> Validated: {yes_price:.4f}")
            app_logger.debug(f"Simulated Y_Bid:{yes_bid:.4f}, Y_Ask:{yes_ask:.4f} | N_Bid:{no_bid:.4f}, N_Ask:{no_ask:.4f}")

            # --- 2. Get Strategy Recommendation ---
            current_market_data = copy.deepcopy(portfolio['market_data']); current_stats = copy.deepcopy(portfolio['stats']); current_balance = portfolio['balance']
            recommendation = None

            try: # Patch config for strategy call
                for key, value in current_config_dict.items():
                    if hasattr(strategy_engine, 'config') and hasattr(strategy_engine.config, key): setattr(strategy_engine.config, key, value)
                    # elif hasattr(config, key): setattr(config, key, value) # Optional backup patch

                recommendation, analysis_details = strategy_engine.calculate_strategy_recommendation(
                    market_name=MARKET_NAME_DEFAULT, yes_bid=yes_bid, yes_ask=yes_ask, no_bid=no_bid, no_ask=no_ask,
                    market_adv=market_adv, market_data=current_market_data, stats=current_stats, current_balance=current_balance)
                portfolio['last_recommendation'] = recommendation
            except Exception as e: app_logger.error(f"Error during calculate_strategy_recommendation at {dt_object}: {e}", exc_info=True)
            finally: # Restore original config values
                for key, value in original_config_values.items():
                     if hasattr(strategy_engine, 'config') and hasattr(strategy_engine.config, key): setattr(strategy_engine.config, key, value)
                     # elif hasattr(config, key): setattr(config, key, value) # Optional backup restore

            if recommendation is None: app_logger.warning(f"Skipping step {dt_object} due to strategy error."); continue

            # --- 3. Simulate Trade Execution ---
            action = recommendation['action_type']; side = recommendation['side']
            shares_rounded = recommendation['shares_rounded']; price = recommendation['price']
            executed_trade = False; trade_details = {}
            zero_share_thresh = current_config_dict.get('ZERO_SHARE_THRESHOLD', config.ZERO_SHARE_THRESHOLD)

            if action != 'HOLD' and shares_rounded > zero_share_thresh:
                app_logger.info(f"Attempting Action: {recommendation['display_text']}")
                cost_proceeds = 0.0; current_stats_mut = portfolio['stats']

                # --- BUY Simulation ---
                if action == 'BUY':
                    cost = shares_rounded * price
                    if current_balance >= cost - zero_thresh:
                        cost_proceeds = -cost; portfolio['balance'] += cost_proceeds; app_logger.debug(f"BUY {side}: Cost={cost:.4f}, New Bal={portfolio['balance']:.4f}")
                        if side == 'YES':
                            new_total = current_stats_mut['yes_shares'] + shares_rounded; current_stats_mut['yes_investment'] += cost; current_stats_mut['yes_avg_price'] = current_stats_mut['yes_investment'] / new_total if new_total > zero_share_thresh else 0.0; current_stats_mut['yes_shares'] = new_total; app_logger.debug(f"Updated YES: Sh={new_total:.{share_decimals}f}, AvgP={current_stats_mut['yes_avg_price']:.4f}")
                            if recommendation.get('calculated_stop_level') is not None: portfolio['market_data']['directional_stop_loss'] = recommendation['calculated_stop_level']; app_logger.info(f"Stop Loss YES updated: {portfolio['market_data']['directional_stop_loss']:.4f}")
                        elif side == 'NO':
                            new_total = current_stats_mut['no_shares'] + shares_rounded; current_stats_mut['no_investment'] += cost; current_stats_mut['no_avg_price'] = current_stats_mut['no_investment'] / new_total if new_total > zero_share_thresh else 0.0; current_stats_mut['no_shares'] = new_total; app_logger.debug(f"Updated NO: Sh={new_total:.{share_decimals}f}, AvgP={current_stats_mut['no_avg_price']:.4f}")
                            if recommendation.get('calculated_stop_level') is not None: portfolio['market_data']['directional_stop_loss'] = recommendation['calculated_stop_level']; app_logger.info(f"Stop Loss NO updated: {portfolio['market_data']['directional_stop_loss']:.4f}")
                        executed_trade = True; trade_details = {'Timestamp': dt_object, 'Action': action, 'Side': side, 'Shares': shares_rounded, 'Price': price, 'Cost': cost, 'Proceeds': 0.0, 'Trigger': recommendation.get('rule_triggered', 'N/A'), 'TriggerReason': recommendation.get('trigger_reason')}
                    else: app_logger.warning(f"Insufficient balance for BUY {side}: Need {cost:.2f}, Have {current_balance:.2f}")

                # --- SELL / SELL_STOP Simulation ---
                elif action in ['SELL', 'SELL_STOP']:
                    proceeds = 0.0; can_sell = False; side_sold_log = side; shares_to_sell = shares_rounded; cost_basis_sold = 0.0; price_exec = 0.0
                    if side == 'YES' and current_stats_mut['yes_shares'] >= shares_to_sell - zero_share_thresh:
                         can_sell = True; price_exec = yes_bid; proceeds = shares_to_sell * price_exec; cost_basis_sold = shares_to_sell * current_stats_mut.get('yes_avg_price', 0.0); app_logger.debug(f"SELL YES: Reducing by {shares_to_sell:.{share_decimals}f} @ {price_exec:.4f}"); current_stats_mut['yes_shares'] -= shares_to_sell; current_stats_mut['yes_investment'] -= cost_basis_sold
                    elif side == 'NO' and current_stats_mut['no_shares'] >= shares_to_sell - zero_share_thresh:
                         can_sell = True; price_exec = no_bid; proceeds = shares_to_sell * price_exec; cost_basis_sold = shares_to_sell * current_stats_mut.get('no_avg_price', 0.0); app_logger.debug(f"SELL NO: Reducing by {shares_to_sell:.{share_decimals}f} @ {price_exec:.4f}"); current_stats_mut['no_shares'] -= shares_to_sell; current_stats_mut['no_investment'] -= cost_basis_sold
                    elif side == 'ALL_YES' and current_stats_mut['yes_shares'] > zero_share_thresh:
                         shares_to_sell = current_stats_mut['yes_shares']; can_sell = True; price_exec = yes_bid; proceeds = shares_to_sell * price_exec; cost_basis_sold = current_stats_mut['yes_investment']; side_sold_log = 'YES'; app_logger.debug(f"SELL STOP ALL_YES: Selling all {shares_to_sell:.{share_decimals}f} @ {price_exec:.4f}. Proc={proceeds:.4f}"); current_stats_mut['yes_investment'] = 0.0; current_stats_mut['yes_shares'] = 0.0
                    elif side == 'ALL_NO' and current_stats_mut['no_shares'] > zero_share_thresh:
                         shares_to_sell = current_stats_mut['no_shares']; can_sell = True; price_exec = no_bid; proceeds = shares_to_sell * price_exec; cost_basis_sold = current_stats_mut['no_investment']; side_sold_log = 'NO'; app_logger.debug(f"SELL STOP ALL_NO: Selling all {shares_to_sell:.{share_decimals}f} @ {price_exec:.4f}. Proc={proceeds:.4f}"); current_stats_mut['no_investment'] = 0.0; current_stats_mut['no_shares'] = 0.0
                    elif side == 'ALL_PAIRS' and current_stats_mut['yes_shares'] >= shares_to_sell - zero_share_thresh and current_stats_mut['no_shares'] >= shares_to_sell - zero_share_thresh:
                         can_sell = True; proceeds_yes = shares_to_sell * yes_bid; proceeds_no = shares_to_sell * no_bid; proceeds = proceeds_yes + proceeds_no; price_exec = price; side_sold_log = 'PAIR'; cost_basis_sold_yes = shares_to_sell * current_stats_mut.get('yes_avg_price', 0.0); cost_basis_sold_no = shares_to_sell * current_stats_mut.get('no_avg_price', 0.0); app_logger.debug(f"SELL STOP ALL_PAIRS: Reducing YES/NO by {shares_to_sell:.{share_decimals}f}"); current_stats_mut['yes_shares'] -= shares_to_sell; current_stats_mut['no_shares'] -= shares_to_sell; current_stats_mut['yes_investment'] -= cost_basis_sold_yes; current_stats_mut['no_investment'] -= cost_basis_sold_no
                    if can_sell:
                        cost_proceeds = proceeds; portfolio['balance'] += cost_proceeds; executed_trade = True; trigger = recommendation.get('rule_triggered', 'N/A'); trigger_reason_detail = recommendation.get('trigger_reason')
                        trade_details = {'Timestamp': dt_object, 'Action': action, 'Side': side_sold_log, 'Shares': shares_to_sell, 'Price': price_exec, 'Cost': 0.0, 'Proceeds': proceeds, 'Trigger': trigger, 'TriggerReason': trigger_reason_detail}
                        app_logger.info(f"Trade Executed: {action} {side_sold_log} {shares_to_sell:.{share_decimals}f} shares. Proceeds={proceeds:.4f}. New Bal={portfolio['balance']:.4f}")
                        if side_sold_log in ['YES', 'PAIR', 'ALL_YES']: current_stats_mut['yes_investment'] = max(0.0, current_stats_mut['yes_investment']); current_stats_mut['yes_avg_price'] = (current_stats_mut['yes_investment'] / current_stats_mut['yes_shares']) if current_stats_mut['yes_shares'] >= zero_share_thresh else 0.0
                        if side_sold_log in ['NO', 'PAIR', 'ALL_NO']: current_stats_mut['no_investment'] = max(0.0, current_stats_mut['no_investment']); current_stats_mut['no_avg_price'] = (current_stats_mut['no_investment'] / current_stats_mut['no_shares']) if current_stats_mut['no_shares'] >= zero_share_thresh else 0.0
                    else: app_logger.warning(f"Trade NOT Executed: {action} {side} {shares_rounded}. Need {shares_to_sell:.{share_decimals}f}, Have Y:{current_stats_mut['yes_shares']:.{share_decimals}f} N:{current_stats_mut['no_shares']:.{share_decimals}f}")

                # --- BUY_ARB Simulation ---
                elif action == 'BUY_ARB' and side == 'PAIR':
                    cost = shares_rounded * price
                    if current_balance >= cost - zero_thresh:
                        cost_proceeds = -cost; portfolio['balance'] += cost_proceeds; cost_yes = shares_rounded * yes_ask; cost_no = shares_rounded * no_ask; app_logger.debug(f"BUY_ARB PAIR: Cost={cost:.4f}, New Bal={portfolio['balance']:.4f}")
                        new_yes = current_stats_mut['yes_shares'] + shares_rounded; current_stats_mut['yes_investment'] += cost_yes; current_stats_mut['yes_avg_price'] = current_stats_mut['yes_investment'] / new_yes if new_yes > zero_share_thresh else 0.0; current_stats_mut['yes_shares'] = new_yes; app_logger.debug(f"Updated YES: Sh={new_yes:.{share_decimals}f}, AvgP={current_stats_mut['yes_avg_price']:.4f}")
                        new_no = current_stats_mut['no_shares'] + shares_rounded; current_stats_mut['no_investment'] += cost_no; current_stats_mut['no_avg_price'] = current_stats_mut['no_investment'] / new_no if new_no > zero_share_thresh else 0.0; current_stats_mut['no_shares'] = new_no; app_logger.debug(f"Updated NO: Sh={new_no:.{share_decimals}f}, AvgP={current_stats_mut['no_avg_price']:.4f}")
                        executed_trade = True; trade_details = {'Timestamp': dt_object, 'Action': action, 'Side': side, 'Shares': shares_rounded, 'Price': price, 'Cost': cost, 'Proceeds': 0.0, 'Trigger': recommendation.get('rule_triggered', 'N/A'), 'TriggerReason': recommendation.get('trigger_reason')}
                    else: app_logger.warning(f"Insufficient balance for BUY_ARB: Need {cost:.2f}, Have {current_balance:.2f}")

                # --- SELL_ARB Simulation ---
                elif action == 'SELL_ARB' and side == 'PAIR':
                    if current_stats_mut['yes_shares'] >= shares_rounded - zero_share_thresh and current_stats_mut['no_shares'] >= shares_rounded - zero_share_thresh:
                         shares_to_sell = shares_rounded; proceeds_yes = shares_to_sell * yes_bid; proceeds_no = shares_to_sell * no_bid; proceeds = proceeds_yes + proceeds_no
                         cost_proceeds = proceeds; portfolio['balance'] += cost_proceeds; app_logger.debug(f"SELL_ARB PAIR: Proceeds={proceeds:.4f}, New Bal={portfolio['balance']:.4f}")
                         cost_basis_sold_yes = shares_to_sell * current_stats_mut.get('yes_avg_price', 0.0); cost_basis_sold_no = shares_to_sell * current_stats_mut.get('no_avg_price', 0.0)
                         current_stats_mut['yes_shares'] -= shares_to_sell; current_stats_mut['yes_investment'] -= cost_basis_sold_yes; current_stats_mut['yes_investment'] = max(0.0, current_stats_mut['yes_investment']); current_stats_mut['yes_avg_price'] = (current_stats_mut['yes_investment'] / current_stats_mut['yes_shares']) if current_stats_mut['yes_shares'] >= zero_share_thresh else 0.0; app_logger.debug(f"Updated YES: Sh={current_stats_mut['yes_shares']:.{share_decimals}f}, AvgP={current_stats_mut['yes_avg_price']:.4f}")
                         current_stats_mut['no_shares'] -= shares_to_sell; current_stats_mut['no_investment'] -= cost_basis_sold_no; current_stats_mut['no_investment'] = max(0.0, current_stats_mut['no_investment']); current_stats_mut['no_avg_price'] = (current_stats_mut['no_investment'] / current_stats_mut['no_shares']) if current_stats_mut['no_shares'] >= zero_share_thresh else 0.0; app_logger.debug(f"Updated NO: Sh={current_stats_mut['no_shares']:.{share_decimals}f}, AvgP={current_stats_mut['no_avg_price']:.4f}")
                         executed_trade = True; trade_details = {'Timestamp': dt_object, 'Action': action, 'Side': side, 'Shares': shares_to_sell, 'Price': price, 'Cost': 0.0, 'Proceeds': proceeds, 'Trigger': recommendation.get('rule_triggered', 'N/A'), 'TriggerReason': recommendation.get('trigger_reason')}
                    else: app_logger.warning(f"Insufficient shares for SELL_ARB PAIR: Have Y:{current_stats_mut['yes_shares']:.{share_decimals}f} N:{current_stats_mut['no_shares']:.{share_decimals}f}, Need {shares_rounded:.{share_decimals}f}")

                # --- Log primary executed trade & Handle Profit Scrape ---
                if executed_trade:
                    for key in ['Shares', 'Price', 'Cost', 'Proceeds']: # Round values for logging
                        if key in trade_details and isinstance(trade_details[key], (float, int)): trade_details[key] = round(trade_details[key], 4)
                    app_logger.info(f"Appending Primary Trade to Log: {trade_details}")
                    trade_log.append(trade_details)

                    # --- Profit Scrape Handling ---
                    scrape_config_enabled = current_config_dict.get('ENABLE_PROFIT_SCRAPE_HEDGE', False)
                    if scrape_config_enabled and trade_details.get('TriggerReason') == 'PROFIT_SCRAPE':
                        proceeds_from_scrape = trade_details['Proceeds']; original_side_sold = trade_details['Side']; side_to_buy = 'NO' if original_side_sold == 'YES' else 'YES'
                        buy_ask_price = no_ask if side_to_buy == 'NO' else yes_ask
                        app_logger.info(f"--- Profit Scrape Follow-up ---"); app_logger.info(f"Scrape Sold: {original_side_sold}. Buying {side_to_buy} with {utils.format_currency(proceeds_from_scrape)} @ Ask {utils.format_price(buy_ask_price)}")
                        if buy_ask_price > zero_thresh and proceeds_from_scrape > zero_thresh:
                            shares_to_buy_unrounded = proceeds_from_scrape / buy_ask_price; shares_to_buy_rounded = share_rounding_func(shares_to_buy_unrounded)
                            if shares_to_buy_rounded > zero_share_thresh:
                                cost_hedge_buy = shares_to_buy_rounded * buy_ask_price
                                if portfolio['balance'] >= cost_hedge_buy - zero_thresh:
                                    portfolio['balance'] -= cost_hedge_buy; app_logger.info(f"PROFIT_SCRAPE_BUY Executed: Buy {utils.format_shares(shares_to_buy_rounded)} {side_to_buy} @ {utils.format_price(buy_ask_price)}. Cost={utils.format_currency(cost_hedge_buy)}. New Bal={utils.format_currency(portfolio['balance'])}")
                                    if side_to_buy == 'YES':
                                        new_yes_total = current_stats_mut['yes_shares'] + shares_to_buy_rounded; current_stats_mut['yes_investment'] += cost_hedge_buy; current_stats_mut['yes_avg_price'] = current_stats_mut['yes_investment'] / new_yes_total if new_yes_total > zero_share_thresh else 0.0; current_stats_mut['yes_shares'] = new_yes_total; app_logger.debug(f"Updated YES (Scrape Buy): Sh={current_stats_mut['yes_shares']:.{share_decimals}f}, AvgP={current_stats_mut['yes_avg_price']:.4f}")
                                    elif side_to_buy == 'NO':
                                        new_no_total = current_stats_mut['no_shares'] + shares_to_buy_rounded; current_stats_mut['no_investment'] += cost_hedge_buy; current_stats_mut['no_avg_price'] = current_stats_mut['no_investment'] / new_no_total if new_no_total > zero_share_thresh else 0.0; current_stats_mut['no_shares'] = new_no_total; app_logger.debug(f"Updated NO (Scrape Buy): Sh={current_stats_mut['no_shares']:.{share_decimals}f}, AvgP={current_stats_mut['no_avg_price']:.4f}")
                                    hedge_trade_details = {'Timestamp': dt_object, 'Action': 'BUY_FROM_PROFIT', 'Side': side_to_buy, 'Shares': shares_to_buy_rounded, 'Price': buy_ask_price, 'Cost': cost_hedge_buy, 'Proceeds': 0.0, 'Trigger': 'Profit Scrape'}
                                    for key_h in ['Shares', 'Price', 'Cost', 'Proceeds']: hedge_trade_details[key_h] = round(hedge_trade_details[key_h], 4)
                                    trade_log.append(hedge_trade_details); app_logger.info(f"Appending Profit Scrape BUY Trade to Log: {hedge_trade_details}")
                                else: app_logger.warning(f"PROFIT_SCRAPE_BUY Failed: Insufficient balance ({utils.format_currency(portfolio['balance'])} < {utils.format_currency(cost_hedge_buy)}) after scrape.")
                            else: app_logger.info(f"PROFIT_SCRAPE_BUY Skipped: Calculated shares rounded to zero ({shares_to_buy_unrounded} unrounded).")
                        else: app_logger.warning(f"PROFIT_SCRAPE_BUY Skipped: Invalid ask price ({utils.format_price(buy_ask_price)}) or zero proceeds ({utils.format_currency(proceeds_from_scrape)}).")

            # --- 4. Update Position State ---
            yes_s = portfolio['stats']['yes_shares']; no_s = portfolio['stats']['no_shares']
            avg_yes_p = portfolio['stats']['yes_avg_price']; avg_no_p = portfolio['stats']['no_avg_price']
            current_state = portfolio['market_data']['position_state']; new_state = current_state
            cost_basis_arb_thresh = current_config_dict.get('COST_BASIS_ARB_THRESHOLD', config.COST_BASIS_ARB_THRESHOLD)
            zero_share_thresh_state = current_config_dict.get('ZERO_SHARE_THRESHOLD', config.ZERO_SHARE_THRESHOLD)
            is_flat = yes_s < zero_share_thresh_state and no_s < zero_share_thresh_state
            is_dir_yes = yes_s >= zero_share_thresh_state and no_s < zero_share_thresh_state
            is_dir_no = no_s >= zero_share_thresh_state and yes_s < zero_share_thresh_state
            is_holding_both = yes_s >= zero_share_thresh_state and no_s >= zero_share_thresh_state
            if is_flat: new_state = 'FLAT'
            elif is_dir_yes: new_state = 'DIRECTIONAL_YES'
            elif is_dir_no: new_state = 'DIRECTIONAL_NO'
            elif is_holding_both: new_state = 'COST_BASIS_ARB' if (avg_yes_p > zero_thresh and avg_no_p > zero_thresh and (avg_yes_p + avg_no_p) < cost_basis_arb_thresh) else 'HEDGED'
            if new_state != current_state:
                app_logger.info(f"State Change: {current_state} -> {new_state}")
                portfolio['market_data']['position_state'] = new_state
                if new_state in ['FLAT', 'HEDGED', 'COST_BASIS_ARB'] and portfolio['market_data']['directional_stop_loss'] is not None:
                    app_logger.info(f"Clearing directional stop loss due to state change to {new_state}.")
                    portfolio['market_data']['directional_stop_loss'] = None
            app_logger.debug(f"Position State at end of step: {portfolio['market_data']['position_state']}")

            # --- 5. Log Performance ---
            current_value_yes = portfolio['stats']['yes_shares'] * yes_bid; current_value_no = portfolio['stats']['no_shares'] * no_bid
            total_portfolio_value = portfolio['balance'] + current_value_yes + current_value_no
            performance_log.append({'Timestamp': dt_object, 'Balance': portfolio['balance'], 'Yes Shares': portfolio['stats']['yes_shares'], 'No Shares': portfolio['stats']['no_shares'], 'Yes Avg Price': portfolio['stats']['yes_avg_price'], 'No Avg Price': portfolio['stats']['no_avg_price'], 'Yes Value': current_value_yes, 'No Value': current_value_no, 'Total Value': total_portfolio_value, 'Position State': portfolio['market_data']['position_state'], 'Stop Level': portfolio['market_data']['directional_stop_loss']})
            app_logger.debug(f"End Step {dt_object}: Bal={portfolio['balance']:.2f}, PortVal={total_portfolio_value:.2f}, State={portfolio['market_data']['position_state']}")

    except Exception as loop_error:
         app_logger.error(f"Critical error during backtest loop at index {index}, timestamp {timestamp}: {loop_error}", exc_info=True)
    finally: # Restore config values
        for key, value in original_config_values.items():
             if hasattr(strategy_engine, 'config') and hasattr(strategy_engine.config, key): setattr(strategy_engine.config, key, value)
             # elif hasattr(config, key): setattr(config, key, value) # Optional backup restore

    # --- Backtest End ---
    final_portfolio_value = performance_log[-1]['Total Value'] if performance_log else initial_balance
    num_trades = len(trade_log); performance_df = pd.DataFrame(performance_log); run_log_output = log_stream.getvalue()
    app_logger.info(f"--- Finished Single Backtest Run ---")
    app_logger.info(f"Final Portfolio Value: {utils.format_currency(final_portfolio_value)}")
    app_logger.info(f"Number of Trades (incl. scrape buys): {num_trades}")
    if not performance_log and final_portfolio_value == initial_balance: app_logger.warning("Backtest loop may have encountered an early error. No performance logged.")
    return final_portfolio_value, num_trades, performance_df, trade_log, portfolio, run_log_output


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Strategy Engine Backtester & Optimizer")
st.write(f"Using Strategy Engine V{config.VERSION}") # Display current config version

# --- File Upload ---
uploaded_files = st.file_uploader(
    "Upload Historical Data CSV(s) (Requires 'Timestamp (UTC)' and 'price' [or contains 'price'])",
    type="csv", accept_multiple_files=True
)

# --- Combined DataFrame Logic ---
combined_df = None
if uploaded_files:
    all_dfs = []
    st.write("### Uploaded Files:")
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")
        try:
            try: df_up = pd.read_csv(uploaded_file)
            except UnicodeDecodeError: uploaded_file.seek(0); df_up = pd.read_csv(uploaded_file, encoding='latin1')
            if 'Timestamp (UTC)' not in df_up.columns: st.error(f"File '{uploaded_file.name}' missing 'Timestamp (UTC)'."); all_dfs = []; break
            price_col_up = 'price' if 'price' in df_up.columns else next((col for col in df_up.columns if 'price' in col.lower()), None)
            if price_col_up is None: st.error(f"File '{uploaded_file.name}' missing price column."); all_dfs = []; break
            all_dfs.append(df_up)
        except Exception as e: st.error(f"Error reading {uploaded_file.name}: {e}"); all_dfs = []; break
    if all_dfs:
        try:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df['Timestamp (UTC)'] = pd.to_numeric(combined_df['Timestamp (UTC)'], errors='coerce')
            combined_df.dropna(subset=['Timestamp (UTC)'], inplace=True)
            price_col_combined = 'price' if 'price' in combined_df.columns else next((col for col in combined_df.columns if 'price' in col.lower()), None)
            if price_col_combined:
                combined_df[price_col_combined] = pd.to_numeric(combined_df[price_col_combined], errors='coerce')
                combined_df.dropna(subset=[price_col_combined], inplace=True)
            else: st.error("Price column not found in combined data."); combined_df = None
            if combined_df is not None and not combined_df.empty:
                 combined_df.sort_values(by='Timestamp (UTC)', inplace=True); combined_df.reset_index(drop=True, inplace=True)
                 st.write("### Combined Data Preview (Sorted, First 5 Rows):"); st.dataframe(combined_df.head())
                 st.success(f"Successfully combined {len(uploaded_files)} file(s) into {len(combined_df)} usable points.")
            elif combined_df is not None and combined_df.empty: st.warning("No valid data points remaining after cleaning."); combined_df = None
        except Exception as e: st.error(f"Error combining/processing files: {e}"); combined_df = None


# --- Parameter Configuration & Optimization ---
st.sidebar.header("Strategy Parameters & Optimization")
enable_optimization = st.sidebar.checkbox("Enable Parameter Optimization (Grid Search)")

# --- Define Parameters to Tune (Reflecting config.py v4.5) ---
params_to_tune = {
    "RISK_PER_TRADE_PCT": ("Risk Per Trade %", 0.01, 0.10, 5, config.RISK_PER_TRADE_PCT),
    "DIRECTIONAL_STOP_LOSS_PCT": ("Directional Stop %", 0.05, 0.30, 5, config.DIRECTIONAL_STOP_LOSS_PCT),
    "ACCUMULATION_STOP_LOSS_PCT": ("Accumulation Stop %", 0.03, 0.25, 4, getattr(config, 'ACCUMULATION_STOP_LOSS_PCT', 0.10)),
    "HEDGED_STOP_LOSS_PCT_BASIS": ("Hedged Stop % Basis", 0.01, 0.15, 4, config.HEDGED_STOP_LOSS_PCT_BASIS),
    "HEDGED_HOLD_AVG_COST_THRESHOLD": ("Hedged Hold Cost Thresh (<)", 0.80, 0.99, 4, config.HEDGED_HOLD_AVG_COST_THRESHOLD),
    "MIN_BUY_PRICE": ("Min Buy Price (>= Ask)", 0.10, 0.50, 5, config.MIN_BUY_PRICE),
    "BUY_THRESHOLD": ("Max Buy Price (<= Ask)", 0.40, 0.90, 6, config.BUY_THRESHOLD),
    "ENABLE_PROFIT_SCRAPE_HEDGE": ("Enable Profit Scrape Hedge", False, True, 2, config.ENABLE_PROFIT_SCRAPE_HEDGE),
    "PROFIT_TAKE_PERCENTAGE_GAIN_THRESHOLD": ("Profit Take/Scrape % Gain Thresh", 0.05, 0.50, 6, config.PROFIT_TAKE_PERCENTAGE_GAIN_THRESHOLD), # Updated Param
    "PROFIT_SCRAPE_SELL_PCT": ("Profit Scrape Sell %", 0.01, 0.25, 5, config.PROFIT_SCRAPE_SELL_PCT),
    "PROFIT_TAKE_SELL_PCT": ("Profit Take Sell % (if Scrape OFF)", 0.10, 1.00, 5, config.PROFIT_TAKE_SELL_PCT),
    "ACCUMULATION_DROP_THRESHOLD": ("Accumulation Drop % >=", 0.02, 0.20, 4, config.ACCUMULATION_DROP_THRESHOLD),
    "HEDGE_PRICE_DROP_THRESHOLD": ("Hedge Drop % >=", 0.03, 0.25, 4, config.HEDGE_PRICE_DROP_THRESHOLD),
    "ENABLE_MARKET_ARBITRAGE": ("Enable Market Arb", False, True, 2, config.ENABLE_MARKET_ARBITRAGE),
    "ARB_THRESHOLD": ("Market Arb Spread Thresh", 0.001, 0.05, 5, config.ARB_THRESHOLD),
    "ARB_BUY_SIZE_PCT_OF_BALANCE": ("Market Arb Size % Bal", 0.01, 0.25, 4, config.ARB_BUY_SIZE_PCT_OF_BALANCE),
    "ENABLE_COST_BASIS_ARBITRAGE": ("Enable Cost Basis Arb", False, True, 2, config.ENABLE_COST_BASIS_ARBITRAGE),
    "COST_BASIS_ARB_THRESHOLD": ("Cost Arb Thresh (< AvgSum)", 0.90, 1.05, 4, config.COST_BASIS_ARB_THRESHOLD),
    "COST_ARB_ACCUM_SIZE_PCT_OF_BALANCE": ("Cost Arb Size % Bal", 0.01, 0.20, 4, config.COST_ARB_ACCUM_SIZE_PCT_OF_BALANCE),
    "ENABLE_ACCUMULATION": ("Enable Accumulation", False, True, 2, config.ENABLE_ACCUMULATION),
    "ENABLE_HEDGING": ("Enable Hedging", False, True, 2, config.ENABLE_HEDGING),
    "HEDGE_MATCH_SHARES": ("Hedge Buys Match Shares?", False, True, 2, config.HEDGE_MATCH_SHARES),
}

config_overrides = {}

# --- Build UI for parameters ---
for key, (name, p_min, p_max, p_steps, p_default) in params_to_tune.items():
    actual_default = p_default if p_default is not None else p_min
    if isinstance(p_default, bool): # Boolean Flags
        if enable_optimization: config_overrides[key] = [True, False]; st.sidebar.markdown(f"**{name}** (Optimize Both)")
        else: config_overrides[key] = st.sidebar.checkbox(name, value=p_default, key=f"{key}_single")
    else: # Numeric Parameters
        if enable_optimization:
            st.sidebar.markdown(f"**{name} Range**"); col1, col2, col3 = st.sidebar.columns(3)
            step_val = max(1e-5, min(0.1, (p_max - p_min) / 10)) if p_max > p_min else 1e-5; format_str = "%.4f" if step_val < 0.01 else "%.3f"
            val_min = col1.number_input("Min", min_value=float(p_min), max_value=float(p_max), value=float(p_min), step=step_val, key=f"{key}_min", format=format_str)
            val_max = col2.number_input("Max", min_value=float(p_min), max_value=float(p_max), value=float(p_max), step=step_val, key=f"{key}_max", format=format_str)
            n_steps = col3.number_input("Steps", min_value=2, max_value=20, value=int(p_steps), step=1, key=f"{key}_steps"); val_max = max(val_min, val_max)
            include_none_option = st.sidebar.checkbox(f"Include 'None' for {name}?", value=(getattr(config, key, None) is None), key=f"{key}_inc_none") if key == 'ACCUMULATION_STOP_LOSS_PCT' else False
            param_values = np.linspace(val_min, val_max, int(n_steps))
            if include_none_option: config_overrides[key] = np.append(param_values, None)
            else: config_overrides[key] = param_values
        else: # Single Run UI
            if key == 'ACCUMULATION_STOP_LOSS_PCT':
                use_specific_stop = st.sidebar.checkbox(f"Use Specific Stop for {name}?", value=(p_default is not None), key=f"{key}_use_specific")
                if use_specific_stop: step_val_s = max(1e-5, min(0.1, (p_max - p_min) / 20)) if p_max > p_min else 1e-5; format_str_s = "%.4f" if step_val_s < 0.01 else "%.3f"; config_overrides[key] = st.sidebar.slider(f"{name} (Specific)", float(p_min), float(p_max), float(actual_default), step_val_s, format=format_str_s, key=f"{key}_single_specific")
                else: config_overrides[key] = None
            else:
                step_val_s = max(1e-5, min(0.1, (p_max - p_min) / 20)) if p_max > p_min else 1e-5; format_str_s = "%.4f" if step_val_s < 0.01 else "%.3f"
                config_overrides[key] = st.sidebar.slider(name, float(p_min), float(p_max), float(actual_default), step_val_s, format=format_str_s, key=f"{key}_single")

st.sidebar.info("Configure parameters for backtest or optimization.")

# --- Run Button Logic & Results Display ---
if combined_df is not None and not combined_df.empty:
    if st.button("Run Backtest / Optimization"):
        st.markdown("---"); run_start_time = time.time()
        if not enable_optimization: # SINGLE RUN
            st.header("Single Backtest Run"); st.write("Running with specified parameters...")
            single_run_params = {k: v for k, v in config_overrides.items() if not isinstance(v, np.ndarray) and not (isinstance(v, list) and len(v)==2 and isinstance(v[0], bool))}
            final_val, n_trades, perf_df, trade_log_list, final_port, logs = run_backtest(combined_df.copy(), INITIAL_BALANCE, single_run_params)
            run_end_time = time.time(); st.write(f"Run completed in {run_end_time - run_start_time:.2f} seconds.")
            if final_val is not None and perf_df is not None and not perf_df.empty:
                st.subheader("Results"); pnl = final_val - INITIAL_BALANCE; pnl_pct = (pnl / INITIAL_BALANCE) * 100 if INITIAL_BALANCE > 0 else 0
                final_row = perf_df.iloc[-1]; final_cash = final_row['Balance']; final_shares_value = final_row['Yes Value'] + final_row['No Value']; batch_pct, batch_num_profit, batch_total = calculate_batch_profitability(perf_df, batch_duration_days=14)
                col1, col2, col3 = st.columns(3); col1.metric("Final Portfolio Value", utils.format_currency(final_val), f"{utils.format_currency(pnl)} ({pnl_pct:.2f}%)"); col2.metric("Final Cash Balance", utils.format_currency(final_cash)); col3.metric("Final Shares Value", utils.format_currency(final_shares_value))
                col_trades, col_batch = st.columns(2); col_trades.metric("Number of Trades", n_trades)
                if batch_total > 0: col_batch.metric("Profitable 2-Week Batches", f"{batch_pct:.1f}%", f"{batch_num_profit} of {batch_total}")
                else: col_batch.metric("Profitable 2-Week Batches", "N/A", "Insufficient Data")
                tab1, tab2, tab3, tab4 = st.tabs(["Equity Curve", "Final State", "Trade Log", "Run Logs"])
                with tab1: fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(perf_df['Timestamp'], perf_df['Total Value'], label='Portfolio Value'); ax.set_xlabel("Time"); ax.set_ylabel("Value ($)"); ax.set_title("Portfolio Value"); ax.legend(); ax.grid(True); plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig)
                with tab2: st.write("Final Portfolio State"); st.text(f"Balance: {utils.format_currency(final_port['balance'])}"); st.text(f"YES: {utils.format_shares(final_port['stats']['yes_shares'])} @ AvgP: {utils.format_price(final_port['stats']['yes_avg_price'])}"); st.text(f"NO: {utils.format_shares(final_port['stats']['no_shares'])} @ AvgP: {utils.format_price(final_port['stats']['no_avg_price'])}"); st.text(f"State: {final_port['market_data']['position_state']}"); st.text(f"Stop: {utils.format_price(final_port['market_data']['directional_stop_loss']) if final_port['market_data']['directional_stop_loss'] is not None else 'None'}")
                with tab3:
                    st.write("Trade Log")
                    if trade_log_list:
                        trade_df = pd.DataFrame(trade_log_list); share_decimals_disp = int(single_run_params.get('SHARE_DECIMALS', config.SHARE_DECIMALS))
                        for col in ['Price', 'Cost', 'Proceeds']:
                            if col in trade_df.columns: trade_df[col] = trade_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)
                        for col in ['Shares']:
                            if col in trade_df.columns: trade_df[col] = trade_df[col].apply(lambda x: f"{x:.{share_decimals_disp}f}" if isinstance(x, (float, int)) else x)
                        if 'TriggerReason' not in trade_df.columns and any('TriggerReason' in d for d in trade_log_list): trade_df['TriggerReason'] = [d.get('TriggerReason') for d in trade_log_list]
                        st.dataframe(trade_df)
                    else: st.info("No trades were executed.")
                with tab4: st.write("Strategy Engine Logs"); st.text_area("Logs", logs, height=400, key="single_run_logs")
            else: st.error("Backtest failed or produced no results."); st.text_area("Run Logs (Error)", logs, height=400, key="single_run_error_logs")
        else: # OPTIMIZATION RUN
            st.header("Parameter Optimization Run"); st.write("Running Grid Search...")
            param_keys = []; param_values_list = []; non_opt_params = {}
            for k, v in config_overrides.items():
                if isinstance(v, np.ndarray) or (isinstance(v, list) and len(v)==2 and isinstance(v[0], bool)): param_values_list.append(v); param_keys.append(k)
                else: non_opt_params[k] = v
            if not param_values_list: st.error("No parameters selected for optimization ranges."); st.stop()
            combinations = list(itertools.product(*param_values_list)); total_combinations = len(combinations)
            st.write(f"Total parameter combinations to test: {total_combinations}")
            if total_combinations == 0: st.warning("No parameter combinations generated."); st.stop()
            if total_combinations > 1000: st.warning(f"Warning: {total_combinations} combinations may take a very long time!")
            results = []; progress_bar = st.progress(0); status_text = st.empty(); results_placeholder = st.empty()
            for i, combo_values in enumerate(combinations):
                current_run_params = dict(zip(param_keys, combo_values)); current_run_params.update(non_opt_params)
                if 'ACCUMULATION_STOP_LOSS_PCT' in current_run_params:
                     val = current_run_params['ACCUMULATION_STOP_LOSS_PCT']
                     if val is None: pass;
                     elif isinstance(val, str) and val.lower() == 'none': current_run_params['ACCUMULATION_STOP_LOSS_PCT'] = None;
                     elif isinstance(val, (float, np.float64, int)): current_run_params['ACCUMULATION_STOP_LOSS_PCT'] = float(val)
                status_text.text(f"Running combination {i+1}/{total_combinations}...")
                final_val, n_trades, perf_df, _, final_port, _ = run_backtest(combined_df.copy(), INITIAL_BALANCE, current_run_params)
                if final_val is not None and perf_df is not None and not perf_df.empty:
                     final_row = perf_df.iloc[-1]; final_cash = final_row['Balance']; final_shares_value = final_row['Yes Value'] + final_row['No Value']; batch_pct, batch_num_profit, batch_total = calculate_batch_profitability(perf_df, batch_duration_days=14)
                     results.append({"params": current_run_params, "final_value": final_val, "final_cash": final_cash, "final_shares_value": final_shares_value, "num_trades": n_trades, "batch_profit_pct": batch_pct, "profitable_batches": batch_num_profit, "total_batches": batch_total})
                else: log_params_fail = {k: (f"{v:.4f}" if isinstance(v, (float, np.float64)) else str(v)) for k, v in current_run_params.items()}; app_logger.warning(f"Combo {i+1} failed (params: {log_params_fail}).")
                progress_bar.progress((i + 1) / total_combinations)
            status_text.text(f"Optimization finished testing {total_combinations} combinations."); run_end_time = time.time(); st.write(f"Optimization completed in {run_end_time - run_start_time:.2f} seconds.")
            if not results: st.error("No successful backtest runs during optimization.")
            else:
                results_df = pd.DataFrame(results); 
                
        def format_params_opt(params_dict): return ', '.join([f"{k}={v:.4f}" if isinstance(v, (float, np.float64)) else f"{k}={v}" for k, v in params_dict.items() if k in param_keys]); results_df['tuned_params_str'] = results_df['params'].apply(format_params_opt); results_df.sort_values(by=["batch_profit_pct", "final_value"], ascending=[False, False], inplace=True)
        st.subheader("Optimization Results (Top 10 by Batch Profitability)")
        display_df = results_df.head(10)[['final_value', 'batch_profit_pct', 'profitable_batches', 'total_batches', 'num_trades', 'tuned_params_str']].copy(); display_df['final_value'] = display_df['final_value'].apply(utils.format_currency); display_df['batch_profit_pct'] = display_df['batch_profit_pct'].apply(lambda x: f"{x:.1f}%"); display_df['batch_info'] = display_df.apply(lambda row: f"{row['profitable_batches']} of {row['total_batches']}" if row['total_batches'] > 0 else "N/A", axis=1)
        results_placeholder.dataframe(display_df[['final_value', 'batch_profit_pct', 'batch_info', 'num_trades', 'tuned_params_str']].set_index('tuned_params_str'))
        st.subheader("Best Parameter Set Found (by Batch Profitability):"); best_result = results_df.iloc[0]; col1, col2, col3 = st.columns(3); col1.metric("Best Final Value", utils.format_currency(best_result['final_value'])); col2.metric("Best Batch Profitability", f"{best_result['batch_profit_pct']:.1f}%", f"{best_result['profitable_batches']} of {best_result['total_batches']}"); col3.metric("Number of Trades (Best)", best_result['num_trades'])
        st.write("Best Parameters (including non-tuned):"); best_params_display = {k: (f"{v:.4f}" if isinstance(v, (float, np.float64)) else ("None" if v is None else str(v))) for k,v in best_result['params'].items()}; st.json(best_params_display)
        if st.checkbox("Rerun backtest with BEST parameters found", key="rerun_best_opt"):
            st.markdown("---"); st.header("Detailed Rerun with Best Parameters")
            final_val_best, n_trades_best, perf_df_best, trade_log_list_best, final_port_best, logs_best = run_backtest(combined_df.copy(), INITIAL_BALANCE, best_result['params'])
            if final_val_best is not None and perf_df_best is not None and not perf_df_best.empty:
                    final_row_best = perf_df_best.iloc[-1]; final_cash_best = final_row_best['Balance']; final_shares_val_best = final_row_best['Yes Value'] + final_row_best['No Value']; batch_pct_best, batch_num_profit_best, batch_total_best = calculate_batch_profitability(perf_df_best, batch_duration_days=14)
                    col1b, col2b, col3b = st.columns(3); col1b.metric("Final Value (Rerun)", utils.format_currency(final_val_best)); col2b.metric("Final Cash (Rerun)", utils.format_currency(final_cash_best)); col3b.metric("Final Shares Value (Rerun)", utils.format_currency(final_shares_val_best))
                    col_trades_best, col_batch_best = st.columns(2); col_trades_best.metric("Trades (Rerun)", n_trades_best)
                    if batch_total_best > 0: col_batch_best.metric("Profitable 2-W Batches (Best)", f"{batch_pct_best:.1f}%", f"{batch_num_profit_best} of {batch_total_best}")
                    else: col_batch_best.metric("Profitable 2-W Batches (Best)", "N/A", "Insufficient Data")
                    tab1b, tab2b, tab3b, tab4b = st.tabs(["Equity Curve (Best)", "Final State (Best)", "Trade Log (Best)", "Run Logs (Best)"])
                    with tab1b: fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(perf_df_best['Timestamp'], perf_df_best['Total Value'], label='Portfolio Value (Best)'); ax.set_xlabel("Time"); ax.set_ylabel("Value ($)"); ax.set_title("Portfolio Value (Best Params)"); ax.legend(); ax.grid(True); plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig)
                    with tab2b: st.write("Final Portfolio State (Best)"); st.text(f"Balance: {utils.format_currency(final_port_best['balance'])}"); st.text(f"YES: {utils.format_shares(final_port_best['stats']['yes_shares'])} @ AvgP: {utils.format_price(final_port_best['stats']['yes_avg_price'])}"); st.text(f"NO: {utils.format_shares(final_port_best['stats']['no_shares'])} @ AvgP: {utils.format_price(final_port_best['stats']['no_avg_price'])}"); st.text(f"State: {final_port_best['market_data']['position_state']}"); st.text(f"Stop: {utils.format_price(final_port_best['market_data']['directional_stop_loss']) if final_port_best['market_data']['directional_stop_loss'] is not None else 'None'}")
                    with tab3b:
                        st.write("Trade Log (Best)")
                        if trade_log_list_best:
                            trade_df_best = pd.DataFrame(trade_log_list_best); share_decimals_disp_best = int(best_result['params'].get('SHARE_DECIMALS', config.SHARE_DECIMALS))
                            for col in ['Price', 'Cost', 'Proceeds']:
                                if col in trade_df_best.columns: trade_df_best[col] = trade_df_best[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)
                            for col in ['Shares']:
                                if col in trade_df_best.columns: trade_df_best[col] = trade_df_best[col].apply(lambda x: f"{x:.{share_decimals_disp_best}f}" if isinstance(x, (float, int)) else x)
                            if 'TriggerReason' not in trade_df_best.columns and any('TriggerReason' in d for d in trade_log_list_best): trade_df_best['TriggerReason'] = [d.get('TriggerReason') for d in trade_log_list_best]
                            st.dataframe(trade_df_best)
                        else: st.info("No trades executed in best run.")
                    with tab4b: st.write("Strategy Engine Logs (Best)"); st.text_area("Logs", logs_best, height=400, key="best_run_logs")
            else: st.error("Rerun with best parameters failed or produced no results.")

else:
    st.info("Please upload one or more valid CSV files to begin.")