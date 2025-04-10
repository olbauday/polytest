##data_manaer.py

import json
import os
import uuid
from datetime import datetime
import traceback
import config
import utils # For validation within manual add
import logging

# --- Module Global Data ---
_all_market_data = {}
_global_current_balance = 1000.0 # Default starting balance
_global_total_realized_pl = 0.0 # Track total realized P/L

# --- Balance Management ---
def get_global_balance():
    """Returns the current global balance."""
    global _global_current_balance
    return _global_current_balance

def set_global_balance(new_balance):
    """Sets the global balance (use with caution, primarily for init)."""
    global _global_current_balance
    try:
        _global_current_balance = float(new_balance)
        return True
    except (ValueError, TypeError):
        logging.error(f"Error: Invalid balance value passed to set_global_balance: {new_balance}")
        return False

def get_total_realized_pl():
    """Returns the total realized P/L across all markets."""
    global _global_total_realized_pl
    return _global_total_realized_pl

# --- Market Data Access ---
def get_market_data(market_name):
    """Safely retrieves data dictionary for a market."""
    return _all_market_data.get(market_name)

def get_all_market_names():
    """Returns a sorted list of all market names."""
    return sorted(list(_all_market_data.keys()))

def market_exists(market_name):
    """Checks if a market exists."""
    return market_name in _all_market_data

# --- Market Structure Management ---
def add_new_market(name, is_test=False):
    """Adds a new market with initial structure."""
    global _all_market_data
    if not name or name in _all_market_data:
        logging.error(f"Error adding market: Invalid name or '{name}' already exists.")
        return False
    _all_market_data[name] = {
        "bets_list": [],
        "transaction_log": [],
        # Store Bid (Sell Price) and Ask (Buy Price) separately
        "last_yes_bid": "", # Price you SELL YES at
        "last_yes_ask": "", # Price you BUY YES at
        "last_no_bid": "",  # Price you SELL NO at
        "last_no_ask": "",  # Price you BUY NO at
        "adv": "",
        "position_state": 'FLAT',
        "directional_stop_loss": None,
        "last_entry_side": None,
        "is_test_market": is_test,
    }
    logging.info(f"Market '{name}' added {'(TEST MARKET)' if is_test else ''}.")
    return True

def delete_market(name):
    """Deletes a market and its data."""
    global _all_market_data
    if name in _all_market_data:
        del _all_market_data[name]
        logging.info(f"Market '{name}' deleted.")
        return True
    else:
        logging.error(f"Error deleting market: '{name}' not found.")
        return False

def update_market_property(market_name, key, value):
    """Updates a simple property (like price, adv) for a market."""
    market_data = get_market_data(market_name)
    if market_data:
        # Specific handling for ADV 'inf'
        if key == 'adv':
            if isinstance(value, str) and value.lower() == 'inf':
                market_data[key] = float('inf')
            elif isinstance(value, (int, float)) and value >= 0:
                 market_data[key] = value
            elif isinstance(value, str) and value.strip() == "":
                 market_data[key] = "" # Allow clearing ADV
            else: # Attempt validation for numeric strings
                 validated_adv = utils.validate_adv(str(value))
                 if validated_adv is not None:
                      market_data[key] = validated_adv
                 else:
                      logging.warning(f"Warning: Invalid ADV value '{value}' not set for market '{market_name}'.")
                      return False
        # Store price strings directly
        elif key in ["last_yes_bid", "last_yes_ask", "last_no_bid", "last_no_ask"]:
             market_data[key] = str(value)
        else:
             market_data[key] = value
        return True
    return False


# --- Holdings (Bets) Management ---
def add_bet(market_name, side, shares, price):
    """Adds a new bet entry to the market's list."""
    market_data = get_market_data(market_name)
    if market_data and shares > config.ZERO_SHARE_THRESHOLD and price > config.ZERO_PRICE_THRESHOLD:
        amount = shares * price
        bet = {
            'id': str(uuid.uuid4()),
            'amount': amount, # Store original cost basis amount
            'price': price,   # Store original price per share
            'side': side
        }
        market_data['bets_list'].append(bet)
        logging.debug(f"Bet added to '{market_name}': {utils.format_shares(shares)} {side} @ {utils.format_price(price)}")
        return True
    logging.error(f"Error adding bet to '{market_name}': Invalid data (Market:{market_data is not None}, Shares:{shares}, Price:{price})")
    return False

def get_bets(market_name):
    """Returns the list of bets for a market."""
    market_data = get_market_data(market_name)
    return market_data.get('bets_list', []) if market_data else []

def set_bets(market_name, new_bets_list):
    """Replaces the bets list (used internally by FIFO and resolution)."""
    market_data = get_market_data(market_name)
    if market_data and isinstance(new_bets_list, list):
        market_data['bets_list'] = new_bets_list
        return True
    return False

def clear_all_bets(market_name):
    """Clears all bet entries for a market (for manual correction)."""
    market_data = get_market_data(market_name)
    if market_data:
        count = len(market_data.get('bets_list', []))
        market_data['bets_list'] = []
        # Also clear related state as holdings are gone
        market_data['position_state'] = 'FLAT'
        market_data['directional_stop_loss'] = None
        market_data['last_entry_side'] = None
        logging.info(f"Cleared {count} bets for market '{market_name}'. State reset to FLAT.")
        return count
    return 0 # Changed from -1 to 0 for consistency

def remove_bet_by_id(market_name, bet_id):
    """Removes a single bet by its UUID (for manual correction)."""
    market_data = get_market_data(market_name)
    if market_data:
        initial_len = len(market_data['bets_list'])
        market_data['bets_list'] = [b for b in market_data['bets_list'] if b.get('id') != bet_id]
        removed = initial_len - len(market_data['bets_list'])
        if removed > 0:
            logging.info(f"Removed bet ID {bet_id} from market '{market_name}'.")
            # Recalculate state after removal
            update_position_state(market_name) # State update is crucial here
            return True
    return False


# --- Transaction Log Management ---
def add_transaction(market_name, tx_type, side, shares, price, cash_flow, is_manual=False):
    """Logs a transaction, updates balance (unless test market), returns success."""
    global _global_current_balance, _global_total_realized_pl
    market_data = get_market_data(market_name)
    if not market_data:
        logging.error(f"Error adding transaction: Market '{market_name}' not found.")
        return False
    is_test = market_data.get('is_test_market', False)

    try:
        # Ensure correct types before calculations/logging
        shares_f = float(shares) if shares is not None else 0.0
        price_f = float(price) if price is not None else 0.0
        cash_flow_f = float(cash_flow) if cash_flow is not None else 0.0

        # Determine current balance BEFORE this transaction affects it
        balance_before_tx = _global_current_balance
        new_balance = balance_before_tx # Default if test market

        # Update balance ONLY if it's NOT a test market
        if not is_test:
            new_balance_precise = balance_before_tx + cash_flow_f
            new_balance = round(new_balance_precise, 2) # Use 2 decimal places for currency
            _global_current_balance = new_balance # Update module global
        else:
            logging.debug(f"Transaction for TEST market '{market_name}' - Balance not changed.")

        log_entry = {
            'ts': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': tx_type,
            'side': side,
            'shares': shares_f, # Log the precise shares involved
            'price': price_f,
            'cash_flow': round(cash_flow_f, 2),
            'balance': new_balance # Log the balance *after* the transaction
        }
        market_data['transaction_log'].append(log_entry)
        logging.info(f"Transaction logged for '{market_name}': {tx_type} {utils.format_shares(shares_f)} {side} @ {utils.format_price(price_f)}, CF: {utils.format_cash_flow(cash_flow_f)}, New Bal: {utils.format_currency(new_balance)}")
        return True
    except Exception as e:
        logging.error(f"Failed to log transaction for '{market_name}': {e}", exc_info=True)
        return False

def get_transactions(market_name):
    """Returns the transaction log for a market."""
    market_data = get_market_data(market_name)
    return market_data.get('transaction_log', []) if market_data else []

def clear_transaction_log(market_name):
    """Clears the transaction log for a market."""
    market_data = get_market_data(market_name)
    if market_data:
        count = len(market_data.get('transaction_log', []))
        market_data['transaction_log'] = []
        logging.info(f"Cleared {count} transactions for market '{market_name}'.")
        return count
    return 0

def update_realized_pl(amount):
    """Adds the given amount to the total realized P/L."""
    global _global_total_realized_pl
    try:
        _global_total_realized_pl += float(amount)
    except (ValueError, TypeError):
        logging.warning(f"Warning: Could not update realized P/L with value: {amount}")


def delete_transaction_by_timestamp(market_name, timestamp_str):
    """
    Deletes a single transaction log entry identified by its timestamp string.
    WARNING: This ONLY removes the log entry. It does NOT adjust balance,
             holdings, or realized P/L. Use ONLY for correcting log errors
             where no other data adjustment is needed. Data inconsistency may result.
    """
    market_data = get_market_data(market_name)
    if not market_data:
        logging.error(f"Delete Log Error: Market '{market_name}' not found.")
        return False

    log = market_data.get('transaction_log', [])
    original_len = len(log)
    index_to_delete = -1

    for i, entry in enumerate(log):
        if entry.get('ts') == timestamp_str:
            index_to_delete = i
            break

    if index_to_delete != -1:
        try:
            deleted_entry = log.pop(index_to_delete)
            logging.warning(f"Manually deleted log entry for '{market_name}' (Timestamp: {timestamp_str}). "
                            f"DETAILS: {deleted_entry}. "
                            f"!! NO automatic adjustments made to balance, holdings, or P/L. Data may be inconsistent !!")
            # No need to update market_data['transaction_log'] directly, list 'log' was modified in place
            return True
        except IndexError:
            logging.error(f"Delete Log Error: Index {index_to_delete} out of bounds for '{market_name}' log (Concurrency issue?).")
            return False
        except Exception as e:
            logging.error(f"Delete Log Error: Unexpected error removing entry for '{market_name}': {e}", exc_info=True)
            return False
    else:
        logging.warning(f"Delete Log Warning: Transaction with timestamp '{timestamp_str}' not found in '{market_name}'.")
        return False

# --- Position Stats & State ---
def calculate_position_stats(market_name):
    """Calculates share counts, total investment, and average prices."""
    stats = {'yes_shares': 0.0, 'no_shares': 0.0, 'yes_avg_price': 0.0, 'no_avg_price': 0.0, 'yes_investment': 0.0, 'no_investment': 0.0}
    market_bets = get_bets(market_name)
    if not market_bets: return stats # Return default if no bets

    yes_total_amount = 0.0; no_total_amount = 0.0
    yes_total_shares = 0.0; no_total_shares = 0.0
    try:
        for bet in market_bets:
            amount = bet.get('amount', 0.0) # Current remaining basis amount
            price = bet.get('price', 0.0)   # Original price for this bet
            side = bet.get('side')

            # Validate data types
            if not isinstance(amount, (int, float)): amount = 0.0
            if not isinstance(price, (int, float)): price = 0.0

            # Use threshold comparison
            if amount < config.ZERO_SHARE_THRESHOLD or price <= config.ZERO_PRICE_THRESHOLD: continue

            # Shares represented by the remaining amount at the original price
            shares_in_bet = amount / price

            if side == 'YES':
                yes_total_shares += shares_in_bet
                yes_total_amount += amount # Sum remaining basis amounts
            elif side == 'NO':
                no_total_shares += shares_in_bet
                no_total_amount += amount

        # Round shares according to config for final stats display/use
        stats['yes_shares'] = config.SHARE_ROUNDING_FUNC(yes_total_shares)
        stats['no_shares'] = config.SHARE_ROUNDING_FUNC(no_total_shares)

        stats['yes_investment'] = round(yes_total_amount, 2)
        stats['no_investment'] = round(no_total_amount, 2)

        # Calculate average prices based on total amounts and precise shares before rounding
        stats['yes_avg_price'] = yes_total_amount / yes_total_shares if yes_total_shares > config.ZERO_SHARE_THRESHOLD else 0.0
        stats['no_avg_price'] = no_total_amount / no_total_shares if no_total_shares > config.ZERO_SHARE_THRESHOLD else 0.0

    except ZeroDivisionError: logging.warning(f"Warning: Zero division during stats calc for '{market_name}'.")
    except Exception as e: logging.error(f"Error calculating stats for '{market_name}': {e}", exc_info=True)
    return stats


def update_position_state(market_name):
    """Recalculates and updates the market's position state in _all_market_data."""
    market_data = get_market_data(market_name)
    if not market_data: return 'FLAT'

    stats = calculate_position_stats(market_name) # Use rounded shares from stats
    yes_s, no_s = stats['yes_shares'], stats['no_shares']
    avg_yes_p, avg_no_p = stats['yes_avg_price'], stats['no_avg_price'] # Use precise avg prices

    hedge_tolerance = getattr(config, 'HEDGE_IMBALANCE_TOLERANCE_SHARES', 1.0)
    old_state = market_data.get('position_state', 'FLAT')
    new_state = 'FLAT' # Default
    zero_threshold = config.ZERO_SHARE_THRESHOLD # Use configured threshold

    # Determine new state based on rounded share counts
    if yes_s <= zero_threshold and no_s <= zero_threshold:
        new_state = 'FLAT'
    elif yes_s > zero_threshold and no_s <= zero_threshold:
        new_state = 'DIRECTIONAL_YES'
    elif yes_s <= zero_threshold and no_s > zero_threshold:
        new_state = 'DIRECTIONAL_NO'
    elif yes_s > zero_threshold and no_s > zero_threshold:
        share_diff = abs(yes_s - no_s)
        if share_diff > hedge_tolerance:
            if yes_s > no_s: new_state = 'DIRECTIONAL_YES'
            else: new_state = 'DIRECTIONAL_NO'
        else:
            cost_sum = avg_yes_p + avg_no_p
            if cost_sum > config.ZERO_PRICE_THRESHOLD and cost_sum < config.COST_BASIS_ARB_THRESHOLD - config.ZERO_PRICE_THRESHOLD:
                new_state = 'COST_BASIS_ARB'
            else:
                new_state = 'HEDGED'
    else:
        logging.warning(f"State Determination Ambiguous: yes_s={yes_s}, no_s={no_s}. Defaulting to FLAT.")
        new_state = 'FLAT'

    # Update state and related fields if state changes or is initialized
    if old_state != new_state or 'position_state' not in market_data:
        market_data['position_state'] = new_state
        logging.info(f"Market '{market_name}' State Update: {old_state} -> {new_state}")
        if not new_state.startswith('DIRECTIONAL'):
            market_data['directional_stop_loss'] = None
            market_data['last_entry_side'] = None
        elif new_state == 'DIRECTIONAL_YES' and market_data.get('last_entry_side') != 'YES':
             market_data['directional_stop_loss'] = None # Clear old stop on direction flip
             market_data['last_entry_side'] = 'YES' # Set current direction
        elif new_state == 'DIRECTIONAL_NO' and market_data.get('last_entry_side') != 'NO':
             market_data['directional_stop_loss'] = None # Clear old stop on direction flip
             market_data['last_entry_side'] = 'NO' # Set current direction

    return new_state


def get_position_state(market_name):
    """Gets the current position state for a market."""
    market_data = get_market_data(market_name)
    return market_data.get('position_state', 'FLAT') if market_data else 'FLAT'

# --- Specific Field Getters/Setters ---
def get_directional_stop(market_name):
    market_data = get_market_data(market_name)
    return market_data.get('directional_stop_loss') if market_data else None

def set_directional_stop(market_name, price):
    market_data = get_market_data(market_name)
    if market_data:
        market_data['directional_stop_loss'] = price
        logging.info(f"Stop Loss for '{market_name}' set to: {utils.format_price(price)}")
        return True
    return False

def get_last_entry_side(market_name):
    market_data = get_market_data(market_name)
    return market_data.get('last_entry_side') if market_data else None

def set_last_entry_side(market_name, side):
    market_data = get_market_data(market_name)
    if market_data and side in ['YES', 'NO', None]:
        market_data['last_entry_side'] = side
        return True
    return False


# --- FIFO Logic ---
def fifo_reduce_holdings(market_name, side_to_reduce, shares_to_reduce_unrounded):
    """Reduces holdings using FIFO. Returns shortfall and cost basis of shares sold."""
    market_data = get_market_data(market_name)
    if not market_data: return shares_to_reduce_unrounded, 0.0

    market_bets = get_bets(market_name) # Get current bets
    shares_to_reduce = config.SHARE_ROUNDING_FUNC(shares_to_reduce_unrounded)

    if shares_to_reduce <= config.ZERO_SHARE_THRESHOLD:
        return 0.0, 0.0 # Nothing to reduce

    shares_remaining_to_reduce = shares_to_reduce
    indices_to_remove = []
    total_cost_basis_sold = 0.0
    temp_bets_copy = [bet.copy() for bet in market_bets] # Work on a copy

    logging.debug(f"  FIFO Reducing {side_to_reduce} by {utils.format_shares(shares_to_reduce)}...")

    for i in range(len(temp_bets_copy)):
        if shares_remaining_to_reduce <= config.ZERO_SHARE_THRESHOLD: break
        bet = temp_bets_copy[i]
        if bet.get('side') == side_to_reduce:
            try:
                orig_p = bet.get('price', 0.0); rem_amt = bet.get('amount', 0.0)
                if not isinstance(orig_p, (int, float)) or orig_p <= config.ZERO_PRICE_THRESHOLD: continue
                if not isinstance(rem_amt, (int, float)) or rem_amt < config.ZERO_SHARE_THRESHOLD: continue

                avail_sh_precise = rem_amt / orig_p
                avail_sh_rounded = config.SHARE_ROUNDING_FUNC(avail_sh_precise)

                # Ensure we don't try to reduce by more than available due to rounding
                sold_sh = min(shares_remaining_to_reduce, avail_sh_rounded)
                if sold_sh < config.ZERO_SHARE_THRESHOLD: continue

                cost_basis_this_lot = sold_sh * orig_p
                total_cost_basis_sold += cost_basis_this_lot

                new_remaining_amount = max(0.0, rem_amt - cost_basis_this_lot)
                temp_bets_copy[i]['amount'] = new_remaining_amount

                shares_remaining_to_reduce -= sold_sh
                shares_remaining_to_reduce = max(0.0, config.SHARE_ROUNDING_FUNC(shares_remaining_to_reduce))

                logging.debug(f"    Reduced {utils.format_shares(sold_sh)} from bet {bet.get('id', 'N/A')[:8]}. CostSold: {utils.format_currency(cost_basis_this_lot)}. RemAmt: {utils.format_currency(new_remaining_amount)}")
                if new_remaining_amount < config.ZERO_SHARE_THRESHOLD:
                    indices_to_remove.append(i)

            except ZeroDivisionError: logging.warning(f"    FIFO Skip bet {bet.get('id', 'N/A')[:8]} (zero price).")
            except Exception as e: logging.error(f"    FIFO ERROR reducing bet {bet.get('id','N/A')}: {e}", exc_info=True); return shares_remaining_to_reduce, total_cost_basis_sold

    # Update the actual market data with the modified bets list
    final_bets_list = [bet for i, bet in enumerate(temp_bets_copy) if i not in indices_to_remove]
    set_bets(market_name, final_bets_list)

    shortfall = max(0.0, shares_remaining_to_reduce)
    if shortfall > config.ZERO_SHARE_THRESHOLD:
         logging.warning(f"    FIFO Warning: Shortfall of {utils.format_shares(shortfall)} shares for {side_to_reduce}.")

    return shortfall, total_cost_basis_sold


# --- Market Resolution ---
def resolve_market(market_name, winning_side):
    """
    Resolves a market, calculating P/L, updating balance, logging, and clearing holdings.
    """
    global _global_current_balance, _global_total_realized_pl
    market_data = get_market_data(market_name)
    if not market_data:
        logging.error(f"Resolve Market Error: Market '{market_name}' not found.")
        return False, "Market not found."
    if market_data.get('position_state', 'FLAT') == 'FLAT':
         logging.warning(f"Resolve Market Info: Market '{market_name}' is already FLAT. No action taken.")
         return True, "Market already FLAT."

    is_test = market_data.get('is_test_market', False)
    stats = calculate_position_stats(market_name)
    winning_shares = 0.0
    winning_investment = 0.0

    if winning_side == 'YES':
        winning_shares = stats.get('yes_shares', 0.0)
        winning_investment = stats.get('yes_investment', 0.0)
    elif winning_side == 'NO':
        winning_shares = stats.get('no_shares', 0.0)
        winning_investment = stats.get('no_investment', 0.0)
    else:
        return False, "Invalid winning side."

    cash_inflow = winning_shares * 1.0 # Winning shares cash out at $1
    market_pnl = cash_inflow - winning_investment # P/L for this market

    # Log the resolution transaction (this handles balance update if not test market)
    tx_success = add_transaction(market_name=market_name,
                                 tx_type=f"RESOLUTION ({winning_side} Wins)",
                                 side=winning_side,
                                 shares=winning_shares,
                                 price=1.0,
                                 cash_flow=cash_inflow)

    if not tx_success:
        logging.error(f"Resolve Market Error: Failed to log RESOLUTION transaction for '{market_name}'. Halting.")
        return False, "Failed to log resolution transaction."

    # Update total realized P/L (only if not a test market)
    if not is_test:
        update_realized_pl(market_pnl)
        logging.info(f"Resolve Market: Updated total realized P/L by {utils.format_currency(market_pnl)} for '{market_name}'.")

    # Clear all holdings (bets) for this market
    set_bets(market_name, [])
    logging.info(f"Resolve Market: Cleared all holdings for '{market_name}'.")

    # Update market state to FLAT
    market_data['position_state'] = 'FLAT'
    market_data['directional_stop_loss'] = None
    market_data['last_entry_side'] = None
    logging.info(f"Resolve Market: State for '{market_name}' set to FLAT.")

    return True, f"Market resolved. P/L: {utils.format_currency(market_pnl)}"


# --- Persistence ---
def save_data():
    """Saves balance, realized P/L, and all market data to JSON."""
    global _all_market_data, _global_current_balance, _global_total_realized_pl
    logging.info(f"Attempting to save data to {config.SAVE_FILE}...")
    try:
        data_to_save = {
            'global_current_balance': _global_current_balance,
            'global_total_realized_pl': _global_total_realized_pl,
            'all_market_data': {}
        }
        for name, market_data in _all_market_data.items():
             market_data_copy = market_data.copy()
             adv_value = market_data_copy.get('adv')
             if adv_value == float('inf'): market_data_copy['adv'] = "inf"
             elif isinstance(adv_value, (int, float)): market_data_copy['adv'] = adv_value
             else: market_data_copy['adv'] = ""

             # Ensure all expected keys exist for robustness
             market_data_copy.setdefault('bets_list', [])
             market_data_copy.setdefault('transaction_log', [])
             # Added bid/ask defaults
             market_data_copy.setdefault('last_yes_bid', "")
             market_data_copy.setdefault('last_yes_ask', "")
             market_data_copy.setdefault('last_no_bid', "")
             market_data_copy.setdefault('last_no_ask', "")
             market_data_copy.setdefault('position_state', 'FLAT')
             market_data_copy.setdefault('directional_stop_loss', None)
             market_data_copy.setdefault('last_entry_side', None)
             market_data_copy.setdefault('is_test_market', False)

             data_to_save['all_market_data'][name] = market_data_copy

        if len(data_to_save['all_market_data']) != len(_all_market_data):
            logging.critical(f"!!! SAVING WARNING: Mismatch in market count during save! Expected {len(_all_market_data)}, Saved {len(data_to_save['all_market_data'])}")

        with open(config.SAVE_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        logging.info(f"Data saved successfully. {len(data_to_save['all_market_data'])} market(s).")
        return True
    except TypeError as te: logging.error(f"Save Error: Data type error saving JSON: {te}", exc_info=True); return False
    except IOError as e: logging.error(f"Save Error: Could not write to file {config.SAVE_FILE}: {e}"); return False
    except Exception as e: logging.error(f"Save Error: Unexpected error saving data: {e}", exc_info=True); return False

def load_data():
    """Loads data, converts ADV 'inf', ensures new fields exist."""
    global _all_market_data, _global_current_balance, _global_total_realized_pl
    _all_market_data = {} # Start fresh
    _global_current_balance = 1000.0 # Default
    _global_total_realized_pl = 0.0 # Default

    if os.path.exists(config.SAVE_FILE):
        logging.info(f"Loading data from {config.SAVE_FILE}...")
        try:
            with open(config.SAVE_FILE, 'r') as f: loaded_data = json.load(f)

            loaded_balance = loaded_data.get('global_current_balance', 1000.0)
            try: _global_current_balance = float(loaded_balance)
            except (ValueError, TypeError): _global_current_balance = 1000.0

            loaded_pl = loaded_data.get('global_total_realized_pl', 0.0)
            try: _global_total_realized_pl = float(loaded_pl)
            except (ValueError, TypeError): _global_total_realized_pl = 0.0

            loaded_markets = loaded_data.get('all_market_data', {})
            for name, market_data in loaded_markets.items():
                 market_data_copy = market_data.copy()

                 adv_value = market_data_copy.get('adv')
                 if isinstance(adv_value, str):
                     if adv_value.lower() == "inf": market_data_copy['adv'] = float('inf')
                     elif adv_value.strip() == "": market_data_copy['adv'] = ""
                     else:
                          try: market_data_copy['adv'] = float(adv_value)
                          except ValueError: market_data_copy['adv'] = ""
                 elif adv_value is None: market_data_copy['adv'] = ""
                 if not isinstance(market_data_copy['adv'], (int, float)) and market_data_copy['adv'] != "":
                       market_data_copy['adv'] = ""

                 # Ensure required fields exist with defaults
                 market_data_copy.setdefault('bets_list', [])
                 market_data_copy.setdefault('transaction_log', [])
                 # Added bid/ask defaults
                 market_data_copy.setdefault('last_yes_bid', "")
                 market_data_copy.setdefault('last_yes_ask', "")
                 market_data_copy.setdefault('last_no_bid', "")
                 market_data_copy.setdefault('last_no_ask', "")
                 market_data_copy.setdefault('position_state', 'FLAT')
                 market_data_copy.setdefault('directional_stop_loss', None)
                 market_data_copy.setdefault('last_entry_side', None)
                 market_data_copy.setdefault('is_test_market', False)

                 _all_market_data[name] = market_data_copy

            logging.info(f"Data loaded successfully. Balance: {utils.format_currency(_global_current_balance)}, Realized P/L: {utils.format_currency(_global_total_realized_pl)}, Markets: {len(_all_market_data)}")
            return True
        except json.JSONDecodeError as e: logging.error(f"Load Error: Reading {config.SAVE_FILE} (corrupted?). Starting fresh.\n{e}"); return False
        except Exception as e: logging.error(f"Load Error: Unexpected error loading data.\n{e}", exc_info=True); return False
    else:
        logging.info("Save file not found. Starting with default balance and empty data.")
        return True