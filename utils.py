# utils.py
import tkinter as tk
# from tkinter import messagebox # Keep UI elements out of utility functions
import config
import logging # Import logging module
import math

# --- Logging Setup ---
def setup_logging():
    """Configures the logging module."""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(config.LOG_FILE), # Log to file
            logging.StreamHandler()                # Log to console
        ]
    )
    logging.info("Logging configured.")

# --- Input Validation ---
def validate_price(price_str, field_name="Price"):
    """Validates price is between 0 and 1 (exclusive). Returns float or None."""
    if not price_str:
        logging.warning(f"Validation Warning: {field_name} empty.")
        return None
    try:
        price = float(price_str)
        if not (0 < price < 1):
            logging.warning(f"Validation Warning: {field_name} must be 0 < price < 1. Got: {price}")
            return None
        return price
    except (ValueError, TypeError):
        logging.warning(f"Validation Warning: Invalid number format for {field_name}: {price_str}")
        return None
    except Exception as e:
        logging.error(f"Unexpected Price validation error: {e}", exc_info=True)
        return None

def validate_shares(shares_str, field_name="Shares"):
    """Validates shares are non-negative. Returns float or None."""
    if not shares_str:
        logging.warning(f"Validation Warning: {field_name} empty.")
        return None
    try:
        shares = float(shares_str)
        if shares < 0:
            logging.warning(f"Validation Warning: {field_name} must be non-negative. Got: {shares}")
            return None
        return shares # Return raw float, rounding happens before execution
    except (ValueError, TypeError):
        logging.warning(f"Validation Warning: Invalid number format for {field_name}: {shares_str}")
        return None
    except Exception as e:
        logging.error(f"Unexpected Shares validation error: {e}", exc_info=True)
        return None

def validate_adv(adv_str, field_name="ADV"):
    """Validates ADV is non-negative or 'inf'. Returns float or float('inf') or None."""
    if not adv_str or adv_str.strip() == "" or adv_str.strip().lower() == "inf":
        return float('inf')
    try:
        adv = float(adv_str)
        if adv < 0:
            logging.warning(f"Validation Warning: {field_name} cannot be negative. Got: {adv}")
            return None
        return adv
    except ValueError:
        logging.warning(f"Validation Warning: Invalid number format for {field_name}: {adv_str}")
        return None
    except Exception as e:
        logging.error(f"Unexpected ADV validation error: {e}", exc_info=True)
        return None

# --- Formatting Helpers ---
def format_shares(shares):
    if shares is None: return "N/A"
    try: return f"{shares:.{config.SHARE_DECIMALS}f}"
    except (ValueError, TypeError): return "Error"

def format_price(price):
    if price is None: return "N/A"
    try: return f"${price:.4f}"
    except (ValueError, TypeError): return "Error"

def format_currency(amount):
    if amount is None: return "N/A"
    try: return f"${amount:,.2f}"
    except (ValueError, TypeError): return "Error"

def format_cash_flow(amount):
    if amount is None: return "N/A"
    try: return f"{'+' if amount >= 0 else '-'}${abs(amount):,.2f}"
    except (ValueError, TypeError): return "Error"

def format_percent(value):
    if value is None: return "N/A"
    try: return f"{value:.2%}"
    except (ValueError, TypeError): return "Error"

# --- Text Widget Helper (If needed for UI) ---
def set_text_widget_content(widget, text, tag=None):
    """Safely updates the content of a Text widget."""
    if widget is None: return
    try:
        widget.config(state='normal')
        widget.delete('1.0', tk.END)
        if text:
            if tag: widget.insert('1.0', text, tag)
            else: widget.insert('1.0', text)
        widget.config(state='disabled')
    except tk.TclError as e:
        logging.warning(f"Text widget update error: {e}") # Handle if widget destroyed
    except Exception as e:
         logging.error(f"Unexpected error setting text widget content: {e}", exc_info=True)

# --- Statistics Update Helper ---
def update_average_price(current_shares, current_avg_price, shares_traded, price_traded):
    """
    Calculates the new average price after a trade using weighted average.
    Assumes shares_traded is positive for buys, could be adapted for sells if needed
    but typically avg price only updated on buys.
    """
    # Ensure numeric types
    try:
        current_shares = float(current_shares) if current_shares is not None else 0.0
        current_avg_price = float(current_avg_price) if current_avg_price is not None else 0.0
        shares_traded = float(shares_traded) if shares_traded is not None else 0.0
        price_traded = float(price_traded) if price_traded is not None else 0.0
    except (ValueError, TypeError) as e:
        logging.error(f"Invalid input type for average price calculation: {e}")
        return current_avg_price # Return old average on error

    # Handle initial buy or invalid inputs
    if shares_traded <= 0 or price_traded <= 0:
        return current_avg_price # No change if no shares traded or price invalid
    if current_shares <= 0:
        return price_traded # First buy, average price is the trade price

    # Weighted average calculation
    old_total_investment = current_shares * current_avg_price
    new_trade_cost = shares_traded * price_traded
    new_total_shares = current_shares + shares_traded
    
    if new_total_shares <= 0: # Avoid division by zero
        return 0.0 # Or handle as appropriate

    new_average = (old_total_investment + new_trade_cost) / new_total_shares
    logging.debug(f"AvgPrice Update: OldShares={format_shares(current_shares)}, OldAvg={format_price(current_avg_price)}, "
                 f"TradedShares={format_shares(shares_traded)}, TradePrice={format_price(price_traded)} -> NewAvg={format_price(new_average)}")
    return new_average

# --- State Determination Logic ---
def determine_new_state(stats):
    """
    Determines the correct position state based on current share holdings and average costs.
    MUST be called *after* stats (shares, avg prices, investments) are updated post-trade.

    Args:
        stats (dict): The updated statistics dictionary for the market.
                      Expected keys: 'yes_shares', 'no_shares', 'yes_avg_price', 'no_avg_price'.

    Returns:
        str: The calculated position state ('FLAT', 'DIRECTIONAL_YES', 'DIRECTIONAL_NO', 'HEDGED', 'COST_BASIS_ARB').
    """
    try:
        yes_s = stats.get('yes_shares', 0.0) or 0.0
        no_s = stats.get('no_shares', 0.0) or 0.0
        avg_yes_p = stats.get('yes_avg_price', 0.0) or 0.0
        avg_no_p = stats.get('no_avg_price', 0.0) or 0.0
        
        # Use a small tolerance to avoid float precision issues with zero check
        tolerance = 1e-9 

        # 1. Check for FLAT state
        if yes_s < tolerance and no_s < tolerance:
            logging.debug("Determined State: FLAT (No shares)")
            return 'FLAT'

        # 2. Check for HEDGED states (requires shares on both sides)
        if yes_s > tolerance and no_s > tolerance:
            imbalance = abs(yes_s - no_s)
            logging.debug(f"State Check: Both sides held. Yes={format_shares(yes_s)}, No={format_shares(no_s)}, Imbalance={format_shares(imbalance)}")
            
            if imbalance <= config.HEDGE_IMBALANCE_TOLERANCE_SHARES:
                # Balanced enough, check for Cost Basis Arb condition
                if avg_yes_p > 0 and avg_no_p > 0: # Need valid average costs
                    cost_sum = avg_yes_p + avg_no_p
                    logging.debug(f"State Check: Balanced. AvgCostSum={format_price(cost_sum)} vs Threshold={format_price(config.COST_BASIS_ARB_THRESHOLD)}")
                    if cost_sum < config.COST_BASIS_ARB_THRESHOLD:
                        logging.debug("Determined State: COST_BASIS_ARB")
                        return 'COST_BASIS_ARB'
                    else:
                        logging.debug("Determined State: HEDGED")
                        return 'HEDGED'
                else:
                    # Balanced but invalid cost basis, default to HEDGED
                    logging.debug("Determined State: HEDGED (Balanced, but avg cost invalid/zero)")
                    return 'HEDGED'
            # else: Imbalanced, fall through to directional checks

        # 3. Check for DIRECTIONAL states (only one side held, or significantly imbalanced)
        if yes_s > no_s:
            logging.debug("Determined State: DIRECTIONAL_YES")
            return 'DIRECTIONAL_YES'
        elif no_s > yes_s:
            logging.debug("Determined State: DIRECTIONAL_NO")
            return 'DIRECTIONAL_NO'
        else:
            # This case should theoretically be caught by HEDGED check if imbalance <= tolerance,
            # but acts as a fallback if tolerance is very small or floats are tricky.
            logging.warning(f"State Determination Ambiguous: yes_s={yes_s}, no_s={no_s}. Defaulting to HEDGED as likely balanced.")
            return 'HEDGED'

    except Exception as e:
        logging.error(f"Error determining new state: {e}", exc_info=True)
        # Fallback: return a default state? Or raise? For now, log error and return FLAT.
        return 'FLAT' # Or maybe the previous state if available?