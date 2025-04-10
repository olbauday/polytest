# strategy_engine.py

import math
import logging
from . import config # Make sure config is imported
# Assuming you have utils or similar for formatting/calculations
# from . import utils

# Get the logger instance (assuming it's configured elsewhere)
app_logger = logging.getLogger('backtester_app') # Or your specific logger name

# --- Helper functions (Placeholder - Assume these exist) ---
def calculate_shares_to_buy_directional(balance, price, risk_pct, stop_loss_pct):
    """Calculates shares based on risk percentage and stop loss."""
    if price <= 0 or stop_loss_pct <= 0 or stop_loss_pct >= 1:
        return 0 # Avoid division by zero or invalid stop loss
    risk_amount = balance * risk_pct
    risk_per_share = price * stop_loss_pct
    if risk_per_share <= config.ZERO_PRICE_THRESHOLD:
         return 0 # Avoid division by zero
    shares = risk_amount / risk_per_share
    # Use config rounding
    return config.SHARE_ROUNDING_FUNC(shares)

def calculate_shares_to_accumulate(balance, price, size_pct):
    """Calculates shares based on a percentage of balance for accumulation."""
    if price <= 0: return 0
    amount_to_invest = balance * size_pct
    shares = amount_to_invest / price
    return config.SHARE_ROUNDING_FUNC(shares)

def calculate_shares_for_hedge(balance, price, size_pct):
    """Calculates shares for hedging."""
    # Similar logic to accumulate, might be the same function call
    if price <= 0: return 0
    amount_to_invest = balance * size_pct
    shares = amount_to_invest / price
    return config.SHARE_ROUNDING_FUNC(shares)

def calculate_shares_for_arb(balance, price, size_pct):
    """Calculates shares for arbitrage trades."""
    if price <= 0: return 0 # Price here is often sum (Y_ask + N_ask) or (Y_bid + N_bid)
    amount_to_invest_or_sell = balance * size_pct # Simplified, might need adjustment based on action
    shares = amount_to_invest_or_sell / price
    return config.SHARE_ROUNDING_FUNC(shares)

# --- Main Strategy Function ---

def calculate_strategy_recommendation(market_name, yes_bid, yes_ask, no_bid, no_ask, market_adv, market_data, stats, current_balance):
    """
    Calculates the recommended trading action based on market state and strategy rules.

    Args:
        market_name (str): Name of the market.
        yes_bid (float): Current highest bid price for YES shares.
        yes_ask (float): Current lowest ask price for YES shares.
        no_bid (float): Current highest bid price for NO shares.
        no_ask (float): Current lowest ask price for NO shares.
        market_adv (float): Available depth value from the market (used for some size calcs, maybe).
        market_data (dict): Current state specific to this market (position_state, stop_loss).
        stats (dict): Current portfolio statistics (share counts, average prices).
        current_balance (float): Current available cash balance.

    Returns:
        tuple: (recommendation, analysis_details)
            recommendation (dict): Details of the recommended action (action_type, side, price, shares, etc.)
                                   Returns None if an error occurs during calculation.
            analysis_details (dict): Supporting information about the decision process.
    """
    recommendation = None
    analysis_details = {
        'market_name': market_name,
        'position_state': market_data.get('position_state', 'UNKNOWN'),
        'directional_stop_loss': market_data.get('directional_stop_loss'),
        'yes_bid': yes_bid, 'yes_ask': yes_ask,
        'no_bid': no_bid, 'no_ask': no_ask,
        'yes_shares': stats.get('yes_shares', 0.0),
        'no_shares': stats.get('no_shares', 0.0),
        'yes_avg_price': stats.get('yes_avg_price', 0.0),
        'no_avg_price': stats.get('no_avg_price', 0.0),
        'balance': current_balance,
        'reasoning': [] # Store steps of the decision logic
    }

    try:
        # --- Initialize default action ---
        action_type = 'HOLD'
        side = 'NONE'
        price = 0.0
        shares_intended = 0.0
        shares_rounded = 0.0
        display_text = "Hold: No suitable action identified."
        calculated_stop_level = None # For passing back calculated stops

        # --- Extract current state ---
        position_state = market_data.get('position_state', 'FLAT')
        directional_stop = market_data.get('directional_stop_loss') # Price level for stop
        yes_shares = stats.get('yes_shares', 0.0)
        no_shares = stats.get('no_shares', 0.0)
        yes_avg_price = stats.get('yes_avg_price', 0.0)
        no_avg_price = stats.get('no_avg_price', 0.0)

        # --- Log entry state ---
        app_logger.debug(f"State Check: {market_name} | State={position_state} | Ys={yes_shares:.2f}@{yes_avg_price:.4f} | Ns={no_shares:.2f}@{no_avg_price:.4f} | Bal={current_balance:.2f}")
        app_logger.debug(f"Market Prices: Y Bid={yes_bid:.4f} Ask={yes_ask:.4f} | N Bid={no_bid:.4f} Ask={no_ask:.4f}")
        analysis_details['reasoning'].append(f"Initial State: {position_state}, Ys: {yes_shares:.2f}, Ns: {no_shares:.2f}")

        # ======================================================================
        # 1. Check Stop Losses FIRST (applies to DIRECTIONAL states primarily)
        # ======================================================================
        if position_state == 'DIRECTIONAL_YES' and directional_stop is not None:
            if yes_bid <= directional_stop: # Use BID price for stop loss check (selling price)
                action_type = 'SELL_STOP'
                side = 'ALL_YES' # Sell all YES shares
                price = yes_bid # Execute at current bid
                shares_intended = yes_shares # Target all shares
                shares_rounded = config.SHARE_ROUNDING_FUNC(shares_intended) # Round actual shares
                display_text = f"STOP LOSS Hit (YES): Sell ALL {shares_rounded:.{config.SHARE_DECIMALS}f} YES @ ~{price:.4f} (Stop was {directional_stop:.4f})"
                app_logger.warning(f"{market_name} - {display_text}")
                analysis_details['reasoning'].append(f"Stop loss triggered for YES at {yes_bid:.4f} (<= {directional_stop:.4f})")
                # Stop further checks, execute the stop loss
                recommendation = {'action_type': action_type, 'side': side, 'price': price, 'shares_intended': shares_intended, 'shares_rounded': shares_rounded, 'display_text': display_text, 'calculated_stop_level': None}
                return recommendation, analysis_details

        elif position_state == 'DIRECTIONAL_NO' and directional_stop is not None:
            if no_bid <= directional_stop: # Use BID price for stop loss check
                action_type = 'SELL_STOP'
                side = 'ALL_NO' # Sell all NO shares
                price = no_bid # Execute at current bid
                shares_intended = no_shares # Target all shares
                shares_rounded = config.SHARE_ROUNDING_FUNC(shares_intended) # Round actual shares
                display_text = f"STOP LOSS Hit (NO): Sell ALL {shares_rounded:.{config.SHARE_DECIMALS}f} NO @ ~{price:.4f} (Stop was {directional_stop:.4f})"
                app_logger.warning(f"{market_name} - {display_text}")
                analysis_details['reasoning'].append(f"Stop loss triggered for NO at {no_bid:.4f} (<= {directional_stop:.4f})")
                 # Stop further checks, execute the stop loss
                recommendation = {'action_type': action_type, 'side': side, 'price': price, 'shares_intended': shares_intended, 'shares_rounded': shares_rounded, 'display_text': display_text, 'calculated_stop_level': None}
                return recommendation, analysis_details

        # ======================================================================
        # 2. Check Arbitrage Opportunities (can occur in any state)
        # ======================================================================
        # --- Check Market Arbitrage (Buy Y+N for < $1 or Sell Y+N for > $1) ---
        buy_arb_opportunity = (yes_ask + no_ask) < (1.0 - config.ARB_THRESHOLD)
        sell_arb_opportunity = (yes_bid + no_bid) > (1.0 + config.ARB_THRESHOLD)

        if buy_arb_opportunity and config.ENABLE_MARKET_ARBITRAGE:
            action_type = 'BUY_ARB'
            side = 'PAIR'
            price = yes_ask + no_ask # Cost per pair
            shares_intended = calculate_shares_for_arb(current_balance, price, config.ARB_BUY_SIZE_PCT_OF_BALANCE)
            shares_rounded = config.SHARE_ROUNDING_FUNC(shares_intended)
            cost = shares_rounded * price
            if shares_rounded > config.ZERO_SHARE_THRESHOLD and current_balance >= cost:
                display_text = f"Market Arb: Buy {shares_rounded:.{config.SHARE_DECIMALS}f} PAIRS @ {price:.4f} (< {1.0 - config.ARB_THRESHOLD:.4f})"
                app_logger.info(f"{market_name} - {display_text}")
                analysis_details['reasoning'].append(f"Market arbitrage buy opportunity detected (Sum Ask={price:.4f})")
                # Prioritize Arb? Or check state first? Let's prioritize simple arb for now.
                recommendation = {'action_type': action_type, 'side': side, 'price': price, 'shares_intended': shares_intended, 'shares_rounded': shares_rounded, 'display_text': display_text, 'calculated_stop_level': None}
                return recommendation, analysis_details
            else:
                app_logger.debug(f"Market Arb Buy condition met, but cannot afford ({cost:.2f} > {current_balance:.2f}) or shares too small ({shares_rounded})")
                analysis_details['reasoning'].append(f"Market arbitrage buy opportunity ignored (affordability/size)")

        # Check Sell Arb only if not doing Buy Arb
        elif sell_arb_opportunity and config.ENABLE_MARKET_ARBITRAGE:
             action_type = 'SELL_ARB'
             side = 'PAIR'
             price = yes_bid + no_bid # Proceeds per pair
             # Base sell size on minimum held shares? Or a %? Let's use a % of min shares for simplicity.
             min_shares = min(yes_shares, no_shares)
             # For backtesting, maybe just try to sell a fixed % of balance equivalent? Or % of held? Let's try % of min held.
             # This sizing needs refinement - config.ARB_SELL_SIZE_PCT_OF_MIN_SHARES ?
             # Let's use the BUY size % applied to current balance / sell price for simplicity here.
             shares_intended = calculate_shares_for_arb(current_balance, price, config.ARB_BUY_SIZE_PCT_OF_BALANCE) # Reusing buy size % logic
             shares_intended = min(shares_intended, min_shares) # Can't sell more than we have
             shares_rounded = config.SHARE_ROUNDING_FUNC(shares_intended)

             if shares_rounded > config.ZERO_SHARE_THRESHOLD and min_shares >= shares_rounded:
                 display_text = f"Market Arb: Sell {shares_rounded:.{config.SHARE_DECIMALS}f} PAIRS @ {price:.4f} (> {1.0 + config.ARB_THRESHOLD:.4f})"
                 app_logger.info(f"{market_name} - {display_text}")
                 analysis_details['reasoning'].append(f"Market arbitrage sell opportunity detected (Sum Bid={price:.4f})")
                 recommendation = {'action_type': action_type, 'side': side, 'price': price, 'shares_intended': shares_intended, 'shares_rounded': shares_rounded, 'display_text': display_text, 'calculated_stop_level': None}
                 return recommendation, analysis_details
             else:
                 app_logger.debug(f"Market Arb Sell condition met, but insufficient shares ({min_shares:.2f} < {shares_rounded}) or shares too small")
                 analysis_details['reasoning'].append(f"Market arbitrage sell opportunity ignored (insufficient shares/size)")


        # ======================================================================
        # 3. Evaluate Actions based on Position State (if no stop/arb triggered)
        # ======================================================================
        action_type = 'HOLD' # Reset action type if no stop/arb
        side = 'NONE'
        price = 0.0
        shares_intended = 0.0
        shares_rounded = 0.0
        display_text = "Hold: No suitable action identified."

        # ----------------------------------------------------------------------
        # State: FLAT - Look for initial entry
        # ----------------------------------------------------------------------
        if position_state == 'FLAT':
            app_logger.debug("State: FLAT - Evaluating basic buy opportunities.")
            analysis_details['reasoning'].append("State is FLAT. Evaluating initial buy.")

            # --- Evaluate BUY YES ---
            # **** MODIFIED CONDITION: Check price is within the range [MIN, MAX] ****
            if config.MIN_BUY_PRICE <= yes_ask <= config.BUY_THRESHOLD:
                app_logger.debug(f"BUY YES condition met: Price {yes_ask:.4f} is within [{config.MIN_BUY_PRICE:.4f}, {config.BUY_THRESHOLD:.4f}]")
                analysis_details['reasoning'].append(f"YES price {yes_ask:.4f} is within buy range [{config.MIN_BUY_PRICE:.4f}, {config.BUY_THRESHOLD:.4f}].")

                # Calculate shares based on risk and calculate potential stop loss level
                shares_intended = calculate_shares_to_buy_directional(
                    current_balance,
                    yes_ask,
                    config.RISK_PER_TRADE_PCT,
                    config.DIRECTIONAL_STOP_LOSS_PCT
                )
                shares_rounded = config.SHARE_ROUNDING_FUNC(shares_intended)
                cost = shares_rounded * yes_ask
                calculated_stop_level = yes_ask * (1.0 - config.DIRECTIONAL_STOP_LOSS_PCT) if config.DIRECTIONAL_STOP_LOSS_PCT else None

                if shares_rounded > config.ZERO_SHARE_THRESHOLD and current_balance >= cost:
                    action_type = 'BUY'
                    side = 'YES'
                    price = yes_ask
                    display_text = f"BUY YES: {shares_rounded:.{config.SHARE_DECIMALS}f} @ {price:.4f} (Stop @ ~{calculated_stop_level:.4f})"
                    app_logger.info(f"{market_name} - {display_text}")
                    analysis_details['reasoning'].append(f"Recommendation: BUY {shares_rounded} YES. Cost={cost:.2f}. Stop={calculated_stop_level:.4f}")
                else:
                    reason_fail = "Affordability" if current_balance < cost else "Size Zero"
                    app_logger.debug(f"BUY YES condition met, but failed ({reason_fail}). Needed {cost:.2f}, have {current_balance:.2f}. Shares: {shares_rounded}")
                    analysis_details['reasoning'].append(f"BUY YES considered but rejected ({reason_fail}).")
                    action_type = 'HOLD' # Ensure hold if buy fails

            # --- Evaluate BUY NO (only if BUY YES wasn't chosen) ---
            # **** MODIFIED CONDITION: Check price is within the range [MIN, MAX] ****
            elif config.MIN_BUY_PRICE <= no_ask <= config.BUY_THRESHOLD:
                app_logger.debug(f"BUY NO condition met: Price {no_ask:.4f} is within [{config.MIN_BUY_PRICE:.4f}, {config.BUY_THRESHOLD:.4f}]")
                analysis_details['reasoning'].append(f"NO price {no_ask:.4f} is within buy range [{config.MIN_BUY_PRICE:.4f}, {config.BUY_THRESHOLD:.4f}].")

                # Calculate shares based on risk and calculate potential stop loss level
                shares_intended = calculate_shares_to_buy_directional(
                    current_balance,
                    no_ask,
                    config.RISK_PER_TRADE_PCT,
                    config.DIRECTIONAL_STOP_LOSS_PCT
                )
                shares_rounded = config.SHARE_ROUNDING_FUNC(shares_intended)
                cost = shares_rounded * no_ask
                calculated_stop_level = no_ask * (1.0 - config.DIRECTIONAL_STOP_LOSS_PCT) if config.DIRECTIONAL_STOP_LOSS_PCT else None

                if shares_rounded > config.ZERO_SHARE_THRESHOLD and current_balance >= cost:
                    action_type = 'BUY'
                    side = 'NO'
                    price = no_ask
                    display_text = f"BUY NO: {shares_rounded:.{config.SHARE_DECIMALS}f} @ {price:.4f} (Stop @ ~{calculated_stop_level:.4f})"
                    app_logger.info(f"{market_name} - {display_text}")
                    analysis_details['reasoning'].append(f"Recommendation: BUY {shares_rounded} NO. Cost={cost:.2f}. Stop={calculated_stop_level:.4f}")
                else:
                    reason_fail = "Affordability" if current_balance < cost else "Size Zero"
                    app_logger.debug(f"BUY NO condition met, but failed ({reason_fail}). Needed {cost:.2f}, have {current_balance:.2f}. Shares: {shares_rounded}")
                    analysis_details['reasoning'].append(f"BUY NO considered but rejected ({reason_fail}).")
                    action_type = 'HOLD' # Ensure hold if buy fails
            else:
                 # Log why no buy happened if checks were performed
                 app_logger.debug(f"Neither YES ({yes_ask:.4f}) nor NO ({no_ask:.4f}) price is within the buy range [{config.MIN_BUY_PRICE:.4f}, {config.BUY_THRESHOLD:.4f}]. Holding.")
                 analysis_details['reasoning'].append(f"Neither side met buy range criteria: YES={yes_ask:.4f}, NO={no_ask:.4f}. Range=[{config.MIN_BUY_PRICE:.4f}, {config.BUY_THRESHOLD:.4f}]")
                 action_type = 'HOLD'

        # ----------------------------------------------------------------------
        # State: DIRECTIONAL_YES - Look to accumulate, take profit, or hedge
        # ----------------------------------------------------------------------
        elif position_state == 'DIRECTIONAL_YES':
            app_logger.debug(f"State: DIRECTIONAL_YES (AvgP: {yes_avg_price:.4f})")
            analysis_details['reasoning'].append(f"State is DIRECTIONAL_YES (AvgP: {yes_avg_price:.4f}).")

            # --- 1. Check Profit Taking ---
            if yes_bid >= config.PROFIT_TAKE_PRICE_THRESHOLD:
                shares_to_sell = yes_shares * config.PROFIT_TAKE_SELL_PCT
                shares_rounded = config.SHARE_ROUNDING_FUNC(shares_to_sell)
                if shares_rounded > config.ZERO_SHARE_THRESHOLD:
                    action_type = 'SELL'
                    side = 'YES'
                    price = yes_bid
                    display_text = f"Profit Take (YES): Sell {shares_rounded:.{config.SHARE_DECIMALS}f} ({config.PROFIT_TAKE_SELL_PCT*100:.0f}%) @ {price:.4f} (Threshold: >{config.PROFIT_TAKE_PRICE_THRESHOLD:.4f})"
                    app_logger.info(f"{market_name} - {display_text}")
                    analysis_details['reasoning'].append(f"Profit taking condition met for YES (Bid {yes_bid:.4f} >= {config.PROFIT_TAKE_PRICE_THRESHOLD:.4f}). Selling {shares_rounded}.")
                    # Profit take overrides other actions except stop loss (already checked)
                    recommendation = {'action_type': action_type, 'side': side, 'price': price, 'shares_intended': shares_to_sell, 'shares_rounded': shares_rounded, 'display_text': display_text, 'calculated_stop_level': directional_stop} # Keep existing stop
                    return recommendation, analysis_details

            # --- 2. Check Accumulation ---
            # Condition: Price drops significantly below average cost, but not stopped out
            price_drop_pct = (yes_avg_price - yes_ask) / yes_avg_price if yes_avg_price > 0 else 0
            should_accumulate = (price_drop_pct >= config.ACCUMULATION_DROP_THRESHOLD and
                                 (directional_stop is None or yes_ask > directional_stop) and # Ensure not below stop
                                 config.ENABLE_ACCUMULATION)

            # Accumulation logic also needs to respect the BUY_THRESHOLD and MIN_BUY_PRICE? Optional. Let's keep it simple for now.
            # Let's add the check: accumulation only if current price is still below original BUY_THRESHOLD ceiling?
            # allow_accum_price = yes_ask <= config.BUY_THRESHOLD # Check if still below max buy price
            allow_accum_price = True # Simpler: Accumulate based on drop, assume initial entry was valid

            if should_accumulate and allow_accum_price:
                # Check optional accumulation stop loss
                accum_stop_triggered = False
                if config.ACCUMULATION_STOP_LOSS_PCT is not None:
                     accum_stop_level = yes_avg_price * (1.0 - config.ACCUMULATION_STOP_LOSS_PCT)
                     if yes_ask <= accum_stop_level:
                          accum_stop_triggered = True
                          app_logger.warning(f"Accumulation prevented: Price {yes_ask:.4f} below accumulation stop level {accum_stop_level:.4f}")
                          analysis_details['reasoning'].append(f"Accumulation prevented by accumulation stop loss ({yes_ask:.4f} <= {accum_stop_level:.4f}).")

                if not accum_stop_triggered:
                     shares_intended = calculate_shares_to_accumulate(current_balance, yes_ask, config.ACCUMULATION_SIZE_PCT_OF_BALANCE)
                     shares_rounded = config.SHARE_ROUNDING_FUNC(shares_intended)
                     cost = shares_rounded * yes_ask
                     if shares_rounded > config.ZERO_SHARE_THRESHOLD and current_balance >= cost:
                          action_type = 'BUY'
                          side = 'YES'
                          price = yes_ask
                          # Recalculate stop based on new average price? Or keep original? Let's keep original for now.
                          # New avg price would be (stats['yes_investment'] + cost) / (yes_shares + shares_rounded)
                          # new_stop = new_avg_price * (1.0 - config.DIRECTIONAL_STOP_LOSS_PCT)
                          display_text = f"Accumulate YES: Buy {shares_rounded:.{config.SHARE_DECIMALS}f} @ {price:.4f} (Drop: {price_drop_pct*100:.1f}%)"
                          app_logger.info(f"{market_name} - {display_text}")
                          analysis_details['reasoning'].append(f"Accumulation condition met (Drop {price_drop_pct*100:.1f}% >= {config.ACCUMULATION_DROP_THRESHOLD*100:.1f}%). Buying {shares_rounded}.")
                          # Accumulation overrides hedging for now
                          recommendation = {'action_type': action_type, 'side': side, 'price': price, 'shares_intended': shares_intended, 'shares_rounded': shares_rounded, 'display_text': display_text, 'calculated_stop_level': directional_stop} # Keep existing stop
                          return recommendation, analysis_details
                     else:
                          reason_fail = "Affordability" if current_balance < cost else "Size Zero"
                          app_logger.debug(f"Accumulate YES condition met, but failed ({reason_fail}). Needed {cost:.2f}, have {current_balance:.2f}. Shares: {shares_rounded}")
                          analysis_details['reasoning'].append(f"Accumulate YES considered but rejected ({reason_fail}).")

            # --- 3. Check Hedging ---
            # Condition: Price drops significantly, consider buying NO shares
            should_hedge = (price_drop_pct >= config.HEDGE_PRICE_DROP_THRESHOLD and config.ENABLE_HEDGING)
            if should_hedge and action_type == 'HOLD': # Only hedge if not accumulating or profit taking
                 shares_intended = calculate_shares_for_hedge(current_balance, no_ask, config.HEDGE_BUY_SIZE_PCT_OF_BALANCE)
                 # Potentially limit hedge shares to match existing YES shares? config.HEDGE_MATCH_SHARES = True/False
                 if config.HEDGE_MATCH_SHARES:
                     shares_intended = min(shares_intended, yes_shares) # Don't buy more NO than we have YES

                 shares_rounded = config.SHARE_ROUNDING_FUNC(shares_intended)
                 cost = shares_rounded * no_ask
                 if shares_rounded > config.ZERO_SHARE_THRESHOLD and current_balance >= cost:
                     action_type = 'BUY'
                     side = 'NO'
                     price = no_ask
                     display_text = f"Hedge YES Position: Buy {shares_rounded:.{config.SHARE_DECIMALS}f} NO @ {price:.4f} (YES Drop: {price_drop_pct*100:.1f}%)"
                     app_logger.info(f"{market_name} - {display_text}")
                     analysis_details['reasoning'].append(f"Hedging condition met (Drop {price_drop_pct*100:.1f}% >= {config.HEDGE_PRICE_DROP_THRESHOLD*100:.1f}%). Buying {shares_rounded} NO.")
                     # Hedging changes state, stop loss might need review (becomes HEDGED state logic)
                     recommendation = {'action_type': action_type, 'side': side, 'price': price, 'shares_intended': shares_intended, 'shares_rounded': shares_rounded, 'display_text': display_text, 'calculated_stop_level': None} # Clear directional stop on hedge
                     return recommendation, analysis_details
                 else:
                     reason_fail = "Affordability" if current_balance < cost else "Size Zero"
                     app_logger.debug(f"Hedge YES condition met, but failed ({reason_fail}). Needed {cost:.2f}, have {current_balance:.2f}. Shares: {shares_rounded}")
                     analysis_details['reasoning'].append(f"Hedge YES considered but rejected ({reason_fail}).")


        # ----------------------------------------------------------------------
        # State: DIRECTIONAL_NO - Symmetric logic to DIRECTIONAL_YES
        # ----------------------------------------------------------------------
        elif position_state == 'DIRECTIONAL_NO':
            app_logger.debug(f"State: DIRECTIONAL_NO (AvgP: {no_avg_price:.4f})")
            analysis_details['reasoning'].append(f"State is DIRECTIONAL_NO (AvgP: {no_avg_price:.4f}).")

             # --- 1. Check Profit Taking ---
            if no_bid >= config.PROFIT_TAKE_PRICE_THRESHOLD:
                shares_to_sell = no_shares * config.PROFIT_TAKE_SELL_PCT
                shares_rounded = config.SHARE_ROUNDING_FUNC(shares_to_sell)
                if shares_rounded > config.ZERO_SHARE_THRESHOLD:
                    action_type = 'SELL'
                    side = 'NO'
                    price = no_bid
                    display_text = f"Profit Take (NO): Sell {shares_rounded:.{config.SHARE_DECIMALS}f} ({config.PROFIT_TAKE_SELL_PCT*100:.0f}%) @ {price:.4f} (Threshold: >{config.PROFIT_TAKE_PRICE_THRESHOLD:.4f})"
                    app_logger.info(f"{market_name} - {display_text}")
                    analysis_details['reasoning'].append(f"Profit taking condition met for NO (Bid {no_bid:.4f} >= {config.PROFIT_TAKE_PRICE_THRESHOLD:.4f}). Selling {shares_rounded}.")
                    recommendation = {'action_type': action_type, 'side': side, 'price': price, 'shares_intended': shares_to_sell, 'shares_rounded': shares_rounded, 'display_text': display_text, 'calculated_stop_level': directional_stop}
                    return recommendation, analysis_details

            # --- 2. Check Accumulation ---
            price_drop_pct = (no_avg_price - no_ask) / no_avg_price if no_avg_price > 0 else 0
            should_accumulate = (price_drop_pct >= config.ACCUMULATION_DROP_THRESHOLD and
                                 (directional_stop is None or no_ask > directional_stop) and
                                 config.ENABLE_ACCUMULATION)
            # allow_accum_price = no_ask <= config.BUY_THRESHOLD # Optional check
            allow_accum_price = True # Simpler

            if should_accumulate and allow_accum_price:
                 # Check optional accumulation stop loss
                accum_stop_triggered = False
                if config.ACCUMULATION_STOP_LOSS_PCT is not None:
                     accum_stop_level = no_avg_price * (1.0 - config.ACCUMULATION_STOP_LOSS_PCT)
                     if no_ask <= accum_stop_level:
                          accum_stop_triggered = True
                          app_logger.warning(f"Accumulation prevented: Price {no_ask:.4f} below accumulation stop level {accum_stop_level:.4f}")
                          analysis_details['reasoning'].append(f"Accumulation prevented by accumulation stop loss ({no_ask:.4f} <= {accum_stop_level:.4f}).")

                if not accum_stop_triggered:
                    shares_intended = calculate_shares_to_accumulate(current_balance, no_ask, config.ACCUMULATION_SIZE_PCT_OF_BALANCE)
                    shares_rounded = config.SHARE_ROUNDING_FUNC(shares_intended)
                    cost = shares_rounded * no_ask
                    if shares_rounded > config.ZERO_SHARE_THRESHOLD and current_balance >= cost:
                        action_type = 'BUY'
                        side = 'NO'
                        price = no_ask
                        display_text = f"Accumulate NO: Buy {shares_rounded:.{config.SHARE_DECIMALS}f} @ {price:.4f} (Drop: {price_drop_pct*100:.1f}%)"
                        app_logger.info(f"{market_name} - {display_text}")
                        analysis_details['reasoning'].append(f"Accumulation condition met (Drop {price_drop_pct*100:.1f}% >= {config.ACCUMULATION_DROP_THRESHOLD*100:.1f}%). Buying {shares_rounded}.")
                        recommendation = {'action_type': action_type, 'side': side, 'price': price, 'shares_intended': shares_intended, 'shares_rounded': shares_rounded, 'display_text': display_text, 'calculated_stop_level': directional_stop}
                        return recommendation, analysis_details
                    else:
                        reason_fail = "Affordability" if current_balance < cost else "Size Zero"
                        app_logger.debug(f"Accumulate NO condition met, but failed ({reason_fail}). Needed {cost:.2f}, have {current_balance:.2f}. Shares: {shares_rounded}")
                        analysis_details['reasoning'].append(f"Accumulate NO considered but rejected ({reason_fail}).")


            # --- 3. Check Hedging ---
            should_hedge = (price_drop_pct >= config.HEDGE_PRICE_DROP_THRESHOLD and config.ENABLE_HEDGING)
            if should_hedge and action_type == 'HOLD':
                 shares_intended = calculate_shares_for_hedge(current_balance, yes_ask, config.HEDGE_BUY_SIZE_PCT_OF_BALANCE)
                 if config.HEDGE_MATCH_SHARES:
                     shares_intended = min(shares_intended, no_shares) # Match YES to existing NO

                 shares_rounded = config.SHARE_ROUNDING_FUNC(shares_intended)
                 cost = shares_rounded * yes_ask
                 if shares_rounded > config.ZERO_SHARE_THRESHOLD and current_balance >= cost:
                     action_type = 'BUY'
                     side = 'YES'
                     price = yes_ask
                     display_text = f"Hedge NO Position: Buy {shares_rounded:.{config.SHARE_DECIMALS}f} YES @ {price:.4f} (NO Drop: {price_drop_pct*100:.1f}%)"
                     app_logger.info(f"{market_name} - {display_text}")
                     analysis_details['reasoning'].append(f"Hedging condition met (Drop {price_drop_pct*100:.1f}% >= {config.HEDGE_PRICE_DROP_THRESHOLD*100:.1f}%). Buying {shares_rounded} YES.")
                     recommendation = {'action_type': action_type, 'side': side, 'price': price, 'shares_intended': shares_intended, 'shares_rounded': shares_rounded, 'display_text': display_text, 'calculated_stop_level': None} # Clear directional stop
                     return recommendation, analysis_details
                 else:
                     reason_fail = "Affordability" if current_balance < cost else "Size Zero"
                     app_logger.debug(f"Hedge NO condition met, but failed ({reason_fail}). Needed {cost:.2f}, have {current_balance:.2f}. Shares: {shares_rounded}")
                     analysis_details['reasoning'].append(f"Hedge NO considered but rejected ({reason_fail}).")


        # ----------------------------------------------------------------------
        # State: HEDGED - Look to unwind hedge or exit based on cost basis stop
        # ----------------------------------------------------------------------
        elif position_state == 'HEDGED':
            app_logger.debug(f"State: HEDGED (Y_AvgP: {yes_avg_price:.4f}, N_AvgP: {no_avg_price:.4f})")
            analysis_details['reasoning'].append(f"State is HEDGED (Y_AvgP: {yes_avg_price:.4f}, N_AvgP: {no_avg_price:.4f}).")
            avg_cost_sum = yes_avg_price + no_avg_price

            # --- 1. Check Hedged Stop Loss (based on avg cost basis) ---
            if avg_cost_sum >= (1.0 + config.HEDGED_STOP_LOSS_PCT_BASIS):
                 action_type = 'SELL_STOP'
                 side = 'ALL_PAIRS' # Sell minimum matching pairs
                 price = yes_bid + no_bid # Exit price per pair
                 shares_to_sell = min(yes_shares, no_shares) # Sell the minimum number held
                 shares_rounded = config.SHARE_ROUNDING_FUNC(shares_to_sell)
                 if shares_rounded > config.ZERO_SHARE_THRESHOLD:
                     display_text = f"STOP LOSS Hit (HEDGED): Sell {shares_rounded:.{config.SHARE_DECIMALS}f} PAIRS @ ~{price:.4f} (Avg Cost Sum {avg_cost_sum:.4f} >= Stop {1.0 + config.HEDGED_STOP_LOSS_PCT_BASIS:.4f})"
                     app_logger.warning(f"{market_name} - {display_text}")
                     analysis_details['reasoning'].append(f"Hedged stop loss triggered (Cost Basis Sum {avg_cost_sum:.4f} >= {1.0 + config.HEDGED_STOP_LOSS_PCT_BASIS:.4f}). Selling {shares_rounded} pairs.")
                     recommendation = {'action_type': action_type, 'side': side, 'price': price, 'shares_intended': shares_to_sell, 'shares_rounded': shares_rounded, 'display_text': display_text, 'calculated_stop_level': None}
                     return recommendation, analysis_details
                 else:
                     analysis_details['reasoning'].append(f"Hedged stop loss triggered but rounded shares are zero.")

            # --- 2. Check if conditions favour unwinding the hedge ---
            # Example: If YES price recovers significantly? Or if NO price drops?
            # This logic needs careful definition based on the goal of the hedge.
            # Simple example: Unwind YES leg if its price gets high again?
            elif yes_bid >= config.PROFIT_TAKE_PRICE_THRESHOLD: # Reuse profit take threshold?
                 shares_to_sell = min(yes_shares, no_shares) * config.HEDGE_UNWIND_SELL_PCT # Sell a % of the pairs
                 shares_rounded = config.SHARE_ROUNDING_FUNC(shares_to_sell)
                 if shares_rounded > config.ZERO_SHARE_THRESHOLD:
                     action_type = 'SELL' # Or 'UNWIND_HEDGE_YES'?
                     side = 'YES' # Sell the YES leg
                     price = yes_bid
                     display_text = f"Unwind Hedge (YES High): Sell {shares_rounded:.{config.SHARE_DECIMALS}f} YES @ {price:.4f}"
                     app_logger.info(f"{market_name} - {display_text}")
                     analysis_details['reasoning'].append(f"Unwinding hedge (YES side) as price {yes_bid:.4f} >= {config.PROFIT_TAKE_PRICE_THRESHOLD:.4f}. Selling {shares_rounded}.")
                     recommendation = {'action_type': action_type, 'side': side, 'price': price, 'shares_intended': shares_to_sell, 'shares_rounded': shares_rounded, 'display_text': display_text, 'calculated_stop_level': None}
                     return recommendation, analysis_details
            # Add symmetric logic for unwinding NO leg if needed

            # --- 3. Hold if cost basis is acceptable and no unwind signal ---
            elif avg_cost_sum < config.HEDGED_HOLD_AVG_COST_THRESHOLD:
                 app_logger.debug(f"Holding HEDGED position, cost basis sum {avg_cost_sum:.4f} < threshold {config.HEDGED_HOLD_AVG_COST_THRESHOLD:.4f}")
                 analysis_details['reasoning'].append(f"Holding HEDGED position, cost basis sum {avg_cost_sum:.4f} is acceptable.")
                 action_type = 'HOLD'
            else:
                 app_logger.debug(f"Holding HEDGED position, cost basis sum {avg_cost_sum:.4f} is acceptable but no unwind signal.")
                 analysis_details['reasoning'].append(f"Holding HEDGED position, cost basis sum {avg_cost_sum:.4f} is acceptable, no unwind signal.")
                 action_type = 'HOLD'


        # ----------------------------------------------------------------------
        # State: COST_BASIS_ARB - Look to add shares if opportunity persists
        # ----------------------------------------------------------------------
        elif position_state == 'COST_BASIS_ARB':
             app_logger.debug(f"State: COST_BASIS_ARB (Y_AvgP: {yes_avg_price:.4f}, N_AvgP: {no_avg_price:.4f}, Sum={yes_avg_price + no_avg_price:.4f})")
             analysis_details['reasoning'].append(f"State is COST_BASIS_ARB (AvgCostSum={yes_avg_price + no_avg_price:.4f}).")

             # Condition: Can we buy more YES+NO for less than the Cost Basis Arb Threshold?
             current_ask_sum = yes_ask + no_ask
             if current_ask_sum < config.COST_BASIS_ARB_THRESHOLD and config.ENABLE_COST_BASIS_ARBITRAGE:
                  shares_intended = calculate_shares_for_arb(current_balance, current_ask_sum, config.COST_ARB_ACCUM_SIZE_PCT_OF_BALANCE)
                  shares_rounded = config.SHARE_ROUNDING_FUNC(shares_intended)
                  cost = shares_rounded * current_ask_sum
                  if shares_rounded > config.ZERO_SHARE_THRESHOLD and current_balance >= cost:
                       action_type = 'BUY_ARB' # Use BUY_ARB action type
                       side = 'PAIR'
                       price = current_ask_sum
                       display_text = f"Cost Basis Arb Accum: Buy {shares_rounded:.{config.SHARE_DECIMALS}f} PAIRS @ {price:.4f} (< {config.COST_BASIS_ARB_THRESHOLD:.4f})"
                       app_logger.info(f"{market_name} - {display_text}")
                       analysis_details['reasoning'].append(f"Cost basis arbitrage accumulation opportunity (Sum Ask={price:.4f} < {config.COST_BASIS_ARB_THRESHOLD:.4f}). Buying {shares_rounded} pairs.")
                       recommendation = {'action_type': action_type, 'side': side, 'price': price, 'shares_intended': shares_intended, 'shares_rounded': shares_rounded, 'display_text': display_text, 'calculated_stop_level': None}
                       return recommendation, analysis_details
                  else:
                       reason_fail = "Affordability" if current_balance < cost else "Size Zero"
                       app_logger.debug(f"Cost Basis Arb Accum condition met, but failed ({reason_fail}). Needed {cost:.2f}, have {current_balance:.2f}. Shares: {shares_rounded}")
                       analysis_details['reasoning'].append(f"Cost Basis Arb Accum considered but rejected ({reason_fail}).")
                       action_type = 'HOLD'
             else:
                 app_logger.debug(f"Holding COST_BASIS_ARB position. Current ask sum {current_ask_sum:.4f} not below threshold {config.COST_BASIS_ARB_THRESHOLD:.4f}.")
                 analysis_details['reasoning'].append(f"Holding COST_BASIS_ARB. No accumulation opportunity (AskSum {current_ask_sum:.4f}).")
                 action_type = 'HOLD'


        # ----------------------------------------------------------------------
        # Fallback: HOLD if no other action taken
        # ----------------------------------------------------------------------
        if action_type == 'HOLD':
            analysis_details['reasoning'].append(f"Final decision: HOLD. Reason: {display_text}")
            app_logger.debug(f"{market_name} - Holding. No action criteria met in state {position_state}.")


        # --- Final Recommendation Assembly ---
        recommendation = {
            'action_type': action_type,
            'side': side,
            'price': price,
            'shares_intended': shares_intended, # Unrounded theoretical shares
            'shares_rounded': shares_rounded,   # Rounded shares for execution
            'display_text': display_text,
            'calculated_stop_level': calculated_stop_level # Pass back stop if calculated for BUY
        }

    except Exception as e:
        app_logger.error(f"Error calculating strategy for {market_name}: {e}", exc_info=True)
        analysis_details['reasoning'].append(f"ERROR during calculation: {e}")
        # Return a safe HOLD recommendation on error
        recommendation = {
            'action_type': 'HOLD', 'side': 'NONE', 'price': 0.0, 'shares_intended': 0.0,
            'shares_rounded': 0.0, 'display_text': f"Error: {e}", 'calculated_stop_level': None
        }
        # analysis_details is returned as is, containing the error reason

    return recommendation, analysis_details

# ============================================================
# Make sure to add MIN_BUY_PRICE to your config.py !
# Example in config.py:
# MIN_BUY_PRICE = 0.40 # Minimum price (ask) required to initiate a BUY action on EITHER side
# ============================================================