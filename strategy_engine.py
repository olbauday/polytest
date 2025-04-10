# strategy_engine.py
import math
import config
import utils
import logging # Import logging module

# --- Sizing Helper ---
def get_max_allocation_pct(adv):
    """Rule 5: Get max allocation % based on ADV."""
    # Simplified ADV check, assumes validation happened before call
    if adv is None or adv == "": adv_f = float('inf')
    else:
        try:
             # Attempt to convert ADV to float if it's not already float('inf')
             adv_f = float(adv) if adv != float('inf') else float('inf')
        except (ValueError, TypeError):
             logging.warning(f"Invalid ADV value '{adv}'. Defaulting max allocation.", exc_info=True)
             adv_f = float('inf') # Default to highest tier if conversion fails


    # Get numeric keys and sort them
    numeric_keys = [k for k in config.ADV_ALLOCATION_MAP if isinstance(k, (int, float)) and k != float('inf')]
    sorted_limits = sorted(numeric_keys)

    # Append infinity if it exists as a key
    if float('inf') in config.ADV_ALLOCATION_MAP:
        sorted_limits.append(float('inf'))

    # Find the appropriate percentage
    for adv_limit in sorted_limits:
        pct = config.ADV_ALLOCATION_MAP[adv_limit]
        try:
            # Compare ADV float with the limit
            if adv_f <= adv_limit:
                #logging.debug(f"ADV check: {adv_f} <= {adv_limit}, using pct {pct}")
                return pct
        except TypeError:
            # This might happen if comparing float('inf') inappropriately, though logic aims to prevent it
             logging.warning(f"TypeError comparing ADV ({adv_f}) with limit ({adv_limit}). Using fallback.", exc_info=True)
             # Fallback to the infinity setting if available, else 0.0
             return config.ADV_ALLOCATION_MAP.get(float('inf'), 0.0)

    # If adv_f is greater than all finite limits but infinity isn't explicitly handled,
    # or if some other edge case occurs, use the infinity setting as a final fallback.
    #logging.debug(f"ADV check: {adv_f} exceeds all finite limits. Using fallback pct.")
    return config.ADV_ALLOCATION_MAP.get(float('inf'), 0.0)


def calculate_trade_size(rule_type, side, buy_price, avg_cost, current_shares_side, total_balance, market_adv, market_stats):
    """
    Calculates shares to trade based on rules, balance, RISK, and allocation limits.
    Returns unrounded shares. Returns 0.0 if sizing conditions not met.
    Uses the BUY PRICE (Ask) for cost calculations for BUY actions.
    """
    # Basic validation
    if total_balance <= 0 or (rule_type not in ['SELL', 'SELL_STOP', 'SELL_ARB'] and (buy_price <= config.ZERO_PRICE_THRESHOLD or buy_price >= 1.0)): # Price check only for buys
        logging.debug(f"Size Calc Skip: Pre-check failed (Balance={utils.format_currency(total_balance)}, BuyPrice={utils.format_price(buy_price)}, Rule={rule_type})")
        return 0.0

    # 1. Determine Max Capital Allocation based on ADV (Rule 5)
    max_alloc_pct = get_max_allocation_pct(market_adv)
    max_capital_for_market = total_balance * max_alloc_pct
    current_investment = market_stats.get('yes_investment', 0.0) + market_stats.get('no_investment', 0.0)
    # Ensure remaining capital doesn't exceed total balance if current investment is somehow negative (shouldn't happen)
    remaining_alloc_capital = min(total_balance, max(0, max_capital_for_market - current_investment))

    logging.debug(f"Size Calc: ADV={market_adv}, MaxAlloc%={utils.format_percent(max_alloc_pct)}, MaxCap={utils.format_currency(max_capital_for_market)}, CurrInv={utils.format_currency(current_investment)}, RemainCap={utils.format_currency(remaining_alloc_capital)}")

    # If no allocation remaining for market, cannot size new trades requiring capital
    # Hedge sizing is based on balance only, not ADV allocation. SELLs don't need this check.
    if remaining_alloc_capital <= config.ZERO_PRICE_THRESHOLD and rule_type not in ['HEDGE', 'SELL', 'SELL_STOP', 'SELL_ARB']:
         logging.debug(f"Size Calc Skip: No remaining capital allocation for market ({utils.format_currency(remaining_alloc_capital)}) for rule {rule_type}")
         return 0.0

    unrounded_shares = 0.0

    # 2. Sizing based on Rule Type (All BUY actions use the buy_price passed)
    if rule_type in ['ENTRY', 'ACCUMULATION']:
        # --- Risk-Based Sizing ---
        risk_capital_per_trade = total_balance * config.RISK_PER_TRADE_PCT

        # Determine Stop Loss Price for *this specific trade* sizing
        # Stop is based on the BUY price (Ask) of the shares being acquired
        stop_loss_price = 0.0
        stop_pct = config.DIRECTIONAL_STOP_LOSS_PCT # Default

        if rule_type == 'ENTRY':
            # Stop relative to current BUY price for a new entry
            stop_loss_price = buy_price * (1.0 - stop_pct)
            logging.debug(f"Size Calc ({rule_type}): Stop based on Entry Buy Price {utils.format_price(buy_price)} * (1 - {stop_pct:.2f})")

        elif rule_type == 'ACCUMULATION':
            # Use specific accumulation stop % if defined, else default
            stop_pct = config.ACCUMULATION_STOP_LOSS_PCT if config.ACCUMULATION_STOP_LOSS_PCT is not None else config.DIRECTIONAL_STOP_LOSS_PCT
            # Stop relative to *current BUY price* for the accumulation buy
            stop_loss_price = buy_price * (1.0 - stop_pct)
            logging.debug(f"Size Calc ({rule_type}): Stop based on Current Buy Price {utils.format_price(buy_price)} * (1 - {stop_pct:.2f})")

        else: # Should not happen based on outer condition
             logging.error(f"Size Calc Error: Unexpected rule '{rule_type}' in risk-sizing block.")
             return 0.0

        # Ensure stop loss is valid (between 0 and buy_price)
        stop_loss_price = max(config.ZERO_PRICE_THRESHOLD, min(stop_loss_price, buy_price - config.ZERO_PRICE_THRESHOLD))
        logging.debug(f"Size Calc ({rule_type}): Entry/Current Buy Price={utils.format_price(buy_price)}, Calculated Stop={utils.format_price(stop_loss_price)}")

        price_risk_per_share = buy_price - stop_loss_price
        if price_risk_per_share <= config.ZERO_PRICE_THRESHOLD: # Avoid division by zero or tiny risk inflating shares
            logging.warning(f"Size Calc Warning ({rule_type}): Minimal price risk per share ({utils.format_currency(price_risk_per_share)}). Cannot size trade.")
            return 0.0

        # Calculate shares based on risk capital and risk per share
        shares_based_on_risk = risk_capital_per_trade / price_risk_per_share
        logging.debug(f"Size Calc ({rule_type}): RiskCap={utils.format_currency(risk_capital_per_trade)}, Risk/Share={utils.format_currency(price_risk_per_share)}, RiskShares={utils.format_shares(shares_based_on_risk)}")

        # Apply Caps: Max shares based on ADV allocation remaining, and max based on total balance
        max_shares_adv = remaining_alloc_capital / buy_price if buy_price > config.ZERO_PRICE_THRESHOLD else 0.0
        max_shares_balance = total_balance / buy_price if buy_price > config.ZERO_PRICE_THRESHOLD else 0.0 # Absolute max based on wallet

        unrounded_shares = max(0.0, min(shares_based_on_risk, max_shares_adv, max_shares_balance))
        logging.debug(f"Size Calc ({rule_type}): MaxSharesADV={utils.format_shares(max_shares_adv)}, MaxSharesBal={utils.format_shares(max_shares_balance)} -> Raw Shares={utils.format_shares(unrounded_shares)}")

    elif rule_type == 'HEDGE':
        # Hedge aims to balance shares. Size = difference in shares.
        share_diff = abs(market_stats.get('yes_shares', 0.0) - market_stats.get('no_shares', 0.0))
        # Target shares needed to balance (buy the difference)
        shares_needed = share_diff
        # Cap by available balance (doesn't use ADV allocation directly, assumes hedging reduces risk)
        max_shares_balance = total_balance / buy_price if buy_price > config.ZERO_PRICE_THRESHOLD else 0.0
        unrounded_shares = max(0.0, min(shares_needed, max_shares_balance))
        logging.debug(f"Size Calc (Hedge): ShareDiff={utils.format_shares(share_diff)}, SharesNeeded={utils.format_shares(shares_needed)}, MaxBalShares={utils.format_shares(max_shares_balance)} -> Raw Shares={utils.format_shares(unrounded_shares)}")

    elif rule_type == 'COST_BASIS_ARB':
        # Size based on a percentage of balance, capped by ADV allocation & balance
        target_capital = total_balance * config.COST_ARB_ACCUM_SIZE_PCT_OF_BALANCE
        # Capital is capped by remaining ADV alloc and total balance
        trade_capital = min(target_capital, remaining_alloc_capital, total_balance)
        unrounded_shares = trade_capital / buy_price if buy_price > config.ZERO_PRICE_THRESHOLD else 0.0
        logging.debug(f"Size Calc (CostArb): TargetCap={utils.format_currency(target_capital)}, AllowedCap={utils.format_currency(trade_capital)} -> Raw Shares={utils.format_shares(unrounded_shares)}")

    elif rule_type == 'BUY_ARB':
         # Size based on a percentage of balance, capped by ADV allocation & balance
         target_capital = total_balance * config.ARB_BUY_SIZE_PCT_OF_BALANCE
         # `buy_price` passed here should be yes_ask + no_ask (the pair buy price)
         # Capital is capped by remaining ADV alloc and total balance
         trade_capital = min(target_capital, remaining_alloc_capital, total_balance)
         # Shares calculation represents the number of PAIRS to buy
         unrounded_shares = trade_capital / buy_price if buy_price > config.ZERO_PRICE_THRESHOLD else 0.0
         logging.debug(f"Size Calc (BuyArb Pair): PairAskPrice={utils.format_price(buy_price)}, TargetCap={utils.format_currency(target_capital)}, AllowedCap={utils.format_currency(trade_capital)} -> Raw Pairs={utils.format_shares(unrounded_shares)}")

    # SELL actions don't require sizing calculations here; the caller determines shares to sell (e.g., all, percentage)
    # The calling function `calculate_strategy_recommendation` determines the shares for SELL/SELL_STOP/SELL_ARB
    # and passes them (unrounded) into the recommendation dictionary.
    # This function is primarily for sizing BUY actions based on capital constraints.
    elif rule_type in ['SELL', 'SELL_STOP', 'SELL_ARB']:
        logging.debug(f"Size Calc ({rule_type}): Sizing determined by caller based on held shares. Returning 0.0.")
        return 0.0 # Return 0 as size is handled elsewhere

    else:
        logging.error(f"Size Calc Error: Unknown rule_type '{rule_type}'")
        return 0.0

    # Final check: cost vs balance (only for BUY actions)
    if rule_type not in ['SELL', 'SELL_STOP', 'SELL_ARB']:
        cost = unrounded_shares * buy_price
        # Use a small tolerance for floating point comparisons
        if cost > total_balance + config.ZERO_PRICE_THRESHOLD:
            logging.warning(f"Size Calc Final Check ({rule_type}): Cost {utils.format_currency(cost)} > Balance {utils.format_currency(total_balance)}. Reducing size.")
            unrounded_shares = total_balance / buy_price if buy_price > config.ZERO_PRICE_THRESHOLD else 0.0
            logging.debug(f"Size Calc Final Check ({rule_type}): Adjusted Raw Shares={utils.format_shares(unrounded_shares)}")

    # Return unrounded shares. The caller (execution logic) should apply final rounding.
    return max(0.0, unrounded_shares)


def calculate_strategy_recommendation(market_name, yes_bid, yes_ask, no_bid, no_ask, market_adv, market_data, stats, current_balance):
    """
    Analyzes market based on STATE, RULES, and RISK using BID/ASK prices.
    Returns a recommendation dictionary and analysis details list (strings).
    Recommendation includes unrounded shares. Caller must round before execution.
    Does NOT modify market_data or balance directly. Assumes 'stats' are up-to-date.
    """
    recommendation = {
        "market": market_name, "action_type": 'HOLD', "side": None,
        "shares_unrounded": 0.0, "shares_rounded": 0.0, # Store both
        "price": 0.0, # Will store Ask for BUYs, Bid for SELLs, Pair sum for ARBs
        "cost_proceeds": 0.0, "rule_triggered": "N/A",
        "calculated_stop_level": None, # For new directional entries/accumulations
        "display_text": "REC: HOLD / Monitor" # Default display text
    }
    analysis_details = [] # List of strings detailing the decision process

    # --- Input Validation & Data Setup ---
    if not market_data:
        analysis_details.append("Error: Market data not found.")
        recommendation['display_text'] = "Error: No market data"
        logging.error(f"Strategy Calc Error for '{market_name}': Market data missing.")
        return recommendation, analysis_details

    # Ensure prices are valid floats for calculations, default to 0 if not
    try: yes_b = float(yes_bid) if yes_bid is not None else 0.0
    except (ValueError, TypeError): yes_b = 0.0
    try: yes_a = float(yes_ask) if yes_ask is not None else 0.0
    except (ValueError, TypeError): yes_a = 0.0
    try: no_b = float(no_bid) if no_bid is not None else 0.0
    except (ValueError, TypeError): no_b = 0.0
    try: no_a = float(no_ask) if no_ask is not None else 0.0
    except (ValueError, TypeError): no_a = 0.0


    current_state = market_data.get('position_state', 'FLAT')
    # Use .get with default 0.0 for robustness
    yes_s = stats.get('yes_shares', 0.0)
    no_s = stats.get('no_shares', 0.0)
    yes_inv = stats.get('yes_investment', 0.0)
    no_inv = stats.get('no_investment', 0.0)
    avg_yes_p = stats.get('yes_avg_price', 0.0)
    avg_no_p = stats.get('no_avg_price', 0.0)
    total_basis = yes_inv + no_inv
    directional_stop_price_level = market_data.get('directional_stop_loss') # The actual price level

    analysis_details.append(f"--- Strategy Eval Start: '{market_name}' (State: {current_state}) ---")
    analysis_details.append(f"  Prices: YES Bid={utils.format_price(yes_b)}, Ask={utils.format_price(yes_a)} | "
                            f"NO Bid={utils.format_price(no_b)}, Ask={utils.format_price(no_a)}")
    analysis_details.append(f"  Shares: YES={utils.format_shares(yes_s)}, NO={utils.format_shares(no_s)}")
    analysis_details.append(f"  AvgCost: YES={utils.format_price(avg_yes_p)}, NO={utils.format_price(avg_no_p)}")
    analysis_details.append(f"  Balance: {utils.format_currency(current_balance)}")
    if directional_stop_price_level is not None:
        analysis_details.append(f"  Current Stop Level: {utils.format_price(directional_stop_price_level)}")


    final_rec_found = False # Flag to stop checking lower priority rules once an action is decided

    # --- RULE PRIORITIES ---

    # --- PRIORITY 1: STOP LOSS ---
    if not final_rec_found:
        stop_triggered = False
        stop_side = None # 'ALL_YES', 'ALL_NO', 'ALL_PAIRS'
        stop_trigger_value = None # Price level or loss percentage
        stop_rule_details = ""
        stop_sell_price = 0.0 # The price used for estimating proceeds
        shares_to_sell_unrounded = 0.0

        # Directional Stop Loss (Trigger on BID price - what you get when you sell)
        if current_state == 'DIRECTIONAL_YES' and yes_s > config.ZERO_SHARE_THRESHOLD and directional_stop_price_level is not None and yes_b <= directional_stop_price_level:
            stop_triggered = True; stop_side = 'ALL_YES'; stop_trigger_value = yes_b; stop_sell_price = yes_b; shares_to_sell_unrounded = yes_s
            stop_rule_details = f"DIRECTIONAL_YES Sell Price (Bid) {utils.format_price(yes_b)} <= Stop Level {utils.format_price(directional_stop_price_level)}"
        elif current_state == 'DIRECTIONAL_NO' and no_s > config.ZERO_SHARE_THRESHOLD and directional_stop_price_level is not None and no_b <= directional_stop_price_level:
            stop_triggered = True; stop_side = 'ALL_NO'; stop_trigger_value = no_b; stop_sell_price = no_b; shares_to_sell_unrounded = no_s
            stop_rule_details = f"DIRECTIONAL_NO Sell Price (Bid) {utils.format_price(no_b)} <= Stop Level {utils.format_price(directional_stop_price_level)}"

        # Hedged/Cost Basis Arb Stop Loss (Based on % loss of cost basis, CONDITIONAL on Avg Cost)
        elif current_state in ['HEDGED', 'COST_BASIS_ARB']:
             num_pairs_held = min(yes_s, no_s)
             # Only proceed if we hold pairs and have a cost basis
             if num_pairs_held > config.ZERO_SHARE_THRESHOLD and total_basis > config.ZERO_PRICE_THRESHOLD:
                 # Calculate avg cost per pair based on the number of pairs held and total investment
                 # Note: This assumes investment primarily relates to the paired shares. If large imbalances exist, this might be skewed.
                 avg_cost_per_pair = total_basis / num_pairs_held

                 # Calculate current market value of the PAIRS based on BID prices
                 current_pair_market_value = num_pairs_held * (yes_b + no_b)
                 # Calculate cost basis attributable to the pairs being evaluated
                 cost_basis_of_pairs = num_pairs_held * avg_cost_per_pair # This simplifies to total_basis if yes_s == no_s
                 # Use the smaller of total basis or calculated pair basis for loss calculation to be conservative
                 relevant_basis = min(total_basis, cost_basis_of_pairs)

                 unrealized_pl_pairs = current_pair_market_value - relevant_basis
                 loss_pct_basis = unrealized_pl_pairs / relevant_basis if relevant_basis > config.ZERO_PRICE_THRESHOLD else 0

                 loss_threshold_met = loss_pct_basis <= -config.HEDGED_STOP_LOSS_PCT_BASIS

                 # Conditional logic
                 apply_stop = False # Default to not applying
                 if loss_threshold_met:
                     analysis_details.append(f"  STOP CHECK ({current_state}): Loss threshold ({utils.format_percent(loss_pct_basis)} <= {utils.format_percent(-config.HEDGED_STOP_LOSS_PCT_BASIS)}) MET.")
                     if config.HEDGED_HOLD_AVG_COST_THRESHOLD > 0 and avg_cost_per_pair < config.HEDGED_HOLD_AVG_COST_THRESHOLD:
                         apply_stop = False # Override: Hold despite loss % due to low avg cost
                         analysis_details.append(f"  STOP CHECK ({current_state}): HOLDING Stop Loss. Avg Cost/Pair ({utils.format_price(avg_cost_per_pair)}) < Hold Threshold ({utils.format_price(config.HEDGED_HOLD_AVG_COST_THRESHOLD)}).")
                         logging.debug(f"Hedged Stop Loss for '{market_name}' ignored due to low avg cost ({utils.format_price(avg_cost_per_pair)}). Loss% was {utils.format_percent(loss_pct_basis)}")
                     else:
                         apply_stop = True # Apply stop loss, threshold met and cost high enough (or threshold disabled)
                         analysis_details.append(f"  STOP CHECK ({current_state}): APPLYING Stop Loss. Avg Cost/Pair ({utils.format_price(avg_cost_per_pair)}) >= Hold Threshold ({utils.format_price(config.HEDGED_HOLD_AVG_COST_THRESHOLD)}).")

                 # Trigger the stop only if conditions met
                 if loss_threshold_met and apply_stop:
                      stop_triggered = True; stop_side = 'ALL_PAIRS'; stop_trigger_value = loss_pct_basis
                      stop_sell_price = yes_b + no_b # Use sum of BIDs for pair sell price estimate
                      shares_to_sell_unrounded = num_pairs_held # Sell all complete pairs
                      stop_rule_details = (f"{current_state} Pair Loss Basis {utils.format_percent(loss_pct_basis)} "
                                           f"<= Threshold {utils.format_percent(-config.HEDGED_STOP_LOSS_PCT_BASIS)} "
                                           f"(Avg Cost {utils.format_price(avg_cost_per_pair)} >= {utils.format_price(config.HEDGED_HOLD_AVG_COST_THRESHOLD)})")
             # else: # No pairs held or no basis, cannot evaluate this stop
             #    analysis_details.append(f"  STOP CHECK ({current_state}): Cannot evaluate. Pairs Held={utils.format_shares(num_pairs_held)}, Total Basis={utils.format_currency(total_basis)}.")

        # Stop Action Block
        if stop_triggered:
             analysis_details.append(f"! STOP LOSS (Priority 1): {stop_rule_details}")
             logging.info(f"STOP LOSS triggered for '{market_name}': {stop_rule_details}")

             # Check if calculated unrounded shares > 0 before rounding
             if shares_to_sell_unrounded <= config.ZERO_SHARE_THRESHOLD:
                  final_rec_found = False # Should not proceed if no shares calculated
                  analysis_details.append(f"  STOP LOSS: Triggered but calculated zero shares to sell for {stop_side}. No action.")
                  logging.info(f"Stop Loss for '{market_name}' triggered but calculated zero shares to sell for {stop_side}.")
             else:
                 rec_shares_rounded = config.SHARE_ROUNDING_FUNC(shares_to_sell_unrounded)

                 # Check if rounded shares > 0
                 if rec_shares_rounded <= config.ZERO_SHARE_THRESHOLD:
                     logging.warning(f"Stop Loss for '{market_name}' resulted in zero rounded shares ({shares_to_sell_unrounded} unrounded). Min holdings likely exist or rounding issue.")
                     final_rec_found = False # Don't make recommendation if shares are zero
                     analysis_details.append(f"  STOP LOSS: Calculated shares {utils.format_shares(shares_to_sell_unrounded)} rounded to zero. No action taken.")
                 else:
                     final_rec_found = True # Set flag as we have a valid action
                     # Estimate proceeds using the relevant BID price(s)
                     estimated_proceeds = rec_shares_rounded * stop_sell_price

                     trigger_display = f"{utils.format_price(stop_trigger_value)}" if stop_side in ['ALL_YES', 'ALL_NO'] else f"{utils.format_percent(stop_trigger_value)}"
                     sell_target_display = stop_side if stop_side != 'ALL_PAIRS' else "PAIRS"

                     recommendation.update({
                         "action_type": "SELL_STOP", "side": stop_side, # Note: 'ALL_PAIRS' needs special handling in execution
                         "shares_unrounded": shares_to_sell_unrounded,
                         "shares_rounded": rec_shares_rounded, # Shares per side, or number of pairs if ALL_PAIRS
                         "price": stop_sell_price, # Bid price or sum of Bids
                         "cost_proceeds": estimated_proceeds, # Positive for proceeds
                         "rule_triggered": f"Stop Loss ({current_state})",
                         "display_text": f"REC (STOP): Sell {sell_target_display} ({utils.format_shares(rec_shares_rounded)} @ ~{utils.format_price(stop_sell_price)}) Trigger: {trigger_display}"
                     })
                     analysis_details.append(f"  REC: Sell {sell_target_display} ({utils.format_shares(rec_shares_rounded)} shares/pairs). Estimated Proceeds: {utils.format_currency(estimated_proceeds)}")


    # --- PRIORITY 2: Market Price Arbitrage ---
    if not final_rec_found:
        # Check BUYING the pair (using ASK prices) - Profit if yes_a + no_a < 1.0
        buy_pair_price = yes_a + no_a
        buy_arb_spread = buy_pair_price - 1.0 # Negative value indicates profit potential

        # Check SELLING the pair (using BID prices) - Profit if yes_b + no_b > 1.0
        sell_pair_price = yes_b + no_b
        sell_arb_spread = sell_pair_price - 1.0 # Positive value indicates profit potential

        arb_check_logged = False

        # --- BUY ARB (Buy pair for < $1) ---
        # Check if we can buy the pair for significantly less than $1
        if buy_pair_price > config.ZERO_PRICE_THRESHOLD and buy_arb_spread < -config.ARB_THRESHOLD:
            if not arb_check_logged:
                 analysis_details.append(f"- Market Arb Check (Priority 2): BuySpread={buy_arb_spread:.4f} (< {-config.ARB_THRESHOLD:.4f}), SellSpread={sell_arb_spread:.4f}")
                 arb_check_logged = True

            profit_per_pair = abs(buy_arb_spread)
            # Size based on ASK price sum (cost to acquire one pair)
            # calculate_trade_size expects side 'PAIR' for arb buys
            pairs_to_buy_unrounded = calculate_trade_size('BUY_ARB', 'PAIR', buy_pair_price, 0, 0, current_balance, market_adv, stats)
            pairs_to_buy_rounded = config.SHARE_ROUNDING_FUNC(pairs_to_buy_unrounded)

            if pairs_to_buy_rounded <= config.ZERO_SHARE_THRESHOLD:
                analysis_details.append(f"  ARB (Buy Opp): Size zero. Check Balance/Alloc/Sizing.")
                logging.debug(f"Market Arb (Buy) for '{market_name}': Sized to zero shares (Unrounded: {pairs_to_buy_unrounded}).")
            else:
                cost = pairs_to_buy_rounded * buy_pair_price # Cost uses ask sum
                guaranteed_profit_at_resolution = pairs_to_buy_rounded * (1.0 - buy_pair_price)
                analysis_details.append(f"  ARB (Buy Opp): Buy {utils.format_shares(pairs_to_buy_rounded)} Pairs @ AskSum ~{utils.format_price(buy_pair_price)}. "
                                         f"Est Cost: {utils.format_currency(cost)}. Est Profit @ Resolve: {utils.format_currency(guaranteed_profit_at_resolution)}.")
                recommendation.update({
                    "action_type": "BUY_ARB", "side": "PAIR", # Execution logic buys YES and NO using asks
                    "shares_unrounded": pairs_to_buy_unrounded,
                    "shares_rounded": pairs_to_buy_rounded,
                    "price": buy_pair_price, # Price is the ask pair sum (cost per pair)
                    "cost_proceeds": -cost, # Negative for cost
                    "rule_triggered": "Market Arb (Buy)",
                    "display_text": f"REC (Arb Buy): Buy {utils.format_shares(pairs_to_buy_rounded)} PAIR @ AskSum ~{utils.format_price(buy_pair_price)} (Cost {utils.format_currency(cost)})"
                })
                final_rec_found = True
                logging.info(f"Market Arb (Buy) triggered for '{market_name}': Buy {utils.format_shares(pairs_to_buy_rounded)} pairs.")

        # --- SELL ARB (Sell pair for > $1, using BID prices) ---
        # Only check sell arb if buy arb didn't trigger and we hold pairs
        elif not final_rec_found and sell_arb_spread > config.ARB_THRESHOLD:
            if not arb_check_logged:
                 analysis_details.append(f"- Market Arb Check (Priority 2): BuySpread={buy_arb_spread:.4f}, SellSpread={sell_arb_spread:.4f} (> {config.ARB_THRESHOLD:.4f})")
                 arb_check_logged = True

            sellable_pairs_unrounded = min(yes_s, no_s)
            sellable_pairs_rounded = config.SHARE_ROUNDING_FUNC(sellable_pairs_unrounded)

            if sellable_pairs_rounded <= config.ZERO_SHARE_THRESHOLD:
                analysis_details.append("  ARB (Sell Opp): Spread exists, but no sufficient matching pairs held.")
                logging.debug(f"Market Arb (Sell) for '{market_name}': Spread > threshold but no sellable pairs ({sellable_pairs_unrounded}).")
            else:
                 # We have pairs to sell and the market price is attractive (> 1 + threshold)
                 current_market_pair_bid_price = sell_pair_price # Price received per pair

                 # Calculate average cost per pair held (considering only paired shares)
                 num_pairs = sellable_pairs_rounded # Use rounded for calculation consistency
                 # Calculate cost basis of the pairs we could sell
                 # This requires knowing the avg price of both yes and no shares that make up the pairs
                 # Simple approximation: use total basis / num pairs if reasonably balanced
                 # A more accurate approach requires tracking cost basis per lot, which is complex.
                 # Let's use the simpler approx. avg cost if basis exists
                 avg_cost_per_pair = 0.0
                 if num_pairs > config.ZERO_SHARE_THRESHOLD and total_basis > config.ZERO_PRICE_THRESHOLD:
                      # Refined avg cost calculation based on *paired* investment
                      # Estimate investment in pairs assumes similar avg costs or uses total basis / pairs held
                      # Let's stick to total_basis / num_pairs_held for simplicity here
                      num_pairs_held = min(yes_s, no_s) # Use unrounded held for cost basis calc
                      if num_pairs_held > config.ZERO_SHARE_THRESHOLD:
                           avg_cost_per_pair = total_basis / num_pairs_held
                      else: avg_cost_per_pair = float('inf') # Cannot determine cost if no pairs

                 # Decision: Sell if market bid > threshold AND (market bid > avg cost OR avg cost unknown/zero)
                 if current_market_pair_bid_price > config.ZERO_PRICE_THRESHOLD and \
                    (avg_cost_per_pair <= config.ZERO_PRICE_THRESHOLD or current_market_pair_bid_price > avg_cost_per_pair):

                    proceeds = sellable_pairs_rounded * current_market_pair_bid_price # Proceeds use bid sum
                    # Estimate P/L based on the simpler avg cost
                    total_cost_basis_sold = sellable_pairs_rounded * avg_cost_per_pair if avg_cost_per_pair != float('inf') else 0
                    realized_pl = proceeds - total_cost_basis_sold

                    profit_condition_met_msg = f"Market BidSum: {utils.format_price(current_market_pair_bid_price)} > Avg Cost/Pair: {utils.format_price(avg_cost_per_pair)}" if avg_cost_per_pair != float('inf') else "Avg Cost Unknown/Zero"

                    analysis_details.append((f"  ARB (Sell Profitable): Sell {utils.format_shares(sellable_pairs_rounded)} Pairs. "
                                             f"{profit_condition_met_msg}. Est. P/L: {utils.format_currency(realized_pl)}."))
                    recommendation.update({
                        "action_type": "SELL_ARB", "side": "PAIR", # Execution logic sells YES and NO using bids
                        "shares_unrounded": sellable_pairs_unrounded, # Target unrounded amount
                        "shares_rounded": sellable_pairs_rounded,    # Rounded amount to actually sell
                        "price": current_market_pair_bid_price, # Price is the bid pair sum (proceeds per pair)
                        "cost_proceeds": proceeds, # Positive for total proceeds
                        "rule_triggered": "Market Arb (Sell - Profitable)",
                        "display_text": f"REC (Arb Profit): Sell {utils.format_shares(sellable_pairs_rounded)} PAIR @ BidSum ~{utils.format_price(current_market_pair_bid_price)} (Est P/L {utils.format_currency(realized_pl)})"
                    })
                    final_rec_found = True
                    logging.info(f"Market Arb (Sell - Profitable) triggered for '{market_name}': Sell {utils.format_shares(sellable_pairs_rounded)} pairs.")
                 else:
                    # Sell Arb spread exists, but it's below our average cost basis. Hold.
                    analysis_details.append((f"  ARB (Sell Hold): Spread ({utils.format_price(current_market_pair_bid_price)}) > Threshold "
                                             f"but <= Avg Cost/Pair {utils.format_price(avg_cost_per_pair)}. Holding."))
                    logging.debug(f"Market Arb (Sell) for '{market_name}': Spread exists but below cost basis ({utils.format_price(avg_cost_per_pair)}). Holding.")

        if arb_check_logged and not final_rec_found:
             analysis_details.append("  ARB: No profitable action identified or possible.")


    # --- PRIORITY 3: Cost Basis Arbitrage Accumulation ---
    # Goal: If we hold pairs bought for < $1 total cost basis, buy more of the *currently cheaper* side to lower overall basis.
    if not final_rec_found and current_state in ['HEDGED', 'COST_BASIS_ARB']:
        cost_arb_accum_possible = False
        # Check if we hold pairs and the sum of average costs is below the threshold
        num_pairs_held = min(yes_s, no_s)
        avg_cost_sum = avg_yes_p + avg_no_p

        # Ensure we have pairs and valid average costs before checking threshold
        if num_pairs_held > config.ZERO_SHARE_THRESHOLD and avg_yes_p > config.ZERO_PRICE_THRESHOLD and avg_no_p > config.ZERO_PRICE_THRESHOLD:
            # Check if the SUM of avg costs qualifies for this state/accumulation strategy
            if avg_cost_sum < config.COST_BASIS_ARB_THRESHOLD:
                cost_arb_accum_possible = True
                analysis_details.append(f"- Cost Arb Accum Check (Priority 3): State={current_state}, AvgCostSum={utils.format_price(avg_cost_sum)} < Threshold ({config.COST_BASIS_ARB_THRESHOLD:.2f})")

                # Determine which side is cheaper to BUY now (use ASK prices)
                accum_side = None
                accum_buy_price = 0.0
                # Prefer buying YES if asks are equal or YES is cheaper (and valid)
                if yes_a > config.ZERO_PRICE_THRESHOLD and (yes_a <= no_a or no_a <= config.ZERO_PRICE_THRESHOLD):
                     accum_side = 'YES'; accum_buy_price = yes_a
                # Otherwise, buy NO if its ask is valid and cheaper
                elif no_a > config.ZERO_PRICE_THRESHOLD and no_a < yes_a:
                     accum_side = 'NO'; accum_buy_price = no_a

                if accum_side is None:
                    analysis_details.append(f"  INFO: Cannot accumulate for Cost Arb, both Ask prices invalid or zero ({utils.format_price(yes_a)}, {utils.format_price(no_a)}).")
                    logging.debug(f"Cost Arb Accum for '{market_name}': Cannot determine side, asks invalid/zero.")
                else:
                    analysis_details.append(f"  INFO: Cheaper side to accumulate: {accum_side} @ Ask {utils.format_price(accum_buy_price)}")
                    # Size the trade using the BUY price (Ask) and specific sizing %
                    # Pass 0 for avg_cost, current_shares_side as they are not used for this sizing type
                    shares_unrounded = calculate_trade_size('COST_BASIS_ARB', accum_side, accum_buy_price, 0, 0, current_balance, market_adv, stats)
                    shares_rounded = config.SHARE_ROUNDING_FUNC(shares_unrounded)

                    if shares_rounded <= config.ZERO_SHARE_THRESHOLD:
                        analysis_details.append(f"  INFO: Cost Arb Accum size zero for {accum_side}. Check Balance/Alloc/Sizing.")
                        logging.debug(f"Cost Arb Accum for '{market_name}': Sized to zero shares for {accum_side} (Unrounded: {shares_unrounded}).")
                    else:
                        cost = shares_rounded * accum_buy_price
                        # Double check cost vs balance (somewhat redundant with sizing check, but safe)
                        if cost <= current_balance + config.ZERO_PRICE_THRESHOLD:
                            recommendation.update({
                                "action_type": "BUY", "side": accum_side,
                                "shares_unrounded": shares_unrounded,
                                "shares_rounded": shares_rounded,
                                "price": accum_buy_price, # Price is the Ask (Buy) price
                                "cost_proceeds": -cost,
                                "rule_triggered": "Cost Basis Arb Accum",
                                "display_text": f"REC (Cost Arb): Buy {utils.format_shares(shares_rounded)} {accum_side} @ Ask {utils.format_price(accum_buy_price)} (Cost: {utils.format_currency(cost)})"
                            })
                            final_rec_found = True
                            analysis_details.append(f"  REC: Buy {utils.format_shares(shares_rounded)} {accum_side} (cheaper side) to improve cost basis.")
                            logging.info(f"Cost Basis Arb Accum triggered for '{market_name}': Buy {utils.format_shares(shares_rounded)} {accum_side}.")
                        else:
                            analysis_details.append(f"  INFO: Insufficient balance for Cost Arb Accum ({utils.format_currency(cost)} vs {utils.format_currency(current_balance)}).")
                            logging.warning(f"Cost Arb Accum for '{market_name}': Insufficient balance ({utils.format_currency(current_balance)}) for cost {utils.format_currency(cost)}.")
            # else: # Avg cost sum is too high, don't accumulate based on this rule
            #    analysis_details.append(f"- Cost Arb Accum Check (Priority 3): AvgCostSum ({utils.format_price(avg_cost_sum)}) >= Threshold ({config.COST_BASIS_ARB_THRESHOLD:.2f}). Skipping.")
        # else: # Don't have pairs or valid avg costs
        #     analysis_details.append(f"- Cost Arb Accum Check (Priority 3): Skipping. Pairs Held={utils.format_shares(num_pairs_held)}, AvgY={utils.format_price(avg_yes_p)}, AvgN={utils.format_price(avg_no_p)}")


    # --- PRIORITY 4: Hedging (Directional Positions Only) ---
    # Goal: If holding a large directional position and its price drops significantly, buy the other side to reduce risk.
    if not final_rec_found and current_state.startswith('DIRECTIONAL'):
         hedge_needed = False
         hedge_reason = ""
         side_to_buy = "" # The side needed to balance (opposite of the large holding)
         hedge_buy_price = 0.0 # ASK price of side to buy
         share_diff = yes_s - no_s # Positive if long YES, Negative if long NO

         # Check if significantly long YES and YES price dropped (use BID price for drop check vs avg cost)
         if share_diff > config.HEDGE_IMBALANCE_TOLERANCE_SHARES: # Significantly long YES
             if avg_yes_p > config.ZERO_PRICE_THRESHOLD and yes_b < avg_yes_p * (1.0 - config.HEDGE_PRICE_DROP_THRESHOLD):
                 hedge_needed = True; side_to_buy = 'NO'; hedge_buy_price = no_a # Need to buy NO at its ASK
                 hedge_reason = f"YES large ({utils.format_shares(yes_s)}) & YES sell price (Bid {utils.format_price(yes_b)}) dropped < {utils.format_percent(1.0 - config.HEDGE_PRICE_DROP_THRESHOLD)} of avg ({utils.format_price(avg_yes_p)})"

         # Check if significantly long NO and NO price dropped (use BID price for drop check vs avg cost)
         elif share_diff < -config.HEDGE_IMBALANCE_TOLERANCE_SHARES: # Significantly long NO
             if avg_no_p > config.ZERO_PRICE_THRESHOLD and no_b < avg_no_p * (1.0 - config.HEDGE_PRICE_DROP_THRESHOLD):
                 hedge_needed = True; side_to_buy = 'YES'; hedge_buy_price = yes_a # Need to buy YES at its ASK
                 hedge_reason = f"NO large ({utils.format_shares(no_s)}) & NO sell price (Bid {utils.format_price(no_b)}) dropped < {utils.format_percent(1.0 - config.HEDGE_PRICE_DROP_THRESHOLD)} of avg ({utils.format_price(avg_no_p)})"

         if hedge_needed:
             analysis_details.append(f"- Hedge Trigger Check (Priority 4): Triggered. Reason: {hedge_reason}.")
             logging.info(f"Hedge triggered for '{market_name}': {hedge_reason}")

             # Check if the ASK price for the side we need to buy is valid
             if hedge_buy_price <= config.ZERO_PRICE_THRESHOLD or hedge_buy_price >= 1.0:
                  analysis_details.append(f"  INFO: Cannot hedge, buy side {side_to_buy} Ask price invalid or >= $1 ({utils.format_price(hedge_buy_price)}).")
                  logging.warning(f"Hedge for '{market_name}': Cannot buy {side_to_buy}, Ask price invalid/high ({utils.format_price(hedge_buy_price)}).")
             else:
                 # Size the hedge trade: aim to buy enough shares to balance the difference
                 # calculate_trade_size handles capping by balance
                 # Pass 0 for avg_cost, current_shares_side as not used for HEDGE sizing
                 shares_unrounded = calculate_trade_size('HEDGE', side_to_buy, hedge_buy_price, 0, 0, current_balance, market_adv, stats)
                 shares_rounded = config.SHARE_ROUNDING_FUNC(shares_unrounded)

                 if shares_rounded <= config.ZERO_SHARE_THRESHOLD:
                      analysis_details.append(f"  INFO: Hedge size zero for {side_to_buy}. Check Balance/Sizing.")
                      logging.debug(f"Hedge for '{market_name}': Sized to zero shares for {side_to_buy} (Unrounded: {shares_unrounded}).")
                 else:
                     cost = shares_rounded * hedge_buy_price
                     # Check balance (redundant with sizing check, but safe)
                     if cost <= current_balance + config.ZERO_PRICE_THRESHOLD:
                         recommendation.update({
                             "action_type": "BUY", "side": side_to_buy,
                             "shares_unrounded": shares_unrounded,
                             "shares_rounded": shares_rounded,
                             "price": hedge_buy_price, # Price is the Ask (Buy) price
                             "cost_proceeds": -cost,
                             "rule_triggered": "Hedge",
                             "display_text": f"REC (Hedge): Buy {utils.format_shares(shares_rounded)} {side_to_buy} @ Ask {utils.format_price(hedge_buy_price)} (Cost: {utils.format_currency(cost)})"
                         })
                         final_rec_found = True
                         analysis_details.append(f"  REC: Buy {utils.format_shares(shares_rounded)} {side_to_buy} to rebalance.")
                         logging.info(f"Hedge recommendation for '{market_name}': Buy {utils.format_shares(shares_rounded)} {side_to_buy}.")
                     else:
                         analysis_details.append(f"  INFO: Insufficient balance for hedge cost ({utils.format_currency(cost)} vs {utils.format_currency(current_balance)}).")
                         logging.warning(f"Hedge for '{market_name}': Insufficient balance ({utils.format_currency(current_balance)}) for cost {utils.format_currency(cost)}.")


    # --- PRIORITY 5: Profit Taking (Directional Positions Only) ---
    # --- MODIFIED RULE: Trigger based on absolute BID price threshold ---
    if not final_rec_found and current_state.startswith('DIRECTIONAL'):
        sell_triggered = False
        side_to_sell = None
        current_shares_held = 0.0
        current_sell_price = 0.0 # Use BID price (what you get)
        avg_price_check = 0.0 # Keep for logging context
        profit_threshold = config.PROFIT_TAKE_PRICE_THRESHOLD

        side_check = current_state.split('_')[-1] # 'YES' or 'NO'

        # Check YES side
        if side_check == 'YES' and yes_s > config.ZERO_SHARE_THRESHOLD:
             current_shares_held = yes_s
             current_sell_price = yes_b # Check BID price
             avg_price_check = avg_yes_p # For logging
             # Trigger condition: BID price > Threshold
             if current_sell_price > profit_threshold:
                 sell_triggered = True
                 side_to_sell = 'YES'

        # Check NO side (only if YES didn't trigger)
        elif side_check == 'NO' and no_s > config.ZERO_SHARE_THRESHOLD:
             current_shares_held = no_s
             current_sell_price = no_b # Check BID price
             avg_price_check = avg_no_p # For logging
             # Trigger condition: BID price > Threshold
             if current_sell_price > profit_threshold:
                 sell_triggered = True
                 side_to_sell = 'NO'

        # Action if triggered
        if sell_triggered and side_to_sell:
             pct_to_sell = config.PROFIT_TAKE_SELL_PCT
             rule_details = (f"Sell Price (Bid {utils.format_price(current_sell_price)}) > "
                             f"Threshold ({utils.format_price(profit_threshold)}). "
                             f"Sell {utils.format_percent(pct_to_sell)} of holdings.")

             analysis_details.append(f"- Profit Taking Check (Priority 5): Triggered on {side_to_sell}.")
             analysis_details.append(f"  - {rule_details}")
             logging.info(f"Profit Taking triggered for '{market_name}' on {side_to_sell}: {rule_details}")

             shares_to_sell_unrounded = current_shares_held * pct_to_sell
             # Apply rounding, ensuring we don't try to sell more than held after rounding
             shares_to_sell_rounded = config.SHARE_ROUNDING_FUNC(shares_to_sell_unrounded)
             current_shares_held_rounded = config.SHARE_ROUNDING_FUNC(current_shares_held) # Round held shares for comparison

             # Ensure final rounded shares don't exceed rounded held shares
             shares_to_sell_final_rounded = min(shares_to_sell_rounded, current_shares_held_rounded)


             if shares_to_sell_final_rounded > config.ZERO_SHARE_THRESHOLD:
                 proceeds = shares_to_sell_final_rounded * current_sell_price # Use BID price for proceeds
                 recommendation.update({
                     "action_type": "SELL", # Generic SELL action
                     "side": side_to_sell,
                     "shares_unrounded": shares_to_sell_unrounded,
                     "shares_rounded": shares_to_sell_final_rounded,
                     "price": current_sell_price, # Price is the Bid (Sell) price
                     "cost_proceeds": proceeds, # Positive for proceeds
                     "rule_triggered": f"Profit Taking (Price > {profit_threshold:.2f})",
                     "display_text": f"REC (Profit Take): Sell {utils.format_shares(shares_to_sell_final_rounded)} {side_to_sell} @ Bid {utils.format_price(current_sell_price)} (Proc: {utils.format_currency(proceeds)})"
                 })
                 final_rec_found = True
                 analysis_details.append(f"  REC: Sell {utils.format_shares(shares_to_sell_final_rounded)} {side_to_sell}.")
                 logging.info(f"Profit Taking recommendation for '{market_name}': Sell {utils.format_shares(shares_to_sell_final_rounded)} {side_to_sell}.")

             else:
                 analysis_details.append(f"  INFO: Profit taking size zero after rounding/calculation (Unrounded: {shares_to_sell_unrounded:.4f}, Held: {current_shares_held:.4f}, Rounded: {shares_to_sell_rounded:.4f}).")
                 logging.debug(f"Profit Taking for '{market_name}': Sized to zero shares after rounding.")
        # else: # Optional: Add logging if check was performed but not triggered
        #    if current_state.startswith('DIRECTIONAL') and side_check:
        #        analysis_details.append(f"- Profit Taking Check (Priority 5): Not triggered for {side_check}. Current Bid: {utils.format_price(current_sell_price)}, Threshold: {utils.format_price(profit_threshold)}")


    # --- PRIORITY 6: Accumulation (Directional Positions Only) ---
    # Goal: If holding a directional position and the price (ASK) drops significantly below average cost, buy more.
    if not final_rec_found and current_state.startswith('DIRECTIONAL'):
        accum_possible = False; accum_side = None; accum_buy_price = 0.0; accum_avg_cost = 0.0; accum_details = []
        side_to_check = current_state.split('_')[-1]

        # Accumulation trigger: Check if ASK price (cost to buy more) has dropped relative to avg cost
        if side_to_check == 'YES' and yes_s > config.ZERO_SHARE_THRESHOLD and avg_yes_p > config.ZERO_PRICE_THRESHOLD:
             # Check ASK price vs avg cost
             if yes_a > config.ZERO_PRICE_THRESHOLD and yes_a < avg_yes_p * (1.0 - config.ACCUMULATION_DROP_THRESHOLD):
                 accum_possible = True; accum_side = 'YES'; accum_buy_price = yes_a; accum_avg_cost = avg_yes_p
                 accum_details.append(f"YES Buy Price (Ask {utils.format_price(yes_a)}) < {utils.format_percent(1.0 - config.ACCUMULATION_DROP_THRESHOLD)} of avg ({utils.format_price(avg_yes_p)}).")
        elif side_to_check == 'NO' and no_s > config.ZERO_SHARE_THRESHOLD and avg_no_p > config.ZERO_PRICE_THRESHOLD:
             # Check ASK price vs avg cost
             if no_a > config.ZERO_PRICE_THRESHOLD and no_a < avg_no_p * (1.0 - config.ACCUMULATION_DROP_THRESHOLD):
                 accum_possible = True; accum_side = 'NO'; accum_buy_price = no_a; accum_avg_cost = avg_no_p
                 accum_details.append(f"NO Buy Price (Ask {utils.format_price(no_a)}) < {utils.format_percent(1.0 - config.ACCUMULATION_DROP_THRESHOLD)} of avg ({utils.format_price(avg_no_p)}).")

        if accum_possible and accum_side:
             analysis_details.append("- Accumulation Check (Priority 6): Triggered.")
             analysis_details.extend([f"  - {d}" for d in accum_details])
             logging.info(f"Accumulation triggered for '{market_name}' on {accum_side}: {' '.join(accum_details)}")

             # Check if buy price is valid (already checked > 0, also check < 1)
             if accum_buy_price >= 1.0:
                  analysis_details.append(f"  INFO: Cannot accumulate {accum_side}, Ask price >= $1 ({utils.format_price(accum_buy_price)}).")
                  logging.warning(f"Accumulation for '{market_name}': Cannot buy {accum_side}, Ask price invalid/high ({utils.format_price(accum_buy_price)}).")
             else:
                 # Size the accumulation trade using the BUY price (Ask) and risk parameters
                 current_side_shares = stats.get(f'{accum_side.lower()}_shares', 0)
                 shares_unrounded = calculate_trade_size('ACCUMULATION', accum_side, accum_buy_price, accum_avg_cost, current_side_shares, current_balance, market_adv, stats)
                 shares_rounded = config.SHARE_ROUNDING_FUNC(shares_unrounded)

                 if shares_rounded <= config.ZERO_SHARE_THRESHOLD:
                     analysis_details.append(f"  INFO: Accumulation size zero for {accum_side}. Check Balance/Alloc/Sizing.")
                     logging.debug(f"Accumulation for '{market_name}': Sized to zero shares for {accum_side} (Unrounded: {shares_unrounded}).")
                 else:
                     cost = shares_rounded * accum_buy_price
                     if cost <= current_balance + config.ZERO_PRICE_THRESHOLD:
                         # Calculate estimated stop loss level based on this trade's BUY price (Ask)
                         # Use specific accumulation stop % if defined, else default directional %
                         stop_pct_accum = config.ACCUMULATION_STOP_LOSS_PCT if config.ACCUMULATION_STOP_LOSS_PCT is not None else config.DIRECTIONAL_STOP_LOSS_PCT
                         stop_loss_for_accum = accum_buy_price * (1.0 - stop_pct_accum)
                         stop_loss_for_accum = max(config.ZERO_PRICE_THRESHOLD, min(stop_loss_for_accum, accum_buy_price - config.ZERO_PRICE_THRESHOLD))

                         recommendation.update({
                             "action_type": "BUY", "side": accum_side,
                             "shares_unrounded": shares_unrounded,
                             "shares_rounded": shares_rounded,
                             "price": accum_buy_price, # Price is Ask (Buy)
                             "cost_proceeds": -cost,
                             "rule_triggered": "Accumulation",
                             "calculated_stop_level": stop_loss_for_accum, # Pass stop level for execution to update market_data
                             "display_text": f"REC (Accumulate): Buy {utils.format_shares(shares_rounded)} {accum_side} @ Ask {utils.format_price(accum_buy_price)} (Cost: {utils.format_currency(cost)}, New Stop: ~{utils.format_price(stop_loss_for_accum)})"
                         })
                         final_rec_found = True
                         analysis_details.append(f"  REC: Buy {utils.format_shares(shares_rounded)} {accum_side} (Risk-Sized). Est. New Stop @ ~{utils.format_price(stop_loss_for_accum)}")
                         logging.info(f"Accumulation recommendation for '{market_name}': Buy {utils.format_shares(shares_rounded)} {accum_side}, Stop ~{utils.format_price(stop_loss_for_accum)}.")
                     else:
                         analysis_details.append(f"  INFO: Insufficient balance for Accumulation ({utils.format_currency(cost)} vs {utils.format_currency(current_balance)}).")
                         logging.warning(f"Accumulation for '{market_name}': Insufficient balance ({utils.format_currency(current_balance)}) for cost {utils.format_currency(cost)}.")


    # --- PRIORITY 7: Initial Entry (FLAT State Only) ---
    # Goal: If the market is FLAT, buy YES or NO if the ASK price is below the entry threshold.
    if not final_rec_found and current_state == 'FLAT':
        entry_side = None; entry_buy_price = 0.0 # Use ASK price for entry

        # Trigger based on ASK price (cost to buy) <= BUY_THRESHOLD and > 0
        buy_yes_triggered = config.ZERO_PRICE_THRESHOLD < yes_a <= config.BUY_THRESHOLD
        buy_no_triggered = config.ZERO_PRICE_THRESHOLD < no_a <= config.BUY_THRESHOLD

        # Decide which side to buy if both are triggered (prefer cheaper)
        if buy_yes_triggered and buy_no_triggered:
             if yes_a <= no_a:
                 entry_side = 'YES'; entry_buy_price = yes_a
             else:
                 entry_side = 'NO'; entry_buy_price = no_a
        elif buy_yes_triggered:
             entry_side = 'YES'; entry_buy_price = yes_a
        elif buy_no_triggered:
             entry_side = 'NO'; entry_buy_price = no_a

        # If an entry condition met
        if entry_side:
            analysis_details.append(f"- Entry Check (Priority 7): State FLAT & {entry_side} Buy Price (Ask {utils.format_price(entry_buy_price)}) <= Threshold ({utils.format_price(config.BUY_THRESHOLD)}).")
            logging.info(f"Entry triggered for '{market_name}' on {entry_side} at Ask {utils.format_price(entry_buy_price)}")

            # Size the entry trade using the BUY price (Ask) and risk parameters
            # Pass 0 for avg_cost, current_shares_side as this is the first entry
            shares_unrounded = calculate_trade_size('ENTRY', entry_side, entry_buy_price, 0, 0, current_balance, market_adv, stats)
            shares_rounded = config.SHARE_ROUNDING_FUNC(shares_unrounded)

            if shares_rounded <= config.ZERO_SHARE_THRESHOLD:
                analysis_details.append(f"  INFO: Entry size zero for {entry_side}. Check Balance/Alloc/Sizing.")
                logging.debug(f"Entry for '{market_name}': Sized to zero shares for {entry_side} (Unrounded: {shares_unrounded}).")
            else:
                cost = shares_rounded * entry_buy_price
                if cost <= current_balance + config.ZERO_PRICE_THRESHOLD:
                     # Calculate stop loss level based on this entry's BUY price (Ask) using the directional stop %
                     stop_loss_for_entry = entry_buy_price * (1.0 - config.DIRECTIONAL_STOP_LOSS_PCT)
                     stop_loss_for_entry = max(config.ZERO_PRICE_THRESHOLD, min(stop_loss_for_entry, entry_buy_price - config.ZERO_PRICE_THRESHOLD))

                     recommendation.update({
                         "action_type": "BUY", "side": entry_side,
                         "shares_unrounded": shares_unrounded,
                         "shares_rounded": shares_rounded,
                         "price": entry_buy_price, # Price is Ask (Buy)
                         "cost_proceeds": -cost,
                         "rule_triggered": "Entry",
                         "calculated_stop_level": stop_loss_for_entry, # Pass stop level for execution to set in market_data
                         "display_text": f"REC (Entry): Buy {utils.format_shares(shares_rounded)} {entry_side} @ Ask {utils.format_price(entry_buy_price)} (Cost: {utils.format_currency(cost)}, Stop: ~{utils.format_price(stop_loss_for_entry)})"
                     })
                     final_rec_found = True
                     analysis_details.append(f"  REC: Initial Buy {utils.format_shares(shares_rounded)} {entry_side} (Risk-Sized). Est. Stop @ ~{utils.format_price(stop_loss_for_entry)}")
                     logging.info(f"Entry recommendation for '{market_name}': Buy {utils.format_shares(shares_rounded)} {entry_side}, Stop ~{utils.format_price(stop_loss_for_entry)}.")
                else:
                     analysis_details.append(f"  INFO: Insufficient balance for Entry ({utils.format_currency(cost)} vs {utils.format_currency(current_balance)}).")
                     logging.warning(f"Entry for '{market_name}': Insufficient balance ({utils.format_currency(current_balance)}) for cost {utils.format_currency(cost)}.")
        # else: # No entry triggered
        #    analysis_details.append(f"- Entry Check (Priority 7): No side met threshold. YES Ask: {utils.format_price(yes_a)}, NO Ask: {utils.format_price(no_a)}, Threshold: {utils.format_price(config.BUY_THRESHOLD)}")


    # --- Default: HOLD ---
    if not final_rec_found:
        analysis_details.append("- No action rule triggered.")
        # Add specific reason if in directional state but stop not hit and no other rule applied
        if current_state.startswith("DIRECTIONAL") and directional_stop_price_level is not None:
            side = current_state.split('_')[-1]
            current_bid = yes_b if side == 'YES' else no_b
            analysis_details.append(f"  INFO: Holding {side}. Current Bid {utils.format_price(current_bid)} above Stop {utils.format_price(directional_stop_price_level)}. No other rules met.")
        elif current_state in ['HEDGED', 'COST_BASIS_ARB']:
             analysis_details.append(f"  INFO: Holding {current_state}. No stop/arb/accum rules met.")
        else: # FLAT
             analysis_details.append("  ACTION: Monitor market for entry opportunities.")

        logging.debug(f"No action recommended for '{market_name}' in state {current_state}.")
        # Keep default HOLD recommendation

    analysis_details.append(f"--- Strategy Eval End: '{market_name}' ---")

    # Log the analysis details at DEBUG level
    for detail in analysis_details:
        logging.debug(detail)

    # Log a summary at INFO level if a recommendation IS made
    if recommendation['action_type'] != 'HOLD':
        logging.info(f"Recommendation for '{market_name}': {recommendation['display_text']}")

    # Ensure shares are non-negative before returning
    recommendation['shares_unrounded'] = max(0.0, recommendation['shares_unrounded'])
    recommendation['shares_rounded'] = max(0.0, recommendation['shares_rounded'])

    return recommendation, analysis_details
