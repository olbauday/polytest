# main_app.py
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import simpledialog
import tkinter.font as tkFont
import traceback
import logging # <-- Import logging
import time # For Tooltip delay

# Import modules
import config
import utils
import data_manager
import strategy_engine

# --- Simple Tooltip Class ---
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave) # Hide tooltip on click
        self.id = None
        self.delay = 500 # ms delay before showing tooltip

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.delay, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self):
        if self.tooltip_window:
            return
        # Calculate position relative to widget
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # Create Toplevel window
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True) # No window decorations
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip_window, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tooltip = self.tooltip_window
        self.tooltip_window = None
        if tooltip:
            tooltip.destroy()

# --- Global GUI Variables ---
root = None
selected_market_name = None
last_recommendation = {} # Cache for execution
default_widget_bg = None # Store default background color

# --- Tkinter StringVars ---
balance_var=None; initial_balance_var=None; selected_market_var=None; adv_var=None;
summary_yes_shares_var=None; summary_no_shares_var=None; summary_yes_inv_var=None;
summary_no_inv_var=None; summary_total_inv_var=None; current_pl_yes_wins_var=None;
current_pl_no_wins_var=None; current_unrealized_pl_var=None; position_state_var = None;
stop_loss_info_var = None; market_arb_var=None; recommendation_var=None;
total_realized_pl_var = None; status_bar_var = None
# NEW StringVars for Bid/Ask prices
yes_bid_var=None; yes_ask_var=None; no_bid_var=None; no_ask_var=None;

# --- Tkinter Widgets (Placeholder references) ---
balance_label=None; entry_initial_balance=None; btn_set_balance=None; pw=None; market_selector_combo=None;
add_market_button=None; delete_market_button=None; left_pane=None; holdings_frame=None;
bets_added_tree=None; bets_scrollbar=None; holdings_buttons_frame=None; remove_mistake_button=None;
clear_mistakes_button=None; add_manual_button=None; right_pane=None; log_frame=None; transaction_log_tree=None;
log_scrollbar=None; clear_log_button=None; summary_analysis_frame=None; summary_frame=None;
summary_yes_shares_label=None; summary_no_shares_label=None; summary_yes_inv_label=None;
summary_no_inv_label=None; summary_total_inv_label=None; current_pl_yes_wins_label=None;
current_pl_no_wins_label=None; current_unrealized_pl_label=None; position_state_label=None;
stop_loss_info_label=None; calc_frame=None; calc_input_frame=None;
# NEW: Bid/Ask entries
entry_yes_bid=None; entry_yes_ask=None; entry_no_bid=None; entry_no_ask=None;
entry_adv=None; calculate_button=None; rec_exec_frame=None;
recommendation_label=None; execute_button=None; analysis_results_frame=None; market_arb_label=None;
arbitrage_recommendation_text=None; analysis_text_container=None; analysis_text_widget=None;
analysis_text_scrollbar=None; disclaimer_label=None; total_realized_pl_label=None; status_bar_label=None
# NEW WIDGETS
resolve_market_button = None; delete_log_button = None

# --- Helper Functions ---
def update_status_bar(message):
    """Updates the status bar text."""
    global status_bar_var
    if status_bar_var:
        try:
            status_bar_var.set(message)
            logging.debug(f"Status Bar: {message}")
        except tk.TclError:
            pass

def update_widget_states():
    """Enables/Disables widgets based on application state."""
    global selected_market_name, last_recommendation, market_selector_combo
    global add_market_button, delete_market_button, resolve_market_button
    global remove_mistake_button, clear_mistakes_button, add_manual_button
    global clear_log_button, delete_log_button
    global calculate_button, execute_button
    global entry_yes_bid, entry_yes_ask, entry_no_bid, entry_no_ask, entry_adv # Bid/Ask entries
    global bets_added_tree, transaction_log_tree

    try:
        market_selected = bool(selected_market_name) and data_manager.market_exists(selected_market_name)
        has_markets = bool(data_manager.get_all_market_names())
        has_holdings = False
        has_log_entries = False
        is_market_flat = True
        log_entry_selected = False

        if market_selected:
            market_data = data_manager.get_market_data(selected_market_name)
            is_market_flat = market_data.get('position_state', 'FLAT') == 'FLAT' if market_data else True
            holdings_items = bets_added_tree.get_children() if bets_added_tree else []
            log_items = transaction_log_tree.get_children() if transaction_log_tree else []
            has_holdings = bool(holdings_items)
            has_log_entries = bool(log_items)
            log_entry_selected = bool(transaction_log_tree.selection()) if transaction_log_tree else False

        # Market controls
        if delete_market_button: delete_market_button.config(state=tk.NORMAL if market_selected else tk.DISABLED)
        if resolve_market_button: resolve_market_button.config(state=tk.NORMAL if market_selected and not is_market_flat else tk.DISABLED)

        # Price/ADV entries & Analysis button
        for widget in [entry_yes_bid, entry_yes_ask, entry_no_bid, entry_no_ask, entry_adv, calculate_button]:
            if widget: widget.config(state=tk.NORMAL if market_selected else tk.DISABLED)

        # Holdings controls
        if remove_mistake_button: remove_mistake_button.config(state=tk.NORMAL if market_selected and has_holdings else tk.DISABLED)
        if clear_mistakes_button: clear_mistakes_button.config(state=tk.NORMAL if market_selected and has_holdings else tk.DISABLED)
        if add_manual_button: add_manual_button.config(state=tk.NORMAL if market_selected else tk.DISABLED)

        # Log controls
        if clear_log_button: clear_log_button.config(state=tk.NORMAL if market_selected and has_log_entries else tk.DISABLED)
        if delete_log_button: delete_log_button.config(state=tk.NORMAL if market_selected and log_entry_selected else tk.DISABLED)

        # Execution button
        action_type = last_recommendation.get("action_type")
        is_actionable = action_type in ['BUY', 'SELL', 'BUY_ARB', 'SELL_ARB', 'SELL_STOP']
        if execute_button: execute_button.config(state=tk.NORMAL if market_selected and is_actionable else tk.DISABLED)

        # Market Selector (Always enabled if markets exist)
        if market_selector_combo: market_selector_combo.config(state="readonly" if has_markets else tk.DISABLED)

    except tk.TclError: pass
    except Exception as e:
        logging.warning(f"Error updating widget states: {e}", exc_info=True)
        update_status_bar("Error updating widget enable/disable states.")


def reset_entry_style(widget):
    """Resets the background color of an entry widget."""
    global default_widget_bg
    if widget:
        try:
            # Attempt to use ttk style if available for Entry
            style = ttk.Style()
            default_entry_bg = style.lookup('TEntry', 'fieldbackground')
            widget.config(background=default_entry_bg)
        except tk.TclError:
            # Fallback for non-ttk or if style lookup fails
            try: widget.config(background=default_widget_bg)
            except tk.TclError: pass


# --- GUI Update Functions ---

def update_balance_display():
    """Updates the global balance and total realized P/L labels."""
    global balance_label, total_realized_pl_label, balance_var, total_realized_pl_var
    try:
        current_balance = data_manager.get_global_balance()
        current_pl = data_manager.get_total_realized_pl()

        if balance_var and balance_label:
            color = "darkgreen" if current_balance >= 0 else "red"
            balance_var.set(f"Global: {utils.format_currency(current_balance)}")
            balance_label.config(foreground=color)

        if total_realized_pl_var and total_realized_pl_label:
            pl_color = "darkgreen" if current_pl >= -0.001 else "red"
            total_realized_pl_var.set(f"Total Realized P/L: {utils.format_currency(current_pl)}")
            total_realized_pl_label.config(foreground=pl_color)

    except Exception as e:
        logging.error(f"Balance display update error: {e}", exc_info=True)
        update_status_bar("Error updating balance display.")

def update_market_selector():
    """Updates the market selection dropdown."""
    global market_selector_combo, selected_market_var, selected_market_name
    if market_selector_combo and selected_market_var:
        try:
            current_selection = selected_market_var.get()
            market_names_raw = data_manager.get_all_market_names()
            market_display_names = [
                f"{'[TEST] ' if data_manager.get_market_data(name).get('is_test_market') else ''}{name}"
                for name in market_names_raw
            ]
            market_selector_combo['values'] = market_display_names

            selected_display_name = None
            if selected_market_name:
                 market_data = data_manager.get_market_data(selected_market_name)
                 prefix = "[TEST] " if market_data and market_data.get('is_test_market') else ""
                 selected_display_name = prefix + selected_market_name

            if selected_display_name and selected_display_name in market_display_names:
                selected_market_var.set(selected_display_name)
            elif market_display_names:
                new_selection = market_display_names[0]
                selected_market_var.set(new_selection)
                selected_market_name = new_selection.replace("[TEST] ", "")
            else:
                selected_market_var.set("")
                selected_market_name = None
        except tk.TclError: pass
        except Exception as e:
            logging.error(f"Error updating market selector: {e}", exc_info=True)
            update_status_bar("Error updating market list.")


def update_holdings_display(market_name):
    """Updates the holdings treeview."""
    global bets_added_tree
    if not bets_added_tree: return
    try:
        for item in bets_added_tree.get_children(): bets_added_tree.delete(item)
    except tk.TclError: pass

    if market_name:
        try:
            bets_data = data_manager.get_bets(market_name)
            for bet in bets_data:
                original_price = bet.get('price', 0.0)
                remaining_amount = bet.get('amount', 0.0)
                side = bet.get('side', 'N/A')
                bet_id = bet.get('id', 'N/A')

                shares_rem_unrounded = 0.0
                if original_price > config.ZERO_PRICE_THRESHOLD and remaining_amount >= config.ZERO_SHARE_THRESHOLD:
                    shares_rem_unrounded = remaining_amount / original_price
                shares_rem_formatted = utils.format_shares(config.SHARE_ROUNDING_FUNC(shares_rem_unrounded))

                bets_added_tree.insert('', tk.END, iid=bet_id, values=(
                    side, shares_rem_formatted, utils.format_currency(remaining_amount), utils.format_price(original_price)
                ))
        except Exception as e:
            logging.error(f"Error updating holdings display for '{market_name}': {e}", exc_info=True)
            update_status_bar(f"Error loading holdings for {market_name}.")


def update_transaction_log_display(market_name):
    """Updates the transaction log treeview."""
    global transaction_log_tree
    if not transaction_log_tree: return
    try:
        selected_iid = transaction_log_tree.selection()[0] if transaction_log_tree.selection() else None
        for item in transaction_log_tree.get_children(): transaction_log_tree.delete(item)
    except tk.TclError:
        selected_iid = None
    except IndexError:
        selected_iid = None


    if market_name:
        try:
            log_data = data_manager.get_transactions(market_name)
            new_selection_found = False
            for i, entry in enumerate(reversed(log_data)):
                entry_ts = entry.get('ts', 'N/A')
                # Use only timestamp as identifier for deletion - assumes reasonable uniqueness
                # If collisions are frequent, need a different identifier passed to delete function
                iid_for_display = entry_ts # Use TS string as IID for selection/deletion lookup

                transaction_log_tree.insert('', tk.END, iid=iid_for_display, values=(
                    entry_ts, entry.get('type', 'N/A'), entry.get('side', 'N/A'),
                    utils.format_shares(entry.get('shares')), utils.format_price(entry.get('price')),
                    utils.format_cash_flow(entry.get('cash_flow')), utils.format_currency(entry.get('balance'))
                ))
                if iid_for_display == selected_iid:
                    transaction_log_tree.selection_set(iid_for_display)
                    new_selection_found = True

            if not new_selection_found:
                 transaction_log_tree.selection_set([]) # Clear selection if old one is gone

        except Exception as e:
             logging.error(f"Error updating tx log display for '{market_name}': {e}", exc_info=True)
             update_status_bar(f"Error loading transaction log for {market_name}.")

    # Update widget states AFTER display update
    update_widget_states()


def update_summary_display(market_name, clear_projections_on_update=True):
    """Updates the summary section, including basis, shares, and optionally projections."""
    global summary_yes_shares_var, summary_no_shares_var, summary_yes_inv_var, summary_no_inv_var
    global summary_total_inv_var, current_pl_yes_wins_var, current_pl_no_wins_var, current_unrealized_pl_var
    global current_pl_yes_wins_label, current_pl_no_wins_label, current_unrealized_pl_label
    global entry_yes_bid, entry_no_bid # Need bid prices for market value calc

    yes_shares_txt = "YES Shares: ---"; no_shares_txt = "NO Shares: ---"
    yes_basis_txt = "YES Basis: ---"; no_basis_txt = "NO Basis: ---"
    total_basis_txt = "Total Basis: ---"

    proj_vars = [current_pl_yes_wins_var, current_pl_no_wins_var, current_unrealized_pl_var]
    proj_defaults = ["Proj Bal (YES): ---", "Proj Bal (NO): ---", "Est. Mkt Val (Sell Bids): --- (Unreal P/L: ---)"]
    proj_labels = [current_pl_yes_wins_label, current_pl_no_wins_label, current_unrealized_pl_label]
    try:
        if clear_projections_on_update:
            for var, default_text in zip(proj_vars, proj_defaults):
                if var: var.set(default_text)
            for lbl in proj_labels:
                if lbl: lbl.config(foreground="black")
    except tk.TclError: pass

    if market_name and data_manager.market_exists(market_name):
        try:
            stats = data_manager.calculate_position_stats(market_name)
            yes_s, no_s = stats['yes_shares'], stats['no_shares']
            yes_inv, no_inv = stats['yes_investment'], stats['no_investment']
            avg_yes_p, avg_no_p = stats['yes_avg_price'], stats['no_avg_price']

            yes_shares_txt = f"YES Shares: {utils.format_shares(yes_s)}"
            no_shares_txt = f"NO Shares: {utils.format_shares(no_s)}"
            yes_basis_txt = f"YES Basis: {utils.format_currency(yes_inv)} (Avg: {utils.format_price(avg_yes_p)})"
            no_basis_txt = f"NO Basis: {utils.format_currency(no_inv)} (Avg: {utils.format_price(avg_no_p)})"
            total_basis = yes_inv + no_inv
            total_basis_txt = f"Total Basis: {utils.format_currency(total_basis)}"

            projections_calculated = False # Flag to track if projections were calculated
            if not clear_projections_on_update:
                has_holdings = yes_s > config.ZERO_SHARE_THRESHOLD or no_s > config.ZERO_SHARE_THRESHOLD
                if has_holdings:
                    yes_b_str = entry_yes_bid.get() if entry_yes_bid else ""; no_b_str = entry_no_bid.get() if entry_no_bid else ""
                    yes_b = utils.validate_price(yes_b_str); no_b = utils.validate_price(no_b_str)

                    if yes_b is not None and no_b is not None:
                        payout_if_yes = yes_s * 1.0; payout_if_no = no_s * 1.0
                        market_pl_if_yes = payout_if_yes - total_basis
                        market_pl_if_no = payout_if_no - total_basis
                        current_balance = data_manager.get_global_balance()
                        market_data = data_manager.get_market_data(market_name)
                        is_test = market_data.get('is_test_market', False) if market_data else False
                        proj_bal_if_yes_disp = f"{utils.format_currency(current_balance + market_pl_if_yes)}{'' if not is_test else ' (Sim.)'}"
                        proj_bal_if_no_disp = f"{utils.format_currency(current_balance + market_pl_if_no)}{'' if not is_test else ' (Sim.)'}"

                        current_market_value = (yes_s * yes_b) + (no_s * no_b)
                        current_unrealized_pl = current_market_value - total_basis
                        unrealized_pl_pct = (current_unrealized_pl / total_basis * 100.0) if total_basis > config.ZERO_PRICE_THRESHOLD else 0.0

                        try:
                            if current_pl_yes_wins_var: current_pl_yes_wins_var.set(f"Proj Bal (YES): {proj_bal_if_yes_disp} (Mkt P/L: {utils.format_cash_flow(market_pl_if_yes)})")
                            if current_pl_no_wins_var: current_pl_no_wins_var.set(f"Proj Bal (NO): {proj_bal_if_no_disp} (Mkt P/L: {utils.format_cash_flow(market_pl_if_no)})")
                            if current_unrealized_pl_var: current_unrealized_pl_var.set(f"Est Mkt Val (Bids): {utils.format_currency(current_market_value)} (Unrl P/L: {utils.format_cash_flow(current_unrealized_pl)}, {unrealized_pl_pct:+.2f}%)") # Shorter Label

                            yes_color = "darkgreen" if market_pl_if_yes >= -0.001 else "red"
                            no_color = "darkgreen" if market_pl_if_no >= -0.001 else "red"
                            unrlz_color = "darkgreen" if current_unrealized_pl >= -0.001 else "red"
                            if current_pl_yes_wins_label: current_pl_yes_wins_label.config(foreground=yes_color)
                            if current_pl_no_wins_label: current_pl_no_wins_label.config(foreground=no_color)
                            if current_unrealized_pl_label: current_unrealized_pl_label.config(foreground=unrlz_color)
                            projections_calculated = True # Mark as successful
                        except tk.TclError: pass

            # If projections weren't calculated, set defaults
            if not projections_calculated:
                 try:
                     for var, default_text in zip(proj_vars, proj_defaults):
                         if var: var.set(default_text)
                     for lbl in proj_labels:
                         if lbl: lbl.config(foreground="black")
                 except tk.TclError: pass

        except Exception as e:
            logging.error(f"Error updating summary display for '{market_name}': {e}", exc_info=True)
            update_status_bar(f"Error calculating summary for {market_name}.")
            yes_shares_txt = "YES Shares: Error"; no_shares_txt = "NO Shares: Error"; yes_basis_txt = "YES Basis: Error"; no_basis_txt = "NO Basis: Error"; total_basis_txt = "Total Basis: Error"
            try:
                for var, default_text in zip(proj_vars, proj_defaults):
                    if var: var.set(default_text.replace("---", "Error"))
                for lbl in proj_labels:
                    if lbl: lbl.config(foreground="red")
            except tk.TclError: pass

    try:
        if summary_yes_shares_var: summary_yes_shares_var.set(yes_shares_txt)
        if summary_no_shares_var: summary_no_shares_var.set(no_shares_txt)
        if summary_yes_inv_var: summary_yes_inv_var.set(yes_basis_txt)
        if summary_no_inv_var: summary_no_inv_var.set(no_basis_txt)
        if summary_total_inv_var: summary_total_inv_var.set(total_basis_txt)
    except tk.TclError: pass


def update_state_display(market_name):
    """Updates the position state label."""
    global position_state_var, position_state_label
    state_text = "State: ---"
    state_color = "black"
    if market_name:
        state = data_manager.get_position_state(market_name)
        state_text = f"State: {state}"
        if state == 'FLAT': state_color = "gray"
        elif state.startswith('DIRECTIONAL'): state_color = "blue"
        elif state == 'HEDGED': state_color = "purple"
        elif state == 'COST_BASIS_ARB': state_color = "darkgreen"
    try:
        if position_state_var: position_state_var.set(state_text)
        if position_state_label: position_state_label.config(foreground=state_color)
    except tk.TclError: pass

def update_stop_loss_display(market_name):
    """Updates the stop loss information label."""
    global stop_loss_info_var, stop_loss_info_label
    stop_info_text = "Stop Info: ---"
    stop_color = "black"
    if market_name:
        market_data = data_manager.get_market_data(market_name)
        if market_data:
            state = market_data.get('position_state', 'FLAT')
            stop_price = market_data.get('directional_stop_loss')

            if state.startswith('DIRECTIONAL') and stop_price is not None:
                side = state.split('_')[-1]
                stop_info_text = f"Stop ({side}): {utils.format_price(stop_price)}"
                stop_color = "red"
            elif state in ['HEDGED', 'COST_BASIS_ARB']:
                stop_thresh = getattr(config, 'HEDGED_STOP_LOSS_PCT_BASIS', 0.05)
                stop_info_text = f"Stop (Pair): Basis Loss > {utils.format_percent(stop_thresh)}"
                stop_color = "red"
            else: stop_info_text = "Stop Info: N/A"
    try:
        if stop_loss_info_var: stop_loss_info_var.set(stop_info_text)
        if stop_loss_info_label: stop_loss_info_label.config(foreground=stop_color)
    except tk.TclError: pass


def update_recommendation_display():
    """Updates the recommendation label based on the cached recommendation."""
    global recommendation_var, recommendation_label, last_recommendation
    if not recommendation_label or not recommendation_var: return
    try:
        rec_text = last_recommendation.get("display_text", "Recommendation: ---")
        action_type = last_recommendation.get("action_type")
        recommendation_var.set(rec_text)
        color = "black"
        if action_type == 'BUY' or action_type == 'BUY_ARB': color = "blue"
        elif action_type == 'SELL' or action_type == 'SELL_ARB': color = "orange"
        elif action_type == 'SELL_STOP': color = "red"
        elif action_type == 'HOLD': color = "darkgreen"
        recommendation_label.config(foreground=color)
        update_widget_states()
    except tk.TclError: pass

def clear_recommendation_cache():
    """Clears the cached recommendation."""
    global last_recommendation
    last_recommendation = {
        "market": None, "action_type": None, "side": None,
        "shares_unrounded": 0.0, "shares_rounded": 0.0,
        "price": 0.0, "cost_proceeds": 0.0, "rule_triggered": None,
        "display_text": "Recommendation: ---", "calculated_stop_level": None
    }
    update_recommendation_display()

def clear_calculation_results(clear_projections=False):
    """Clears analysis text widgets AND optionally projection labels."""
    global market_arb_var, market_arb_label, arbitrage_recommendation_text, analysis_text_widget
    global entry_yes_bid, entry_yes_ask, entry_no_bid, entry_no_ask, entry_adv # Updated entries
    try:
        if market_arb_var: market_arb_var.set("Market Arb?: ---")
        if market_arb_label: market_arb_label.config(foreground="black")
        utils.set_text_widget_content(arbitrage_recommendation_text, "---")
        utils.set_text_widget_content(analysis_text_widget, "---")

        if clear_projections:
            proj_vars = [current_pl_yes_wins_var, current_pl_no_wins_var, current_unrealized_pl_var]
            proj_defaults = ["Proj Bal (YES): ---", "Proj Bal (NO): ---", "Est. Mkt Val (Sell Bids): --- (Unreal P/L: ---)"]
            for var, default_text in zip(proj_vars, proj_defaults):
                 if var: var.set(default_text)
            proj_labels = [current_pl_yes_wins_label, current_pl_no_wins_label, current_unrealized_pl_label]
            for lbl in proj_labels:
                if lbl: lbl.config(foreground="black")

        # Clear input error highlighting
        for entry in [entry_yes_bid, entry_yes_ask, entry_no_bid, entry_no_ask, entry_adv]:
            reset_entry_style(entry)

    except tk.TclError: pass
    except Exception as e:
        logging.warning(f"Minor error clearing calculation results display: {e}", exc_info=True)

def clear_all_displays():
     """Clears Holdings, Log, Summary, Projections, State, Stops, and Analysis Results."""
     logging.debug("Clearing all displays...")
     update_holdings_display(None)
     update_transaction_log_display(None)
     update_summary_display(None, clear_projections_on_update=True)
     update_state_display(None)
     update_stop_loss_display(None)
     clear_calculation_results(clear_projections=False)
     update_widget_states()
     update_status_bar("Display cleared.")

def update_all_displays_for_market(market_name):
    """Updates Holdings, Log, Summary, State, and Stops for the given market."""
    if not market_name or not data_manager.market_exists(market_name):
        logging.warning(f"Update displays skipped: market '{market_name}' invalid/not found.")
        update_status_bar(f"Market '{market_name}' not found or invalid.")
        clear_all_displays()
        return
    logging.debug(f"Updating all displays for market: {market_name}")
    update_status_bar(f"Loading data for market: {market_name}...")
    update_holdings_display(market_name)
    update_transaction_log_display(market_name) # Updates log widgets state too
    update_summary_display(market_name, clear_projections_on_update=True) # Clear proj initially
    update_state_display(market_name)
    update_stop_loss_display(market_name)
    update_widget_states() # General update
    update_status_bar(f"Market '{market_name}' loaded.")


# --- GUI Callbacks / Event Handlers ---

def set_initial_balance_callback(event=None):
    """Callback to set the initial balance."""
    global entry_initial_balance
    update_status_bar("Setting initial balance...")
    try:
        initial_bal_str = entry_initial_balance.get()
        if not initial_bal_str:
            messagebox.showerror("Input Error", "Initial balance cannot be empty."); update_status_bar("Set Balance Error: Input empty."); return
        initial_bal = float(initial_bal_str)
        if initial_bal < 0:
            messagebox.showerror("Input Error", "Initial balance cannot be negative."); update_status_bar("Set Balance Error: Value negative."); return

        if data_manager.set_global_balance(initial_bal):
            update_balance_display()
            logging.info(f"Global Balance set to: {utils.format_currency(initial_bal)}")
            messagebox.showinfo("Balance Set", f"Global Balance set to: {utils.format_currency(initial_bal)}")
            update_status_bar(f"Global balance set to {utils.format_currency(initial_bal)}.")
            run_analysis_if_possible()
        else:
            messagebox.showerror("Error", "Failed to set balance (internal error)."); update_status_bar("Set Balance Error: Failed internally.")
    except ValueError:
        messagebox.showerror("Input Error", "Invalid number format for balance."); update_status_bar("Set Balance Error: Invalid number.")
    except Exception as e:
        logging.error(f"Error setting balance: {e}", exc_info=True)
        messagebox.showerror("Error", f"An unexpected error occurred: {e}"); update_status_bar("Set Balance Error: Unexpected error.")
    finally: update_widget_states()

def add_new_market_callback():
    """Callback to add a new market."""
    global selected_market_name, market_selector_combo, entry_yes_bid # Focus target changed
    update_status_bar("Adding new market...")
    is_test = messagebox.askyesno("Market Type", "Is this a TEST market?\n(Test markets will not affect global balance)")
    type_str = "(TEST)" if is_test else ""

    new_name = simpledialog.askstring("New Market", "Enter unique market name:")
    if not new_name or not new_name.strip(): update_status_bar("Add Market cancelled or name empty."); return
    new_name = new_name.strip()

    if data_manager.market_exists(new_name):
        messagebox.showerror("Error", f"Market '{new_name}' already exists."); update_status_bar(f"Add Market Error: '{new_name}' exists."); return

    if data_manager.add_new_market(new_name, is_test):
        selected_market_name = new_name
        update_market_selector()
        market_data = data_manager.get_market_data(new_name)
        prefix = "[TEST] " if market_data and market_data.get('is_test_market') else ""
        display_name = prefix + new_name
        if selected_market_var: selected_market_var.set(display_name)

        switch_market_callback(force_switch=True)
        logging.info(f"Market '{new_name}' {type_str} added.")
        messagebox.showinfo("Market Added", f"Market '{new_name}' {type_str} added.")
        update_status_bar(f"Market '{new_name}' {type_str} added.")
        if entry_yes_bid: entry_yes_bid.focus_set() # Focus on YES Bid
    else:
        messagebox.showerror("Error", f"Failed to add market '{new_name}' (internal error)."); update_status_bar(f"Add Market Error: Failed to add '{new_name}'.")
    update_widget_states()


def delete_selected_market_callback():
    """Callback to delete the selected market."""
    global selected_market_name, selected_market_var, market_selector_combo
    global entry_yes_bid, entry_yes_ask, entry_no_bid, entry_no_ask, entry_adv # Added bid/ask entries
    market_display_name_to_delete = selected_market_var.get()
    if not market_display_name_to_delete: messagebox.showerror("Error", "No market selected for deletion."); update_status_bar("Delete Market Error: None selected."); return

    market_name_to_delete = market_display_name_to_delete.replace("[TEST] ", "")

    if not data_manager.market_exists(market_name_to_delete):
        messagebox.showerror("Error", f"Selected market '{market_name_to_delete}' internal data not found."); update_status_bar(f"Delete Market Error: '{market_name_to_delete}' not found internally."); update_market_selector(); return

    confirm_msg = f"Permanently delete market '{market_display_name_to_delete}' and ALL its data?\n\nTHIS CANNOT BE UNDONE."
    if messagebox.askyesno("Confirm Delete", confirm_msg, icon='warning', default=messagebox.NO):
        update_status_bar(f"Deleting market '{market_name_to_delete}'...")
        if data_manager.delete_market(market_name_to_delete):
            logging.info(f"Market '{market_name_to_delete}' deleted via callback.")
            original_selection = selected_market_name
            if selected_market_name == market_name_to_delete: selected_market_name = None

            update_market_selector()

            new_selection_display = selected_market_var.get() if selected_market_var else ""
            if new_selection_display:
                new_internal_name = new_selection_display.replace("[TEST] ", "")
                if original_selection == market_name_to_delete or new_internal_name != original_selection:
                     selected_market_name = new_internal_name
                     switch_market_callback(force_switch=True)
                else: selected_market_name = new_internal_name
            else:
                selected_market_name = None; clear_all_displays(); clear_recommendation_cache()
                for entry in [entry_yes_bid, entry_yes_ask, entry_no_bid, entry_no_ask, entry_adv]: # Clear all price entries
                     if entry: entry.delete(0, tk.END)

            messagebox.showinfo("Market Deleted", f"Market '{market_display_name_to_delete}' deleted.")
            update_status_bar(f"Market '{market_display_name_to_delete}' deleted.")
        else:
            messagebox.showerror("Error", f"Failed to delete market '{market_name_to_delete}' (internal error)."); update_status_bar(f"Delete Market Error: Failed to delete '{market_name_to_delete}'.")
    else: update_status_bar("Market deletion cancelled.")
    update_widget_states()


def resolve_market_callback():
    """Callback to resolve the currently selected market."""
    global selected_market_name
    if not selected_market_name: messagebox.showerror("Error", "No market selected to resolve."); update_status_bar("Resolve Error: No market selected."); return

    market_data = data_manager.get_market_data(selected_market_name)
    if not market_data: messagebox.showerror("Internal Error", f"Cannot find data for market '{selected_market_name}'."); update_status_bar(f"Resolve Error: Data not found for {selected_market_name}."); return
    if market_data.get('position_state', 'FLAT') == 'FLAT': messagebox.showinfo("Info", f"Market '{selected_market_name}' is already FLAT."); update_status_bar(f"Resolve Info: Market '{selected_market_name}' already flat."); return

    dialog = tk.Toplevel(root); dialog.title(f"Resolve Market: {selected_market_name}"); dialog.geometry("300x150"); dialog.transient(root); dialog.grab_set()
    ttk.Label(dialog, text=f"Which side won market '{selected_market_name}'?", wraplength=280).pack(pady=10)
    result = tk.StringVar(); button_frame = ttk.Frame(dialog); button_frame.pack(pady=10)
    def set_result(side): result.set(side); dialog.destroy()
    ttk.Button(button_frame, text="YES Wins", command=lambda: set_result("YES")).pack(side=tk.LEFT, padx=10)
    ttk.Button(button_frame, text="NO Wins", command=lambda: set_result("NO")).pack(side=tk.LEFT, padx=10)
    ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=10)
    root.wait_window(dialog)

    winning_side = result.get()
    if not winning_side: update_status_bar("Market resolution cancelled."); return

    update_status_bar(f"Calculating resolution for '{selected_market_name}' ({winning_side} Wins)...")

    stats = data_manager.calculate_position_stats(selected_market_name)
    win_shares = stats.get(f'{winning_side.lower()}_shares', 0.0); win_basis = stats.get(f'{winning_side.lower()}_investment', 0.0)
    cash_inflow = win_shares * 1.0; market_pnl = cash_inflow - win_basis
    is_test = market_data.get('is_test_market', False); test_str = " (TEST)" if is_test else ""

    confirm_msg = f"Resolve market '{selected_market_name}'{test_str} as {winning_side} Wins?\n\n" \
                  f"- Winning Shares ({winning_side}): {utils.format_shares(win_shares)}\n" \
                  f"- Cash Inflow (@ $1.00): {utils.format_currency(cash_inflow)}\n" \
                  f"- Cost Basis ({winning_side}): {utils.format_currency(win_basis)}\n" \
                  f"----------------------------------\n" \
                  f"- Realized P/L (Market): {utils.format_cash_flow(market_pnl)}\n\n" \
                  f"This will clear ALL holdings for this market and {'update Global Balance/P&L.' if not is_test else 'NOT update Global Balance/P&L.'}\n\nProceed?"

    if messagebox.askyesno("Confirm Market Resolution", confirm_msg, icon='question'):
        update_status_bar(f"Resolving market '{selected_market_name}' as {winning_side}...")
        success, message = data_manager.resolve_market(selected_market_name, winning_side)
        if success:
            logging.info(f"Market '{selected_market_name}' resolved successfully as {winning_side}.")
            messagebox.showinfo("Resolution Complete", f"Market '{selected_market_name}' resolved.\n{message}")
            update_status_bar(f"Market '{selected_market_name}' resolved ({winning_side} Wins).")
            update_all_displays_for_market(selected_market_name); update_balance_display()
            clear_calculation_results(clear_projections=True); clear_recommendation_cache()
        else:
            logging.error(f"Market resolution failed for '{selected_market_name}': {message}")
            messagebox.showerror("Resolution Error", f"Failed to resolve market '{selected_market_name}':\n{message}")
            update_status_bar(f"Resolution FAILED for '{selected_market_name}'.")
            update_all_displays_for_market(selected_market_name); update_balance_display()
    else: update_status_bar("Market resolution cancelled by user.")
    update_widget_states()


def switch_market_callback(event=None, force_switch=False):
    """Handles switching the active market in the UI."""
    global selected_market_name, selected_market_var, default_widget_bg
    global entry_yes_bid, entry_yes_ask, entry_no_bid, entry_no_ask, entry_adv # Bid/Ask entries
    global holdings_frame, summary_frame

    new_selected_display_name = selected_market_var.get() if selected_market_var else ""
    new_selected_internal_name = new_selected_display_name.replace("[TEST] ", "")

    if not force_switch and new_selected_internal_name == selected_market_name: return

    if not new_selected_display_name or not data_manager.market_exists(new_selected_internal_name):
        if selected_market_name:
            logging.info("Clearing displays due to invalid/empty market selection.")
            update_status_bar("Market selection cleared or invalid.")
            selected_market_name = None; clear_all_displays(); clear_recommendation_cache()
            for entry in [entry_yes_bid, entry_yes_ask, entry_no_bid, entry_no_ask, entry_adv]: # Clear all price entries
                 if entry: entry.delete(0, tk.END)
            if holdings_frame: holdings_frame.config(background=default_widget_bg)
            if summary_frame: summary_frame.config(background=default_widget_bg)
        if market_selector_combo and not data_manager.market_exists(new_selected_internal_name):
             if selected_market_var: selected_market_var.set("")
        selected_market_name = None; update_widget_states(); return

    logging.info(f"Switching Market Display: {selected_market_name} -> {new_selected_internal_name}")
    update_status_bar(f"Switching to market: {new_selected_internal_name}...")
    selected_market_name = new_selected_internal_name

    market_data = data_manager.get_market_data(selected_market_name)
    if not market_data:
         messagebox.showerror("Error", f"Failed to load data for market '{selected_market_name}'.")
         update_status_bar(f"Error loading data for {selected_market_name}."); clear_all_displays(); clear_recommendation_cache()
         if holdings_frame: holdings_frame.config(background=default_widget_bg)
         if summary_frame: summary_frame.config(background=default_widget_bg)
         update_widget_states(); return

    is_test = market_data.get('is_test_market', False); target_bg = "alice blue" if is_test else default_widget_bg
    try:
        if holdings_frame: holdings_frame.config(background=target_bg)
        if summary_frame: summary_frame.config(background=target_bg)
    except tk.TclError: pass

    update_all_displays_for_market(selected_market_name)

    # Populate Bid/Ask and ADV entry fields
    entries_data = {
        entry_yes_bid: market_data.get("last_yes_bid", ""),
        entry_yes_ask: market_data.get("last_yes_ask", ""),
        entry_no_bid: market_data.get("last_no_bid", ""),
        entry_no_ask: market_data.get("last_no_ask", ""),
    }
    for entry, value in entries_data.items():
        if entry: entry.delete(0, tk.END); entry.insert(0, value)

    adv_value = market_data.get("adv", ""); adv_display_str = ""
    if adv_value == float('inf'): adv_display_str = "inf"
    elif isinstance(adv_value, (int, float)): adv_display_str = str(adv_value)
    else: adv_display_str = str(adv_value)
    if entry_adv: entry_adv.delete(0, tk.END); entry_adv.insert(0, adv_display_str)

    clear_calculation_results(clear_projections=False); clear_recommendation_cache()
    run_analysis_if_possible()


def run_analysis_if_possible():
    """Checks if prices are valid and runs analysis if a market is selected."""
    global selected_market_name
    global entry_yes_bid, entry_yes_ask, entry_no_bid, entry_no_ask # Bid/Ask entries
    if not selected_market_name: return

    # --- REMOVED FLAT CHECK START ---
    # market_data = data_manager.get_market_data(selected_market_name)
    # if market_data and market_data.get('position_state', 'FLAT') == 'FLAT':
    #      logging.debug(f"Analysis skipped for '{selected_market_name}': Market is FLAT.")
    #      clear_calculation_results(clear_projections=True); clear_recommendation_cache()
    #      update_status_bar("Analysis skipped: Market is FLAT."); return
    # --- REMOVED FLAT CHECK END ---

    # Validate all four price fields silently
    prices_valid = True
    price_entries = {
        'YES Bid': entry_yes_bid, 'YES Ask': entry_yes_ask,
        'NO Bid': entry_no_bid, 'NO Ask': entry_no_ask
    }
    for name, entry in price_entries.items():
        if entry:
            price_str = entry.get()
            if utils.validate_price(price_str) is None:
                prices_valid = False
                break # No need to check further

    if prices_valid:
        logging.debug(f"Prices valid, proceeding with analysis for '{selected_market_name}'...")
        calculate_strategy_callback()
    else:
        logging.debug("Analysis skipped: One or more prices not valid.")
        clear_calculation_results(clear_projections=True); clear_recommendation_cache()
        update_status_bar("Analysis skipped: Invalid price(s).")


def calculate_strategy_callback(event=None):
    """Callback to run the strategy analysis."""
    global selected_market_name, last_recommendation, analysis_text_widget, arbitrage_recommendation_text
    global entry_yes_bid, entry_yes_ask, entry_no_bid, entry_no_ask, entry_adv # Bid/Ask entries

    if not selected_market_name: messagebox.showerror("Error", "No market selected."); update_status_bar("Analysis Error: No market selected."); return

    # --- REMOVED FLAT CHECK START ---
    # market_data = data_manager.get_market_data(selected_market_name)
    # if market_data and market_data.get('position_state', 'FLAT') == 'FLAT':
    #      messagebox.showinfo("Info", f"Market '{selected_market_name}' is currently FLAT. No analysis needed.") # This message was misleading too
    #      update_status_bar("Analysis skipped: Market is FLAT."); clear_calculation_results(clear_projections=True); clear_recommendation_cache(); return
    # --- REMOVED FLAT CHECK END ---

    update_status_bar(f"Running strategy analysis for '{selected_market_name}'...")

    # Reset potential error highlighting
    price_entries = [entry_yes_bid, entry_yes_ask, entry_no_bid, entry_no_ask, entry_adv]
    for entry in price_entries: reset_entry_style(entry)

    # Get and validate inputs
    yes_b_str = entry_yes_bid.get(); yes_a_str = entry_yes_ask.get()
    no_b_str = entry_no_bid.get(); no_a_str = entry_no_ask.get()
    adv_str = entry_adv.get()

    yes_b = utils.validate_price(yes_b_str); yes_a = utils.validate_price(yes_a_str)
    no_b = utils.validate_price(no_b_str); no_a = utils.validate_price(no_a_str)
    market_adv = utils.validate_adv(adv_str)

    errors = []
    price_validation = {'YES Bid': (yes_b, entry_yes_bid, "last_yes_bid"), 'YES Ask': (yes_a, entry_yes_ask, "last_yes_ask"),
                        'NO Bid': (no_b, entry_no_bid, "last_no_bid"), 'NO Ask': (no_a, entry_no_ask, "last_no_ask")}
    for name, (price_val, entry_widget, data_key) in price_validation.items():
        if price_val is None:
            errors.append(f"Invalid {name}.")
            if entry_widget: entry_widget.config(background='pink')
            data_manager.update_market_property(selected_market_name, data_key, "") # Clear invalid stored price

    if market_adv is None:
        errors.append("Invalid Market ADV.")
        if entry_adv: entry_adv.config(background='pink')
        if adv_str.strip() != "" and adv_str.strip().lower() != "inf":
             data_manager.update_market_property(selected_market_name, "adv", "")
             if adv_var: adv_var.set("")

    # Spread Check (Ask must be >= Bid)
    if yes_b is not None and yes_a is not None and yes_a < yes_b - config.ZERO_PRICE_THRESHOLD:
        errors.append("YES Ask cannot be lower than YES Bid.")
        if entry_yes_ask: entry_yes_ask.config(background='lightcoral')
        if entry_yes_bid: entry_yes_bid.config(background='lightcoral')
    if no_b is not None and no_a is not None and no_a < no_b - config.ZERO_PRICE_THRESHOLD:
        errors.append("NO Ask cannot be lower than NO Bid.")
        if entry_no_ask: entry_no_ask.config(background='lightcoral')
        if entry_no_bid: entry_no_bid.config(background='lightcoral')


    if errors:
        error_msg = "Input Error(s):\n- " + "\n- ".join(errors) + "\nCannot analyze."
        messagebox.showerror("Input Error", error_msg); clear_calculation_results(clear_projections=True); clear_recommendation_cache()
        update_status_bar("Analysis Error: Invalid input(s)."); update_widget_states(); return

    # --- Input valid, proceed ---
    data_manager.update_market_property(selected_market_name, "last_yes_bid", yes_b_str)
    data_manager.update_market_property(selected_market_name, "last_yes_ask", yes_a_str)
    data_manager.update_market_property(selected_market_name, "last_no_bid", no_b_str)
    data_manager.update_market_property(selected_market_name, "last_no_ask", no_a_str)
    if market_adv == float('inf'): data_manager.update_market_property(selected_market_name, "adv", 'inf')
    elif market_adv == "": data_manager.update_market_property(selected_market_name, "adv", "")
    else: data_manager.update_market_property(selected_market_name, "adv", market_adv)


    stats = data_manager.calculate_position_stats(selected_market_name)
    current_balance = data_manager.get_global_balance()
    market_data = data_manager.get_market_data(selected_market_name) # Reload after potential updates
    if not market_data:
         messagebox.showerror("Error", f"Could not retrieve data for market '{selected_market_name}'.")
         update_status_bar(f"Analysis Error: Failed to get data for {selected_market_name}."); return

    try:
        logging.info(f"Calculating strategy for {selected_market_name}...")
        recommendation, analysis_details = strategy_engine.calculate_strategy_recommendation(
            market_name=selected_market_name, yes_bid=yes_b, yes_ask=yes_a, no_bid=no_b, no_ask=no_a, # Pass all prices
            market_adv=market_adv, market_data=market_data, stats=stats,
            current_balance=current_balance
        )
        logging.info(f"Recommendation for {selected_market_name}: {recommendation.get('action_type')}, Side: {recommendation.get('side')}, Shares: {recommendation.get('shares_rounded')}")

        last_recommendation = recommendation; update_recommendation_display()
        utils.set_text_widget_content(analysis_text_widget, "\n".join(analysis_details))
        update_summary_display(selected_market_name, clear_projections_on_update=False) # Update with projections
        update_state_display(selected_market_name); update_stop_loss_display(selected_market_name)

        arb_rec_line = next((line for line in analysis_details if line.strip().startswith("ARB (")), None)
        if arb_rec_line:
            tag = "blue" if "Buy Opp" in arb_rec_line else "orange" if "Sell" in arb_rec_line else None
            utils.set_text_widget_content(arbitrage_recommendation_text, arb_rec_line.strip(), tag)
        else:
             arb_check_line = next((line for line in analysis_details if "Market Arb Check" in line), None)
             if arb_check_line: utils.set_text_widget_content(arbitrage_recommendation_text, "Market Arb: No action needed/possible.")
             else: utils.set_text_widget_content(arbitrage_recommendation_text, "---")


        update_status_bar("Strategy analysis complete.")
        if recommendation.get("action_type") in ['BUY', 'SELL', 'BUY_ARB', 'SELL_ARB', 'SELL_STOP']:
            if execute_button: execute_button.focus_set()

    except Exception as e:
        logging.error(f"Error during analysis: {e}", exc_info=True)
        messagebox.showerror("Strategy Engine Error", f"Error during analysis:\n{e}\n\nCheck logs for details.")
        clear_calculation_results(clear_projections=True); clear_recommendation_cache()
        update_status_bar("Analysis Error: Strategy engine failed."); update_widget_states()


def execute_action_callback():
    """Callback to execute the cached recommended action."""
    global last_recommendation, selected_market_name
    global entry_yes_bid, entry_yes_ask, entry_no_bid, entry_no_ask # Need current prices

    rec = last_recommendation
    market_name = rec.get("market")
    action_type = rec.get("action_type")
    side = rec.get("side") # Get side early for price checks

    if not selected_market_name or market_name != selected_market_name:
        messagebox.showwarning("Execution Warning", "Recommendation may be outdated or for a different market.\nPlease re-run analysis before executing."); update_status_bar("Execute Warning: Recommendation outdated."); return
    if action_type not in ['BUY', 'SELL', 'BUY_ARB', 'SELL_ARB', 'SELL_STOP']:
        messagebox.showinfo("Execution Info", f"No actionable recommendation ({action_type}) to execute."); update_status_bar(f"Execute Info: No action ({action_type})."); return

    # --- Get CURRENT Bid/Ask prices for execution ---
    current_yes_bid = utils.validate_price(entry_yes_bid.get()); current_yes_ask = utils.validate_price(entry_yes_ask.get())
    current_no_bid = utils.validate_price(entry_no_bid.get()); current_no_ask = utils.validate_price(entry_no_ask.get())

    # Validate essential prices needed for the specific action
    prices_ok = True
    if action_type in ['BUY', 'BUY_ARB']: # Need Ask(s)
        if side == 'YES' and current_yes_ask is None: prices_ok = False
        if side == 'NO' and current_no_ask is None: prices_ok = False
        if side == 'PAIR' and (current_yes_ask is None or current_no_ask is None): prices_ok = False
    if action_type in ['SELL', 'SELL_ARB']: # Need Bid(s)
        if side == 'YES' and current_yes_bid is None: prices_ok = False
        if side == 'NO' and current_no_bid is None: prices_ok = False
        if side == 'PAIR' and (current_yes_bid is None or current_no_bid is None): prices_ok = False
    if action_type == 'SELL_STOP': # Need Bid(s) for the side being stopped
        if side == 'ALL_YES' and current_yes_bid is None: current_yes_bid = 0.0 # Assume 0 if crashed
        if side == 'ALL_NO' and current_no_bid is None: current_no_bid = 0.0 # Assume 0 if crashed
        if side == 'ALL_PAIRS':
            if current_yes_bid is None: current_yes_bid = 0.0
            if current_no_bid is None: current_no_bid = 0.0
        # Price check is implicitly ok because we assigned 0 if None for stops
        prices_ok = True

    if not prices_ok:
         messagebox.showerror("Execution Error", f"Cannot execute {action_type} {side}: Invalid current market price(s) needed.\nPlease refresh prices and re-analyze."); update_status_bar(f"Execute Error: Invalid current price(s) for {action_type} {side}."); return

    # Determine Execution Price based on action and side
    shares_rec_rounded = rec.get("shares_rounded", 0.0)
    exec_price_yes = 0.0; exec_price_no = 0.0; exec_price_display = 0.0 # Used for confirmation dialog

    if action_type == 'BUY':
        exec_price_yes = current_yes_ask if side == 'YES' else 0.0
        exec_price_no = current_no_ask if side == 'NO' else 0.0
        exec_price_display = exec_price_yes if side == 'YES' else exec_price_no
    elif action_type == 'SELL':
        exec_price_yes = current_yes_bid if side == 'YES' else 0.0
        exec_price_no = current_no_bid if side == 'NO' else 0.0
        exec_price_display = exec_price_yes if side == 'YES' else exec_price_no
    elif action_type == 'BUY_ARB':
        exec_price_yes = current_yes_ask; exec_price_no = current_no_ask
        exec_price_display = exec_price_yes + exec_price_no
    elif action_type == 'SELL_ARB':
        exec_price_yes = current_yes_bid; exec_price_no = current_no_bid
        exec_price_display = exec_price_yes + exec_price_no
    elif action_type == 'SELL_STOP':
        exec_price_yes = current_yes_bid if side in ['ALL_YES', 'ALL_PAIRS'] else 0.0
        exec_price_no = current_no_bid if side in ['ALL_NO', 'ALL_PAIRS'] else 0.0
        # Display price depends on what's being sold
        if side == 'ALL_YES': exec_price_display = exec_price_yes
        elif side == 'ALL_NO': exec_price_display = exec_price_no
        elif side == 'ALL_PAIRS': exec_price_display = exec_price_yes + exec_price_no

    rule = rec.get('rule_triggered', 'N/A')
    calculated_stop_level = rec.get('calculated_stop_level')
    is_stop_action = action_type == 'SELL_STOP'

    # Shares check: Ensure > 0 for non-stop actions
    if not is_stop_action and shares_rec_rounded <= config.ZERO_SHARE_THRESHOLD:
         messagebox.showerror("Execution Error", f"Invalid shares: {utils.format_shares(shares_rec_rounded)}."); update_status_bar("Execute Error: Zero shares."); clear_recommendation_cache(); return

    market_data = data_manager.get_market_data(market_name)
    if not market_data: messagebox.showerror("Internal Error", f"Market data missing for '{market_name}'."); update_status_bar(f"Execute Error: Missing data for {market_name}."); return
    current_stats = data_manager.calculate_position_stats(market_name)
    current_state = data_manager.get_position_state(market_name)
    current_balance = data_manager.get_global_balance()
    is_test_market = market_data.get('is_test_market', False)

    # 2. Confirmation Dialog
    test_str = " (TEST)" if is_test_market else ""
    confirm_title = f"Confirm Execution{test_str}"; confirm_msg = f"MARKET: '{market_name}'{test_str}\nRULE: {rule}\nSTATE: {current_state}\n==============================\nPROPOSED ACTION:\n\n"
    can_proceed = True

    # --- Build Confirmation Messages using Current Prices ---
    if action_type == 'BUY':
        required_cost = shares_rec_rounded * exec_price_display
        if not is_test_market and required_cost > current_balance + config.ZERO_PRICE_THRESHOLD:
             messagebox.showerror("Balance Error", f"Insufficient Balance: {utils.format_currency(current_balance)}\nRequired (@ Ask ~{utils.format_price(exec_price_display)}): {utils.format_currency(required_cost)}")
             update_status_bar("Execute Error: Insufficient balance for BUY."); can_proceed = False
        else:
            stop_msg = f"~{utils.format_price(calculated_stop_level)}" if calculated_stop_level is not None else "N/A"
            confirm_msg += f"TYPE: BUY\nSIDE: {side}\nSHARES: {utils.format_shares(shares_rec_rounded)}\nPRICE (Ask): {utils.format_price(exec_price_display)}\nEST. COST: {utils.format_currency(required_cost)}\nEST. STOP LVL: {stop_msg}\n\nExecute?"
    elif action_type == 'SELL':
        available_shares = current_stats['yes_shares'] if side == 'YES' else current_stats['no_shares']
        if shares_rec_rounded > available_shares + config.ZERO_SHARE_THRESHOLD:
            messagebox.showerror("Execution Error", f"Need: {utils.format_shares(shares_rec_rounded)} {side}\nAvailable: {utils.format_shares(available_shares)}")
            update_status_bar("Execute Error: Insufficient shares for SELL."); can_proceed = False
        else:
            proceeds_est = shares_rec_rounded * exec_price_display
            if not is_test_market and proceeds_est < 0.01 and not messagebox.askyesno("Low Proceeds Warning", f"Est. proceeds (@ Bid ~{utils.format_price(exec_price_display)}) are low ({utils.format_currency(proceeds_est)}).\nContinue?"): can_proceed = False
            if can_proceed: confirm_msg += f"TYPE: SELL\nSIDE: {side}\nSHARES: {utils.format_shares(shares_rec_rounded)}\nPRICE (Bid): {utils.format_price(exec_price_display)}\nEST. PROCEEDS: {utils.format_currency(proceeds_est)}\n\nExecute?"
    elif action_type == 'BUY_ARB':
         required_cost = shares_rec_rounded * exec_price_display # exec_price_display = yes_ask + no_ask
         if not is_test_market and required_cost > current_balance + config.ZERO_PRICE_THRESHOLD:
             messagebox.showerror("Balance Error", f"Insufficient Balance: {utils.format_currency(current_balance)}\nPair Cost (@ Asks ~{utils.format_price(exec_price_display)}): {utils.format_currency(required_cost)}")
             update_status_bar("Execute Error: Insufficient balance for ARB BUY."); can_proceed = False
         else:
             est_profit = shares_rec_rounded * abs(1.0 - exec_price_display)
             confirm_msg += f"TYPE: BUY ARBITRAGE PAIR\nPAIRS: {utils.format_shares(shares_rec_rounded)}\n(Current Asks: Y~{utils.format_price(exec_price_yes)}, N~{utils.format_price(exec_price_no)}, Sum~{utils.format_price(exec_price_display)})\nPAIR COST (Y+N): {utils.format_currency(required_cost)}\nPOTENTIAL PROFIT: {utils.format_currency(est_profit)}\n\nExecute?"
    elif action_type == 'SELL_ARB':
          available_yes, available_no = current_stats['yes_shares'], current_stats['no_shares']
          if shares_rec_rounded > available_yes + config.ZERO_SHARE_THRESHOLD or shares_rec_rounded > available_no + config.ZERO_SHARE_THRESHOLD:
               messagebox.showerror("Execution Error", f"Need: {utils.format_shares(shares_rec_rounded)} pairs\nHave: YES {utils.format_shares(available_yes)}, NO {utils.format_shares(available_no)}")
               update_status_bar("Execute Error: Insufficient shares for ARB SELL."); can_proceed = False
          else:
              proceeds_est = shares_rec_rounded * exec_price_display # exec_price_display = yes_bid + no_bid
              avg_yes_p, avg_no_p = current_stats['yes_avg_price'], current_stats['no_avg_price']; pl_text = ""
              if avg_yes_p > config.ZERO_PRICE_THRESHOLD and avg_no_p > config.ZERO_PRICE_THRESHOLD:
                  basis_cost = shares_rec_rounded * (avg_yes_p + avg_no_p); est_realized_pl = proceeds_est - basis_cost; pl_text = f"EST. REALIZED P/L: {utils.format_currency(est_realized_pl)}\n"
              confirm_msg += f"TYPE: SELL ARBITRAGE PAIR\nPAIRS: {utils.format_shares(shares_rec_rounded)}\n(Current Bids Sum: ~{utils.format_price(exec_price_display)})\nEST. PROCEEDS: {utils.format_currency(proceeds_est)}\n{pl_text}\nExecute?"
    elif action_type == 'SELL_STOP':
         trigger_val = rec.get('price') # Get trigger value from original rec (which is Bid sum or single Bid)
         confirm_msg += f"TYPE: SELL (STOP LOSS TRIGGERED!)\n"
         stop_shares = 0.0 # Initialize shares to be sold by the stop

         if side == 'ALL_YES':
             stop_shares = current_stats['yes_shares']; proceeds_est = stop_shares * exec_price_yes
             if stop_shares > config.ZERO_SHARE_THRESHOLD:
                 confirm_msg += f"-> Sell ALL YES: {utils.format_shares(stop_shares)} @ Bid ~{utils.format_price(exec_price_yes)}\n-> Est Proceeds: {utils.format_currency(proceeds_est)}\n"
                 if isinstance(trigger_val, (int, float)) and config.ZERO_PRICE_THRESHOLD < trigger_val < 1.0: confirm_msg += f"-> Trigger: YES Bid <= {utils.format_price(trigger_val)}\n"
             else: confirm_msg += "-> No YES shares to sell.\n"; can_proceed = False # Cannot proceed if no shares
         elif side == 'ALL_NO':
             stop_shares = current_stats['no_shares']; proceeds_est = stop_shares * exec_price_no
             if stop_shares > config.ZERO_SHARE_THRESHOLD:
                 confirm_msg += f"-> Sell ALL NO: {utils.format_shares(stop_shares)} @ Bid ~{utils.format_price(exec_price_no)}\n-> Est Proceeds: {utils.format_currency(proceeds_est)}\n"
                 if isinstance(trigger_val, (int, float)) and config.ZERO_PRICE_THRESHOLD < trigger_val < 1.0: confirm_msg += f"-> Trigger: NO Bid <= {utils.format_price(trigger_val)}\n"
             else: confirm_msg += "-> No NO shares to sell.\n"; can_proceed = False # Cannot proceed if no shares
         elif side == 'ALL_PAIRS':
             stop_shares = min(current_stats['yes_shares'], current_stats['no_shares']) # Shares per side (number of pairs)
             proceeds_est = stop_shares * exec_price_display # exec_price_display = yes_bid + no_bid
             if stop_shares > config.ZERO_SHARE_THRESHOLD:
                 confirm_msg += f"-> Sell Pairs: {utils.format_shares(stop_shares)} @ Bids Sum ~{utils.format_price(exec_price_display)}\n   (YES {utils.format_shares(stop_shares)} @ ~{utils.format_price(exec_price_yes)} + NO {utils.format_shares(stop_shares)} @ ~{utils.format_price(exec_price_no)})\n-> Est Total Proceeds: {utils.format_currency(proceeds_est)}\n"
                 # Trigger value for HEDGED stop is % loss (negative float)
                 hedged_trigger_pct = rec.get('price') # price stores loss_pct_basis for this type
                 if isinstance(hedged_trigger_pct, (int, float)) and hedged_trigger_pct < 0: confirm_msg += f"-> Trigger: Basis Loss % ~ {utils.format_percent(hedged_trigger_pct)}\n"
             else: confirm_msg += "-> No PAIRS to sell.\n"; can_proceed = False # Cannot proceed if no shares
         else: confirm_msg += f"-> Unknown Stop Side: {side}\n"; can_proceed = False

         if can_proceed: confirm_msg += "\nExecute this STOP SALE?"
         else: update_status_bar("Execute Error: Stop triggered but no shares to sell.")

    else: messagebox.showerror("Internal Error", f"Unknown action type '{action_type}'"); update_status_bar(f"Execute Error: Unknown action type {action_type}."); can_proceed = False

    if not can_proceed: update_status_bar("Execution halted before confirmation."); return
    if not messagebox.askyesno(confirm_title, confirm_msg): logging.info("Execution cancelled by user."); update_status_bar("Execution cancelled by user."); return

    # 3. Execute Action using Data Manager and CURRENT prices
    logging.info(f"Executing {action_type} for {market_name}..."); update_status_bar(f"Executing {action_type} for '{market_name}'...")
    execution_successful = True; error_details = ""

    try:
        if action_type == 'BUY':
            cost_final = shares_rec_rounded * exec_price_display; cash_flow = -abs(cost_final)
            if data_manager.add_transaction(market_name, action_type, side, shares_rec_rounded, exec_price_display, cash_flow):
                if not data_manager.add_bet(market_name, side, shares_rec_rounded, exec_price_display):
                     error_details = "Logged Tx but failed adding Holding."; logging.critical(f"CRITICAL: {error_details} {market_name}!"); execution_successful = False
                else:
                     if calculated_stop_level is not None:
                          if data_manager.set_directional_stop(market_name, calculated_stop_level): logging.info(f"Stop set to {utils.format_price(calculated_stop_level)}.")
                          else: logging.error(f"Failed setting stop for {market_name}.")
                          data_manager.set_last_entry_side(market_name, side)
                     else: data_manager.set_directional_stop(market_name, None) # Ensure stop is clear if not calculated
            else: error_details = "Could not log Tx."; logging.error(f"Exec failed {market_name}: {error_details}"); execution_successful = False

        elif action_type == 'SELL':
            proceeds_final = shares_rec_rounded * exec_price_display; cash_flow = abs(proceeds_final)
            shortfall, cost_basis_sold = data_manager.fifo_reduce_holdings(market_name, side, shares_rec_rounded)
            actual_shares_sold = shares_rec_rounded - shortfall
            if shortfall > config.ZERO_SHARE_THRESHOLD: error_details += f" FIFO Shortfall: {utils.format_shares(shortfall)} {side}."; logging.warning(error_details)
            if actual_shares_sold > config.ZERO_SHARE_THRESHOLD:
                actual_cash_flow = cash_flow * (actual_shares_sold / shares_rec_rounded if shares_rec_rounded > config.ZERO_SHARE_THRESHOLD else 1) # Avoid ZeroDivisionError
                if data_manager.add_transaction(market_name, action_type, side, actual_shares_sold, exec_price_display, actual_cash_flow):
                    if not is_test_market and cost_basis_sold is not None:
                       realized_pl = actual_cash_flow - cost_basis_sold; data_manager.update_realized_pl(realized_pl)
                       logging.info(f"Realized P/L on SELL {market_name}: {utils.format_currency(realized_pl)}")
                else: error_details += " FIFO done but failed log Tx."; logging.critical(f"CRITICAL: {error_details} {market_name}!"); execution_successful = False
            elif execution_successful: logging.warning(f"SELL resulted in 0 shares sold (FIFO) {market_name}."); error_details += " 0 shares sold (FIFO)."

        elif action_type == 'BUY_ARB':
            pairs_final = shares_rec_rounded; cost_yes = pairs_final * exec_price_yes; cost_no = pairs_final * exec_price_no
            log_ok_y = data_manager.add_transaction(market_name, "BUY", "YES", pairs_final, exec_price_yes, -cost_yes)
            if log_ok_y:
                 log_ok_n = data_manager.add_transaction(market_name, "BUY", "NO", pairs_final, exec_price_no, -cost_no)
                 if log_ok_n:
                      bet_ok_y = data_manager.add_bet(market_name, 'YES', pairs_final, exec_price_yes); bet_ok_n = data_manager.add_bet(market_name, 'NO', pairs_final, exec_price_no)
                      if not bet_ok_y or not bet_ok_n: error_details = "Arb Txs logged but failed adding Holdings."; logging.critical(f"CRITICAL: {error_details} {market_name}!"); execution_successful = False
                 else: error_details = "Could not log ARB BUY NO Tx."; logging.error(f"Exec failed {market_name}: {error_details}"); execution_successful = False
            else: error_details = "Could not log ARB BUY YES Tx."; logging.error(f"Exec failed {market_name}: {error_details}"); execution_successful = False

        elif action_type == 'SELL_ARB':
            pairs_final = shares_rec_rounded
            shortfall_y, cost_basis_y = data_manager.fifo_reduce_holdings(market_name, 'YES', pairs_final)
            shortfall_n, cost_basis_n = data_manager.fifo_reduce_holdings(market_name, 'NO', pairs_final)
            if shortfall_y > config.ZERO_SHARE_THRESHOLD or shortfall_n > config.ZERO_SHARE_THRESHOLD:
                 error_details = f"SELL ARB FIFO shortfall. Y:{utils.format_shares(shortfall_y)}, N:{utils.format_shares(shortfall_n)}."; logging.warning(f"{error_details} {market_name}. Halted."); execution_successful = False
            else:
                 proceeds_yes = pairs_final * exec_price_yes; proceeds_no = pairs_final * exec_price_no
                 log_ok_y = data_manager.add_transaction(market_name, "SELL", "YES", pairs_final, exec_price_yes, proceeds_yes)
                 if log_ok_y:
                      log_ok_n = data_manager.add_transaction(market_name, "SELL", "NO", pairs_final, exec_price_no, proceeds_no)
                      if log_ok_n:
                           if not is_test_market and cost_basis_y is not None and cost_basis_n is not None:
                              realized_pl = (proceeds_yes + proceeds_no) - (cost_basis_y + cost_basis_n); data_manager.update_realized_pl(realized_pl)
                              logging.info(f"Realized P/L on SELL ARB {market_name}: {utils.format_currency(realized_pl)}")
                      else: error_details = "SELL ARB reduced (Y ok) but failed log NO Tx."; logging.critical(f"CRITICAL: {error_details} {market_name}!"); execution_successful = False
                 else: error_details = "SELL ARB reduced but failed log YES Tx."; logging.critical(f"CRITICAL: {error_details} {market_name}!"); execution_successful = False

        elif action_type == 'SELL_STOP':
            sides_to_sell = []
            # Get current shares AGAIN right before FIFO, as state might be slightly off if rapid events
            latest_stats = data_manager.calculate_position_stats(market_name)
            if side == 'ALL_YES': shares = latest_stats['yes_shares']; price = exec_price_yes
            elif side == 'ALL_NO': shares = latest_stats['no_shares']; price = exec_price_no
            elif side == 'ALL_PAIRS': pairs = min(latest_stats['yes_shares'], latest_stats['no_shares']); price_y = exec_price_yes; price_n = exec_price_no

            if side == 'ALL_YES' and shares > config.ZERO_SHARE_THRESHOLD: sides_to_sell.append(('YES', shares, price))
            elif side == 'ALL_NO' and shares > config.ZERO_SHARE_THRESHOLD: sides_to_sell.append(('NO', shares, price))
            elif side == 'ALL_PAIRS' and pairs > config.ZERO_SHARE_THRESHOLD: sides_to_sell.extend([('YES', pairs, price_y), ('NO', pairs, price_n)])

            total_shortfall = 0; total_proceeds = 0; total_cost_basis = 0; stop_log_errors = 0
            for side_code, shares_unrounded, sell_price in sides_to_sell:
                shares_final = config.SHARE_ROUNDING_FUNC(shares_unrounded)
                if shares_final <= config.ZERO_SHARE_THRESHOLD: continue
                shortfall, cost_basis_sold = data_manager.fifo_reduce_holdings(market_name, side_code, shares_final)
                actual_shares = shares_final - shortfall; total_shortfall += shortfall
                if actual_shares > config.ZERO_SHARE_THRESHOLD:
                     actual_proceeds = actual_shares * sell_price; total_proceeds += actual_proceeds
                     if cost_basis_sold is not None: total_cost_basis += cost_basis_sold
                     if not data_manager.add_transaction(market_name, "SELL (STOP)", side_code, actual_shares, sell_price, actual_proceeds):
                         stop_log_errors += 1; error_details += f" Failed log STOP {side_code} Tx."; logging.critical(f"CRITICAL: Stop FIFO done but failed log {side_code} Tx {market_name}!"); execution_successful = False

            if stop_log_errors > 0: error_details = f"STOP SELL failed log {stop_log_errors} Tx(s)."
            elif total_shortfall > config.ZERO_SHARE_THRESHOLD: error_details += f" STOP SELL shortfall: {utils.format_shares(total_shortfall)}."; logging.warning(error_details)

            # Update P/L only if execution was successful overall AND not a test market
            if execution_successful and not is_test_market and total_cost_basis is not None:
                 realized_pl = total_proceeds - total_cost_basis; data_manager.update_realized_pl(realized_pl)
                 logging.info(f"Realized P/L on STOP LOSS ({side}) for {market_name}: {utils.format_currency(realized_pl)}")

            # Clear directional stop info regardless of partial success (stop was triggered)
            data_manager.set_directional_stop(market_name, None)
            data_manager.set_last_entry_side(market_name, None)

        # --- Post-Execution (Common) ---
        if execution_successful:
            success_msg = f"{action_type} action completed successfully for '{market_name}'."
            if error_details: success_msg += f" Warnings: {error_details}"
            logging.info(success_msg)
            update_status_bar(f"{action_type} successful for '{market_name}'.")
            messagebox.showinfo("Executed", success_msg)
        else:
            fail_msg = f"{action_type} action FAILED for '{market_name}'. Reason: {error_details}"
            logging.error(fail_msg)
            update_status_bar(f"{action_type} FAILED for '{market_name}'. Check logs.")
            messagebox.showerror("Execution Error", f"{fail_msg}\n\nData might be inconsistent. Check holdings, log, and balance carefully.")

        # Update state, UI, clear recommendation AFTER execution attempt
        final_state = data_manager.update_position_state(market_name)
        logging.info(f"State for {market_name} after execution attempt: {final_state}")
        update_all_displays_for_market(market_name)
        update_balance_display()
        clear_calculation_results(clear_projections=True)
        clear_recommendation_cache()

    except Exception as e:
        logging.error(f"Unexpected error during {action_type} execution: {e}", exc_info=True)
        messagebox.showerror("Critical Execution Error", f"Unexpected error during {action_type} execution:\n{e}\n\nCHECK DATA INTEGRITY (Holdings/Log/Balance).")
        update_status_bar(f"CRITICAL ERROR during {action_type}. Check logs & data!")
        try:
             # Attempt to refresh state and UI even after error
             data_manager.update_position_state(market_name)
             update_all_displays_for_market(market_name)
             update_balance_display()
        except Exception as update_e: logging.error(f"Error during post-error UI update: {update_e}", exc_info=True)
        clear_recommendation_cache() # Clear potentially bad recommendation


def remove_selected_bet_callback():
    """Callback for removing a holding entry manually (mistake correction)."""
    global selected_market_name, bets_added_tree
    if not selected_market_name: messagebox.showerror("Error", "No market selected."); return
    if not bets_added_tree: return

    selected_items = bets_added_tree.selection()
    if not selected_items: messagebox.showwarning("Selection Error", "No holding entries selected."); update_status_bar("Remove Holding: No selection."); return

    item_details = []
    for item_id in selected_items:
        try: values = bets_added_tree.item(item_id, 'values'); item_details.append(f"- {values[0]} {values[1]} @ ~{values[3]}")
        except (tk.TclError, IndexError): item_details.append(f"- ID: {item_id} (details unavailable)")

    confirm_msg = f"Manually remove {len(selected_items)} selected holding entr{'y' if len(selected_items) == 1 else 'ies'} for '{selected_market_name}'?\n\n" + \
                  "\n".join(item_details) + "\n\n** MISTAKE CORRECTION ONLY. **\n- Does NOT adjust balance or log.\n- Recalculate strategy afterwards.\n\nProceed?"
    if messagebox.askyesno("Confirm Manual Removal", confirm_msg, icon='warning', default=messagebox.NO):
        update_status_bar(f"Removing {len(selected_items)} selected holdings (manual)...")
        removed_count = 0; success = True; errors = []
        for item_id in selected_items:
             if not isinstance(item_id, str): logging.warning(f"Skipping non-string item ID: {item_id}"); continue
             if not data_manager.remove_bet_by_id(selected_market_name, item_id): success = False; errors.append(item_id[:8]+"..."); logging.warning(f"Failed remove bet ID {item_id}")
             else: removed_count += 1
        logging.info(f"Manual removal bets {selected_market_name}. Succeeded: {removed_count}")

        if removed_count > 0:
            update_holdings_display(selected_market_name); update_summary_display(selected_market_name, True); update_state_display(selected_market_name); update_stop_loss_display(selected_market_name)
            clear_calculation_results(clear_projections=False); clear_recommendation_cache()
            msg = f"Removed {removed_count} holding entr{'y' if removed_count == 1 else 'ies'}. "
            if not success: msg += f"Failed {len(errors)} entr{'y' if len(errors)==1 else 'ies'}. "
            msg += "Re-analyze recommended."; messagebox.showinfo("Removal Complete", msg); update_status_bar(f"Removed {removed_count} holdings manually.")
        elif not success: messagebox.showerror("Error", f"Failed remove selected entr{'y' if len(errors)==1 else 'ies'}. Check logs."); update_status_bar("Manual removal failed.")
        else: messagebox.showinfo("Not Found", "Selected entries not found."); update_status_bar("Manual removal: Selected items not found.")
        update_widget_states()

def clear_all_bets_callback():
    """Callback to clear all holdings for the market (mistake correction)."""
    global selected_market_name
    if not selected_market_name: messagebox.showerror("Error", "No market selected."); return

    holdings_count = len(bets_added_tree.get_children()) if bets_added_tree else 0
    if holdings_count == 0: messagebox.showinfo("Info", "No holdings exist."); update_status_bar("Clear Holdings: None exist."); return

    confirm_msg_1 = f"Clear ALL ({holdings_count}) holding entries for '{selected_market_name}'?\n\n** MISTAKE CORRECTION ONLY. **\n- Does NOT adjust balance or log.\n- State reset to FLAT.\n- Recalculate strategy afterwards.\n\nProceed?"
    if messagebox.askyesno("Confirm Clear All Holdings", confirm_msg_1, icon='warning', default=messagebox.NO):
        confirm_msg_2 = f"ARE YOU ABSOLUTELY SURE?\n\nIrreversibly delete all holding records for '{selected_market_name}'?"
        if messagebox.askyesno("Final Confirmation", confirm_msg_2, icon='error', default=messagebox.NO):
            update_status_bar(f"Clearing ALL holdings for '{selected_market_name}' (manual)...")
            cleared_count = data_manager.clear_all_bets(selected_market_name)
            logging.info(f"Cleared {cleared_count} bets for {selected_market_name}.")
            if cleared_count >= 0:
                 update_holdings_display(selected_market_name); update_summary_display(selected_market_name, True); update_state_display(selected_market_name); update_stop_loss_display(selected_market_name)
                 clear_calculation_results(clear_projections=False); clear_recommendation_cache()
                 messagebox.showinfo("Cleared", f"Cleared {cleared_count} holdings. State FLAT. Re-analyze."); update_status_bar(f"Cleared {cleared_count} holdings manually.")
            else: messagebox.showerror("Error", f"Failed clear holdings for '{selected_market_name}'."); update_status_bar(f"Error clearing holdings.")
            update_widget_states()
        else: update_status_bar("Clear all holdings cancelled (final).")
    else: update_status_bar("Clear all holdings cancelled.")


def clear_transaction_log_callback():
     """Callback to clear the transaction log."""
     global selected_market_name, transaction_log_tree
     if not selected_market_name: messagebox.showerror("Error", "No market selected."); return

     log_count = len(transaction_log_tree.get_children()) if transaction_log_tree else 0
     if log_count == 0: messagebox.showinfo("Empty", "Transaction log already empty."); update_status_bar("Clear Log: Log empty."); return

     confirm_msg = f"Clear transaction log ({log_count} entries) for '{selected_market_name}'?\n\n** CANNOT be undone. **\n- Does NOT change balance/holdings.\n\nProceed?"
     if messagebox.askyesno("Confirm Clear Log", confirm_msg, icon='warning', default=messagebox.NO):
         update_status_bar(f"Clearing transaction log for '{selected_market_name}'...")
         cleared_count = data_manager.clear_transaction_log(selected_market_name)
         logging.info(f"Cleared transaction log ({cleared_count} entries) for {selected_market_name}.")
         if cleared_count >= 0:
             update_transaction_log_display(selected_market_name); messagebox.showinfo("Log Cleared", f"Log cleared ({cleared_count} entries)."); update_status_bar(f"Log cleared for '{selected_market_name}'.")
         else: messagebox.showerror("Error", f"Failed clear log for '{selected_market_name}'."); update_status_bar(f"Error clearing log.")
     else: update_status_bar("Clear log cancelled.")


def delete_selected_log_entry_callback():
    """Callback to delete a single selected transaction log entry (mistake correction)."""
    global selected_market_name, transaction_log_tree
    if not selected_market_name: messagebox.showerror("Error", "No market selected."); update_status_bar("Delete Log Error: No market."); return
    if not transaction_log_tree: return

    selected_items = transaction_log_tree.selection()
    if not selected_items: messagebox.showwarning("Selection Error", "No transaction log entry selected."); update_status_bar("Delete Log: No selection."); return
    if len(selected_items) > 1: messagebox.showwarning("Selection Error", "Select only ONE entry to delete."); update_status_bar("Delete Log: Multiple selected."); return

    item_iid = selected_items[0] # This should be the timestamp string based on update_transaction_log_display
    try:
        item_values = transaction_log_tree.item(item_iid, 'values'); timestamp_str = item_values[0]
        entry_details = f"- {item_values[0]} | {item_values[1]} | {item_values[2]} | {item_values[3]} sh @ {item_values[4]} | CF: {item_values[5]}"
    except (tk.TclError, IndexError): messagebox.showerror("Error", "Could not retrieve details of selected log entry."); update_status_bar("Delete Log Error: Cannot get details."); return

    confirm_msg = f"Manually delete this log entry for '{selected_market_name}'?\n\n{entry_details}\n\n** MISTAKE CORRECTION ONLY. **\n** DOES NOT adjust Balance, Holdings, or P/L. **\n** Data WILL likely become INCONSISTENT! **\n\nProceed?"
    if messagebox.askyesno("Confirm Manual Log Deletion", confirm_msg, icon='error', default=messagebox.NO):
        update_status_bar(f"Deleting log entry {timestamp_str} (manual)...")
        deleted = data_manager.delete_transaction_by_timestamp(selected_market_name, timestamp_str)
        if deleted:
            logging.warning(f"Manually deleted log entry {timestamp_str}. Data may be inconsistent.")
            update_transaction_log_display(selected_market_name); messagebox.showinfo("Deletion Complete", "Log entry deleted.\nBalance/Holdings NOT adjusted."); update_status_bar(f"Deleted log entry {timestamp_str} manually.")
        else: messagebox.showerror("Error", f"Failed delete log entry {timestamp_str}. Check logs."); update_status_bar(f"Manual log deletion FAILED for {timestamp_str}."); update_transaction_log_display(selected_market_name)
    else: update_status_bar("Manual log deletion cancelled.")


def add_manual_holding_callback():
    """Callback to manually add a holding entry and corresponding transaction."""
    global selected_market_name
    if not selected_market_name: messagebox.showerror("Error", "No market selected."); return

    market_data = data_manager.get_market_data(selected_market_name); is_test = market_data.get('is_test_market', False); test_str = " (TEST)" if is_test else ""
    update_status_bar(f"Adding manual holding for '{selected_market_name}'{test_str}...")

    try:
        side = simpledialog.askstring("Manual Holding - Step 1/3", "Enter Side (YES or NO):", parent=root)
        if not side or side.upper() not in ['YES', 'NO']: update_status_bar("Manual add cancelled (side)."); return
        side = side.upper()

        shares_str = simpledialog.askstring("Manual Holding - Step 2/3", f"Enter Shares ({config.SHARE_DECIMALS} decimals):", parent=root)
        shares_val = utils.validate_shares(shares_str)
        if shares_val is None or shares_val <= config.ZERO_SHARE_THRESHOLD: messagebox.showerror("Input Error", "Invalid/zero shares."); update_status_bar("Manual add error: Invalid shares."); return
        shares_final = config.SHARE_ROUNDING_FUNC(shares_val)

        price_str = simpledialog.askstring("Manual Holding - Step 3/3", "Enter Price (0-1):", parent=root)
        price_val = utils.validate_price(price_str)
        if price_val is None: messagebox.showerror("Input Error", "Invalid price."); update_status_bar("Manual add error: Invalid price."); return

        cost = shares_final * price_val; cash_flow = -abs(cost)

        confirm_msg = f"Manually add to '{selected_market_name}'{test_str}?\n\nSIDE: {side}\nSHARES: {utils.format_shares(shares_final)}\nPRICE: {utils.format_price(price_val)}\nCOST (CF): {utils.format_cash_flow(cash_flow)}\n\n" \
                      f"** WILL add transaction record **\n** {'WILL adjust Global Balance.' if not is_test else 'WILL NOT adjust Global Balance (Test Market).' } **\n\nUse to record external trades/correct errors."
        if not messagebox.askyesno("Confirm Manual Add", confirm_msg, parent=root): update_status_bar("Manual add cancelled (confirm)."); return

        update_status_bar(f"Executing manual add for '{selected_market_name}'...")
        tx_success = data_manager.add_transaction(selected_market_name, "MANUAL_ADD", side, shares_final, price_val, cash_flow, is_manual=True)
        if tx_success:
             bet_success = data_manager.add_bet(selected_market_name, side, shares_final, price_val)
             if bet_success:
                  data_manager.update_position_state(selected_market_name)
                  update_all_displays_for_market(selected_market_name); update_balance_display(); clear_calculation_results(); clear_recommendation_cache()
                  logging.info(f"Manual holding added {selected_market_name}: {shares_final} {side} @ {price_val}")
                  messagebox.showinfo("Success", "Manual holding/transaction added."); update_status_bar(f"Manual holding added for '{selected_market_name}'.")
             else:
                  logging.critical(f"CRITICAL: Manual Tx logged, but failed add bet {selected_market_name}!")
                  messagebox.showerror("CRITICAL Error", "Tx logged, but FAILED add bet holding. Data INCONSISTENT!"); update_status_bar("CRITICAL ERROR: Manual add failed - data inconsistent!")
                  update_all_displays_for_market(selected_market_name); update_balance_display()
        else: messagebox.showerror("Error", "Failed log manual transaction."); update_status_bar("Manual add error: Failed log Tx.")

    except Exception as e: logging.error(f"Error during manual add: {e}", exc_info=True); messagebox.showerror("Error", f"Unexpected error during manual add: {e}"); update_status_bar("Manual add error: Unexpected.")
    finally: update_widget_states()


def on_closing():
    """Handle the window closing event."""
    logging.info("Closing application...")
    update_status_bar("Saving data and closing...")
    if data_manager.save_data():
        logging.info("Data saved successfully."); update_status_bar("Data saved. Exiting."); root.destroy()
    else:
        logging.error("!!! CRITICAL: Error saving data on closing.")
        update_status_bar("Error saving data!")
        if messagebox.askyesno("Save Error", "CRITICAL ERROR: Failed to save data!\nLatest changes may be lost.\n\nQuit anyway?", icon='error', default=messagebox.NO): root.destroy()
        else: update_status_bar("Save error acknowledged. Close aborted."); return


def on_treeview_select(event):
    """Callback function when a Treeview selection changes."""
    if event.widget == transaction_log_tree: update_widget_states()


# --- GUI Construction ---
def build_gui(root_window):
    """Creates and lays out all the Tkinter widgets."""
    global root, balance_var, initial_balance_var, selected_market_var, adv_var
    global summary_yes_shares_var, summary_no_shares_var, summary_yes_inv_var, summary_no_inv_var
    global summary_total_inv_var, current_pl_yes_wins_var, current_pl_no_wins_var, current_unrealized_pl_var
    global position_state_var, stop_loss_info_var, market_arb_var, recommendation_var, total_realized_pl_var
    global yes_bid_var, yes_ask_var, no_bid_var, no_ask_var # New price vars
    global balance_label, entry_initial_balance, btn_set_balance, pw, market_selector_combo
    global add_market_button, delete_market_button, left_pane, holdings_frame, bets_added_tree
    global bets_scrollbar, holdings_buttons_frame, remove_mistake_button, clear_mistakes_button, add_manual_button
    global right_pane, log_frame, transaction_log_tree, log_scrollbar, clear_log_button
    global summary_analysis_frame, summary_frame, summary_yes_shares_label, summary_no_shares_label
    global summary_yes_inv_label, summary_no_inv_label, summary_total_inv_label, current_pl_yes_wins_label
    global current_pl_no_wins_label, current_unrealized_pl_label, position_state_label, stop_loss_info_label
    global calc_frame, calc_input_frame
    global entry_yes_bid, entry_yes_ask, entry_no_bid, entry_no_ask, entry_adv # New price entries
    global calculate_button, rec_exec_frame, recommendation_label, execute_button, analysis_results_frame
    global market_arb_label, arbitrage_recommendation_text, analysis_text_container, analysis_text_widget
    global analysis_text_scrollbar, disclaimer_label, total_realized_pl_label
    global status_bar_var, status_bar_label, default_widget_bg
    global resolve_market_button, delete_log_button

    root = root_window
    root.title(f"Trading Strategy Engine v{config.VERSION} (Bid/Ask)"); root.geometry("1000x1000")

    default_widget_bg = root.cget('bg');
    try: test_frame = ttk.Frame(root); default_widget_bg = test_frame.cget('background'); test_frame.destroy();
    except: pass

    # --- StringVars ---
    balance_var=tk.StringVar(); initial_balance_var=tk.StringVar(); selected_market_var=tk.StringVar(); adv_var=tk.StringVar();
    summary_yes_shares_var=tk.StringVar(); summary_no_shares_var=tk.StringVar(); summary_yes_inv_var=tk.StringVar(); summary_no_inv_var=tk.StringVar();
    summary_total_inv_var=tk.StringVar(); current_pl_yes_wins_var=tk.StringVar(); current_pl_no_wins_var=tk.StringVar(); current_unrealized_pl_var=tk.StringVar();
    position_state_var = tk.StringVar(); stop_loss_info_var = tk.StringVar(); market_arb_var=tk.StringVar(); recommendation_var=tk.StringVar();
    total_realized_pl_var = tk.StringVar(); status_bar_var = tk.StringVar(value="Initializing...")
    # New Price StringVars
    yes_bid_var=tk.StringVar(); yes_ask_var=tk.StringVar(); no_bid_var=tk.StringVar(); no_ask_var=tk.StringVar();

    # --- Styles & Fonts ---
    style=ttk.Style(root); style.theme_use('clam')
    default_font=tkFont.nametofont("TkDefaultFont"); bold_font=tkFont.Font(family=default_font.cget("family"),size=default_font.cget("size"),weight='bold'); italic_font=tkFont.Font(family=default_font.cget("family"),size=default_font.cget("size"),slant='italic'); small_font=tkFont.Font(family=default_font.cget("family"),size=max(8,default_font.cget("size")-2)); small_italic_font=tkFont.Font(family=small_font.cget("family"),size=small_font.cget("size"),slant='italic'); disclaimer_font=tkFont.Font(family=default_font.cget("family"),size=max(7,default_font.cget("size")-3),slant='italic'); recommendation_font=tkFont.Font(family=default_font.cget("family"),size=default_font.cget("size")+1,weight='bold'); state_font = tkFont.Font(family=default_font.cget("family"), size=default_font.cget("size"), weight='bold'); status_font = tkFont.Font(family=default_font.cget("family"), size=max(8, default_font.cget("size")-2))

    # --- Top Frame: Market Selection & Balance ---
    top_frame=ttk.Frame(root,padding="10"); top_frame.pack(side=tk.TOP, fill="x",padx=5,pady=5)
    market_frame=ttk.Frame(top_frame); market_frame.pack(side=tk.LEFT,padx=5)
    ttk.Label(market_frame,text="Market:").pack(side=tk.LEFT)
    market_selector_combo=ttk.Combobox(market_frame,textvariable=selected_market_var,width=35,state="readonly"); market_selector_combo.pack(side=tk.LEFT,padx=5); market_selector_combo.bind("<<ComboboxSelected>>", switch_market_callback); Tooltip(market_selector_combo, "Select active market. '[TEST]' markets don't affect Global Balance.")
    add_market_button=ttk.Button(market_frame,text="Add",command=add_new_market_callback,width=5); add_market_button.pack(side=tk.LEFT,padx=(0,2)); Tooltip(add_market_button, "Add new market definition.")
    delete_market_button=ttk.Button(market_frame,text="Del",command=delete_selected_market_callback,width=5); delete_market_button.pack(side=tk.LEFT,padx=(0,2)); Tooltip(delete_market_button, "Delete selected market and ALL data. CANNOT BE UNDONE.")
    resolve_market_button = ttk.Button(market_frame, text="Resolve...", command=resolve_market_callback, width=8); resolve_market_button.pack(side=tk.LEFT, padx=(5, 5)); Tooltip(resolve_market_button, "Resolve selected market (declare winner).\nClears holdings, logs Tx, updates P/L & Bal (unless TEST).")

    balance_pl_frame=ttk.Frame(top_frame); balance_pl_frame.pack(side=tk.RIGHT,padx=5)
    balance_frame=ttk.Frame(balance_pl_frame); balance_frame.pack(side=tk.TOP, anchor='e')
    ttk.Label(balance_frame,text="Set Init Bal $:").pack(side=tk.LEFT,padx=(10, 5))
    entry_initial_balance=ttk.Entry(balance_frame,textvariable=initial_balance_var,width=10); entry_initial_balance.pack(side=tk.LEFT,padx=5); entry_initial_balance.bind("<Return>", set_initial_balance_callback); Tooltip(entry_initial_balance, "Enter starting global balance. Press Enter or 'Set'.")
    btn_set_balance=ttk.Button(balance_frame,text="Set",command=set_initial_balance_callback,width=5); btn_set_balance.pack(side=tk.LEFT,padx=5); Tooltip(btn_set_balance, "Set initial global balance.")
    balance_label=ttk.Label(balance_frame,textvariable=balance_var,font=tkFont.Font(weight='bold', size=11)); balance_label.pack(side=tk.LEFT,padx=10); Tooltip(balance_label, "Current Global Balance (non-TEST markets).")
    total_realized_pl_label = ttk.Label(balance_pl_frame, textvariable=total_realized_pl_var, font=tkFont.Font(weight='normal', size=10)); total_realized_pl_label.pack(side=tk.TOP, anchor='e', padx=10, pady=(2,0)); Tooltip(total_realized_pl_label, "Total realized P/L across all closed positions (non-TEST).")

    # --- Paned Window Layout ---
    pw=ttk.PanedWindow(root,orient=tk.HORIZONTAL); pw.pack(fill=tk.BOTH,expand=True,padx=10,pady=(0,5))

    # --- Left Pane: Holdings ---
    left_pane=ttk.Frame(pw,padding="5",width=480); left_pane.pack_propagate(False); pw.add(left_pane,weight=1)
    holdings_frame=ttk.LabelFrame(left_pane,text="Holdings (Current Market)",padding="10"); holdings_frame.pack(pady=5,fill="both",expand=True)
    bets_cols=('side','shares','cost','price'); bets_added_tree=ttk.Treeview(holdings_frame,columns=bets_cols,show='headings',selectmode='extended',height=15); bets_added_tree.heading('side',text='Side'); bets_added_tree.heading('shares',text='Rem. Shares'); bets_added_tree.heading('cost',text='Rem. Cost $'); bets_added_tree.heading('price',text='Orig. Price'); bets_added_tree.column('side',width=50,anchor=tk.CENTER); bets_added_tree.column('shares',width=100,anchor=tk.E); bets_added_tree.column('cost',width=100,anchor=tk.E); bets_added_tree.column('price',width=85,anchor=tk.E); bets_scrollbar=ttk.Scrollbar(holdings_frame,orient=tk.VERTICAL,command=bets_added_tree.yview); bets_added_tree.configure(yscroll=bets_scrollbar.set); bets_added_tree.grid(row=0,column=0,sticky='nsew'); bets_scrollbar.grid(row=0,column=1,sticky='ns'); holdings_frame.grid_rowconfigure(0,weight=1); holdings_frame.grid_columnconfigure(0,weight=1); Tooltip(bets_added_tree, "Current open positions.\nRem. Cost = Remaining cost basis.\nOrig. Price = Acquisition price.")
    holdings_buttons_frame=ttk.Frame(holdings_frame); holdings_buttons_frame.grid(row=1,column=0,columnspan=2,sticky='ew',pady=(10,0))
    add_manual_button = ttk.Button(holdings_buttons_frame, text="Add Manual...", command=add_manual_holding_callback); add_manual_button.pack(side=tk.LEFT,padx=5); Tooltip(add_manual_button, "Manually add holding & transaction.\nAdjusts balance (unless TEST).")
    remove_mistake_button=ttk.Button(holdings_buttons_frame,text="Remove Sel. (Mistake)",command=remove_selected_bet_callback); remove_mistake_button.pack(side=tk.LEFT,padx=5); Tooltip(remove_mistake_button, "Remove selected holding entries ONLY.\nDoes NOT change balance/log.\nUse for data correction.")
    clear_mistakes_button=ttk.Button(holdings_buttons_frame,text="Clear All (Mistake)",command=clear_all_bets_callback); clear_mistakes_button.pack(side=tk.LEFT,padx=5); Tooltip(clear_mistakes_button, "Remove ALL holding entries ONLY.\nDoes NOT change balance/log.\nUse ONLY if ALL holdings data is wrong.")

    # --- Right Pane: Log, Summary, Analysis ---
    right_pane=ttk.Frame(pw,padding="5"); pw.add(right_pane,weight=2)

    # Transaction Log (Top Right)
    log_frame=ttk.LabelFrame(right_pane,text="Transaction Log (Current Market)",padding="10"); log_frame.pack(pady=(5,5), fill="both", expand=True, side=tk.TOP, anchor='n'); log_frame.rowconfigure(0, weight=1); log_frame.columnconfigure(0, weight=1)
    log_cols=('ts','type','side','shares','price','cash_flow','balance'); transaction_log_tree=ttk.Treeview(log_frame,columns=log_cols,show='headings',height=12, selectmode='browse'); transaction_log_tree.heading('ts',text='Timestamp'); transaction_log_tree.heading('type',text='Type'); transaction_log_tree.heading('side',text='Side'); transaction_log_tree.heading('shares',text='Shares'); transaction_log_tree.heading('price',text='Price'); transaction_log_tree.heading('cash_flow',text='Cash Flow'); transaction_log_tree.heading('balance',text='Global Bal.'); transaction_log_tree.column('ts',width=135); transaction_log_tree.column('type',width=80,anchor=tk.W); transaction_log_tree.column('side',width=45,anchor=tk.CENTER); transaction_log_tree.column('shares',width=90,anchor=tk.E); transaction_log_tree.column('price',width=75,anchor=tk.E); transaction_log_tree.column('cash_flow',width=90,anchor=tk.E); transaction_log_tree.column('balance',width=100,anchor=tk.E); log_scrollbar=ttk.Scrollbar(log_frame,orient=tk.VERTICAL,command=transaction_log_tree.yview); transaction_log_tree.configure(yscroll=log_scrollbar.set); transaction_log_tree.grid(row=0,column=0,sticky='nsew'); log_scrollbar.grid(row=0,column=1,sticky='ns'); Tooltip(transaction_log_tree, "Transaction history.\nCash Flow: +inflows (SELLS), -outflows (BUYS).\nGlobal Bal.: Balance AFTER Tx (non-TEST)."); transaction_log_tree.bind('<<TreeviewSelect>>', on_treeview_select)
    log_buttons_frame = ttk.Frame(log_frame); log_buttons_frame.grid(row=1,column=0,columnspan=2,sticky='e',pady=(5,0))
    delete_log_button = ttk.Button(log_buttons_frame, text="Delete Sel. Log (Mistake)", command=delete_selected_log_entry_callback); delete_log_button.pack(side=tk.LEFT, padx=5); Tooltip(delete_log_button, "Delete selected log entry ONLY.\n!! DOES NOT CHANGE BALANCE/HOLDINGS !!\nUse ONLY for correcting purely erroneous log entries.\nData inconsistency likely if used incorrectly.")
    clear_log_button=ttk.Button(log_buttons_frame,text="Clear Entire Log",command=clear_transaction_log_callback); clear_log_button.pack(side=tk.LEFT, padx=5); Tooltip(clear_log_button, "Clear all entries from log.\nDoes NOT affect holdings/balance.")

    # Summary & Analysis Area (Bottom Right)
    summary_analysis_frame = ttk.Frame(right_pane); summary_analysis_frame.pack(pady=5, fill="both", expand=True, side=tk.BOTTOM, anchor='s'); summary_analysis_frame.rowconfigure(1, weight=1); summary_analysis_frame.columnconfigure(0, weight=1)
    summary_frame=ttk.LabelFrame(summary_analysis_frame,text="Position Summary & Projections",padding="10"); summary_frame.grid(row=0, column=0, pady=0, sticky="ew")
    s_row1=ttk.Frame(summary_frame); s_row1.pack(fill='x', pady=(0, 2)); summary_yes_shares_label = ttk.Label(s_row1,textvariable=summary_yes_shares_var, font=bold_font); summary_yes_shares_label.pack(side=tk.LEFT,anchor=tk.W,padx=5); Tooltip(summary_yes_shares_label, "Total YES shares held."); summary_no_shares_label = ttk.Label(s_row1,textvariable=summary_no_shares_var, font=bold_font); summary_no_shares_label.pack(side=tk.LEFT,anchor=tk.W,padx=25); Tooltip(summary_no_shares_label, "Total NO shares held.")
    s_row2=ttk.Frame(summary_frame); s_row2.pack(fill='x', pady=(0, 5)); summary_yes_inv_label = ttk.Label(s_row2,textvariable=summary_yes_inv_var,font=small_font); summary_yes_inv_label.pack(side=tk.LEFT,anchor=tk.W,padx=5); Tooltip(summary_yes_inv_label, "YES cost basis & avg price."); summary_no_inv_label = ttk.Label(s_row2,textvariable=summary_no_inv_var,font=small_font); summary_no_inv_label.pack(side=tk.LEFT,anchor=tk.W,padx=25); Tooltip(summary_no_inv_label, "NO cost basis & avg price.")
    ttk.Separator(summary_frame,orient='horizontal').pack(fill='x',pady=5)
    s_row3=ttk.Frame(summary_frame); s_row3.pack(fill='x',pady=(5,5)); summary_total_inv_label = ttk.Label(s_row3,textvariable=summary_total_inv_var,font=tkFont.Font(weight='bold', size=small_font.cget('size'))); summary_total_inv_label.pack(side=tk.LEFT,anchor=tk.W,padx=5); Tooltip(summary_total_inv_label, "Total combined cost basis."); position_state_label = ttk.Label(s_row3, textvariable=position_state_var, font=state_font); position_state_label.pack(side=tk.LEFT, anchor=tk.W, padx=25); Tooltip(position_state_label, "Current position state."); stop_loss_info_label = ttk.Label(s_row3, textvariable=stop_loss_info_var, font=italic_font); stop_loss_info_label.pack(side=tk.LEFT, anchor=tk.W, padx=25); Tooltip(stop_loss_info_label, "Active stop-loss info.")
    ttk.Separator(summary_frame,orient='horizontal').pack(fill='x',pady=5)
    s_row4=ttk.Frame(summary_frame); s_row4.pack(fill='x', pady=(5, 2)); current_pl_yes_wins_label=ttk.Label(s_row4,textvariable=current_pl_yes_wins_var,font=default_font); current_pl_yes_wins_label.pack(side=tk.LEFT,anchor=tk.W,padx=5); Tooltip(current_pl_yes_wins_label, "Projected Global Balance if YES wins."); current_pl_no_wins_label=ttk.Label(s_row4,textvariable=current_pl_no_wins_var,font=default_font); current_pl_no_wins_label.pack(side=tk.LEFT,anchor=tk.W,padx=25); Tooltip(current_pl_no_wins_label, "Projected Global Balance if NO wins.")
    s_row5=ttk.Frame(summary_frame); s_row5.pack(fill='x',pady=(2,5)); current_unrealized_pl_label=ttk.Label(s_row5,textvariable=current_unrealized_pl_var,font=italic_font, wraplength=500); current_unrealized_pl_label.pack(side=tk.LEFT,anchor=tk.W,padx=5); Tooltip(current_unrealized_pl_label, "Est Market Value (using BIDs) & Unrealized P/L.")

    # Strategy Engine & Analysis Frame
    calc_frame=ttk.LabelFrame(summary_analysis_frame,text="Strategy Engine & Analysis",padding="10"); calc_frame.grid(row=1, column=0, pady=(10,0), sticky="nsew"); calc_frame.rowconfigure(3, weight=1); calc_frame.columnconfigure(0, weight=1)
    # Input Row (Bid/Ask)
    calc_input_frame=ttk.Frame(calc_frame); calc_input_frame.grid(row=0, column=0, sticky="ew", pady=5);
    ttk.Label(calc_input_frame,text="YES Bid: (SELL)").grid(row=0,column=0,padx=5,pady=3,sticky=tk.W); entry_yes_bid=ttk.Entry(calc_input_frame,width=8, textvariable=yes_bid_var); entry_yes_bid.grid(row=0,column=1,padx=5,pady=3,sticky=tk.W); entry_yes_bid.bind("<Return>", calculate_strategy_callback); Tooltip(entry_yes_bid, "YES Sell Price (Bid). Press Enter or Run Analysis. Bid for selling/checking stops")
    ttk.Label(calc_input_frame,text="YES Ask: (BUY)").grid(row=1,column=0,padx=5,pady=3,sticky=tk.W); entry_yes_ask=ttk.Entry(calc_input_frame,width=8, textvariable=yes_ask_var); entry_yes_ask.grid(row=1,column=1,padx=5,pady=3,sticky=tk.W); entry_yes_ask.bind("<Return>", calculate_strategy_callback); Tooltip(entry_yes_ask, "YES Buy Price (Ask). Press Enter or Run Analysis. Ask for buying")
    ttk.Label(calc_input_frame,text="NO Bid: (SELL)").grid(row=0,column=2,padx=(15,5),pady=3,sticky=tk.W); entry_no_bid=ttk.Entry(calc_input_frame,width=8, textvariable=no_bid_var); entry_no_bid.grid(row=0,column=3,padx=5,pady=3,sticky=tk.W); entry_no_bid.bind("<Return>", calculate_strategy_callback); Tooltip(entry_no_bid, "NO Sell Price (Bid). Press Enter or Run Analysis. Bid for selling/checking stops")
    ttk.Label(calc_input_frame,text="NO Ask: (BUY)").grid(row=1,column=2,padx=(15,5),pady=3,sticky=tk.W); entry_no_ask=ttk.Entry(calc_input_frame,width=8, textvariable=no_ask_var); entry_no_ask.grid(row=1,column=3,padx=5,pady=3,sticky=tk.W); entry_no_ask.bind("<Return>", calculate_strategy_callback); Tooltip(entry_no_ask, "NO Buy Price (Ask). Press Enter or Run Analysis. Ask for buying")
    ttk.Label(calc_input_frame,text="ADV $:").grid(row=0,column=4,padx=(15,5),pady=3,sticky=tk.W); entry_adv=ttk.Entry(calc_input_frame,width=10, textvariable=adv_var); entry_adv.grid(row=0,column=5,padx=5,pady=3,sticky=tk.W); entry_adv.bind("<Return>", calculate_strategy_callback); Tooltip(entry_adv, "Estimated Market ADV $ (numeric or 'inf').")
    calculate_button=ttk.Button(calc_input_frame,text="Run Analysis",command=calculate_strategy_callback); calculate_button.grid(row=1, column=4, columnspan=2, padx=(15,5), pady=5, sticky="w"); Tooltip(calculate_button, "Analyze position & market conditions.")

    # Recommendation & Execution Row
    rec_exec_frame = ttk.Frame(calc_frame); rec_exec_frame.grid(row=1, column=0, sticky="ew", pady=(5,10)); rec_exec_frame.columnconfigure(0, weight=1)
    recommendation_label = ttk.Label(rec_exec_frame, textvariable=recommendation_var, font=recommendation_font, anchor=tk.W, justify=tk.LEFT, wraplength=550); recommendation_label.grid(row=0, column=0, sticky="ew", padx=5); Tooltip(recommendation_label, "Recommended action from last analysis.")
    execute_button = ttk.Button(rec_exec_frame, text="Execute Action", command=execute_action_callback); execute_button.grid(row=0, column=1, padx=10, sticky="e"); Tooltip(execute_button, "Execute recommended action (uses CURRENT prices).")
    # Analysis Results Area
    analysis_results_frame=ttk.Frame(calc_frame, padding=(0, 5, 0, 0)); analysis_results_frame.grid(row=3, column=0, sticky='nsew', pady=(5,0)); analysis_results_frame.columnconfigure(0,weight=1); analysis_results_frame.rowconfigure(2, weight=1)
    market_arb_label=ttk.Label(analysis_results_frame,textvariable=market_arb_var,font=default_font); market_arb_label.grid(row=0,column=0,sticky='ew',pady=1,padx=5); Tooltip(market_arb_label, "Indicates Market Arb opportunities (Buy Ask Sum < $1 or Sell Bid Sum > $1).")
    arbitrage_recommendation_text=tk.Text(analysis_results_frame,height=2,wrap=tk.WORD,relief=tk.FLAT,font=small_italic_font,state='disabled',background=default_widget_bg); arbitrage_recommendation_text.grid(row=1,column=0,sticky='ew',pady=(0,5),padx=5); arbitrage_recommendation_text.tag_configure("bold",font=tkFont.Font(size=small_font.cget('size'), weight='bold')); arbitrage_recommendation_text.tag_configure("blue",foreground="blue"); arbitrage_recommendation_text.tag_configure("orange",foreground="orange"); Tooltip(arbitrage_recommendation_text, "Specific recommendation for market arbitrage.")
    analysis_text_container = ttk.LabelFrame(analysis_results_frame, text="Analysis & Rule Details"); analysis_text_container.grid(row=2, column=0, sticky='nsew', pady=(5, 5), padx=5); analysis_text_container.grid_rowconfigure(0, weight=1); analysis_text_container.grid_columnconfigure(0, weight=1); Tooltip(analysis_text_container, "Detailed log of rules and calculations.")
    analysis_text_widget=tk.Text(analysis_text_container, height=8, wrap=tk.WORD, relief=tk.FLAT, font=small_font, state='disabled', background=default_widget_bg); analysis_text_widget.grid(row=0, column=0, sticky='nsew')
    analysis_text_scrollbar = ttk.Scrollbar(analysis_text_container, orient=tk.VERTICAL, command=analysis_text_widget.yview); analysis_text_scrollbar.grid(row=0, column=1, sticky='ns'); analysis_text_widget.configure(yscrollcommand=analysis_text_scrollbar.set)
    disclaimer_label=ttk.Label(analysis_results_frame,text=f"Strategy v{config.VERSION}. NOT financial advice. Verify logic. Data saved on exit.",font=disclaimer_font,justify=tk.LEFT,foreground="gray"); disclaimer_label.grid(row=4,column=0,sticky='sw',pady=(5,0),padx=5)

    # --- Status Bar ---
    status_bar_label = ttk.Label(root, textvariable=status_bar_var, relief=tk.SUNKEN, anchor=tk.W, font=status_font, padding=(5, 2))
    status_bar_label.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=0)


# --- Application Initialization ---
def initialize_app():
    """Loads data and sets up the initial UI state."""
    global selected_market_name, initial_balance_var, market_selector_combo, selected_market_var

    utils.setup_logging(); logging.info(f"Initializing Trading Strategy Engine v{config.VERSION}...")
    update_status_bar("Loading data...")

    if not data_manager.load_data():
        logging.warning("Failed load. Starting fresh."); update_status_bar("Load failed. Starting fresh.")
        data_manager.set_global_balance(1000.0); messagebox.showinfo("Fresh Start", "No previous data/failed load.\nStarting fresh (Balance $1000).")
    else: update_status_bar("Data loaded successfully.")

    initial_balance_var.set(f"{data_manager.get_global_balance():.2f}"); update_balance_display()
    update_market_selector()

    market_names_display = market_selector_combo['values'] if market_selector_combo else []
    if market_names_display:
        first_display_name = market_names_display[0]; first_internal_name = first_display_name.replace("[TEST] ", "")
        if selected_market_var: selected_market_var.set(first_display_name)
        selected_market_name = first_internal_name
        switch_market_callback(force_switch=True)
    else: clear_all_displays(); clear_recommendation_cache(); update_status_bar("No markets defined. Use 'Add'.")

    update_widget_states(); logging.info("Application Ready.")
    if market_names_display: update_status_bar("Application Ready.")


# --- Main Execution ---
if __name__ == "__main__":
    main_root = tk.Tk()
    build_gui(main_root)
    initialize_app()
    main_root.protocol("WM_DELETE_WINDOW", on_closing)
    main_root.mainloop()