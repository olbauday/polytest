# utilsbacktest.py
import config

def format_currency(amount):
    try:
        return f"${amount:,.2f}"
    except (TypeError, ValueError):
        return str(amount)

def format_price(price):
    try:
        # Adjust decimals based on typical price range (e.g., more for < 1.0)
        if price is not None and abs(price) < 1.0:
             decimals = 4
        else:
             decimals = 2
        return f"{price:.{decimals}f}"
    except (TypeError, ValueError):
        return str(price)

def format_shares(shares):
    try:
        return f"{shares:.{config.SHARE_DECIMALS}f}"
    except (TypeError, ValueError):
        return str(shares)

def format_percent(value):
    try:
        return f"{value:.2%}"
    except (TypeError, ValueError):
        return str(value)

# Add any other utility functions your strategy_engine might implicitly rely on