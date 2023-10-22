import sqlite3


def has_other_trader_escrow_items(discord_id: str, channel_id: str) -> bool:
    # Connect to the database
    conn = sqlite3.connect("trading_bot.db")
    cursor = conn.cursor()

    # Retrieve the ID of the trader with the given discord_id
    cursor.execute("SELECT id FROM traders WHERE discord_id=?", (discord_id,))
    trader_id = cursor.fetchone()

    # If trader does not exist, return False
    if not trader_id:
        conn.close()
        return False

    trader_id = trader_id[0] # get the actual ID value

    # Retrieve the trade data with the given channel ID
    cursor.execute("SELECT trader1_id, trader2_id FROM trades WHERE channel_id=?", (channel_id,))
    trade_data = cursor.fetchone()

    # If no trade exists for the given channel, return False
    if not trade_data:
        conn.close()
        return False

    trader1_id, trader2_id = trade_data

    # Identify the other trader
    other_trader_id = trader1_id if trader_id == trader2_id else trader2_id

    # Check for items with the status "in escrow" for the other trader
    cursor.execute("SELECT id FROM items WHERE trader_id=? AND status='in escrow'", (other_trader_id,))
    items = cursor.fetchall()

    # Close the connection to the database
    conn.close()

    # Return True if there are items with the status "in escrow" for the other trader, otherwise return False
    return len(items) > 0


def has_untraded_items(discord_id: str, channel_id: str) -> bool:
    # Connect to the database
    conn = sqlite3.connect("trading_bot.db")
    cursor = conn.cursor()

    # Retrieve the ID of the trader with the given discord_id
    cursor.execute("SELECT id FROM traders WHERE discord_id=?", (discord_id,))
    trader_id = cursor.fetchone()

    # If trader does not exist, return False
    if not trader_id:
        conn.close()
        return False

    trader_id = trader_id[0]  # get the actual ID value

    # Retrieve the trade ID with the given channel ID
    cursor.execute("SELECT id FROM trades WHERE channel_id=?", (channel_id,))
    trade_id = cursor.fetchone()

    # If no trade exists for the given channel, return False
    if not trade_id:
        conn.close()
        return False

    trade_id = trade_id[0]  # get the actual trade ID value

    # Check for items with the tag "not traded" for the trader
    cursor.execute(
        "SELECT id FROM items WHERE trade_id=? AND trader_id=? AND status='not traded'",
        (trade_id, trader_id),
    )
    items = cursor.fetchall()

    # Close the connection to the database
    conn.close()

    # Return True if there are untraded items, otherwise return False
    return len(items) > 0
