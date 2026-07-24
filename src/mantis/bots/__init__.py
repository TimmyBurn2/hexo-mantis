"""mantis.bots — BotProtocol + in-repo bots + the ONE rung resolver (design §a.1).

Public API: `BotProtocol`, `RungUnresolvable`, `RandomBot`, `resolve_bot`. External bot
adapters (sealbot/kraken/strix) are WP12-R property; at HEAD `resolve_bot` raises
`RungUnresolvable` for all three (0/6 ladder-rung census verdict, DESIGN.md).
"""
from __future__ import annotations

from mantis.bots.protocol import BotProtocol, RungUnresolvable
from mantis.bots.random_bot import RandomBot
from mantis.bots.resolve import resolve_bot

__all__ = ["BotProtocol", "RandomBot", "RungUnresolvable", "resolve_bot"]
