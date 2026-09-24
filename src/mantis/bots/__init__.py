"""mantis.bots — BotProtocol, the in-repo random bot, the strix adapter and the ONE rung resolver."""
from __future__ import annotations

from mantis.bots.protocol import BotProtocol, RungUnresolvable
from mantis.bots.random_bot import RandomBot
from mantis.bots.resolve import resolve_bot

__all__ = ["BotProtocol", "RandomBot", "RungUnresolvable", "resolve_bot"]
