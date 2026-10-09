"""TabSage: a plan-and-act agent that answers questions over CSV tables with SQL.

Author: Rupa Srivalli
"""
from .agent import Agent, AgentResult
from .tools import Database

__all__ = ["Agent", "AgentResult", "Database"]
__version__ = "0.1.0"
