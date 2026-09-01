"""Independent sell-only Paper risk-exit authority."""

from .contracts import PaperError, WRITER_ID, writer_registration
from .execution import calculate_fees, calculate_sell_shares, execute_sell
from .runtime import (
    account_register,
    account_status,
    risk_exit_preview,
    risk_exit_run,
    verify_account,
    writer_status,
)

__all__ = [
    "PaperError",
    "WRITER_ID",
    "account_register",
    "account_status",
    "calculate_fees",
    "calculate_sell_shares",
    "execute_sell",
    "risk_exit_preview",
    "risk_exit_run",
    "verify_account",
    "writer_registration",
    "writer_status",
]
