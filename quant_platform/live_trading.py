from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Order:
    symbol: str
    quantity: int
    side: str


class BrokerAdapter:
    def submit_order(self, order: Order) -> None:
        raise NotImplementedError("Implement broker adapter for live trading.")


class SimulationAdapter(BrokerAdapter):
    def submit_order(self, order: Order) -> None:
        print(f"Simulated order: {order}")
