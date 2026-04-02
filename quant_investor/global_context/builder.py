from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_investor.branch_contracts import BranchResult, UnifiedDataBundle
from quant_investor.contracts import GlobalContext


class GlobalContextBuilder:
    """Build and cache a shared market context for the research DAG."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        schema_version: str = "global-context.v1",
    ) -> None:
        self.cache_dir = Path(cache_dir or Path(".cache") / "quant_investor" / "global_context")
        self.schema_version = schema_version

    def cache_path_for(self, market: str, latest_trade_date: str, universe_hash: str) -> Path:
        return self.cache_dir / market.upper() / latest_trade_date / f"{universe_hash}.json"

    def build(
        self,
        *,
        data_bundle: UnifiedDataBundle,
        phase1_context: dict[str, Any] | None = None,
        branch_results: dict[str, BranchResult] | None = None,
        calibrated_signals: dict[str, Any] | None = None,
        risk_result: Any | None = None,
        force_refresh: bool = False,
    ) -> GlobalContext:
        latest_trade_date = str(
            data_bundle.metadata.get("end_date")
            or data_bundle.metadata.get("latest_trade_date")
            or pd.Timestamp.now().normalize().strftime("%Y%m%d")
        )
        universe = list(data_bundle.symbols)
        universe_hash = self._hash_universe(universe)
        cache_key = f"{data_bundle.market.upper()}:{latest_trade_date}:{universe_hash}"
        cache_path = self.cache_path_for(data_bundle.market, latest_trade_date, universe_hash)

        if not force_refresh and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if str(payload.get("schema_version", "")) != self.schema_version:
                    raise ValueError("schema_version_mismatch")
                context = GlobalContext.from_dict(payload)
                context.cache_key = cache_key
                context.cache_path = str(cache_path)
                return context
            except Exception:
                pass

        context = self._compute_context(
            data_bundle=data_bundle,
            phase1_context=dict(phase1_context or {}),
            branch_results=dict(branch_results or {}),
            calibrated_signals=dict(calibrated_signals or {}),
            risk_result=risk_result,
            cache_key=cache_key,
            cache_path=str(cache_path),
            latest_trade_date=latest_trade_date,
            universe_hash=universe_hash,
        )
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(context.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # Cache is an optimization only; return the freshly built context on write failures.
            pass
        return context

    def _compute_context(
        self,
        *,
        data_bundle: UnifiedDataBundle,
        phase1_context: dict[str, Any],
        branch_results: dict[str, BranchResult],
        calibrated_signals: dict[str, Any],
        risk_result: Any | None,
        cache_key: str,
        cache_path: str,
        latest_trade_date: str,
        universe_hash: str,
    ) -> GlobalContext:
        liquidity_snapshot = self._build_liquidity_snapshot(data_bundle)
        correlation_matrix = self._build_correlation_matrix(data_bundle)
        quant_symbol_scores: dict[str, float] = {}
        quant_signal = calibrated_signals.get("quant")
        if quant_signal is not None:
            quant_symbol_scores = {
                str(symbol): float(score)
                for symbol, score in getattr(quant_signal, "symbol_convictions", {}).items()
            }
        if not quant_symbol_scores:
            quant_symbol_scores = {
                str(symbol): float(score)
                for symbol, score in dict(phase1_context.get("quant_symbol_scores", {})).items()
            }

        macro_signal = dict(phase1_context.get("macro_signals", {}))
        macro_regime = {
            "regime": str(
                phase1_context.get("macro_regime")
                or macro_signal.get("macro_regime")
                or data_bundle.macro_data.get("risk_level")
                or "default"
            ),
            "score": float(
                phase1_context.get("macro_score")
                or macro_signal.get("macro_score")
                or getattr(branch_results.get("macro"), "score", 0.0)
                or 0.0
            ),
            "signals": macro_signal,
        }

        risk_budget = {}
        if risk_result is not None:
            position_sizing = getattr(risk_result, "position_sizing", None)
            risk_budget = {
                "risk_level": getattr(risk_result, "risk_level", ""),
                "cash_ratio": getattr(position_sizing, "cash_ratio", None),
                "target_exposure": getattr(position_sizing, "target_exposure", None),
                "max_position_size": getattr(risk_result, "max_position_size", None),
            }

        completeness_passed = not bool(data_bundle.metadata.get("data_gate", {}).get("blocked_symbols", []))
        source_metadata = {
            "symbol_provenance": data_bundle.symbol_provenance(),
            "data_source_status": data_bundle.metadata.get("data_source_status", ""),
        }

        return GlobalContext(
            market=str(data_bundle.market or ""),
            as_of_date=str(data_bundle.metadata.get("end_date", latest_trade_date)),
            latest_trade_date=latest_trade_date,
            universe=universe_hash and list(data_bundle.symbols) or list(data_bundle.symbols),
            universe_hash=universe_hash,
            completeness_passed=completeness_passed,
            data_gate=dict(data_bundle.metadata.get("data_gate", {})),
            macro_regime=macro_regime,
            quant_factor_scores=quant_symbol_scores,
            style_exposures=dict(phase1_context.get("factor_exposures", {})),
            liquidity_snapshot=liquidity_snapshot,
            risk_budget=risk_budget,
            correlation_matrix=correlation_matrix,
            phase1_context=dict(phase1_context),
            source_metadata=source_metadata,
            cache_key=cache_key,
            cache_path=cache_path,
            schema_version=self.schema_version,
        )

    @staticmethod
    def _hash_universe(universe: list[str]) -> str:
        payload = ",".join(sorted(str(item) for item in universe))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _build_liquidity_snapshot(data_bundle: UnifiedDataBundle) -> dict[str, dict[str, Any]]:
        snapshot: dict[str, dict[str, Any]] = {}
        for symbol, df in data_bundle.symbol_data.items():
            if df is None or df.empty:
                snapshot[symbol] = {"liquidity_score": 0.0, "last_amount": 0.0, "last_volume": 0.0}
                continue
            amount = float(df["amount"].iloc[-1]) if "amount" in df.columns else 0.0
            volume = float(df["volume"].iloc[-1]) if "volume" in df.columns else 0.0
            avg_amount = float(df["amount"].tail(20).mean()) if "amount" in df.columns else amount
            liquidity_score = max(0.0, min(1.0, avg_amount / 100_000_000.0))
            snapshot[symbol] = {
                "last_amount": amount,
                "last_volume": volume,
                "avg_amount_20d": avg_amount,
                "liquidity_score": liquidity_score,
            }
        return snapshot

    @staticmethod
    def _build_correlation_matrix(data_bundle: UnifiedDataBundle) -> dict[str, dict[str, float]]:
        frames = []
        for symbol, df in data_bundle.symbol_data.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            series = df.set_index("date")["close"].pct_change().rename(symbol)
            frames.append(series)
        if not frames:
            return {}
        returns = pd.concat(frames, axis=1).dropna(how="all").fillna(0.0)
        if returns.empty:
            return {}
        return returns.corr().fillna(0.0).to_dict()
