from __future__ import annotations

from datetime import date
from typing import cast

import pandas as pd

from core.contracts import Asset, CorporateAction, CorporateActionType
from core.errors import DataQualityError

from .common import normalize_currency_code, normalize_dataframe_columns

_REQUIRED_COLUMNS = ("action_type", "source")


def normalize_corporate_actions(
    frame: pd.DataFrame,
    asset: Asset,
) -> tuple[CorporateAction, ...]:
    work = normalize_dataframe_columns(frame)
    missing = [column for column in _REQUIRED_COLUMNS if column not in work.columns]
    if missing:
        raise DataQualityError(f"missing corporate-action columns: {missing}")

    for column in ("announced_at", "ex_date", "record_date", "payable_date"):
        if column in work.columns:
            work[column] = pd.to_datetime(work[column], errors="coerce", utc=True).dt.date

    for column in ("ratio_numerator", "ratio_denominator", "cash_amount"):
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    if "currency" in work.columns:
        work["currency"] = work["currency"].apply(
            lambda value: None if pd.isna(value) else normalize_currency_code(str(value))
        )

    if work["action_type"].isna().any() or work["source"].isna().any():
        raise DataQualityError("corporate action contains missing required values")

    unique_columns = [
        "action_type",
        "announced_at",
        "ex_date",
        "record_date",
        "payable_date",
        "ratio_numerator",
        "ratio_denominator",
        "cash_amount",
        "new_symbol",
        "source_event_id",
    ]
    unique_columns = [column for column in unique_columns if column in work.columns]
    if work.duplicated(subset=unique_columns).any():
        raise DataQualityError("duplicate corporate actions detected")

    work = work.sort_values(["ex_date", "announced_at"], na_position="last").reset_index(drop=True)

    actions: list[CorporateAction] = []
    for row in work.itertuples(index=False):
        action_type = str(row.action_type).strip().lower()
        try:
            parsed_type = CorporateActionType(action_type)
        except ValueError as exc:
            raise DataQualityError(f"unsupported corporate action type: {action_type}") from exc

        def optional_value(row: object, name: str) -> object | None:
            if not hasattr(row, name):
                return None
            value = getattr(row, name)
            return None if pd.isna(value) else value

        actions.append(
            CorporateAction(
                asset=asset,
                action_type=parsed_type,
                announced_at=cast(date | None, optional_value(row, "announced_at")),
                ex_date=cast(date | None, optional_value(row, "ex_date")),
                record_date=cast(date | None, optional_value(row, "record_date")),
                payable_date=cast(date | None, optional_value(row, "payable_date")),
                ratio_numerator=cast(float | None, optional_value(row, "ratio_numerator")),
                ratio_denominator=cast(float | None, optional_value(row, "ratio_denominator")),
                cash_amount=cast(float | None, optional_value(row, "cash_amount")),
                currency=cast(str | None, optional_value(row, "currency")),
                new_symbol=cast(str | None, optional_value(row, "new_symbol")),
                related_symbol=cast(str | None, optional_value(row, "related_symbol")),
                source=str(row.source).strip(),
                source_event_id=cast(str | None, optional_value(row, "source_event_id")),
            )
        )

    return tuple(actions)


def apply_split_adjustments(
    frame: pd.DataFrame,
    actions: tuple[CorporateAction, ...],
) -> pd.DataFrame:
    required = ("timestamp", "open", "high", "low", "close", "volume")
    work = normalize_dataframe_columns(frame)
    missing = [column for column in required if column not in work.columns]
    if missing:
        raise DataQualityError(f"missing market columns for split adjustment: {missing}")

    adjusted = work.copy()
    timestamps = pd.to_datetime(adjusted["timestamp"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise DataQualityError("invalid market timestamps for corporate-action adjustment")
    adjusted["timestamp"] = timestamps

    split_actions = [
        action
        for action in actions
        if action.action_type
        in {
            CorporateActionType.SPLIT,
            CorporateActionType.REVERSE_SPLIT,
        }
        and action.ex_date is not None
    ]
    split_actions.sort(key=lambda action: action.ex_date or pd.Timestamp.min.date())

    for action in split_actions:
        factor = action.split_factor
        if factor <= 0:
            raise DataQualityError("split factor must be positive")
        mask = adjusted["timestamp"].dt.date < action.ex_date
        for column in ("open", "high", "low", "close"):
            adjusted.loc[mask, column] = adjusted.loc[mask, column] / factor
        adjusted.loc[mask, "volume"] = adjusted.loc[mask, "volume"] * factor

    return adjusted
