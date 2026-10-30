from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any


DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%m-%d-%Y")


def _parse_float(value: Any) -> float | None:
    """Parse a value to float, returning None if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_date(value: Any) -> dt.datetime | None:
    """Parse a value to datetime using multiple common date formats. Returns None if parsing fails."""
    text = str(value or "").strip()
    if not text:
        return None

    for fmt in DATE_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def _normalize_sales_rows(sales_data: Any) -> list[dict[str, Any]]:
    """Convert sales_data to a consistent list of dicts, supporting pandas DataFrames, lists, and dicts."""
    if hasattr(sales_data, "to_dict") and hasattr(sales_data, "head"):
        try:
            return list(sales_data.to_dict(orient="records"))
        except Exception:
            return []

    if isinstance(sales_data, list):
        return [row for row in sales_data if isinstance(row, dict)]

    if isinstance(sales_data, dict):
        return [sales_data]

    return []


def _extract_categorical_filters(query: str, rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Extract categorical filters (Region, Product, Gender) from the query by checking for field values mentioned in it."""
    normalized = query.lower()

    values_by_field: dict[str, set[str]] = {
        "Region": set(),
        "Product": set(),
        "Customer_Gender": set(),
    }

    for row in rows:
        for field in values_by_field:
            value = str(row.get(field, "")).strip()
            if value:
                values_by_field[field].add(value)

    filters: dict[str, list[str]] = {}
    for field, values in values_by_field.items():
        matched: list[str] = []
        for value in sorted(values):
            token = value.lower()
            if re.search(rf"\b{re.escape(token)}\b", normalized):
                matched.append(value)
        if matched:
            filters[field] = matched

    return filters


def _extract_age_filter(query: str) -> dict[str, int] | None:
    """Extract age range filters from query patterns like 'between X and Y', 'over X', 'under X'. Returns None if no age filter detected."""
    normalized = query.lower()

    between_match = re.search(r"between\s+(\d{1,3})\s+and\s+(\d{1,3})", normalized)
    if between_match:
        lower = int(between_match.group(1))
        upper = int(between_match.group(2))
        return {"min": min(lower, upper), "max": max(lower, upper)}

    over_match = re.search(r"(over|older than|above|greater than)\s+(\d{1,3})", normalized)
    if over_match:
        return {"min_exclusive": int(over_match.group(2))}

    under_match = re.search(r"(under|younger than|below|less than)\s+(\d{1,3})", normalized)
    if under_match:
        return {"max_exclusive": int(under_match.group(2))}

    return None


def _apply_filters(
    rows: list[dict[str, Any]],
    categorical_filters: dict[str, list[str]],
    age_filter: dict[str, int] | None,
) -> list[dict[str, Any]]:
    """Apply categorical and age filters to rows, returning only rows that match all criteria."""
    filtered = rows

    for field, allowed_values in categorical_filters.items():
        allowed_set = {value.strip() for value in allowed_values}
        filtered = [
            row
            for row in filtered
            if str(row.get(field, "")).strip() in allowed_set
        ]

    if age_filter:
        min_inclusive = age_filter.get("min")
        max_inclusive = age_filter.get("max")
        min_exclusive = age_filter.get("min_exclusive")
        max_exclusive = age_filter.get("max_exclusive")

        next_filtered: list[dict[str, Any]] = []
        for row in filtered:
            age = _parse_float(row.get("Customer_Age"))
            if age is None:
                continue

            if min_inclusive is not None and age < min_inclusive:
                continue
            if max_inclusive is not None and age > max_inclusive:
                continue
            if min_exclusive is not None and age <= min_exclusive:
                continue
            if max_exclusive is not None and age >= max_exclusive:
                continue
            next_filtered.append(row)

        filtered = next_filtered

    return filtered


def _evenly_sample_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    """Sample up to max_rows from the list, evenly distributed across the full range to maintain coverage."""
    if len(rows) <= max_rows:
        return rows
    if max_rows <= 1:
        return [rows[0]]

    last_index = len(rows) - 1
    denominator = max_rows - 1
    sampled: list[dict[str, Any]] = []
    for slot in range(max_rows):
        index = int(round((slot * last_index) / denominator))
        sampled.append(rows[index])
    return sampled


def _build_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregates on the row set: total sales by region/product, monthly sales, and average customer satisfaction."""
    sales_by_region: dict[str, float] = {}
    sales_by_product: dict[str, float] = {}
    monthly_sales: dict[str, float] = {}

    satisfaction_sum = 0.0
    satisfaction_count = 0

    for row in rows:
        sales = _parse_float(row.get("Sales"))
        if sales is None:
            continue

        region = str(row.get("Region", "")).strip() or "Unknown"
        product = str(row.get("Product", "")).strip() or "Unknown"

        sales_by_region[region] = sales_by_region.get(region, 0.0) + sales
        sales_by_product[product] = sales_by_product.get(product, 0.0) + sales

        parsed_date = _parse_date(row.get("Date"))
        if parsed_date is not None:
            month_key = parsed_date.strftime("%Y-%m")
            monthly_sales[month_key] = monthly_sales.get(month_key, 0.0) + sales

        satisfaction = _parse_float(row.get("Customer_Satisfaction"))
        if satisfaction is not None:
            satisfaction_sum += satisfaction
            satisfaction_count += 1

    avg_satisfaction = (satisfaction_sum / satisfaction_count) if satisfaction_count else None

    return {
        "sales_by_region": sorted(sales_by_region.items(), key=lambda item: item[1], reverse=True),
        "sales_by_product": sorted(sales_by_product.items(), key=lambda item: item[1], reverse=True),
        "monthly_sales": sorted(monthly_sales.items(), key=lambda item: item[0]),
        "avg_customer_satisfaction": avg_satisfaction,
    }


def build_sales_context(
    query: str,
    sales_data: Any,
    max_sample_rows: int = 220,
) -> dict[str, Any]:
    """Build a compact, query-aware sales context with aggregates plus sampled rows."""
    all_rows = _normalize_sales_rows(sales_data)
    categorical_filters = _extract_categorical_filters(query, all_rows)
    age_filter = _extract_age_filter(query)

    filtered_rows = _apply_filters(all_rows, categorical_filters, age_filter)
    if not filtered_rows:
        filtered_rows = all_rows

    sampled_rows = _evenly_sample_rows(filtered_rows, max_rows=max_sample_rows)

    context = {
        "query": query,
        "row_count_total": len(all_rows),
        "row_count_filtered": len(filtered_rows),
        "applied_filters": {
            "categorical": categorical_filters,
            "age": age_filter,
        },
        "aggregates": _build_aggregates(filtered_rows),
        "sample_rows": sampled_rows,
    }
    return context


def serialize_sales_context(query: str, sales_data: Any, max_sample_rows: int = 220) -> str:
    """Build query-aware sales context and serialize it to JSON for use in LLM prompts."""
    context = build_sales_context(query=query, sales_data=sales_data, max_sample_rows=max_sample_rows)
    return json.dumps(context, ensure_ascii=True)
