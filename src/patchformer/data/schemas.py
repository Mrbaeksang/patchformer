"""Closed toy schemas intentionally partitioned by benchmark split.

The splits do not share a schema ID, domain, field-name family, or instruction
template family.  This is a small but auditable guard against the most obvious
synthetic-data leakage; it is not evidence of real-world generalization.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

JsonObject = dict[str, Any]


def _leaf(type_name: str | list[str], title: str, **extra: Any) -> JsonObject:
    return {"type": type_name, "title": title, **extra}


def _object(properties: JsonObject, required: list[str], title: str) -> JsonObject:
    return {
        "type": "object",
        "title": title,
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# ``split`` is benchmark metadata rather than part of the JSON schema itself.
SCHEMA_CATALOG: tuple[JsonObject, ...] = (
    {
        "schema_id": "orders-v1",
        "domain": "orders",
        "split": "train",
        "schema": _object(
            {
                "order": _object(
                    {
                        "order_id": _leaf("string", "order identifier"),
                        "status": _leaf(
                            "string",
                            "order status",
                            enum=["pending", "preparing", "shipped", "cancelled"],
                        ),
                        "delivery_date": _leaf("string", "delivery date"),
                        "priority": _leaf("number", "delivery priority"),
                        "gift_wrap": _leaf("boolean", "gift wrapping"),
                        "customer_note": _leaf(["string", "null"], "customer note"),
                    },
                    ["order_id", "status", "delivery_date", "priority"],
                    "order",
                )
            },
            ["order"],
            "order state",
        ),
    },
    {
        "schema_id": "catalog-v1",
        "domain": "catalog",
        "split": "train",
        "schema": _object(
            {
                "item": _object(
                    {
                        "sku": _leaf("string", "stock keeping unit"),
                        "availability": _leaf("boolean", "store availability"),
                        "quantity": _leaf("number", "inventory quantity"),
                        "department": _leaf(
                            "string", "store department", enum=["books", "home", "outdoors"]
                        ),
                        "featured": _leaf("boolean", "featured flag"),
                        "internal_note": _leaf(["string", "null"], "internal note"),
                    },
                    ["sku", "availability", "quantity", "department"],
                    "catalog item",
                )
            },
            ["item"],
            "catalog state",
        ),
    },
    {
        "schema_id": "support-tickets-v1",
        "domain": "support",
        "split": "validation",
        "schema": _object(
            {
                "ticket": _object(
                    {
                        "ticket_code": _leaf("string", "ticket code"),
                        "stage": _leaf(
                            "string", "case stage", enum=["new", "assigned", "resolved", "closed"]
                        ),
                        "severity": _leaf("number", "case severity"),
                        "escalated": _leaf("boolean", "escalation flag"),
                        "resolution_note": _leaf(["string", "null"], "resolution note"),
                        "callback_requested": _leaf("boolean", "callback request"),
                    },
                    ["ticket_code", "stage", "severity", "escalated"],
                    "support ticket",
                )
            },
            ["ticket"],
            "support state",
        ),
    },
    {
        "schema_id": "profiles-v1",
        "domain": "profiles",
        "split": "test",
        "schema": _object(
            {
                "profile": _object(
                    {
                        "member_key": _leaf("string", "member key"),
                        "contact_channel": _leaf(
                            "string", "contact channel", enum=["email", "sms", "none"]
                        ),
                        "marketing_score": _leaf("number", "marketing score"),
                        "verified": _leaf("boolean", "verification flag"),
                        "display_alias": _leaf(["string", "null"], "display alias"),
                        "digest_enabled": _leaf("boolean", "digest preference"),
                    },
                    ["member_key", "contact_channel", "marketing_score", "verified"],
                    "member profile",
                )
            },
            ["profile"],
            "profile state",
        ),
    },
)


def schemas_for_split(split: str) -> list[JsonObject]:
    """Return detached schema records assigned exclusively to ``split``."""
    selected = [record for record in SCHEMA_CATALOG if record["split"] == split]
    if not selected:
        raise ValueError(f"unknown or empty split: {split!r}")
    return deepcopy(selected)


def all_split_names() -> tuple[str, ...]:
    return ("train", "validation", "test")
