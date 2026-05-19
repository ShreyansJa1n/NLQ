from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FewShotExample:
    question: str
    sql: str


SQLITE_FEW_SHOT: tuple[FewShotExample, ...] = (
    FewShotExample(
        question="List the 5 most recent transactions.",
        sql="SELECT id, user_id, amount_cents, occurred_on\n"
        "FROM transactions\n"
        "ORDER BY occurred_on DESC\n"
        "LIMIT 5;",
    ),
    FewShotExample(
        question="How much did each user spend in total?",
        sql="SELECT u.id, u.name, SUM(t.amount_cents) AS total_cents\n"
        "FROM users u\n"
        "JOIN transactions t ON t.user_id = u.id\n"
        "GROUP BY u.id, u.name\n"
        "ORDER BY total_cents DESC;",
    ),
    FewShotExample(
        question="Which categories had no transactions last month?",
        sql="SELECT c.id, c.label\n"
        "FROM categories c\n"
        "LEFT JOIN transactions t\n"
        "  ON t.category_id = c.id\n"
        "  AND t.occurred_on >= date('now', 'start of month', '-1 month')\n"
        "  AND t.occurred_on <  date('now', 'start of month')\n"
        "WHERE t.id IS NULL\n"
        "ORDER BY c.label;",
    ),
)


def few_shot_for(dialect: str) -> tuple[FewShotExample, ...]:
    if dialect == "sqlite":
        return SQLITE_FEW_SHOT
    raise ValueError(f"No few-shot examples for dialect: {dialect}")
