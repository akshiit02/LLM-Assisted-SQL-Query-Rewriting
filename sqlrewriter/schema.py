"""Benchmark schema definition (single source of truth for every backend).

The schema models a retail / order-management database with ten related
tables.  It is deliberately rich enough to support every query complexity
level used by the study (simple lookups, multi-way joins, correlated
subqueries, CTEs, window functions, set operations ...).

DDL for each dialect and the *index configurations* used in the experiment
(RQ5) are rendered from the definitions below.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Column:
    name: str
    type: str  # logical type: INT, DECIMAL(p,s), VARCHAR(n), DATE
    nullable: bool = True


@dataclass(frozen=True)
class ForeignKey:
    columns: tuple[str, ...]
    ref_table: str
    ref_columns: tuple[str, ...]


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()
    description: str = ""

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)


@dataclass(frozen=True)
class Index:
    name: str
    table: str
    columns: tuple[str, ...]
    unique: bool = False


@dataclass
class IndexConfiguration:
    name: str
    description: str
    indexes: list[Index] = field(default_factory=list)


def _c(name: str, type_: str, nullable: bool = True) -> Column:
    return Column(name, type_, nullable)


TABLES: tuple[Table, ...] = (
    Table(
        "regions",
        (_c("region_id", "INT", False), _c("name", "VARCHAR(40)", False)),
        ("region_id",),
        description="Geographic sales regions.",
    ),
    Table(
        "nations",
        (
            _c("nation_id", "INT", False),
            _c("name", "VARCHAR(40)", False),
            _c("region_id", "INT", False),
        ),
        ("nation_id",),
        (ForeignKey(("region_id",), "regions", ("region_id",)),),
        "Countries, each belonging to one region.",
    ),
    Table(
        "customers",
        (
            _c("customer_id", "INT", False),
            _c("name", "VARCHAR(80)", False),
            _c("email", "VARCHAR(120)", False),
            _c("nation_id", "INT", False),
            _c("segment", "VARCHAR(20)", False),
            _c("signup_date", "DATE", False),
            _c("credit_limit", "DECIMAL(10,2)"),
            _c("is_active", "INT", False),
        ),
        ("customer_id",),
        (ForeignKey(("nation_id",), "nations", ("nation_id",)),),
        "Customers. credit_limit is NULL for roughly 5% of rows.",
    ),
    Table(
        "suppliers",
        (
            _c("supplier_id", "INT", False),
            _c("name", "VARCHAR(80)", False),
            _c("nation_id", "INT", False),
            _c("rating", "DECIMAL(3,1)"),
        ),
        ("supplier_id",),
        (ForeignKey(("nation_id",), "nations", ("nation_id",)),),
        "Product suppliers. rating may be NULL.",
    ),
    Table(
        "categories",
        (
            _c("category_id", "INT", False),
            _c("name", "VARCHAR(60)", False),
            _c("parent_category_id", "INT"),
        ),
        ("category_id",),
        (ForeignKey(("parent_category_id",), "categories", ("category_id",)),),
        "Two-level product category hierarchy (top-level rows have NULL parent).",
    ),
    Table(
        "products",
        (
            _c("product_id", "INT", False),
            _c("name", "VARCHAR(100)", False),
            _c("category_id", "INT", False),
            _c("supplier_id", "INT"),
            _c("unit_price", "DECIMAL(10,2)", False),
            _c("cost_price", "DECIMAL(10,2)", False),
            _c("stock_qty", "INT", False),
            _c("discontinued", "INT", False),
            _c("launch_date", "DATE", False),
        ),
        ("product_id",),
        (
            ForeignKey(("category_id",), "categories", ("category_id",)),
            ForeignKey(("supplier_id",), "suppliers", ("supplier_id",)),
        ),
        "Product catalogue. supplier_id is NULL for ~3% of products (house brand).",
    ),
    Table(
        "orders",
        (
            _c("order_id", "INT", False),
            _c("customer_id", "INT", False),
            _c("order_date", "DATE", False),
            _c("status", "VARCHAR(12)", False),
            _c("ship_mode", "VARCHAR(12)", False),
            _c("total_amount", "DECIMAL(12,2)", False),
            _c("discount_code", "VARCHAR(12)"),
            _c("shipped_date", "DATE"),
        ),
        ("order_id",),
        (ForeignKey(("customer_id",), "customers", ("customer_id",)),),
        "Order headers (2021-2025). status in PENDING/SHIPPED/DELIVERED/CANCELLED/RETURNED.",
    ),
    Table(
        "order_items",
        (
            _c("order_id", "INT", False),
            _c("line_no", "INT", False),
            _c("product_id", "INT", False),
            _c("quantity", "INT", False),
            _c("unit_price", "DECIMAL(10,2)", False),
            _c("discount", "DECIMAL(4,2)", False),
        ),
        ("order_id", "line_no"),
        (
            ForeignKey(("order_id",), "orders", ("order_id",)),
            ForeignKey(("product_id",), "products", ("product_id",)),
        ),
        "Order lines. discount is a fraction between 0 and 0.30.",
    ),
    Table(
        "payments",
        (
            _c("payment_id", "INT", False),
            _c("order_id", "INT", False),
            _c("payment_date", "DATE", False),
            _c("amount", "DECIMAL(12,2)", False),
            _c("method", "VARCHAR(12)", False),
            _c("status", "VARCHAR(10)", False),
        ),
        ("payment_id",),
        (ForeignKey(("order_id",), "orders", ("order_id",)),),
        "Payments against orders (0..2 per order). status in OK/FAILED/REFUNDED.",
    ),
    Table(
        "reviews",
        (
            _c("review_id", "INT", False),
            _c("product_id", "INT", False),
            _c("customer_id", "INT", False),
            _c("rating", "INT", False),
            _c("review_date", "DATE", False),
            _c("helpful_votes", "INT", False),
        ),
        ("review_id",),
        (
            ForeignKey(("product_id",), "products", ("product_id",)),
            ForeignKey(("customer_id",), "customers", ("customer_id",)),
        ),
        "Product reviews, rating 1..5.",
    ),
)

TABLES_BY_NAME = {t.name: t for t in TABLES}

# Tables in dependency order (parents first) – used for loading & dropping.
LOAD_ORDER = [t.name for t in TABLES]


# ---------------------------------------------------------------------------
# Index configurations (independent variable for RQ5)
# ---------------------------------------------------------------------------

_FK_INDEXES = [
    Index("ix_nations_region", "nations", ("region_id",)),
    Index("ix_customers_nation", "customers", ("nation_id",)),
    Index("ix_suppliers_nation", "suppliers", ("nation_id",)),
    Index("ix_categories_parent", "categories", ("parent_category_id",)),
    Index("ix_products_category", "products", ("category_id",)),
    Index("ix_products_supplier", "products", ("supplier_id",)),
    Index("ix_orders_customer", "orders", ("customer_id",)),
    Index("ix_items_product", "order_items", ("product_id",)),
    Index("ix_payments_order", "payments", ("order_id",)),
    Index("ix_reviews_product", "reviews", ("product_id",)),
    Index("ix_reviews_customer", "reviews", ("customer_id",)),
]

_TUNED_EXTRA = [
    Index("ix_orders_date", "orders", ("order_date",)),
    Index("ix_orders_status_date", "orders", ("status", "order_date")),
    Index("ix_orders_cust_date", "orders", ("customer_id", "order_date", "total_amount")),
    Index("ix_items_prod_qty", "order_items", ("product_id", "quantity", "unit_price")),
    Index("ix_customers_segment", "customers", ("segment", "nation_id")),
    Index("ix_products_cat_price", "products", ("category_id", "unit_price")),
    Index("ix_payments_status", "payments", ("status", "order_id")),
    Index("ix_reviews_prod_rating", "reviews", ("product_id", "rating")),
]

INDEX_CONFIGURATIONS: dict[str, IndexConfiguration] = {
    "baseline": IndexConfiguration(
        "baseline", "Primary keys only (no secondary indexes).", []
    ),
    "fk_indexes": IndexConfiguration(
        "fk_indexes", "Primary keys + one index per foreign-key column.", list(_FK_INDEXES)
    ),
    "tuned": IndexConfiguration(
        "tuned",
        "FK indexes + composite/covering indexes for the workload's common predicates.",
        list(_FK_INDEXES) + list(_TUNED_EXTRA),
    ),
}

ALL_SECONDARY_INDEXES: list[Index] = list(_FK_INDEXES) + list(_TUNED_EXTRA)


# ---------------------------------------------------------------------------
# DDL rendering
# ---------------------------------------------------------------------------

def _render_type(logical: str, dialect: str) -> str:
    t = logical.upper()
    if dialect == "oracle":
        if t == "INT":
            return "NUMBER(10)"
        if t.startswith("DECIMAL"):
            return t.replace("DECIMAL", "NUMBER")
        if t.startswith("VARCHAR"):
            return t.replace("VARCHAR", "VARCHAR2")
        if t == "DATE":
            return "DATE"
    elif dialect == "sqlite":
        if t == "INT":
            return "INTEGER"
        if t.startswith("DECIMAL"):
            return "REAL"
        if t.startswith("VARCHAR"):
            return "TEXT"
        if t == "DATE":
            return "TEXT"  # ISO-8601 'YYYY-MM-DD'; lexical order == date order
    raise ValueError(f"Unsupported type {logical!r} for dialect {dialect!r}")


def create_table_ddl(table: Table, dialect: str) -> str:
    lines = []
    for col in table.columns:
        null = "" if col.nullable else " NOT NULL"
        lines.append(f"    {col.name} {_render_type(col.type, dialect)}{null}")
    pk_name = f"pk_{table.name}"
    lines.append(f"    CONSTRAINT {pk_name} PRIMARY KEY ({', '.join(table.primary_key)})")
    for i, fk in enumerate(table.foreign_keys, start=1):
        lines.append(
            f"    CONSTRAINT fk_{table.name}_{i} FOREIGN KEY ({', '.join(fk.columns)}) "
            f"REFERENCES {fk.ref_table} ({', '.join(fk.ref_columns)})"
        )
    body = ",\n".join(lines)
    return f"CREATE TABLE {table.name} (\n{body}\n)"


def create_index_ddl(index: Index) -> str:
    unique = "UNIQUE " if index.unique else ""
    return f"CREATE {unique}INDEX {index.name} ON {index.table} ({', '.join(index.columns)})"


def schema_ddl(dialect: str) -> list[str]:
    return [create_table_ddl(t, dialect) for t in TABLES]


def sqlglot_schema(dialect: str = "oracle") -> dict[str, dict[str, str]]:
    """Schema mapping in the shape sqlglot's optimizer expects."""
    mapping: dict[str, dict[str, str]] = {}
    for t in TABLES:
        mapping[t.name] = {c.name: _sqlglot_type(c.type) for c in t.columns}
    return mapping


def _sqlglot_type(logical: str) -> str:
    t = logical.upper()
    if t == "INT":
        return "INT"
    if t.startswith("DECIMAL"):
        return t
    if t.startswith("VARCHAR"):
        return t
    return "DATE"


def write_oracle_scripts(directory: str) -> list[str]:
    """Write standalone Oracle SQL*Plus scripts (handy for DBAs / manual setup)."""
    import os

    os.makedirs(directory, exist_ok=True)
    written = []
    schema_path = os.path.join(directory, "01_schema.sql")
    with open(schema_path, "w", encoding="utf-8") as fh:
        fh.write("-- Generated by sqlrewriter.schema – benchmark schema for Oracle\n\n")
        for ddl in schema_ddl("oracle"):
            fh.write(ddl + ";\n\n")
    written.append(schema_path)
    for cfg in INDEX_CONFIGURATIONS.values():
        path = os.path.join(directory, f"02_indexes_{cfg.name}.sql")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"-- Index configuration '{cfg.name}': {cfg.description}\n\n")
            for ix in cfg.indexes:
                fh.write(create_index_ddl(ix) + ";\n")
            fh.write("\nBEGIN DBMS_STATS.GATHER_SCHEMA_STATS(ownname => USER); END;\n/\n")
        written.append(path)
    drop_path = os.path.join(directory, "99_drop.sql")
    with open(drop_path, "w", encoding="utf-8") as fh:
        fh.write("-- Drops every benchmark table\n\n")
        for name in reversed(LOAD_ORDER):
            fh.write(f"DROP TABLE {name} CASCADE CONSTRAINTS PURGE;\n")
    written.append(drop_path)
    return written
