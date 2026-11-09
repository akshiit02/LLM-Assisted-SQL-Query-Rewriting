"""Deterministic synthetic data generator for the benchmark schema.

``generate(scale, seed)`` returns rows for every table.  The same
(scale, seed) always yields byte-identical data, so experiments on SQLite and
Oracle are reproducible and comparable.

Scale factor 1.0 produces roughly:
    customers 5k, suppliers 200, products 2k, orders 50k,
    order_items ~150k, payments ~55k, reviews 20k

The data intentionally contains features that make query rewriting
non-trivial:
  * NULLs in nullable columns (credit_limit, supplier_id, rating,
    discount_code, shipped_date, parent_category_id) – these break naive
    ``NOT IN`` -> ``NOT EXISTS`` style rewrites;
  * skewed (Zipf-like) customer and product popularity;
  * customers without orders and products without sales (anti-join cases);
  * duplicate values that punish rewrites that drop DISTINCT or
    turn semi-joins into plain joins.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass

REGIONS = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
NATIONS = [
    ("ALGERIA", 0), ("ARGENTINA", 1), ("BRAZIL", 1), ("CANADA", 1), ("EGYPT", 4),
    ("ETHIOPIA", 0), ("FRANCE", 3), ("GERMANY", 3), ("INDIA", 2), ("INDONESIA", 2),
    ("IRAN", 4), ("IRAQ", 4), ("JAPAN", 2), ("JORDAN", 4), ("KENYA", 0),
    ("MOROCCO", 0), ("MOZAMBIQUE", 0), ("PERU", 1), ("CHINA", 2), ("ROMANIA", 3),
    ("SAUDI ARABIA", 4), ("VIETNAM", 2), ("RUSSIA", 3), ("UNITED KINGDOM", 3),
    ("UNITED STATES", 1),
]
TOP_CATEGORIES = [
    "Electronics", "Home", "Garden", "Toys", "Books", "Sports", "Fashion", "Grocery",
]
SUB_CATEGORY_SUFFIXES = ["Essentials", "Premium", "Accessories", "Outlet"]
SEGMENTS = ["CONSUMER", "CORPORATE", "SMALL_BIZ", "GOVERNMENT"]
SEGMENT_WEIGHTS = [0.55, 0.25, 0.15, 0.05]
ORDER_STATUSES = ["DELIVERED", "SHIPPED", "PENDING", "CANCELLED", "RETURNED"]
ORDER_STATUS_WEIGHTS = [0.58, 0.17, 0.10, 0.09, 0.06]
SHIP_MODES = ["STANDARD", "EXPRESS", "PICKUP", "FREIGHT"]
SHIP_MODE_WEIGHTS = [0.6, 0.25, 0.1, 0.05]
PAY_METHODS = ["CARD", "UPI", "WALLET", "BANK", "COD"]
DISCOUNT_CODES = ["WELCOME10", "FESTIVE20", "LOYAL5", "FLASH15", "BULK25"]
ADJECTIVES = ["Smart", "Classic", "Ultra", "Eco", "Pro", "Mini", "Max", "Prime", "Lite", "Nova"]
NOUNS = ["Widget", "Lamp", "Chair", "Kettle", "Drone", "Novel", "Racket", "Jacket", "Blender",
         "Speaker", "Planter", "Puzzle", "Backpack", "Monitor", "Tea"]
FIRST = ["Aarav", "Diya", "Liam", "Emma", "Noah", "Olivia", "Arjun", "Mei", "Lucas", "Sofia",
         "Kabir", "Ananya", "Hiro", "Fatima", "Mateo", "Zara", "Ivan", "Chloe", "Omar", "Priya"]
LAST = ["Sharma", "Smith", "Garcia", "Chen", "Kumar", "Muller", "Rossi", "Tanaka", "Silva",
        "Khan", "Patel", "Brown", "Nguyen", "Ivanov", "Dubois", "Singh", "Lee", "Haddad"]

START_DATE = dt.date(2021, 1, 1)
END_DATE = dt.date(2025, 12, 31)
_DAY_SPAN = (END_DATE - START_DATE).days


@dataclass
class Dataset:
    scale: float
    seed: int
    tables: dict[str, list[tuple]]

    def row_counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.tables.items()}


def _zipf_weights(n: int, s: float) -> list[float]:
    return [1.0 / ((i + 1) ** s) for i in range(n)]


def _cumulative(weights: list[float]) -> list[float]:
    total = 0.0
    out = []
    for w in weights:
        total += w
        out.append(total)
    return out


def _rand_date(rng: random.Random, start: dt.date = START_DATE, span: int = _DAY_SPAN) -> dt.date:
    return start + dt.timedelta(days=rng.randint(0, span))


def generate(scale: float = 1.0, seed: int = 42) -> Dataset:
    if scale <= 0:
        raise ValueError("scale must be positive")
    rng = random.Random(seed)

    n_customers = max(100, int(5000 * scale))
    n_suppliers = max(20, int(200 * scale))
    n_products = max(200, int(2000 * scale))
    n_orders = max(1000, int(50000 * scale))
    n_reviews = max(400, int(20000 * scale))

    tables: dict[str, list[tuple]] = {}

    tables["regions"] = [(i + 1, name) for i, name in enumerate(REGIONS)]
    tables["nations"] = [(i + 1, name, region + 1) for i, (name, region) in enumerate(NATIONS)]

    # --- customers -------------------------------------------------------
    customers = []
    for cid in range(1, n_customers + 1):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        nation = rng.randint(1, len(NATIONS))
        segment = rng.choices(SEGMENTS, SEGMENT_WEIGHTS)[0]
        signup = _rand_date(rng, dt.date(2019, 1, 1), (dt.date(2025, 6, 30) - dt.date(2019, 1, 1)).days)
        credit = None if rng.random() < 0.05 else round(rng.uniform(500, 50000), 2)
        active = 0 if rng.random() < 0.10 else 1
        customers.append((cid, name, f"user{cid}@example.com", nation, segment, signup, credit, active))
    tables["customers"] = customers

    # --- suppliers -------------------------------------------------------
    suppliers = []
    for sid in range(1, n_suppliers + 1):
        rating = None if rng.random() < 0.08 else round(rng.uniform(1.0, 5.0), 1)
        suppliers.append((sid, f"Supplier#{sid:05d}", rng.randint(1, len(NATIONS)), rating))
    tables["suppliers"] = suppliers

    # --- categories (8 top-level + 32 children) ------------------------
    categories = []
    cat_id = 0
    top_ids = []
    for top in TOP_CATEGORIES:
        cat_id += 1
        top_ids.append(cat_id)
        categories.append((cat_id, top, None))
    leaf_ids = []
    for parent_idx, top in enumerate(TOP_CATEGORIES):
        for suffix in SUB_CATEGORY_SUFFIXES:
            cat_id += 1
            leaf_ids.append(cat_id)
            categories.append((cat_id, f"{top} {suffix}", top_ids[parent_idx]))
    tables["categories"] = categories

    # --- products --------------------------------------------------------
    products = []
    prices: dict[int, float] = {}
    for pid in range(1, n_products + 1):
        name = f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)} {pid}"
        category = rng.choice(leaf_ids) if rng.random() < 0.95 else rng.choice(top_ids)
        supplier = None if rng.random() < 0.03 else rng.randint(1, n_suppliers)
        price = round(rng.lognormvariate(3.3, 0.9), 2) + 1.0
        price = round(min(price, 2500.0), 2)
        cost = round(price * rng.uniform(0.45, 0.85), 2)
        stock = rng.randint(0, 800)
        discontinued = 1 if rng.random() < 0.07 else 0
        launch = _rand_date(rng, dt.date(2018, 1, 1), (dt.date(2025, 6, 30) - dt.date(2018, 1, 1)).days)
        prices[pid] = price
        products.append((pid, name, category, supplier, price, cost, stock, discontinued, launch))
    tables["products"] = products

    # Popularity distributions. ~8% of customers and ~10% of products never
    # appear in orders (anti-join test cases).
    ordering_customers = list(range(1, n_customers + 1))
    rng.shuffle(ordering_customers)
    ordering_customers = ordering_customers[: int(n_customers * 0.92)]
    cust_cum = _cumulative(_zipf_weights(len(ordering_customers), 0.6))

    selling_products = list(range(1, n_products + 1))
    rng.shuffle(selling_products)
    selling_products = selling_products[: int(n_products * 0.90)]
    prod_cum = _cumulative(_zipf_weights(len(selling_products), 0.8))

    # --- orders, order_items, payments ----------------------------------
    orders, items, payments = [], [], []
    payment_id = 0
    for oid in range(1, n_orders + 1):
        cust = rng.choices(ordering_customers, cum_weights=cust_cum)[0]
        odate = _rand_date(rng)
        status = rng.choices(ORDER_STATUSES, ORDER_STATUS_WEIGHTS)[0]
        ship_mode = rng.choices(SHIP_MODES, SHIP_MODE_WEIGHTS)[0]
        discount_code = rng.choice(DISCOUNT_CODES) if rng.random() < 0.18 else None
        shipped = (
            odate + dt.timedelta(days=rng.randint(1, 10))
            if status in ("SHIPPED", "DELIVERED", "RETURNED")
            else None
        )
        n_lines = rng.choices([1, 2, 3, 4, 5], [0.30, 0.28, 0.20, 0.13, 0.09])[0]
        total = 0.0
        used = set()
        line_no = 0
        for _ in range(n_lines):
            pid = rng.choices(selling_products, cum_weights=prod_cum)[0]
            if pid in used:
                continue
            used.add(pid)
            line_no += 1
            qty = rng.choices([1, 2, 3, 4, 5, 10], [0.5, 0.2, 0.12, 0.08, 0.06, 0.04])[0]
            unit = prices[pid]
            disc = rng.choice([0.0, 0.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
            total += qty * unit * (1 - disc)
            items.append((oid, line_no, pid, qty, unit, disc))
        total = round(total, 2)
        orders.append((oid, cust, odate, status, ship_mode, total, discount_code, shipped))

        if status == "CANCELLED":
            if rng.random() < 0.5:
                payment_id += 1
                payments.append((payment_id, oid, odate, total, rng.choice(PAY_METHODS), "FAILED"))
            continue
        if status == "PENDING" and rng.random() < 0.6:
            continue  # not yet paid
        method = rng.choice(PAY_METHODS)
        if rng.random() < 0.10:
            payment_id += 1
            payments.append((payment_id, oid, odate, total, method, "FAILED"))
        payment_id += 1
        pay_status = "REFUNDED" if status == "RETURNED" else "OK"
        pay_date = odate + dt.timedelta(days=rng.randint(0, 2))
        payments.append((payment_id, oid, pay_date, total, method, pay_status))

    tables["orders"] = orders
    tables["order_items"] = items
    tables["payments"] = payments

    # --- reviews -----------------------------------------------------------
    reviews = []
    for rid in range(1, n_reviews + 1):
        pid = rng.choices(selling_products, cum_weights=prod_cum)[0]
        cid = rng.choices(ordering_customers, cum_weights=cust_cum)[0]
        rating = rng.choices([1, 2, 3, 4, 5], [0.07, 0.08, 0.15, 0.32, 0.38])[0]
        rdate = _rand_date(rng)
        votes = int(rng.expovariate(0.25))
        reviews.append((rid, pid, cid, rating, rdate, votes))
    tables["reviews"] = reviews

    return Dataset(scale=scale, seed=seed, tables=tables)
