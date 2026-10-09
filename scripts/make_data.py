"""Generate the small sample dataset used by the demo and the eval harness.

The data is synthetic and deterministic (fixed random seed), so every run of the
eval harness sees exactly the same tables.

    python scripts/make_data.py
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

CITIES = ["Vijayawada", "Hyderabad", "Bengaluru", "Chennai", "Mumbai", "Pune", "Visakhapatnam"]
SEGMENTS = ["Standard", "Premium"]
FIRST = ["Aarav", "Diya", "Kiran", "Meera", "Rohan", "Sneha", "Arjun", "Lakshmi", "Vikram", "Ananya",
         "Rahul", "Priya", "Sai", "Kavya", "Nikhil", "Divya", "Harsha", "Pooja", "Teja", "Swathi"]
LAST = ["Reddy", "Rao", "Sharma", "Iyer", "Naidu", "Patel", "Kumar", "Menon", "Varma", "Gupta"]

PRODUCTS = [
    ("Wireless Mouse", "Electronics", 799), ("Mechanical Keyboard", "Electronics", 3499),
    ("USB-C Hub", "Electronics", 1899), ("Noise Cancelling Headphones", "Electronics", 8999),
    ("27in Monitor", "Electronics", 15999), ("Webcam HD", "Electronics", 2499),
    ("Office Chair", "Furniture", 7499), ("Standing Desk", "Furniture", 18999),
    ("Bookshelf", "Furniture", 4599), ("Desk Lamp", "Furniture", 1299),
    ("Notebook Pack", "Stationery", 349), ("Gel Pens (10)", "Stationery", 199),
    ("Planner 2025", "Stationery", 599), ("Sticky Notes", "Stationery", 149),
    ("Python Crash Course", "Books", 1499), ("Deep Learning Book", "Books", 2999),
    ("Clean Code", "Books", 1799), ("Designing Data-Intensive Apps", "Books", 2499),
    ("Water Bottle", "Lifestyle", 699), ("Backpack", "Lifestyle", 2299),
]


def write(name, header, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


customers = []
for cid in range(1, 61):
    customers.append([
        cid,
        f"{random.choice(FIRST)} {random.choice(LAST)}",
        random.choice(CITIES),
        (date(2024, 1, 1) + timedelta(days=random.randint(0, 500))).isoformat(),
        random.choices(SEGMENTS, weights=[3, 1])[0],
    ])
write("customers.csv", ["customer_id", "name", "city", "signup_date", "segment"], customers)

products = [[i + 1, n, c, p] for i, (n, c, p) in enumerate(PRODUCTS)]
write("products.csv", ["product_id", "name", "category", "price"], products)

# Customers 51-60 never order, so "customers with no orders" is a real question.
orders = []
for oid in range(1, 601):
    orders.append([
        oid,
        random.randint(1, 50),
        random.randint(1, len(PRODUCTS)),
        random.choices([1, 2, 3, 4, 5], weights=[50, 25, 12, 8, 5])[0],
        (date(2024, 7, 1) + timedelta(days=random.randint(0, 456))).isoformat(),
        random.choices(["completed", "cancelled", "returned"], weights=[82, 11, 7])[0],
    ])
write("orders.csv", ["order_id", "customer_id", "product_id", "quantity", "order_date", "status"], orders)

print(f"Wrote {len(customers)} customers, {len(products)} products, {len(orders)} orders to {OUT}")
