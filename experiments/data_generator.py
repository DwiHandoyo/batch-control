"""
Data Generator - Generates load on PostgreSQL for system identification experiments.

Inserts and updates order data at controlled rates to create measurable load
on the CQRS pipeline.
"""

import os
import sys
import time
import json
import random
import string
import logging
import argparse
import psycopg2
from datetime import datetime
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data-generator')

PRODUCT_NAMES = [
    'Mechanical Keyboard', 'Wireless Mouse', 'USB-C Hub', 'Monitor Stand',
    'Webcam HD', 'Headset Gaming', 'Laptop Stand', 'Cable Organizer',
    'Desk Lamp', 'Mouse Pad XL', 'External SSD', 'Power Bank',
    'Screen Protector', 'Phone Holder', 'Bluetooth Speaker',
]

STATUSES = ['pending', 'confirmed', 'processing', 'shipped', 'delivered', 'completed', 'cancelled']

CITIES = [
    'Jakarta', 'Bandung', 'Surabaya', 'Yogyakarta', 'Semarang',
    'Medan', 'Makassar', 'Denpasar', 'Malang', 'Bogor',
]


def random_order() -> dict:
    """Generate a random order row."""
    first_names = ['Andi', 'Budi', 'Citra', 'Dewi', 'Eko', 'Fira', 'Gani', 'Hana', 'Irfan', 'Joko']
    last_names = ['Pratama', 'Santoso', 'Wijaya', 'Kusuma', 'Hidayat', 'Saputra', 'Lestari', 'Nugroho']

    first = random.choice(first_names)
    last = random.choice(last_names)
    customer_name = f"{first} {last}"
    customer_email = f"{first.lower()}.{last.lower()}{random.randint(1,999)}@example.com"

    product_name = random.choice(PRODUCT_NAMES)
    quantity = random.randint(1, 10)
    unit_price = round(random.uniform(25000, 500000), 2)
    total_price = round(unit_price * quantity, 2)
    status = random.choice(STATUSES)

    city = random.choice(CITIES)
    shipping_address = f"Jl. {''.join(random.choices(string.ascii_uppercase, k=1))}. {''.join(random.choices(string.ascii_lowercase, k=8)).title()} No. {random.randint(1, 200)}, {city}"

    metadata = json.dumps({
        "source": random.choice(["web", "mobile", "api"]),
        "priority": random.choice(["low", "normal", "high"]),
        "notes": ''.join(random.choices(string.ascii_letters + ' ', k=random.randint(10, 50))).strip(),
    })

    return {
        'customer_name': customer_name,
        'customer_email': customer_email,
        'product_name': product_name,
        'quantity': quantity,
        'unit_price': unit_price,
        'total_price': total_price,
        'status': status,
        'shipping_address': shipping_address,
        'metadata': metadata,
    }


def insert_batch(cursor, batch_size: int, **kwargs):
    """Insert a batch of orders."""
    values = []
    params = []
    for _ in range(batch_size):
        order = random_order()
        values.append("(%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)")
        params.extend([
            order['customer_name'], order['customer_email'], order['product_name'],
            order['quantity'], order['unit_price'], order['total_price'],
            order['status'], order['shipping_address'], order['metadata'],
        ])

    sql = (
        "INSERT INTO orders (customer_name, customer_email, product_name, "
        "quantity, unit_price, total_price, status, shipping_address, metadata) VALUES "
        + ','.join(values)
    )
    cursor.execute(sql, params)


def update_batch(cursor, batch_size: int, **kwargs) -> int:
    """
    Update random existing orders.
    Returns actual number of rows updated.
    """
    cursor.execute(
        "SELECT id FROM orders ORDER BY random() LIMIT %s",
        (batch_size,)
    )
    rows = cursor.fetchall()

    if not rows:
        return 0

    for (row_id,) in rows:
        order = random_order()
        cursor.execute(
            "UPDATE orders SET "
            "customer_name = %s, customer_email = %s, product_name = %s, "
            "quantity = %s, unit_price = %s, total_price = %s, "
            "status = %s, shipping_address = %s, metadata = %s::jsonb "
            "WHERE id = %s",
            (
                order['customer_name'], order['customer_email'], order['product_name'],
                order['quantity'], order['unit_price'], order['total_price'],
                order['status'], order['shipping_address'], order['metadata'],
                row_id,
            )
        )

    return len(rows)


def run_constant_rate(conn, rate: int, duration_sec: int, **kwargs):
    """Insert at a constant rate (rows/second) for a given duration."""
    logger.info(f"Constant rate: {rate} rows/sec for {duration_sec}s")
    cursor = conn.cursor()

    start = time.time()
    total_inserted = 0

    while time.time() - start < duration_sec:
        batch_start = time.time()

        insert_batch(cursor, rate)
        conn.commit()
        total_inserted += rate

        elapsed = time.time() - batch_start
        sleep_time = max(0, 1.0 - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    cursor.close()
    logger.info(f"Inserted {total_inserted} rows in {time.time() - start:.1f}s")
    return total_inserted


def run_step_profile(conn, steps: list, **kwargs):
    """
    Run a step profile: list of (rate, duration_sec) tuples.
    Example: [(10, 30), (100, 30), (10, 30)] = low-high-low
    """
    logger.info(f"Step profile with {len(steps)} steps")
    total = 0
    for i, (rate, duration) in enumerate(steps):
        logger.info(f"Step {i+1}/{len(steps)}: rate={rate} rows/sec, duration={duration}s")
        total += run_constant_rate(conn, rate, duration)
    logger.info(f"Step profile complete. Total inserted: {total}")
    return total


def run_mixed_rate(conn, rate: int, duration_sec: int, update_ratio: float = 0.5, **kwargs):
    """
    Run mixed INSERT + UPDATE at a constant total rate.

    Args:
        rate: Total operations per second (INSERT + UPDATE combined)
        duration_sec: Duration in seconds
        update_ratio: Proportion of updates (0.0 = all INSERT, 1.0 = all UPDATE)
    """
    update_count = max(0, int(rate * update_ratio))
    insert_count = rate - update_count
    logger.info(f"Mixed rate: {rate} ops/sec ({insert_count} INSERT + {update_count} UPDATE) for {duration_sec}s")
    cursor = conn.cursor()

    start = time.time()
    total_inserted = 0
    total_updated = 0

    while time.time() - start < duration_sec:
        batch_start = time.time()

        # INSERT portion
        if insert_count > 0:
            insert_batch(cursor, insert_count)
            total_inserted += insert_count

        # UPDATE portion
        if update_count > 0:
            actually_updated = update_batch(cursor, update_count)
            total_updated += actually_updated
            # If not enough rows to update, insert the remainder
            if actually_updated < update_count:
                shortfall = update_count - actually_updated
                insert_batch(cursor, shortfall)
                total_inserted += shortfall

        conn.commit()

        elapsed = time.time() - batch_start
        sleep_time = max(0, 1.0 - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    cursor.close()
    total_time = time.time() - start
    logger.info(f"Mixed complete. Inserted {total_inserted}, Updated {total_updated} in {total_time:.1f}s")
    return total_inserted + total_updated


def run_ramp_profile(conn, start_rate: int, end_rate: int, duration_sec: int, **kwargs):
    """Linearly ramp from start_rate to end_rate over duration."""
    logger.info(f"Ramp: {start_rate} -> {end_rate} rows/sec over {duration_sec}s")
    cursor = conn.cursor()

    start = time.time()
    total_inserted = 0

    while time.time() - start < duration_sec:
        elapsed = time.time() - start
        progress = elapsed / duration_sec
        current_rate = int(start_rate + (end_rate - start_rate) * progress)
        current_rate = max(1, current_rate)

        batch_start = time.time()
        insert_batch(cursor, current_rate)
        conn.commit()
        total_inserted += current_rate

        batch_elapsed = time.time() - batch_start
        sleep_time = max(0, 1.0 - batch_elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    cursor.close()
    logger.info(f"Ramp complete. Inserted {total_inserted} rows in {time.time() - start:.1f}s")
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description='Data Generator for System Identification')
    parser.add_argument('--host', default=os.getenv('POSTGRES_HOST', 'localhost'))
    parser.add_argument('--port', type=int, default=int(os.getenv('POSTGRES_PORT', 5433)))
    parser.add_argument('--db', default=os.getenv('POSTGRES_DB', 'cqrs_write'))
    parser.add_argument('--user', default=os.getenv('POSTGRES_USER', 'postgres'))
    parser.add_argument('--password', default=os.getenv('POSTGRES_PASSWORD', 'postgres'))

    subparsers = parser.add_subparsers(dest='mode', help='Generator mode')

    # Constant rate
    p_const = subparsers.add_parser('constant', help='Constant insert rate')
    p_const.add_argument('--rate', type=int, required=True, help='Rows per second')
    p_const.add_argument('--duration', type=int, required=True, help='Duration in seconds')

    # Step profile
    p_step = subparsers.add_parser('step', help='Step profile (low-high-low)')
    p_step.add_argument('--rates', type=int, nargs='+', required=True, help='Rates for each step')
    p_step.add_argument('--durations', type=int, nargs='+', required=True, help='Duration for each step')

    # Mixed INSERT + UPDATE
    p_mixed = subparsers.add_parser('mixed', help='Mixed INSERT + UPDATE at constant rate')
    p_mixed.add_argument('--rate', type=int, required=True, help='Total operations per second')
    p_mixed.add_argument('--duration', type=int, required=True, help='Duration in seconds')
    p_mixed.add_argument('--update-ratio', type=float, default=0.5,
                         help='Proportion of updates (0.0-1.0, default 0.5)')

    # Ramp
    p_ramp = subparsers.add_parser('ramp', help='Linear ramp')
    p_ramp.add_argument('--start-rate', type=int, required=True)
    p_ramp.add_argument('--end-rate', type=int, required=True)
    p_ramp.add_argument('--duration', type=int, required=True, help='Duration in seconds')

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        sys.exit(1)

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host=args.host, port=args.port, dbname=args.db,
        user=args.user, password=args.password
    )
    logger.info(f"Connected to PostgreSQL at {args.host}:{args.port}/{args.db}")

    try:
        if args.mode == 'constant':
            run_constant_rate(conn, args.rate, args.duration)
        elif args.mode == 'mixed':
            run_mixed_rate(conn, args.rate, args.duration, args.update_ratio)
        elif args.mode == 'step':
            if len(args.rates) != len(args.durations):
                logger.error("Number of rates must match number of durations")
                sys.exit(1)
            steps = list(zip(args.rates, args.durations))
            run_step_profile(conn, steps)
        elif args.mode == 'ramp':
            run_ramp_profile(conn, args.start_rate, args.end_rate, args.duration)
    finally:
        conn.close()
        logger.info("PostgreSQL connection closed")


if __name__ == '__main__':
    main()
