"""Per-product fee snapshot loader (one fixed value per product; see catalog/fees_by_product_*.provenance.json)."""
import csv
import hashlib
import json
import math
from collections import namedtuple

Fee = namedtuple('Fee', 'product mult tick open_rate open_fixed close_rate close_fixed close_today_rate close_today_fixed')
_COLS = ('mult', 'tick', 'open_rate', 'open_fixed_yuan', 'close_rate', 'close_fixed_yuan',
         'close_today_rate', 'close_today_fixed_yuan')


def load_fee_table(path, expected_sha256=None):
    with open(path, 'rb') as handle:
        raw = handle.read()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Fee table hash mismatch')
    table = {}
    for row in csv.DictReader(raw.decode('utf-8').splitlines()):
        product = row['product'].strip().lower()
        if not product or product in table:
            raise ValueError('Duplicate or empty product in fee table')
        values = [float(row[c]) for c in _COLS]
        if any(not math.isfinite(v) or v < 0 for v in values) or values[0] <= 0 or values[1] <= 0:
            raise ValueError('Invalid fee row for ' + product)
        table[product] = Fee(product, *values)
    if not table:
        raise ValueError('Empty fee table')
    return table


def check_provenance(csv_path, provenance_path):
    with open(provenance_path, encoding='utf-8') as handle:
        provenance = json.load(handle)
    return load_fee_table(csv_path, provenance['sha256'])


def fee_yuan(fee, kind, price, lots=1):
    """Fee in yuan for `lots` at `price`. kind: open | close | close_today."""
    if kind not in ('open', 'close', 'close_today'):
        raise ValueError('Unknown fee kind')
    if not math.isfinite(price) or price <= 0 or lots <= 0:
        raise ValueError('Positive finite price and lots required')
    rate, fixed = {'open': (fee.open_rate, fee.open_fixed), 'close': (fee.close_rate, fee.close_fixed),
                   'close_today': (fee.close_today_rate, fee.close_today_fixed)}[kind]
    return (rate * price * fee.mult + fixed) * lots
