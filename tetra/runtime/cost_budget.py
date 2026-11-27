"""Conservative admission control for text-only paid requests, shared across workers."""
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import time
import uuid

from tetra.runtime.download import atomic_json

MAX_REQUEST_BYTES = 350_000
PRICE_CEILING = {'prompt': 5, 'completion': 20, 'request': 0}


class BudgetExceeded(RuntimeError):
    """No request was sent because size, accounting or monetary limits blocked it."""


class RequestTooLarge(BudgetExceeded):
    """The serialized request exceeds the transport admission limit."""


def request_reserve(body: dict, prices: dict | None = None) -> tuple[int, float]:
    """UTF-8 bytes are a deliberately conservative token proxy, not a tokenizer."""
    encoded_bytes = len(json.dumps(body, ensure_ascii=True).encode('utf-8'))
    if encoded_bytes > MAX_REQUEST_BYTES:
        raise RequestTooLarge('Request exceeds 350000-byte limit; no silent truncation or API call.')
    if type(body.get('max_tokens')) is not int or body['max_tokens'] < 1:
        raise BudgetExceeded('A bounded positive output-token limit is required.')
    # Includes a large serialization margin. Price ceilings are sent to the router.
    rates = prices or PRICE_CEILING
    reserve = ((encoded_bytes + 4096) * rates['prompt']
               + body['max_tokens'] * rates['completion']) / 1_000_000
    return encoded_bytes, reserve


class CostBudget:
    def __init__(self, path: Path, case_id: str, case_limit: float = 2.0,
                 batch_limit: float = 10.0, uncapped: bool = False):
        self.path = Path(path).resolve()
        self.case_id = case_id
        self.uncapped = uncapped
        self.case_limit = float(case_limit)
        self.batch_limit = float(batch_limit)
        if not case_id or not all(math.isfinite(x) and x > 0 for x in (self.case_limit, self.batch_limit)):
            raise ValueError('Finite positive limits and a case ID are required.')
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def locked(self):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                data = json.loads(self.path.read_text())
                if data['case_limit'] != self.case_limit or data['batch_limit'] != self.batch_limit:
                    raise BudgetExceeded('Budget limits changed within an existing campaign.')
            else:
                data = {'case_limit': self.case_limit, 'batch_limit': self.batch_limit,
                        'entries': {}, 'blocked': False}
            yield data
            atomic_json(self.path, data)

    def reserve(self, amount: float) -> str:
        if not math.isfinite(amount) or amount <= 0:
            raise BudgetExceeded('Invalid cost reservation.')
        with self.locked() as data:
            total = sum(entry['accounted_usd'] for entry in data['entries'].values())
            case_total = sum(entry['accounted_usd'] for entry in data['entries'].values()
                             if entry['case_id'] == self.case_id)
            if data['blocked'] or (not self.uncapped and (total + amount > self.batch_limit or case_total + amount > self.case_limit)):
                raise BudgetExceeded('Cost budget cannot cover the next request reservation; no request sent.')
            if self.uncapped:
                data['uncapped_user_authorized'] = True
            ticket = uuid.uuid4().hex
            data['entries'][ticket] = {'case_id': self.case_id, 'reserved_usd': amount,
                                       'accounted_usd': amount, 'status': 'in_flight', 'epoch': time.time()}
        return ticket

    def settle(self, ticket: str, actual: float | None) -> None:
        with self.locked() as data:
            entry = data['entries'][ticket]
            if actual is None:
                entry['status'] = 'charge_unknown_reserved'
            elif type(actual) not in (int, float) or not math.isfinite(actual) or actual < 0:
                entry['status'] = 'charge_invalid_reserved'
                data['blocked'] = True
            else:
                entry.update(status='settled', accounted_usd=actual)
                if actual > entry['reserved_usd'] + 1e-9:
                    data['blocked'] = True
                    entry['reservation_exceeded'] = True


def from_environment() -> CostBudget | None:
    path = os.environ.get('TETRA_COST_LEDGER')
    if not path:
        return None
    return CostBudget(Path(path), os.environ.get('TETRA_COST_CASE_ID', ''),
                      float(os.environ.get('TETRA_CASE_BUDGET_USD', '2')),
                      float(os.environ.get('TETRA_BATCH_BUDGET_USD', '10')),
                      uncapped=os.environ.get('TETRA_UNCAPPED_EXPERIMENT') == '1')
