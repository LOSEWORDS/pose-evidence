import tempfile
import unittest
from pathlib import Path


class CostBudgetTests(unittest.TestCase):
    def test_pending_requests_reserve_shared_budget(self):
        from tetra.runtime.cost_budget import CostBudget, BudgetExceeded
        with tempfile.TemporaryDirectory() as root:
            a = CostBudget(Path(root)/'budget.json', 'a', 0.5, 0.5)
            b = CostBudget(Path(root)/'budget.json', 'b', 0.5, 0.5)
            ticket = a.reserve(0.3)
            with self.assertRaises(BudgetExceeded):
                b.reserve(0.3)
            a.settle(ticket, 0.1)
            b.reserve(0.3)

    def test_unknown_charge_keeps_reservation(self):
        from tetra.runtime.cost_budget import CostBudget, BudgetExceeded
        with tempfile.TemporaryDirectory() as root:
            budget = CostBudget(Path(root)/'budget.json', 'a', 0.5, 1)
            ticket = budget.reserve(0.4)
            budget.settle(ticket, None)
            with self.assertRaises(BudgetExceeded):
                budget.reserve(0.2)

    def test_oversized_request_stops_before_network(self):
        from tetra.runtime.cost_budget import request_reserve, BudgetExceeded
        with self.assertRaises(BudgetExceeded):
            request_reserve({'messages': [{'content': 'x'*350001}], 'max_tokens': 8192})

    def test_reported_overrun_blocks_future_requests(self):
        from tetra.runtime.cost_budget import CostBudget, BudgetExceeded
        with tempfile.TemporaryDirectory() as root:
            budget = CostBudget(Path(root)/'budget.json', 'a', 2, 10)
            ticket = budget.reserve(0.1)
            budget.settle(ticket, 0.2)
            with self.assertRaises(BudgetExceeded):
                budget.reserve(0.1)

    def test_budget_cannot_be_bypassed_by_alternative_provider(self):
        import os
        from unittest.mock import patch
        from tetra.runtime.llm import create_client
        from tetra.runtime.cost_budget import BudgetExceeded
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
                'TETRA_LLM_PROVIDER': 'unsupported', 'TETRA_COST_LEDGER': str(Path(root)/'budget.json')}):
            with self.assertRaises(ValueError):
                create_client()

    def test_budgeted_request_enforces_router_prices_and_records_actual_cost(self):
        import os
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        from tetra.runtime.llm import OpenRouterClient
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
                'OPENROUTER_API_KEY': 'test-only', 'TETRA_COST_LEDGER': str(Path(root)/'budget.json'),
                'TETRA_COST_CASE_ID': 'test'}, clear=True):
            response = SimpleNamespace(status_code=200, headers={}, json=lambda: {
                'choices': [{'message': {'role': 'assistant', 'content': 'ok'}}], 'usage': {'cost': 0.01}})
            with patch('tetra.runtime.llm.requests.post', return_value=response) as post:
                OpenRouterClient().complete([], [])
            self.assertEqual(post.call_args.kwargs['json']['provider']['max_price'],
                             {'prompt': 5, 'completion': 20, 'request': 0})
            data = json.loads((Path(root)/'budget.json').read_text())
            self.assertEqual(sum(row['accounted_usd'] for row in data['entries'].values()), 0.01)
