import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class DeepSeekCurrentTests(unittest.TestCase):
    def test_explicit_native_provider_and_thinking_protocol(self):
        from tetra.runtime.llm import create_client
        with patch.dict(os.environ, {'TETRA_LLM_PROVIDER': 'deepseek', 'DEEPSEEK_API_KEY': 'test-only'}, clear=True):
            client = create_client()
        self.assertEqual(client.model, 'deepseek-flash')
        response = SimpleNamespace(status_code=200, headers={}, json=lambda: {
            'model': 'deepseek-flash', 'choices': [{'message': {'role': 'assistant', 'content': None,
                'reasoning_content': 'opaque protocol', 'tool_calls': []}}],
            'usage': {'prompt_tokens': 1000, 'completion_tokens': 100}})
        with patch('tetra.runtime.llm.requests.post', return_value=response) as post:
            result = client.complete([], [{'type': 'function', 'function': {'name': 'test'}}])
        self.assertEqual(post.call_args.args[0], 'https://api.deepseek.com/chat/completions')
        body = post.call_args.kwargs['json']
        self.assertEqual(body['thinking'], {'type': 'enabled'})
        self.assertNotIn('provider', body)
        self.assertNotEqual(body.get('tool_choice'), 'required')
        self.assertEqual(result['message']['reasoning_content'], 'opaque protocol')
        self.assertAlmostEqual(result['usage']['cost'], 0.00042)
        self.assertEqual(result['usage']['cost_basis'], 'deepseek_published_peak_rate_upper_estimate')

class SessionModelTests(unittest.TestCase):
    def test_provider_change_cannot_resume_existing_session(self):
        from tetra.runtime.llm import bind_session_model
        state = {}
        with patch.dict(os.environ, {'TETRA_LLM_PROVIDER': 'deepseek'}, clear=True):
            bind_session_model(state)
            self.assertEqual(state['llm_config']['model'], 'deepseek-flash')
        with patch.dict(os.environ, {'TETRA_LLM_PROVIDER': 'openrouter'}, clear=True):
            with self.assertRaises(ValueError):
                bind_session_model(state)

    def test_old_usage_requires_new_session(self):
        from tetra.runtime.llm import bind_session_model
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                bind_session_model({'llm_calls': 2, 'messages': []})
