"""Explicit LLM provider selection; OpenRouter Flash defaults, no model fallback."""
import os
import json
import time
import random
from pathlib import Path
import requests
from tetra.runtime.cost_budget import from_environment, request_reserve, PRICE_CEILING, BudgetExceeded

def read_api_key(variable, key_file=None):
    """Resolve one explicitly configured credential without searching the checkout."""
    key = os.environ.get(variable, "").strip()
    if key:
        return key
    configured_file = key_file or os.environ.get(f"{variable}_FILE")
    if configured_file:
        key = Path(configured_file).expanduser().read_text().strip()
        if key:
            return key
    raise ValueError(f"Set {variable} or provide a nonempty key file via {variable}_FILE.")


class ModelServiceError(RuntimeError):
    """Allowlisted diagnostics only: never credentials, prompts or response bodies."""
    def __init__(self, details):
        self.details = details
        super().__init__('Model request failed: ' + json.dumps(details, sort_keys=True))


class OpenRouterClient:
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    provider_name = "openrouter"
    price_ceiling = PRICE_CEILING

    def prepare_body(self, body):
        return body

    def normalize_usage(self, usage):
        return usage

    def __init__(self,key_file=None,model=None):
        self._key = read_api_key('OPENROUTER_API_KEY', key_file)
        self.model = model or os.environ.get('TETRA_LLM_MODEL', 'deepseek/deepseek-v4-flash')
        self.effort=os.environ.get('TETRA_REASONING_EFFORT','high')
        if self.effort not in ('low','medium','high','xhigh','max'):raise ValueError('Unsupported reasoning effort')
        self.max_tokens=int(os.environ.get('TETRA_LLM_MAX_TOKENS','8192'))
        if not 1024<=self.max_tokens<=32768:raise ValueError('LLM max tokens must be 1024..32768')
    def complete(self,messages,tools):
        body={'model':self.model,'messages':messages,'max_tokens':self.max_tokens,'stream':False,
              'reasoning':{'effort':self.effort},'provider':{'require_parameters':True}}
        if tools:body.update(tools=tools,tool_choice='required')
        body = self.prepare_body(body)
        budget = from_environment()
        if budget and self.provider_name == 'openrouter':
            body['provider']['max_price'] = PRICE_CEILING.copy()
        payload_bytes, reservation = request_reserve(body, self.price_ceiling)
        events = []
        transient = {408, 429, 500, 502, 503, 504, 529}
        for attempt in range(1, 5):
            ticket = budget.reserve(reservation) if budget else None
            start = time.monotonic()
            details = {'attempts': attempt, 'retryable': False}
            retry_after = 0.0
            try:
                response = requests.post(self.endpoint,
                                         headers={'Authorization': 'Bearer ' + self._key},
                                         json=body, timeout=(10, 180))
                details['http_status'] = response.status_code
                try:
                    data = response.json()
                except (ValueError, TypeError, AttributeError):
                    data = None
                if response.status_code != 200:
                    details.update(category='http_error', retryable=response.status_code in transient)
                elif isinstance(data, dict) and data.get('error'):
                    error = data['error']
                    code = error.get('code') if isinstance(error, dict) else None
                    code = code if type(code) is int else None
                    details.update(category='provider_error', provider_error_code=code,
                                   retryable=code in transient)
                else:
                    try:
                        choice = data['choices'][0]
                        raw = choice['message']
                        if not isinstance(raw, dict) or raw.get('role') != 'assistant':
                            raise ValueError()
                        message = {key: value for key, value in raw.items()
                                   if key in ('role', 'content', 'tool_calls', 'reasoning_details') or (key == 'reasoning_content' and self.provider_name == 'deepseek')}
                        usage = data.get('usage')
                        if usage is None:
                            usage = {}
                        if not isinstance(usage, dict):
                            raise ValueError()
                        usage = self.normalize_usage(usage)
                        if budget:
                            budget.settle(ticket, usage.get('cost'))
                        details.update(category='success', elapsed_seconds=time.monotonic() - start,
                                       payload_bytes=payload_bytes, reserved_usd=reservation if budget else None)
                        events.append(details.copy())
                        self._record(details)
                        return {'message': message, 'usage': usage, 'model': data.get('model', self.model),
                                'provider': data.get('provider', self.provider_name), 'finish_reason': choice.get('finish_reason'),
                                'transport_attempts': events}
                    except (ValueError, KeyError, IndexError, TypeError):
                        details.update(category='invalid_completion', retryable=False)
                try:
                    retry_after = min(30.0, max(0.0, float(getattr(response, 'headers', {}).get('Retry-After', 0))))
                except (ValueError, TypeError):
                    pass
            except requests.RequestException as error:
                details.update(category='network_error', exception_type=type(error).__name__,
                               retryable=isinstance(error, (requests.Timeout, requests.ConnectionError)))
            if budget:
                # Rejected authentication/credit/validation/rate-limit requests have no generation.
                rejected = details.get('http_status') in (400, 401, 402, 403, 404, 422, 429)
                budget.settle(ticket, 0.0 if rejected else None)
            details['elapsed_seconds'] = time.monotonic() - start
            events.append(details.copy())
            self._record(details)
            if not details['retryable'] or attempt == 4:
                raise ModelServiceError(dict(details, transport_attempts=events)) from None
            time.sleep(max(retry_after, 2 ** attempt + random.uniform(0, 1)))

    def _record(self, event):
        path = os.environ.get('TETRA_LLM_EVENT_LOG')
        if path:
            try:
                with Path(path).open('a') as stream:
                    stream.write(json.dumps(dict(event, epoch=time.time(), model=self.model)) + '\n')
            except OSError:
                # Returned transport diagnostics remain authoritative if telemetry cannot be written.
                pass


class DeepSeekClient(OpenRouterClient):
    """Native thinking/tool protocol with the same retry and budget safeguards."""
    endpoint = 'https://api.deepseek.com/chat/completions'
    provider_name = 'deepseek'
    price_ceiling = {'prompt': 0.30, 'completion': 1.20, 'request': 0}

    def __init__(self, key_file=None, model=None):
        self._key = read_api_key('DEEPSEEK_API_KEY', key_file)
        self.model = model or os.environ.get('TETRA_LLM_MODEL', 'deepseek-flash')
        if self.model != 'deepseek-flash':
            raise BudgetExceeded('Native DeepSeek cost profile is validated only for deepseek-flash.')
        self.effort = os.environ.get('TETRA_REASONING_EFFORT', 'high')
        if self.effort not in ('low', 'high', 'max'):
            raise ValueError('DeepSeek reasoning effort must be low, high or max.')
        self.max_tokens = int(os.environ.get('TETRA_LLM_MAX_TOKENS', '8192'))
        if not 1024 <= self.max_tokens <= 32768:
            raise ValueError('LLM max tokens must be 1024..32768')

    def prepare_body(self, body):
        body.pop('provider', None)
        body.pop('reasoning', None)
        body.pop('tool_choice', None)
        body.update(thinking={'type': 'enabled'}, reasoning_effort=self.effort)
        return body

    def normalize_usage(self, usage):
        usage = dict(usage)
        prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
        if type(prompt) is int and type(completion) is int and min(prompt, completion) >= 0:
            # Use cache-miss peak prices for every token: conservative, not an actual invoice.
            usage['cost'] = (prompt * self.price_ceiling['prompt']
                             + completion * self.price_ceiling['completion']) / 1_000_000
            usage['cost_basis'] = 'deepseek_published_peak_rate_upper_estimate'
            usage['price_checked_date'] = '2026-11-17'
        else:
            usage.pop('cost', None)
            usage['cost_basis'] = 'unknown_reserved'
        return usage


def create_client():
    provider = os.environ.get('TETRA_LLM_PROVIDER', 'openrouter').lower()
    if provider == 'openrouter':
        return OpenRouterClient()
    if provider == 'deepseek':
        return DeepSeekClient()
    raise ValueError('Unknown LLM provider; use openrouter or deepseek')


def bind_session_model(state):
    """Prevent implicit continuation with a different model or reasoning protocol."""
    provider = os.environ.get('TETRA_LLM_PROVIDER', 'openrouter').lower()
    config = {'provider': provider,
              'model': os.environ.get('TETRA_LLM_MODEL', 'deepseek-flash' if provider == 'deepseek' else 'deepseek/deepseek-v4-flash'),
              'reasoning_effort': os.environ.get('TETRA_REASONING_EFFORT', 'high'),
              'max_tokens': int(os.environ.get('TETRA_LLM_MAX_TOKENS', '8192'))}
    previous = state.get('llm_config')
    if previous is None and state.get('llm_calls', 0):
        raise ValueError('Legacy model session cannot migrate implicitly; start a new comparison run.')
    if previous is not None and previous != config:
        raise ValueError('Session model configuration changed; start a new comparison run.')
    state['llm_config'] = config
