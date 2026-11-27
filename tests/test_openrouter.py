import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from tetra.runtime.llm import OpenRouterClient,create_client
class OpenRouterTests(unittest.TestCase):
 def test_default_provider_uses_openrouter_key_only(self):
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{},clear=True):
   key=Path(d)/'key';key.write_text('test-openrouter-secret')
   with patch.dict(os.environ,{'OPENROUTER_API_KEY_FILE':str(key),'DEEPSEEK_API_KEY':'wrong-provider'}):
    c=create_client();self.assertIsInstance(c,OpenRouterClient);self.assertEqual(c.model,'deepseek/deepseek-v4-flash');self.assertEqual(c._key,'test-openrouter-secret')
 def test_tool_protocol_reasoning_metadata_is_preserved_without_plaintext_field(self):
  with patch.dict(os.environ,{'OPENROUTER_API_KEY':'test-secret'},clear=True):c=OpenRouterClient()
  details=[{'type':'reasoning.encrypted','data':'opaque','signature':'sig'}]
  response=SimpleNamespace(status_code=200,json=lambda:{'model':c.model,'provider':'Anthropic','usage':{'total_tokens':10},'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':None,'reasoning':'not-for-report','reasoning_details':details,'tool_calls':[{'id':'t1','type':'function','function':{'name':'check','arguments':'{}'}}]}}]})
  with patch('tetra.runtime.llm.requests.post',return_value=response) as post:
   result=c.complete([{'role':'user','content':'test'}],[{'type':'function','function':{'name':'check'}}])
  url=post.call_args.args[0];body=post.call_args.kwargs['json']
  self.assertEqual(url,'https://openrouter.ai/api/v1/chat/completions');self.assertNotIn('thinking',body);self.assertEqual(body['reasoning']['effort'],'high');self.assertEqual(body['tool_choice'],'required')
  self.assertEqual(result['message']['reasoning_details'],details);self.assertNotIn('reasoning',result['message']);self.assertEqual(result['provider'],'Anthropic')
 def test_error_does_not_leak_response_or_credentials(self):
  with patch.dict(os.environ,{'OPENROUTER_API_KEY':'secret-value'},clear=True):c=OpenRouterClient()
  with patch('tetra.runtime.llm.requests.post',return_value=SimpleNamespace(status_code=401,text='secret-value')):
   with self.assertRaises(RuntimeError) as e:c.complete([],[])
  self.assertNotIn('secret-value',str(e.exception))
 def test_network_exception_is_sanitized(self):
  import requests
  with patch.dict(os.environ,{'OPENROUTER_API_KEY':'secret-value'},clear=True):c=OpenRouterClient()
  with patch('tetra.runtime.llm.requests.post',side_effect=requests.RequestException('secret-value')):
   with self.assertRaises(RuntimeError) as e:c.complete([],[])
  self.assertNotIn('secret-value',str(e.exception))
 def test_unknown_provider_fails_without_fallback(self):
  with patch.dict(os.environ,{'TETRA_LLM_PROVIDER':'invalid'},clear=True):
   with self.assertRaises(ValueError):create_client()

class UsageTests(unittest.TestCase):
 def test_null_usage_is_normalized(self):
  with patch.dict(os.environ,{'OPENROUTER_API_KEY':'test'},clear=True):c=OpenRouterClient()
  response=SimpleNamespace(status_code=200,json=lambda:{'usage':None,'choices':[{'message':{'role':'assistant','content':'ok'}}]})
  with patch('tetra.runtime.llm.requests.post',return_value=response):self.assertEqual(c.complete([],[])['usage'],{})
 def test_invalid_usage_is_reportable_error(self):
  with patch.dict(os.environ,{'OPENROUTER_API_KEY':'test'},clear=True):c=OpenRouterClient()
  response=SimpleNamespace(status_code=200,json=lambda:{'usage':'invalid','choices':[{'message':{'role':'assistant','content':'ok'}}]})
  with patch('tetra.runtime.llm.requests.post',return_value=response):
   with self.assertRaises(RuntimeError):c.complete([],[])
