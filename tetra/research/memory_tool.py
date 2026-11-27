"""Wire contract for updating working memory, shared by scientist runners."""

TEXT = {'type': 'string'}
REFS = {'type': 'array', 'items': TEXT}
INTERPRETATION = {
    'type': 'object',
    'properties': {'claim': TEXT, 'evidence_ids': REFS,
                   'support_scope': {'type': 'string', 'enum': ['scientific', 'workflow', 'execution'], 'description': 'Default scientific. Workflow facts concern planning, selection or audit completion, not chemical/physical validity.'},
                   'status': {'type': 'string', 'enum': ['untested', 'supported', 'contradicted', 'unresolved']}},
    'required': ['claim', 'evidence_ids', 'status'], 'additionalProperties': False,
}
LESSON = {
    'type': 'object', 'properties': {'claim': TEXT, 'scope': TEXT, 'evidence_ids': REFS,
        'support_scope': {'type': 'string', 'enum': ['scientific', 'workflow', 'execution']}},
    'required': ['claim', 'scope', 'evidence_ids'], 'additionalProperties': False,
}
MEMORY_PARAMETERS = {'working_memory': TEXT,
                      'hypotheses': {'type': 'array', 'items': INTERPRETATION},
                      'lessons': {'type': 'array', 'items': LESSON}, 'evidence_ids': REFS}

MEMORY_DESCRIPTION = 'Integrate feedback into the Lead Board. Cite every new evidence ID. Supply the full current hypotheses and lessons; previous versions remain archived. Lessons are optional and session-scoped.'

def memory_tool_schema() -> dict:
    return {'type': 'function', 'function': {
        'name': 'update_memory', 'description': MEMORY_DESCRIPTION,
        'parameters': {'type': 'object', 'properties': MEMORY_PARAMETERS,
                       'required': list(MEMORY_PARAMETERS), 'additionalProperties': False},
    }}
