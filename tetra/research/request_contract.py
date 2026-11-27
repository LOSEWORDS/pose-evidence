"""Persist and enforce the requested number of deliverable structures."""


def set_delivery_requirement(state, count, source_quote):
    if type(count) is not int or not 1 <= count <= 25:
        raise ValueError('Delivery count must be between 1 and 25.')
    if not isinstance(source_quote, str) or not source_quote.strip() or source_quote not in state['goal']:
        raise ValueError('Quote the explicit delivery request verbatim.')
    requirement = {'count': count, 'source_quote': source_quote}
    if state.get('delivery_requirement', requirement) != requirement:
        raise ValueError('Delivery requirement is already recorded.')
    state['delivery_requirement'] = requirement
    return requirement


def cap_selection(state, selection):
    requirement = state.get('delivery_requirement')
    if not requirement:
        return selection
    ids = selection['selected_ids']
    count = requirement['count']
    return {**selection, 'selected_ids': ids[:count],
            'alternative_ids': ids[count:] + selection.get('alternative_ids', []),
            'requested_delivery_count': count}
