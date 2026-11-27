"""PubMed and Europe PMC retrieval with research source provenance."""
import hashlib
import re
import time
from xml.etree import ElementTree

import requests

BASE_URL = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
SOURCE_ID = re.compile(r'[A-Z0-9]+:[A-Za-z0-9_.-]+')


def _request(parameters):
    transient_statuses = {408, 429, 500, 502, 503, 504}
    for attempt in range(3):
        try:
            response = requests.get(BASE_URL, params=parameters, timeout=(5, 15))
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise
        except requests.HTTPError as exc:
            if attempt == 2 or exc.response.status_code not in transient_statuses:
                raise
        time.sleep(1 << attempt)


PUBMED_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'


def _pubmed_request(endpoint, parameters):
    response = requests.get(PUBMED_URL + endpoint, params=parameters, timeout=(5, 15))
    response.raise_for_status()
    return response


def _pubmed_records(ids):
    if not ids:
        return []
    response = _pubmed_request('efetch.fcgi', {'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'xml'})
    rows = []
    for entry in ElementTree.fromstring(response.content).findall('PubmedArticle'):
        article = entry.find('./MedlineCitation/Article')
        pmid = entry.findtext('./MedlineCitation/PMID')
        if article is None or not pmid:
            continue
        title = article.find('ArticleTitle')
        abstract = [''.join(node.itertext()) for node in article.findall('./Abstract/AbstractText')]
        doi = entry.findtext("./PubmedData/ArticleIdList/ArticleId[@IdType='doi']")
        rows.append({'source': 'MED', 'id': pmid, 'title': ''.join(title.itertext()) if title is not None else None,
                     'doi': doi, 'pubYear': article.findtext('./Journal/JournalIssue/PubDate/Year'),
                     'pmcid': entry.findtext("./PubmedData/ArticleIdList/ArticleId[@IdType='pmc']"),
                     'abstractText': '\n'.join(abstract)})
    return rows


def _search_pubmed(query, limit):
    ids = _pubmed_request('esearch.fcgi', {'db': 'pubmed', 'term': query,
                          'retmax': limit, 'sort': 'relevance', 'retmode': 'json'}).json()['esearchresult']['idlist']
    by_id = {row['id']: row for row in _pubmed_records(ids)}
    return [by_id[pmid] for pmid in ids if pmid in by_id]


def _rank_hits(rows, family_terms):
    """Prefer comparable binding structures over studies that only mention the family."""
    unique = {}
    for row in rows:
        key = (row.get('doi') or f"{row['source']}:{row['id']}").casefold()
        unique.setdefault(key, row)

    def score(row):
        title = (row.get('title') or '').casefold()
        abstract = (row.get('abstractText') or '').casefold()
        family = int(any(term in title for term in family_terms)) * 2
        family += int(any(term in abstract for term in family_terms))
        structural = ('crystal structure', 'binding mode', 'binding site',
                      'co-crystal', 'structural basis', 'bound inhibitor', 'in complex with')
        return family, sum(2 * (term in title) + (term in abstract) for term in structural)

    return sorted(unique.values(), key=score, reverse=True)


def search_literature(query, limit=5, *, broader_query=None, family_terms=(), fetch=_request):
    """Return discovery metadata. Reading a source is a separate operation."""
    if not isinstance(query, str) or not query.strip() or len(query) > 1000:
        raise ValueError('Provide a literature query of 1–1000 characters.')
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError('Literature result limit must be between 1 and 20.')
    if (not isinstance(family_terms, (list, tuple)) or len(family_terms) > 5
            or any(not isinstance(term, str) or not re.fullmatch(r"[\w -]{2,80}", term.strip())
                   for term in family_terms)):
        raise ValueError('Provide up to five protein or family names in family_terms.')
    terms = tuple(term.strip().casefold() for term in family_terms)
    family_clause = ' OR '.join(f'"{term}"' for term in terms)

    def scoped(term):
        return f'({term.strip()}) AND ({family_clause})' if terms else term.strip()

    queries = [scoped(query)]
    if broader_query is not None:
        if not isinstance(broader_query, str) or not broader_query.strip() or len(broader_query) > 1000:
            raise ValueError('Provide a broader query of 1–1000 characters.')
        if ' '.join(broader_query.casefold().split()) != ' '.join(query.casefold().split()):
            queries.append(scoped(broader_query))

    attempts = []
    size = max(limit, 20) if terms else limit
    for term in queries:
        if fetch is _request:
            rows = _search_pubmed(term, size)
        else:
            payload = {'query': term, 'pageSize': size, 'format': 'json'}
            rows = fetch(payload)['resultList']['result']
        attempts.append({'query': term, 'hit_count': len(rows)})
        if rows:
            break
    rows = _rank_hits(rows, terms) if terms else rows
    return {'query': query, 'matched_query': term if rows else None, 'attempts': attempts,
            'family_terms': list(terms), 'results': [
        {'source_id': f"{row['source']}:{row['id']}",
         'title': row.get('title'), 'doi': row.get('doi'),
         'publication_year': row.get('pubYear'),
         'has_pmc_fulltext': bool(row.get('pmcid')),
         'abstract_excerpt': (row.get('abstractText') or '')[:500]}
        for row in rows[:limit]
    ]}



def _pmc_sections(pmcid):
    if not re.fullmatch(r'PMC[0-9]+', pmcid):
        raise ValueError('Invalid PMC article ID.')
    response = _pubmed_request('efetch.fcgi', {'db': 'pmc', 'id': pmcid[3:], 'retmode': 'xml'})
    article = ElementTree.fromstring(response.content)
    body = article.find('.//body')
    if body is None:
        return []
    sections = []
    for section in body.findall('sec') or [body]:
        title = section.find('title')
        text = ' '.join(' '.join(section.itertext()).split())
        if text:
            sections.append({'section': ''.join(title.itertext()) if title is not None else 'Body',
                             'text': text})
    for node in [*body.iter('fig'), *body.iter('table-wrap')]:
        label = node.findtext('label') or node.get('id') or node.tag
        text = ' '.join(' '.join(node.itertext()).split())
        if text:
            sections.append({'section': label, 'text': text})
    return sections


def read_structure_details(source, terms):
    """Retrieve bounded, attributable passages for a structural question."""
    if (not isinstance(terms, list) or not 1 <= len(terms) <= 6
            or any(not isinstance(term, str) or not 1 <= len(term.strip()) <= 80
                   for term in terms)):
        raise ValueError('Provide one to six short structural search terms.')
    source_id = source['source_id']
    pmcid = source.get('pmcid')
    if (not pmcid and source_id.startswith('MED:')
            and source.get('fulltext_status') != 'no_pmc_fulltext_identifier'):
        try:
            records = _pubmed_records([source_id.split(':', 1)[1]])
        except (requests.RequestException, ElementTree.ParseError) as error:
            return {'source_id': source_id, 'terms': terms, 'url': source.get('url'),
                    'status': 'unavailable', 'reason': f'PubMed link lookup failed: {error}'}
        pmcid = next((row.get('pmcid') for row in records
                      if 'MED:' + row['id'] == source_id), None)
    result = {'source_id': source_id, 'terms': terms, 'pmcid': pmcid, 'url': source.get('url')}
    if not pmcid:
        return {**result, 'status': 'unavailable', 'reason': 'No accessible PMC full text. Search for a comparable structural study.'}
    try:
        sections = _pmc_sections(pmcid)
    except (requests.RequestException, ElementTree.ParseError, ValueError) as error:
        return {**result, 'status': 'unavailable', 'reason': f'PMC retrieval failed: {error}'}
    needles = [term.strip().casefold() for term in terms]
    passages = []
    for row in sections:
        text = row['text']
        matches = [text.casefold().find(term) for term in needles]
        starts = [index for index in matches if index >= 0]
        if starts:
            start = max(0, min(starts) - 250)
            excerpt = text[start:start + 950]
            passages.append((len(starts), {'section': row['section'], 'text': excerpt}))
    matches = [row for _, row in sorted(passages, key=lambda pair: -pair[0])[:5]]
    return {**result, 'url': f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/',
            'status': 'retrieved' if matches else 'no_matching_passages',
            'passages': matches}


PREDICTION_OUTCOMES = {'prediction_completed', 'prediction_failed', 'prediction_import_failed'}


def current_literature_round(observations):
    start = next((index for index in range(len(observations) - 1, -1, -1)
                  if observations[index]['action'] in PREDICTION_OUTCOMES), -1)
    return observations[start + 1:]


def pending_literature_sources(observations):
    observations = current_literature_round(observations)
    latest = next((row for row in reversed(observations) if row['action'] == 'search_literature'), {})
    read = {row.get('result', {}).get('source_id') for row in observations
            if row['action'] == 'read_literature'}
    return list(dict.fromkeys(hit['source_id'] for hit in latest.get('result', {}).get('results', [])
                              if hit['source_id'] not in read))


def pending_reference_sources(observations):
    """Return citation groups with pre-screened PDBs and no direct measurement yet."""
    recent = current_literature_round(observations)
    attempted = {(row.get('arguments') or {}).get('source_id') for row in recent
                 if row['action'] == 'read_structure_details'
                 and (row.get('arguments') or {}).get('route') == 'reference_coordinates'}
    return {hit['source_id']: hit for row in recent if row['action'] == 'search_literature'
            and (row.get('arguments') or {}).get('route') == 'sequence'
            for hit in row['result'].get('results', [])
            if hit.get('pdb_ids') and hit['source_id'] not in attempted}


def structural_followup(observations):
    """Require one targeted read after screening and a search if that read found no details."""
    recent = current_literature_round(observations)
    if any(row['action'] == 'read_structure_details'
           and row.get('result', {}).get('status') == 'retrieved'
           and any(draft.get('target_positions') for draft in row['result'].get('pocket_drafts', []))
           for row in recent):
        return None
    screened = {row.get('result', {}).get('source_id') for row in recent
                if row['action'] == 'read_literature' and row.get('result', {}).get('status') == 'retrieved'}
    if not screened:
        return None
    detailed = [(i, row['result']) for i, row in enumerate(recent)
                if row['action'] == 'read_structure_details'
                and row.get('result', {}).get('source_id') in screened]
    if not detailed:
        return 'read_structure_details', 'Use read_structure_details on a screened source before registering a prediction.'
    if any(result.get('status') == 'retrieved' for _, result in detailed):
        return None
    alternatives = recent[detailed[0][0] + 1:]
    searches = [row for row in alternatives if row['action'] == 'search_literature']
    if not searches:
        return 'search_literature', 'Search for a comparable structural study after the detailed read found no usable passage.'
    alternate_hits = {hit['source_id'] for row in searches[-1:]
                      for hit in row.get('result', {}).get('results', [])}
    previously_read = {row.get('arguments', {}).get('source_id') for row in recent[:detailed[0][0]]
                       if row['action'] == 'read_literature'}
    if (alternate_hits - previously_read and
            not any(row['action'] == 'read_literature'
                    and row.get('arguments', {}).get('source_id') in alternate_hits - previously_read
                    for row in alternatives)):
        return 'read_literature', 'Read a new hit from the comparable-structure search before registering a prediction.'
    if not alternate_hits:
        if len(recent) != len(observations):
            return None
        return 'search_literature', 'The latest search returned no usable sources. Try a different structure query.'
    new_sources = (screened - previously_read) & alternate_hits
    if not new_sources:
        if len(recent) != len(observations):
            return None
        return 'search_literature', 'The latest sources could not be screened. Try a different search.'
    if new_sources and not any(result.get('source_id') in new_sources for _, result in detailed):
        return 'read_structure_details', 'Use read_structure_details on one newly screened comparable study.'
    return None


def structural_followup_blocker(observations):
    requirement = structural_followup(observations)
    return requirement[1] if requirement else None


def read_literature(source_id, *, fetch=_request):
    """Return the abstract text supplied by the source, with provenance."""
    if not isinstance(source_id, str) or not SOURCE_ID.fullmatch(source_id):
        raise ValueError('Use a source ID returned by literature search.')
    source, article_id = source_id.split(':', 1)
    if fetch is _request and source == 'MED' and article_id.isdigit():
        rows = _pubmed_records([article_id])
    else:
        rows = fetch({'query': f'SRC:{source} AND EXT_ID:"{article_id}"',
                      'resultType': 'core', 'pageSize': 5, 'format': 'json'})['resultList']['result']
    row = next((item for item in rows
                if item.get('source') == source and item.get('id') == article_id), None)
    if row is None:
        return {'source_id': source_id, 'status': 'unavailable', 'reason': 'Source not found.'}
    abstract = row.get('abstractText')
    result = {'source_id': source_id, 'title': row.get('title'), 'doi': row.get('doi'),
              'publication_date': row.get('firstPublicationDate'),
              'url': f'https://pubmed.ncbi.nlm.nih.gov/{article_id}/' if source == 'MED' else
              f'https://europepmc.org/article/{source}/{article_id}'}
    pmcid = row.get('pmcid')
    if fetch is _request and pmcid:
        try:
            sections = _pmc_sections(pmcid)
        except (requests.RequestException, ElementTree.ParseError, ValueError):
            result['fulltext_status'] = 'retrieval_failed'
        else:
            if sections:
                ordered = sorted(sections, key=lambda item: not any(word in item['section'].lower()
                                 for word in ('result', 'structure', 'binding', 'discussion')))
                text = '\n\n'.join(section['text'] for section in ordered)
                excerpt = text[:12000]
                return {**result, 'status': 'retrieved', 'section': 'full_text_excerpt',
                        'pmcid': pmcid, 'url': f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/',
                        'text': excerpt, 'truncated': len(excerpt) < len(text),
                        'sha256': hashlib.sha256(excerpt.encode()).hexdigest()}
            result['fulltext_status'] = 'body_unavailable'
    else:
        result['fulltext_status'] = 'no_pmc_fulltext_identifier'
    if pmcid:
        result['pmcid'] = pmcid
    if not abstract:
        return {**result, 'status': 'unavailable', 'reason': 'No abstract supplied.'}
    return {**result, 'status': 'retrieved', 'section': 'abstract', 'text': abstract,
            'sha256': hashlib.sha256(abstract.encode()).hexdigest()}


def initial_search_stalled(observations):
    """Recognize two unsuccessful searches before the first prediction."""
    searches = [row.get('result', {}) for row in observations
                if row.get('action') == 'search_literature']
    return len(searches) >= 2 and all(not result.get('results') for result in searches[-2:])
