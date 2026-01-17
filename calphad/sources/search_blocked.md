# Literature search blocked

Both Crossref/OpenAlex searches failed or were blocked. Details: crossref: <urlopen error Tunnel connection failed: 403 Forbidden>, openalex: <urlopen error Tunnel connection failed: 403 Forbidden>

Evidence (2026-01-15):
- curl to api.crossref.org returned "HTTP/1.1 403 Forbidden" with server=envoy and "CONNECT tunnel failed".
- curl to api.openalex.org returned "HTTP/1.1 403 Forbidden" with server=envoy and "CONNECT tunnel failed".

Suggestion: run calphad/tools/lit_search.py locally with outbound access and set CROSSREF_MAILTO / OPENALEX_MAILTO / OPENALEX_API_KEY as needed, or provide a manual candidates list to the pipeline with --candidates calphad/sources/manual_candidates.json.
