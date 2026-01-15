
## Runtime

time_utc: 2026-01-15T23:25:29.298704Z
python: 3.10.19
platform: Linux-6.12.13-x86_64-with-glibc2.39

## Environment presence (no secrets)

CROSSREF_MAILTO: false
OPENALEX_MAILTO: false
OPENALEX_API_KEY: false
UNPAYWALL_EMAIL: false

## DNS probes

$ getent hosts api.crossref.org
54.210.207.171  api.crossref.org
54.209.43.164   api.crossref.org
3.211.170.214   api.crossref.org
52.87.33.164    api.crossref.org
$ getent hosts api.openalex.org
2606:4700:10::6814:1ae5 api.openalex.org
2606:4700:10::ac42:9f88 api.openalex.org
$ getent hosts api.unpaywall.org
35.71.145.101   api.unpaywall.org
75.2.97.79      api.unpaywall.org
99.83.151.71    api.unpaywall.org
13.248.132.87   api.unpaywall.org

## Crossref probes

$ curl -sS -L -D - -o /dev/null -w HTTP_STATUS:%{http_code}
URL_EFFECTIVE:%{url_effective}
 https://api.crossref.org/works?query=fe-b&rows=1&mailto=
HTTP/1.1 403 Forbidden
content-length: 16
content-type: text/plain
date: Thu, 15 Jan 2026 23:25:29 GMT
server: envoy
connection: close

HTTP_STATUS:000
URL_EFFECTIVE:https://api.crossref.org/works?query=fe-b&rows=1&mailto=
curl: (56) CONNECT tunnel failed, response 403
$ curl -sS -L -D - -o /dev/null -w HTTP_STATUS:%{http_code}
URL_EFFECTIVE:%{url_effective}
 https://api.crossref.org/works?query=fe-b&rows=1
HTTP/1.1 403 Forbidden
content-length: 16
content-type: text/plain
date: Thu, 15 Jan 2026 23:25:30 GMT
server: envoy
connection: close

HTTP_STATUS:000
URL_EFFECTIVE:https://api.crossref.org/works?query=fe-b&rows=1
curl: (56) CONNECT tunnel failed, response 403

## OpenAlex probes

$ curl -sS -L -D - -o /dev/null -w HTTP_STATUS:%{http_code}
URL_EFFECTIVE:%{url_effective}
 https://api.openalex.org/works?search=fe-b&per-page=1&mailto=
HTTP/1.1 403 Forbidden
content-length: 16
content-type: text/plain
date: Thu, 15 Jan 2026 23:25:30 GMT
server: envoy
connection: close

HTTP_STATUS:000
URL_EFFECTIVE:https://api.openalex.org/works?search=fe-b&per-page=1&mailto=
curl: (56) CONNECT tunnel failed, response 403

## Interpretation

Both Crossref and OpenAlex probes return `HTTP/1.1 403 Forbidden` with `server: envoy` and `CONNECT tunnel failed`, which indicates the outbound proxy/WAF is blocking these domains in this cloud environment. This appears independent of `mailto` and without custom headers, so the block is not fixable in-code here. Use a local environment with outbound access or provide a manual candidates list to continue.
