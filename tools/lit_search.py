#!/usr/bin/env python3
"""Search OpenAlex and arXiv and write results into lit/.

Stdlib only (no extra dependencies). Usage:

    python3 tools/lit_search.py "infinite relational model collapsed variational Bayes" \
        --issue FEW-365 -n 10 [--source openalex|arxiv|both] [--since 2023] [--dry-run]

    python3 tools/lit_search.py venues --days 14      # weekly venue scan (see below)

Writes:
    lit/notes/<ISSUE>-<slug>.md   a note with the query and a table of hits
    lit/papers.bib                BibTeX entries (appended, deduplicated by DOI/arXiv/OpenAlex ID)

The OpenAlex API key is read from $OPENALEX_API_KEY or ~/.config/openalex/api_key.
It is never written to disk by this script and must never be committed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIT = ROOT / "lit"
BIB = LIT / "papers.bib"
NOTES = LIT / "notes"
UA = "ISBM-lit-search/0.1 (https://github.com/Tinnifo/ISBM; mailto:tinnifo@users.noreply.github.com)"


def api_key() -> str | None:
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        return key.strip()
    p = Path.home() / ".config" / "openalex" / "api_key"
    if p.exists():
        return p.read_text().strip() or None
    return None


def http_get(url: str, retries: int = 2) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed: {redact(url)}: {last}")


def redact(url: str) -> str:
    return re.sub(r"(api_key=)[^&]+", r"\1***", url)


# ---------------------------------------------------------------- OpenAlex
ARXIV_SOURCE = "S4306400194"  # OpenAlex source ID of arXiv


def openalex(query: str, n: int, since: int | None, arxiv_only: bool = False) -> list[dict]:
    params = {"search": query, "per-page": str(n),
              "select": "id,doi,title,display_name,publication_year,authorships,primary_location,ids,cited_by_count,type"}
    filters = []
    if since:
        filters.append(f"from_publication_date:{since}-01-01")
    if arxiv_only:
        filters.append(f"primary_location.source.id:{ARXIV_SOURCE}")
    if filters:
        params["filter"] = ",".join(filters)
    key = api_key()
    if key:
        params["api_key"] = key
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    data = json.loads(http_get(url))
    out = []
    for w in data.get("results", []):
        loc = w.get("primary_location") or {}
        src = (loc.get("source") or {}).get("display_name")
        ids = w.get("ids") or {}
        arxiv = None
        for v in [loc.get("landing_page_url") or "", ids.get("arxiv") or ""]:
            m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", v)
            if m:
                arxiv = m.group(1)
        out.append({
            "source": "openalex",
            "title": w.get("display_name") or w.get("title") or "",
            "authors": [a["author"]["display_name"] for a in w.get("authorships", []) if a.get("author")],
            "year": w.get("publication_year"),
            "doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
            "openalex": (w.get("id") or "").rsplit("/", 1)[-1] or None,
            "arxiv": arxiv,
            "venue": src,
            "cited_by": w.get("cited_by_count"),
            "url": w.get("doi") or w.get("id"),
        })
    return out


# ---------------------------------------------------------------- arXiv
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def arxiv(query: str, n: int, since: int | None) -> list[dict]:
    """arXiv export API; if it is rate-limited (HTTP 429, common from shared IPs),
    fall back to arXiv-hosted works in OpenAlex."""
    try:
        return arxiv_api(query, n, since)
    except Exception as e:  # noqa: BLE001
        print(f"warning: arXiv API unavailable ({e}); using OpenAlex arXiv index", file=sys.stderr)
        res = openalex(query, n, since, arxiv_only=True)
        for p in res:
            p["source"] = "arxiv(openalex)"
        return res


def arxiv_api(query: str, n: int, since: int | None) -> list[dict]:
    terms = " AND ".join(f"all:{t}" for t in re.findall(r"\w+", query))
    params = {"search_query": terms, "start": "0", "max_results": str(n * 2 if since else n),
              "sortBy": "relevance"}
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    root = ET.fromstring(http_get(url))
    out = []
    for e in root.findall("a:entry", ATOM):
        aid = e.findtext("a:id", default="", namespaces=ATOM).rsplit("/abs/", 1)[-1]
        aid = re.sub(r"v\d+$", "", aid)
        year = int(e.findtext("a:published", default="0000", namespaces=ATOM)[:4] or 0)
        if since and year < since:
            continue
        out.append({
            "source": "arxiv",
            "title": " ".join(e.findtext("a:title", default="", namespaces=ATOM).split()),
            "authors": [a.findtext("a:name", default="", namespaces=ATOM) for a in e.findall("a:author", ATOM)],
            "year": year,
            "doi": e.findtext("arxiv:doi", default=None, namespaces=ATOM),
            "openalex": None,
            "arxiv": aid,
            "venue": "arXiv " + (e.find("arxiv:primary_category", ATOM).get("term") if e.find("arxiv:primary_category", ATOM) is not None else ""),
            "cited_by": None,
            "url": f"https://arxiv.org/abs/{aid}",
        })
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- output
def ascii_word(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z]", "", s)


STOP = {"a", "an", "the", "of", "on", "for", "and", "in", "to", "with", "via", "by"}


def bib_key(p: dict) -> str:
    last = ascii_word((p["authors"][0].split()[-1] if p["authors"] else "anon")).lower() or "anon"
    word = next((ascii_word(w).lower() for w in p["title"].split() if w.lower() not in STOP and ascii_word(w)), "x")
    return f"{last}{p['year'] or ''}{word}"


def bib_entry(p: dict, key: str) -> str:
    esc = lambda s: (s or "").replace("{", "").replace("}", "")  # noqa: E731
    fields = [("title", "{" + esc(p["title"]) + "}"), ("author", " and ".join(p["authors"][:20]) or "Anonymous"),
              ("year", str(p["year"] or ""))]
    if p.get("venue"):
        fields.append(("journal" if not p["venue"].startswith("arXiv") else "howpublished", esc(p["venue"])))
    if p.get("doi"):
        fields.append(("doi", p["doi"]))
    if p.get("arxiv"):
        fields += [("eprint", p["arxiv"]), ("archivePrefix", "arXiv")]
    if p.get("openalex"):
        fields.append(("openalex", p["openalex"]))
    if p.get("url"):
        fields.append(("url", p["url"]))
    kind = "article" if p.get("doi") else "misc"
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
    return f"@{kind}{{{key},\n{body}\n}}\n"


def existing_ids(text: str) -> set[str]:
    ids = set()
    for f in ("doi", "eprint", "openalex"):
        ids |= {m.lower() for m in re.findall(rf"{f} = \{{([^}}]+)\}}", text)}
    return ids


def merge(results: list[dict]) -> list[dict]:
    """Deduplicate across sources by DOI, arXiv ID, or normalised title."""
    seen: dict[str, dict] = {}
    for p in results:
        tkey = re.sub(r"\W", "", p["title"].lower())[:60]
        keys = [k for k in (p.get("doi") and p["doi"].lower(), p.get("arxiv"), tkey) if k]
        hit = next((seen[k] for k in keys if k in seen), None)
        if hit:
            for f in ("doi", "arxiv", "openalex", "venue", "cited_by"):
                if not hit.get(f) and p.get(f):
                    hit[f] = p[f]
            hit["source"] = "openalex+arxiv" if hit["source"] != p["source"] else hit["source"]
        else:
            hit = p
        for k in keys:
            seen[k] = hit
    uniq = []
    for p in seen.values():
        if p not in uniq:
            uniq.append(p)
    return uniq


def ident(p: dict) -> str:
    parts = []
    if p.get("doi"):
        parts.append(f"doi:{p['doi']}")
    if p.get("arxiv"):
        parts.append(f"arXiv:{p['arxiv']}")
    if p.get("openalex"):
        parts.append(f"OpenAlex:{p['openalex']}")
    return " · ".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--issue", default="FEW-000", help="Linear issue ID the search belongs to")
    ap.add_argument("-n", type=int, default=10, help="results per source")
    ap.add_argument("--source", choices=["openalex", "arxiv", "both"], default="both")
    ap.add_argument("--since", type=int, help="only works from this year on")
    ap.add_argument("--dry-run", action="store_true", help="print, don't write lit/")
    a = ap.parse_args(argv)

    results, errors = [], []
    if a.source in ("openalex", "both"):
        try:
            results += openalex(a.query, a.n, a.since)
        except Exception as e:  # noqa: BLE001
            errors.append(f"OpenAlex: {e}")
    if a.source in ("arxiv", "both"):
        try:
            results += arxiv(a.query, a.n, a.since)
        except Exception as e:  # noqa: BLE001
            errors.append(f"arXiv: {e}")
    for e in errors:
        print("warning:", e, file=sys.stderr)
    papers = merge(results)
    if not papers:
        print("no results", file=sys.stderr)
        return 1

    recent_cut = dt.date.today().year - 3
    for p in papers:
        p["key"] = bib_key(p)
        flag = " (recent)" if (p["year"] or 0) >= recent_cut else ""
        print(f"[{p['source']}] {p['year']}{flag}  {p['title'][:90]}\n    {ident(p)}  key={p['key']}")
    if a.dry_run:
        return 0

    LIT.mkdir(exist_ok=True)
    NOTES.mkdir(exist_ok=True)
    bib_text = BIB.read_text() if BIB.exists() else ""
    have = existing_ids(bib_text)
    keys_used = set(re.findall(r"^@\w+\{([^,]+),", bib_text, flags=re.M))
    added = 0
    with BIB.open("a") as f:
        for p in papers:
            ids = {x.lower() for x in (p.get("doi"), p.get("arxiv"), p.get("openalex")) if x}
            if ids & have:
                continue
            key = p["key"]
            i = 0
            while key in keys_used:
                i += 1
                key = f"{p['key']}{chr(96 + i)}"
            p["key"] = key
            keys_used.add(key)
            have |= ids
            f.write("\n" + bib_entry(p, key))
            added += 1

    slug = "-".join(re.findall(r"[a-z0-9]+", a.query.lower()))[:50]
    note = NOTES / f"{a.issue}-{slug}.md"
    today = dt.date.today().isoformat()
    lines = [f"# {a.issue}: literature search", "",
             f"- **Query:** `{a.query}`", f"- **Date:** {today}",
             f"- **Sources:** {a.source}" + (f", since {a.since}" if a.since else ""),
             f"- **Linear:** https://linear.app/tinnifo/issue/{a.issue}", "",
             f"Recent = published {recent_cut} or later (the ≥1-recent-citation rule).", "",
             "| Year | Title | Authors | IDs | Bib key | Cited by |", "|---|---|---|---|---|---|"]
    for p in sorted(papers, key=lambda p: -(p["year"] or 0)):
        au = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
        yr = f"**{p['year']}**" if (p["year"] or 0) >= recent_cut else str(p["year"])
        title = p["title"].replace("|", "/")
        lines.append(f"| {yr} | {title} | {au} | {ident(p)} | `{p['key']}` | {p['cited_by'] if p['cited_by'] is not None else ''} |")
    lines += ["", "## Relevance (fill in)", "", "- Which works ground which hypothesis, and why:", ""]
    if errors:
        lines += ["> [!warning] Search errors", *[f"> {e}" for e in errors], ""]
    with note.open("a" if note.exists() else "w") as f:
        if note.stat().st_size if note.exists() else 0:
            f.write("\n---\n\n")
        f.write("\n".join(lines))
    print(f"\nwrote {note.relative_to(ROOT)} and {added} new entr{'y' if added == 1 else 'ies'} in {BIB.relative_to(ROOT)}")
    return 0


# ================================================================ venues mode
# Weekly literature scan of leading venues (owned by Roxana, see AGENTS.md).
#
#   python3 tools/lit_search.py venues [--days 14] [--topics lit/scan_topics.json] [-n 15] [--dry-run]
#
# Sources:
#   OpenAlex  works filtered by venue source ID + from_publication_date (recent window).
#             Proceedings coverage lags badly (ICML/ICLR/AISTATS/UAI/JMLR show ~0 works for
#             2024-26 in OpenAlex as of 2026-11), so OpenAlex alone is not enough.
#   arXiv     export API, (cat:stat.ML OR cat:cs.LG) AND topic terms, submitted in the window.
#             Falls back to OpenAlex's arXiv index (S4306400194) when the API returns 429.
#   OpenReview api2 /notes/search, accepted papers (venueid) of ICLR, NeurIPS, ICML for this
#             year and last year. Static once published, so new-to-us filtering uses seen.txt.
# Every hit is scored against each active FEW hypothesis issue in lit/scan_topics.json: it
# must match >= min_core domain terms ("core") AND >= min_method method terms ("method"). Papers already
# listed in lit/scans/seen.txt are skipped, so each weekly digest only shows new papers.

VENUE_SOURCES = {  # OpenAlex source IDs (looked up 2026-11-10 via /sources?search=)
    "NeurIPS": "S4363606243",   # neural information processing systems
    "ICLR": "S4306419637",      # International Conference on Learning Representations
    "ICML": "S4306419644",      # International Conference on Machine Learning
    "AISTATS": "S4306419146",   # International Conference on Artificial Intelligence and Statistics
    "UAI": "S4306421103",       # Uncertainty in Artificial Intelligence
    "JMLR": "S118988714",       # Journal of Machine Learning Research (ISSN 1532-4435)
}
OPENREVIEW_VENUES = ["ICLR.cc/{y}/Conference", "NeurIPS.cc/{y}/Conference", "ICML.cc/{y}/Conference"]
ARXIV_CATS = ["stat.ML", "cs.LG"]
SCANS = LIT / "scans"
SEEN = SCANS / "seen.txt"
TOPICS = LIT / "scan_topics.json"


def _oa_venues(query: str, since: dt.date, n: int) -> list[dict]:
    params = {"search": query, "per-page": str(n),
              "filter": f"from_publication_date:{since.isoformat()},primary_location.source.id:" + "|".join(VENUE_SOURCES.values()),
              "select": "id,doi,display_name,publication_year,publication_date,authorships,primary_location,ids,abstract_inverted_index"}
    key = api_key()
    if key:
        params["api_key"] = key
    data = json.loads(http_get("https://api.openalex.org/works?" + urllib.parse.urlencode(params)))
    rev = {v: k for k, v in VENUE_SOURCES.items()}
    out = []
    for w in data.get("results", []):
        src = ((w.get("primary_location") or {}).get("source") or {})
        inv = w.get("abstract_inverted_index") or {}
        words = sorted(((i, t) for t, idx in inv.items() for i in idx))
        out.append({"id": f"OpenAlex:{w['id'].rsplit('/', 1)[-1]}" + (f" · doi:{w['doi'].replace('https://doi.org/', '')}" if w.get("doi") else ""),
                    "uid": w["id"].rsplit("/", 1)[-1], "title": w.get("display_name") or "",
                    "venue": rev.get(src.get("id", "").rsplit("/", 1)[-1], src.get("display_name") or "?"),
                    "date": w.get("publication_date") or str(w.get("publication_year")),
                    "abstract": " ".join(t for _, t in words), "url": w.get("doi") or w["id"], "via": "OpenAlex"})
    return out


def _arxiv_recent(terms: list[str], since: dt.date, n: int) -> list[dict]:
    cats = " OR ".join(f"cat:{c}" for c in ARXIV_CATS)
    tq = " OR ".join(f'abs:"{t}"' if " " in t else f"abs:{t}" for t in terms)
    rng = f"submittedDate:[{since:%Y%m%d}0000 TO {dt.date.today():%Y%m%d}2359]"
    params = {"search_query": f"({cats}) AND ({tq}) AND {rng}", "start": "0", "max_results": str(n),
              "sortBy": "submittedDate", "sortOrder": "descending"}
    root = ET.fromstring(http_get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)))
    out = []
    for e in root.findall("a:entry", ATOM):
        aid = re.sub(r"v\d+$", "", e.findtext("a:id", default="", namespaces=ATOM).rsplit("/abs/", 1)[-1])
        cat = e.find("arxiv:primary_category", ATOM)
        out.append({"id": f"arXiv:{aid}", "uid": aid,
                    "title": " ".join(e.findtext("a:title", default="", namespaces=ATOM).split()),
                    "venue": "arXiv " + (cat.get("term") if cat is not None else ""),
                    "date": e.findtext("a:published", default="", namespaces=ATOM)[:10],
                    "abstract": " ".join(e.findtext("a:summary", default="", namespaces=ATOM).split()),
                    "url": f"https://arxiv.org/abs/{aid}", "via": "arXiv API"})
    return out


def _arxiv_recent_fallback(query: str, since: dt.date, n: int) -> list[dict]:
    params = {"search": query, "per-page": str(n),
              "filter": f"from_publication_date:{since.isoformat()},primary_location.source.id:{ARXIV_SOURCE}",
              "select": "id,display_name,publication_date,primary_location,ids,abstract_inverted_index"}
    key = api_key()
    if key:
        params["api_key"] = key
    data = json.loads(http_get("https://api.openalex.org/works?" + urllib.parse.urlencode(params)))
    out = []
    for w in data.get("results", []):
        url = (w.get("primary_location") or {}).get("landing_page_url") or ""
        m = re.search(r"([0-9]{4}\.[0-9]{4,5})", url)
        aid = m.group(1) if m else w["id"].rsplit("/", 1)[-1]
        inv = w.get("abstract_inverted_index") or {}
        words = sorted(((i, t) for t, idx in inv.items() for i in idx))
        out.append({"id": f"arXiv:{aid}" if m else f"OpenAlex:{aid}", "uid": aid, "title": w.get("display_name") or "",
                    "venue": "arXiv", "date": w.get("publication_date") or "", "abstract": " ".join(t for _, t in words),
                    "url": f"https://arxiv.org/abs/{aid}" if m else w["id"], "via": "OpenAlex arXiv index"})
    return out


def _openreview(query: str, n: int) -> list[dict]:
    y = dt.date.today().year
    out = []
    for tmpl in OPENREVIEW_VENUES:
        for year in (y, y - 1):
            vid = tmpl.format(y=year)
            params = {"term": query, "venueid": vid, "limit": str(n), "type": "terms"}
            try:
                data = json.loads(http_get("https://api2.openreview.net/notes/search?" + urllib.parse.urlencode(params), retries=1))
            except Exception as e:  # noqa: BLE001
                print(f"warning: OpenReview {vid}: {e}", file=sys.stderr)
                continue
            for note in data.get("notes", []):
                c = note.get("content", {})
                val = lambda k: (c.get(k) or {}).get("value", "") if isinstance(c.get(k), dict) else (c.get(k) or "")  # noqa: E731
                ms = note.get("pdate") or note.get("odate") or note.get("cdate")
                out.append({"id": f"OpenReview:{note['forum']}", "uid": note["forum"], "title": val("title"),
                            "venue": val("venue") or vid, "date": dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date().isoformat() if ms else str(year),
                            "abstract": val("abstract"), "url": f"https://openreview.net/forum?id={note['forum']}", "via": "OpenReview"})
            time.sleep(0.5)
    return out


def _score(p: dict, topic: dict) -> list[str]:
    """Matched terms, or [] unless the paper hits >= min_core domain terms AND
    >= min_method method terms (generic words like 'variational' alone are not enough)."""
    text = (p["title"] + " " + p.get("abstract", "")).lower()
    core = [k for k in topic["core"] if k.lower() in text]
    meth = [k for k in topic.get("method", []) if k.lower() in text]
    if len(core) < topic.get("min_core", 1) or len(meth) < topic.get("min_method", 1):
        return []
    return core + meth


def scan_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="lit_search.py venues", description="Weekly venue scan → lit/scans/YYYY-MM-DD.md")
    ap.add_argument("--days", type=int, default=14, help="recency window for OpenAlex and arXiv (default 14)")
    ap.add_argument("--topics", default=str(TOPICS), help="JSON list of active hypothesis topics")
    ap.add_argument("-n", type=int, default=15, help="results per source and topic query")
    ap.add_argument("--no-openreview", action="store_true")
    ap.add_argument("--include-seen", action="store_true", help="don't skip papers from earlier scans")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    topics = json.loads(Path(a.topics).read_text())
    since = dt.date.today() - dt.timedelta(days=a.days)
    seen = set(SEEN.read_text().split()) if SEEN.exists() and not a.include_seen else set()
    hits: dict[str, dict] = {}
    log: list[str] = []

    def add(papers, source_name):
        log.append(f"{source_name}: {len(papers)} raw")
        for p in papers:
            hits.setdefault(p["uid"], p)

    for t in topics:
        for q in t["queries"]:
            for name, fn in (("OpenAlex venues", lambda: _oa_venues(q, since, a.n)),):
                try:
                    add(fn(), f"{name} [{t['issue']}] '{q}'")
                except Exception as e:  # noqa: BLE001
                    log.append(f"{name} '{q}': ERROR {e}")
            if not a.no_openreview:
                add(_openreview(q, max(5, a.n // 2)), f"OpenReview [{t['issue']}] '{q}'")
        try:
            add(_arxiv_recent(t["core"][:6], since, a.n * 2), f"arXiv API [{t['issue']}]")
        except Exception as e:  # noqa: BLE001
            log.append(f"arXiv API [{t['issue']}]: {e}; falling back to OpenAlex arXiv index")
            for q in t["queries"]:
                try:
                    add(_arxiv_recent_fallback(q, since, a.n), f"OpenAlex arXiv index [{t['issue']}] '{q}'")
                except Exception as e2:  # noqa: BLE001
                    log.append(f"OpenAlex arXiv index '{q}': ERROR {e2}")
        time.sleep(1)

    rows = []
    for p in hits.values():
        if p["uid"] in seen:
            continue
        best = None
        for t in topics:
            m = _score(p, t)
            if m and (best is None or len(m) > len(best[1])):
                best = (t, m)
        if best:
            rows.append((best[0], best[1], p))
    rows.sort(key=lambda r: (r[0]["issue"], -len(r[1]), r[2]["date"]), reverse=False)

    today = dt.date.today().isoformat()
    lines = [f"# Venue scan {today}", "",
             f"Owner: Roxana · window: OpenAlex/arXiv since {since.isoformat()} ({a.days} days); OpenReview: accepted ICLR/NeurIPS/ICML {dt.date.today().year - 1}–{dt.date.today().year}, new-to-us only.",
             "Venues: " + ", ".join(f"{k} ({v})" for k, v in VENUE_SOURCES.items()) + f"; arXiv {', '.join(ARXIV_CATS)}.", "",
             "Relevance lines are **auto-generated from keyword matches**; Roxana confirms or rewrites each one, then turns relevant papers into a `type/lit-review` issue or a comment on the hypothesis issue, plus a vault lab-journal entry.", ""]
    for t in topics:
        mine = [r for r in rows if r[0] is t]
        lines += [f"## {t['issue']}: {t['label']}", ""]
        if not mine:
            lines += ["_No new matching papers._", ""]
            continue
        lines += ["| ID | Venue | Date | Title | Relevance (auto) | Keep? |", "|---|---|---|---|---|---|"]
        for _, m, p in sorted(mine, key=lambda r: (-len(r[1]), r[2]["date"]), reverse=False):
            title = p["title"].replace("|", "/")
            rel = f"Matches {t['issue']} on: " + ", ".join(m[:5])
            lines.append(f"| [{p['id']}]({p['url']}) | {p['venue']} | {p['date']} | {title} | {rel} | [ ] |")
        lines.append("")
    lines += ["<details><summary>Source log</summary>", "", *[f"- {l}" for l in log], "", "</details>", ""]
    digest = "\n".join(lines)
    print(digest)
    if a.dry_run:
        return 0
    SCANS.mkdir(parents=True, exist_ok=True)
    out = SCANS / f"{today}.md"
    out.write_text(digest)
    with SEEN.open("a") as f:
        for _, _, p in rows:
            f.write(p["uid"] + "\n")
    print(f"wrote {out.relative_to(ROOT)} ({len(rows)} papers)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "venues":
        sys.exit(scan_main(sys.argv[2:]))
    sys.exit(main())
