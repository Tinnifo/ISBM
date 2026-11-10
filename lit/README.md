# lit/

Literature for the ISBM project. No Zotero: this folder (and Tinni's vault) is the library.

- `papers.bib`: BibTeX, one entry per work, keyed `firstauthorYEARword`. Each entry
  carries `doi`, `eprint` (arXiv) and/or `openalex` fields.
- `notes/FEW-123-<slug>.md`: one note per issue: the query, the hits, and which works
  ground which hypothesis.

## Rule

Every hypothesis cites **≥ 2 works by ID** (DOI / arXiv / OpenAlex), **≥ 1 from the last 3 years**.

## Search

```bash
# key: export OPENALEX_API_KEY=...  or  ~/.config/openalex/api_key (chmod 600). Never commit it.
python3 tools/lit_search.py "infinite relational model collapsed variational Bayes" \
    --issue FEW-365 -n 10            # OpenAlex + arXiv, writes lit/notes + lit/papers.bib
python3 tools/lit_search.py "stochastic block model variational" --source arxiv --since 2023
python3 tools/lit_search.py "..." --dry-run   # print only
```

PDFs are not committed (copyright/size); link them by DOI/arXiv instead.
