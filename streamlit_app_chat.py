from __future__ import annotations

import html
import json
import os
import re
import textwrap
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import streamlit as st

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Trade Guide", layout="wide")


# ----------------------------
# Utilities
# ----------------------------
def _norm_ws(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(s: Any) -> str:
    if s is None:
        return ""
    txt = html.unescape(str(s))
    txt = _TAG_RE.sub(" ", txt)
    return _norm_ws(txt)


def _trunc(s: str, max_chars: int) -> str:
    s = s or ""
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def _wrap(text: str, width: int, indent: str = "") -> str:
    text = _norm_ws(text)
    if not text:
        return ""
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _import_openai():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        st.error(
            "Missing dependency: openai. Install with: pip install openai\n\n"
            f"Import error: {e}"
        )
        st.stop()
    return OpenAI


def _import_elasticsearch():
    try:
        from elasticsearch import Elasticsearch  # type: ignore
    except Exception as e:
        st.error(
            "Missing dependency: elasticsearch. Install with: pip install elasticsearch\n\n"
            f"Import error: {e}"
        )
        st.stop()
    return Elasticsearch


def _import_psycopg2():
    try:
        import psycopg2  # type: ignore
    except Exception as e:
        st.error(
            "Missing dependency: psycopg2 (recommended: psycopg2-binary). "
            "Install with: pip install psycopg2-binary\n\n"
            f"Import error: {e}"
        )
        st.stop()
    return psycopg2


def _es_search(es, *, index: str, body: Dict[str, Any], request_timeout: Optional[int] = None) -> Dict[str, Any]:
    # Compatible across elasticsearch python client versions.
    try:
        if request_timeout is None:
            return es.search(index=index, body=body)  # type: ignore[arg-type]
        return es.search(index=index, body=body, request_timeout=request_timeout)  # type: ignore[arg-type]
    except TypeError:
        # Some clients may not accept request_timeout or body kw in the same way
        try:
            if request_timeout is None:
                return es.search(index=index, **body)  # type: ignore[arg-type]
            return es.search(index=index, request_timeout=request_timeout, **body)  # type: ignore[arg-type]
        except Exception:
            return es.search(index=index, body=body)  # type: ignore[arg-type]


# ----------------------------
# Clients
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str):
    OpenAI = _import_openai()
    return OpenAI(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_es_client(url: str, user: str, password: str):
    Elasticsearch = _import_elasticsearch()
    if user and password:
        return Elasticsearch(url, basic_auth=(user, password))
    return Elasticsearch(url)


# ----------------------------
# OpenAI calls
# ----------------------------
def embed_texts_openai(client, texts: List[str], model: str, batch_size: int = 128) -> List[List[float]]:
    vectors: List[List[float]] = []
    if not texts:
        return vectors

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(
                    model=model,
                    input=batch,
                    encoding_format="float",
                )
                vectors.extend([item.embedding for item in resp.data])
                break
            except Exception:
                attempt += 1
                if attempt >= 6:
                    raise
                time.sleep(min(2**attempt, 30))

    return vectors


def llm_call_1_extract(client, user_query: str, model: str = "gpt-4.1") -> Dict[str, Any]:
    # Kept for full functionality, but never shown in UI.
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "trade_type": {"type": ["string", "null"], "enum": ["ixrac", "idxal", "transit", None]},
            "vehicle": {"type": ["string", "null"], "enum": ["avtomobil", "deniz", "demiryolu", "hava", None]},
            "actor": {
                "type": ["string", "null"],
                "enum": ["hamisi", "dasiyici", "istehsalci", "idxalatci", "ixracatci", None],
            },
            "intent": {
                "type": "string",
                "enum": ["steps", "details", "fee_duration", "result", "organization", "digitization", "general"],
            },
            "hs_query": {"type": "string"},
            "keywords": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 12},
            "hs_broad_terms": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 8},
        },
        "required": ["trade_type", "vehicle", "actor", "intent", "hs_query", "keywords", "hs_broad_terms"],
    }

    prompt = f"""
You are a router. Your job is to extract structured routing info from the user query.

trade_type:
- ixrac: exporting
- idxal: importing
- transit: transiting
- null only if it cannot be determined from the query

vehicle:
- avtomobil, deniz, demiryolu, hava, or null

actor:
- hamisi (no filter)
- dasiyici (carrier)
- istehsalci (producer)
- idxalatci (importer)
- ixracatci (exporter)
- null only if cannot be determined

intent:
- steps, details, fee_duration, result, organization, digitization, general

hs_query:
- short product phrase (Azerbaijani if possible). Do not include trade_type words.

keywords:
- 3-12 keywords and category terms relevant for HS code search and routing.

hs_broad_terms:
- 0-8 broader HS category hints (hypernyms), only when confident.

Output MUST be STRICT JSON matching the schema.

User query:
{user_query}
""".strip()

    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "router_extract",
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        return json.loads(resp.output_text)
    except Exception:
        resp = client.responses.create(model=model, input=prompt)
        txt = resp.output_text or ""
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            raise ValueError("LLM extract failed to return JSON.")
        return json.loads(m.group(0))


def call_llm_answer_stream(client, question: str, context: str, model: str = "gpt-4.1"):
    system = (
        "You are a trade process assistant. "
        "The context contains one or more process guides. "
        "Use ONLY the provided context; do not add outside knowledge. "
        "Do not mention retrieval, embeddings, vector search, scoring, or databases. "
        "Treat each SECTION as a separate process guide. "
        "When the user asks for steps, return them as an ordered list. "
        "If multiple processes could apply, present the best match first and label alternatives. "
        "If the context lacks required details, say exactly what is missing and what you would need."
    )

    user = f"""Question:
{question}

Context:
{context}

Instructions:
- Answer using only the context above.
- If the user asks for steps, output numbered steps.
- If the answer is incomplete, state missing fields (for example: required documents, fees, where to apply, eligibility, or step-by-step actions).
"""

    def _gen():
        with client.responses.stream(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        ) as s:
            for event in s:
                if getattr(event, "type", None) == "response.output_text.delta":
                    # Yield exactly what the API streams (closest to token-by-token in Streamlit)
                    yield event.delta
            s.get_final_response()

    return _gen()


# ----------------------------
# Mapping and retrieval
# ----------------------------
def map_llm1_filters_to_db_params(extracted: Dict[str, Any]) -> Dict[str, Any]:
    trade_type = extracted.get("trade_type")
    vehicle = extracted.get("vehicle")
    actor = extracted.get("actor")

    trade_map = {"ixrac": "EXPORT", "idxal": "IMPORT", "transit": "TRANSIT"}
    vehicle_map = {"deniz": "1", "avtomobil": "2", "demiryolu": "3", "hava": "4"}
    actor_map = {
        "hamisi": None,
        "dasiyici": "CARRIER",
        "idxalatci": "IMPORTER",
        "istehsalci": "PRODUCER",
        "ixracatci": "EXPORTER",
        None: None,
    }

    trade_es: Optional[str] = trade_map.get(trade_type) if trade_type else None
    vehicle_id: Optional[str] = vehicle_map.get(vehicle) if vehicle else None
    party_role_es: Optional[str] = actor_map.get(actor, None)

    in_vehicle_id: Optional[str] = None
    out_vehicle_id: Optional[str] = None

    if trade_es == "IMPORT":
        in_vehicle_id = vehicle_id
    elif trade_es == "EXPORT":
        out_vehicle_id = vehicle_id
    elif trade_es == "TRANSIT":
        in_vehicle_id = vehicle_id
        out_vehicle_id = vehicle_id

    hs_query = extracted.get("hs_query", "")
    keywords = extracted.get("keywords") or []
    hs_query_list = [hs_query] + list(keywords)

    return {
        "trade_type_es": trade_es,
        "vehicle_id": vehicle_id,
        "in_vehicle_id": in_vehicle_id,
        "out_vehicle_id": out_vehicle_id,
        "party_role_es": party_role_es,
        "intent": extracted.get("intent"),
        "hs_query": [x for x in hs_query_list if _norm_ws(x)],
        "hs_broad_terms": extracted.get("hs_broad_terms") or [],
    }


def extract_hscodes(resp: Dict[str, Any], top_k: int = 10) -> List[str]:
    hits = (resp.get("hits") or {}).get("hits") or []
    out: List[str] = []
    for h in hits[:top_k]:
        src = h.get("_source") or {}
        code = src.get("code")
        if code:
            out.append(str(code))
    return out


def es_search_hscodes_tiered(
    es,
    query_text: str,
    trade_type_filter: Optional[str] = None,
    index_name: str = "flattened_hscodes_v6",
    size: int = 20,
    request_timeout: int = 30,
    tradings_is_nested: bool = False,
    tradings_path: str = "tradings",
    tradings_field: str = "tradings.tradeType.keyword",
) -> Dict[str, Any]:
    trade_map = {"ixrac": "EXPORT", "idxal": "IMPORT", "transit": "TRANSIT"}
    t = (trade_type_filter or "").strip()
    trade_es = trade_map.get(t.lower()) or (t if t in {"EXPORT", "IMPORT", "TRANSIT"} else None)

    filter_clauses: List[Dict[str, Any]] = []
    if trade_es:
        if tradings_is_nested:
            filter_clauses.append({"nested": {"path": tradings_path, "query": {"term": {tradings_field: trade_es}}}})
        else:
            filter_clauses.append({"term": {tradings_field: trade_es}})

    should: List[Dict[str, Any]] = [
        {"match": {"name_az_d4_expanded": {"query": query_text, "operator": "and", "fuzziness": "0", "boost": 300_000_000}}},
        {"match": {"name_az_d3_expanded": {"query": query_text, "operator": "and", "fuzziness": "0", "boost": 250_000_000}}},
        {"match": {"name_az_d2_expanded": {"query": query_text, "operator": "and", "fuzziness": "0", "boost": 200_000_000}}},
        {"match": {"name_az_d4_expanded": {"query": query_text, "operator": "and", "fuzziness": "1", "prefix_length": 3, "boost": 150_000_000}}},
        {"match": {"name_az_d3_expanded": {"query": query_text, "operator": "and", "fuzziness": "1", "prefix_length": 3, "boost": 120_000_000}}},
        {"match": {"name_az_d2_expanded": {"query": query_text, "operator": "and", "fuzziness": "1", "prefix_length": 3, "boost": 90_000_000}}},
        {"match": {"name_az_d4_expanded": {"query": query_text, "operator": "and", "boost": 50_000_000}}},
        {"match": {"name_az_d3_expanded": {"query": query_text, "operator": "and", "boost": 30_000_000}}},
        {"match": {"name_az_d2_expanded": {"query": query_text, "operator": "and", "boost": 20_000_000}}},
    ]

    body: Dict[str, Any] = {
        "size": size,
        "track_total_hits": False,
        "query": {"bool": {"filter": filter_clauses, "should": should, "minimum_should_match": 1}},
    }

    return _es_search(es, index=index_name, body=body, request_timeout=request_timeout)


# ----------------------------
# Postgres resolver (unchanged, not shown in UI)
# ----------------------------
FLOW_COMMODITY_TBL = "flow_commodity"
FLOWS_TBL = "flow"
FLOW_COMBINATIONS_TBL = "flow_combination"
FLOW_COMB_PROCESS_TBL = "flow_combination_process"
PROCESS_PARTY_TBL = "process_party"

FLOW_COMMODITY_FLOW_ID_COL = "flow_id"
FLOW_COMMODITY_HSCODE_COL = "hs_code"

FLOWS_ID_COL = "id"
FLOWS_TRADE_TYPE_COL = "trade_type"

FLOW_COMB_ID_COL = "id"
FLOW_COMB_FLOW_ID_COL = "flow_id"
FLOW_COMB_INVEH_COL = "in_vehicle_id"
FLOW_COMB_OUTVEH_COL = "out_vehicle_id"

FLOW_COMB_PROCESS_COMB_ID_COL = "flow_combination_id"
FLOW_COMB_PROCESS_PROCESS_ID_COL = "process_id"

PROCESS_PARTY_PROCESS_ID_COL = "process_id"
PROCESS_PARTY_ROLE_COL = "party_role"


def _dedupe_keep_order(xs: Sequence[int]) -> List[int]:
    seen: set[int] = set()
    out: List[int] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _to_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def make_pg_conn(
    host: str,
    port: int,
    dbname: str,
    user: str,
    password: str,
    connect_timeout: int = 10,
    statement_timeout_ms: int = 30000,
):
    psycopg2 = _import_psycopg2()
    return psycopg2.connect(
        host=host,
        port=int(port),
        dbname=dbname,
        user=user,
        password=password,
        connect_timeout=int(connect_timeout),
        options=f"-c statement_timeout={int(statement_timeout_ms)}",
    )


def _build_combination_where_and_params(in_vehicle_id: Any, out_vehicle_id: Any) -> Tuple[str, List[Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    inv = _to_int_or_none(in_vehicle_id)
    outv = _to_int_or_none(out_vehicle_id)

    if inv is not None:
        clauses.append(f"{FLOW_COMB_INVEH_COL} = %s")
        params.append(inv)
    if outv is not None:
        clauses.append(f"{FLOW_COMB_OUTVEH_COL} = %s")
        params.append(outv)

    if not clauses:
        return "1=1", []

    return " AND ".join(clauses), params


def db_get_flow_ids_by_hscodes(conn, hscodes: Sequence[str]) -> List[int]:
    hscodes = [str(x) for x in hscodes if _norm_ws(x)]
    if not hscodes:
        return []
    sql = f"""
        SELECT DISTINCT {FLOW_COMMODITY_FLOW_ID_COL}
        FROM {FLOW_COMMODITY_TBL}
        WHERE {FLOW_COMMODITY_HSCODE_COL} = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (hscodes,))
        rows = cur.fetchall()
    return [int(r[0]) for r in rows if r and r[0] is not None]


def db_filter_flow_ids_by_trade_type(conn, flow_ids: Sequence[int], trade_type_es: Optional[str]) -> List[int]:
    if not flow_ids:
        return []
    if not trade_type_es:
        return list(flow_ids)

    sql = f"""
        SELECT {FLOWS_ID_COL}
        FROM {FLOWS_TBL}
        WHERE {FLOWS_ID_COL} = ANY(%s)
          AND {FLOWS_TRADE_TYPE_COL} = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (list(flow_ids), trade_type_es))
        rows = cur.fetchall()
    return [int(r[0]) for r in rows if r and r[0] is not None]


def db_choose_single_flow_id(flow_ids: Sequence[int]) -> Optional[int]:
    flow_ids = list(flow_ids)
    if not flow_ids:
        return None
    return int(sorted(flow_ids)[0])


def db_get_flow_combination_id(conn, flow_id: int, in_vehicle_id: Any, out_vehicle_id: Any) -> Optional[int]:
    where_sql, params = _build_combination_where_and_params(in_vehicle_id, out_vehicle_id)

    sql = f"""
        SELECT {FLOW_COMB_ID_COL}
        FROM {FLOW_COMBINATIONS_TBL}
        WHERE {FLOW_COMB_FLOW_ID_COL} = %s
          AND {where_sql}
        ORDER BY {FLOW_COMB_ID_COL} ASC
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, [int(flow_id)] + params)
        row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else None


def db_get_process_ids_by_combination_id(conn, comb_id: int) -> List[int]:
    sql = f"""
        SELECT {FLOW_COMB_PROCESS_PROCESS_ID_COL}
        FROM {FLOW_COMB_PROCESS_TBL}
        WHERE {FLOW_COMB_PROCESS_COMB_ID_COL} = %s
        ORDER BY {FLOW_COMB_PROCESS_PROCESS_ID_COL} ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (int(comb_id),))
        rows = cur.fetchall()
    return [int(r[0]) for r in rows if r and r[0] is not None]


def db_filter_process_ids_by_party_role(conn, process_ids: Sequence[int], party_role_es: Optional[str]) -> List[int]:
    if not process_ids:
        return []
    if not party_role_es:
        return list(process_ids)

    sql = f"""
        SELECT DISTINCT {PROCESS_PARTY_PROCESS_ID_COL}
        FROM {PROCESS_PARTY_TBL}
        WHERE {PROCESS_PARTY_PROCESS_ID_COL} = ANY(%s)
          AND {PROCESS_PARTY_ROLE_COL} = %s
        ORDER BY {PROCESS_PARTY_PROCESS_ID_COL} ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (list(process_ids), party_role_es))
        rows = cur.fetchall()
    return [int(r[0]) for r in rows if r and r[0] is not None]


def resolve_process_ids_pg_dbready(conn, filters: Dict[str, Any], hscodes: Sequence[str]) -> Dict[str, Any]:
    flow_ids = _dedupe_keep_order(db_get_flow_ids_by_hscodes(conn, hscodes))
    flow_ids_tt = _dedupe_keep_order(db_filter_flow_ids_by_trade_type(conn, flow_ids, filters.get("trade_type_es")))

    flow_id = db_choose_single_flow_id(flow_ids_tt)
    if flow_id is None:
        return {"flow_id": None, "flow_combination_id": None, "process_ids": []}

    comb_id = db_get_flow_combination_id(conn, flow_id, filters.get("in_vehicle_id"), filters.get("out_vehicle_id"))
    if comb_id is None:
        return {"flow_id": flow_id, "flow_combination_id": None, "process_ids": []}

    process_ids = _dedupe_keep_order(db_get_process_ids_by_combination_id(conn, comb_id))
    process_ids = _dedupe_keep_order(db_filter_process_ids_by_party_role(conn, process_ids, filters.get("party_role_es")))

    return {"flow_id": flow_id, "flow_combination_id": comb_id, "process_ids": process_ids}


# ----------------------------
# Process ES retrieval
# ----------------------------
def es_retrieve_docs(
    es,
    index_name: str,
    query: str,
    query_vec: Optional[List[float]],
    top_k: int,
    source_includes: List[str],
    process_ids: Optional[List[int]] = None,
    process_id_field: str = "process_id",
    semantic: bool = True,
    keyword_fallback: bool = True,
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []

    if semantic and query_vec is not None:
        filters: List[Dict[str, Any]] = []
        if process_ids:
            filters.append({"terms": {process_id_field: process_ids}})
        q = {"bool": {"filter": filters}} if filters else {"match_all": {}}
        body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": q,
                    "script": {"source": "cosineSimilarity(params.q, 'embedding') + 1.0", "params": {"q": query_vec}},
                }
            },
            "_source": {"includes": source_includes, "excludes": ["embedding"]},
        }
        resp = _es_search(es, index=index_name, body=body, request_timeout=30)
        hits = (resp.get("hits") or {}).get("hits") or []
        docs = [h.get("_source") or {} for h in hits]

    if (not docs) and keyword_fallback:
        filters: List[Dict[str, Any]] = []
        if process_ids:
            filters.append({"terms": {process_id_field: process_ids}})
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "filter": filters,
                    "should": [
                        {"multi_match": {"query": query, "fields": ["info^3", "organization.name^2"], "type": "best_fields"}},
                        {
                            "nested": {
                                "path": "execution_steps",
                                "query": {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["execution_steps.name^2", "execution_steps.description"],
                                        "type": "best_fields",
                                        "operator": "and",
                                        "fuzziness": "AUTO",
                                    }
                                },
                                "score_mode": "avg",
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            },
            "_source": {"includes": source_includes, "excludes": ["embedding"]},
        }
        resp = _es_search(es, index=index_name, body=body, request_timeout=30)
        hits = (resp.get("hits") or {}).get("hits") or []
        docs = [h.get("_source") or {} for h in hits]

    return docs


def format_context_from_docs(
    docs: List[Dict[str, Any]],
    query: str,
    width: int = 120,
    max_process_desc_chars: int = 3000,
    max_step_desc_chars: int = 1000,
    limit_chars: int = 20000,
) -> str:
    lines: List[str] = []
    lines.append("BEGIN FLOW")
    lines.append(_wrap(f"Query: {query}", width=width))
    lines.append("")

    def step_key(st: Dict[str, Any]) -> Tuple[int, str]:
        o = st.get("order", None)
        o2 = 10**9 if o is None else int(o) if str(o).isdigit() else 10**9
        n = _norm_ws(st.get("name", ""))
        return (o2, n)

    for doc in docs:
        if not isinstance(doc, dict):
            continue

        name = _norm_ws(doc.get("name") or doc.get("title") or doc.get("process_name") or "Process")
        desc = _strip_html(doc.get("description") or doc.get("process_description") or "")

        org = doc.get("organization") or {}
        org_name = _norm_ws(org.get("name")) if isinstance(org, dict) else ""

        steps = doc.get("execution_steps", []) or []
        if not isinstance(steps, list):
            steps = []

        lines.append("BEGIN SECTION")
        lines.append("PROCESS")
        lines.append(_wrap(f"Name: {name}", width=width))

        if desc:
            lines.append("")
            lines.append("Description")
            lines.append(_wrap(_trunc(desc, max_process_desc_chars), width=width, indent="  "))

        if org_name:
            lines.append("")
            lines.append("Organization")
            lines.append(_wrap(org_name, width=width, indent="  "))

        lines.append("")
        lines.append("Execution steps")
        if not steps:
            lines.append(_wrap("No steps available.", width=width, indent="  "))
        else:
            for idx, stp in enumerate(sorted([s for s in steps if isinstance(s, dict)], key=step_key), start=1):
                st_name = _norm_ws(stp.get("name") or f"Step {idx}")
                st_desc = _strip_html(stp.get("description") or "")
                lines.append(_wrap(f"{idx}. {st_name}", width=width, indent="  "))
                if st_desc:
                    lines.append(_wrap(_trunc(st_desc, max_step_desc_chars), width=width, indent="     "))

        lines.append("")
        lines.append("END SECTION")
        lines.append("")

        if limit_chars and sum(len(x) + 1 for x in lines) >= limit_chars:
            break

    lines.append("END FLOW")
    out = "\n".join(lines).rstrip()
    if limit_chars and len(out) > limit_chars:
        out = out[:limit_chars].rstrip()
    return out


def run_pipeline_answer_stream(
    query: str,
    *,
    openai_client,
    openai_model_router: str,
    openai_model_answer: str,
    openai_embed_model: str,
    es,
    process_index: str,
    hs_index: str,
    pg_cfg: Dict[str, Any],
    top_k_docs: int,
    top_k_hs: int,
    semantic: bool,
    keyword_fallback: bool,
):
    # Full pipeline runs, but we only stream the final answer in UI.
    llm1 = llm_call_1_extract(openai_client, query, model=openai_model_router)
    mapped = map_llm1_filters_to_db_params(llm1)

    product_index_query = " ".join(mapped.get("hs_query") or [])
    hs_resp = es_search_hscodes_tiered(
        es=es,
        query_text=product_index_query,
        trade_type_filter=mapped.get("trade_type_es"),
        index_name=hs_index,
        size=max(top_k_hs, 1),
    )
    hs_codes = extract_hscodes(hs_resp, top_k=top_k_hs)

    process_ids: List[int] = []
    if hs_codes and pg_cfg.get("enabled"):
        try:
            conn = make_pg_conn(
                host=pg_cfg["host"],
                port=int(pg_cfg["port"]),
                dbname=pg_cfg["dbname"],
                user=pg_cfg["user"],
                password=pg_cfg["password"],
            )
            try:
                db_out = resolve_process_ids_pg_dbready(
                    conn,
                    {
                        "trade_type_es": mapped.get("trade_type_es"),
                        "in_vehicle_id": mapped.get("in_vehicle_id"),
                        "out_vehicle_id": mapped.get("out_vehicle_id"),
                        "party_role_es": mapped.get("party_role_es"),
                    },
                    hs_codes,
                )
                process_ids = list(db_out.get("process_ids") or [])
            finally:
                conn.close()
        except Exception:
            process_ids = []

    q_vec = embed_texts_openai(openai_client, [query], model=openai_embed_model, batch_size=1)[0]

    src_includes = [
        "name",
        "title",
        "description",
        "organization",
        "execution_steps",
        "info",
    ]
    docs = es_retrieve_docs(
        es=es,
        index_name=process_index,
        query=query,
        query_vec=q_vec,
        top_k=top_k_docs,
        source_includes=src_includes,
        process_ids=process_ids if process_ids else None,
        process_id_field="process_id",
        semantic=semantic,
        keyword_fallback=keyword_fallback,
    )

    ctx = format_context_from_docs(docs=docs, query=query)
    return call_llm_answer_stream(openai_client, question=query, context=ctx, model=openai_model_answer)


# ----------------------------
# UI (ONLY ANSWER, streamed)
# ----------------------------
st.title("Trade Guide")

with st.sidebar:
    st.header("Configuration")

    st.subheader("OpenAI")
    api_key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    openai_model_router = st.text_input("Router model", value=os.getenv("OPENAI_ROUTER_MODEL", "gpt-4.1"))
    openai_model_answer = st.text_input("Answer model", value=os.getenv("OPENAI_ANSWER_MODEL", "gpt-4.1"))
    openai_embed_model = st.text_input("Embedding model", value=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"))

    st.subheader("Elasticsearch")
    es_url = st.text_input("ES URL", value=os.getenv("ES_URL", "http://localhost:9200"))
    es_user = st.text_input("ES user (optional)", value=os.getenv("ES_USER", ""))
    es_pass = st.text_input("ES password (optional)", type="password", value=os.getenv("ES_PASS", ""))
    process_index = st.text_input("Process index", value=os.getenv("ES_PROCESS_INDEX", "process_vector_db_openai"))
    hs_index = st.text_input("HS index", value=os.getenv("ES_HS_INDEX", "flattened_hscodes_v6"))

    st.subheader("PostgreSQL (optional)")
    pg_enabled = st.checkbox("Enable Postgres routing", value=False)
    pg_host = st.text_input("PG host", value=os.getenv("PGHOST", "localhost"))
    pg_port = st.number_input("PG port", min_value=1, max_value=65535, value=int(os.getenv("PGPORT", "5432")))
    pg_db = st.text_input("PG database", value=os.getenv("PGDATABASE", "trade_guide_flow"))
    pg_user = st.text_input("PG user", value=os.getenv("PGUSER", ""))
    pg_pass = st.text_input("PG password", type="password", value=os.getenv("PGPASSWORD", ""))

    st.subheader("Retrieval settings")
    top_k_docs = st.slider("Top K process docs", 1, 30, 10)
    top_k_hs = st.slider("Top K HS candidates", 1, 20, 5)
    semantic = st.checkbox("Semantic search", value=True)
    keyword_fallback = st.checkbox("Keyword fallback", value=True)

if not api_key:
    st.info("Enter your OpenAI API key in the sidebar.")
    st.stop()

oai = get_openai_client(api_key)
es = get_es_client(es_url, es_user, es_pass)

pg_cfg = {
    "enabled": bool(pg_enabled and pg_user and pg_pass and pg_db),
    "host": pg_host,
    "port": int(pg_port),
    "dbname": pg_db,
    "user": pg_user,
    "password": pg_pass,
}

query = st.text_area("Query", value="", height=90, placeholder="Ask your question...")
run = st.button("Run", type="primary", disabled=not bool(_norm_ws(query)))

st.markdown("## Answer")
if run:
    try:
        stream = run_pipeline_answer_stream(
            query.strip(),
            openai_client=oai,
            openai_model_router=openai_model_router,
            openai_model_answer=openai_model_answer,
            openai_embed_model=openai_embed_model,
            es=es,
            process_index=process_index,
            hs_index=hs_index,
            pg_cfg=pg_cfg,
            top_k_docs=top_k_docs,
            top_k_hs=top_k_hs,
            semantic=semantic,
            keyword_fallback=keyword_fallback,
        )
        st.write_stream(stream)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
