# serve.py
import os
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from functools import lru_cache
import mcp.types as types

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse

# --- Weaviate client imports (v4) ---
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

# OpenAI client per descrizioni immagini
from openai import OpenAI

_OPENAI_CLIENT = None
if os.environ.get("OPENAI_API_KEY"):
    _OPENAI_CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
else:
    print("[query-caption] WARNING: OPENAI_API_KEY non impostata, niente descrizioni testuali per le query.")

# In-memory stato Vertex
_VERTEX_HEADERS: Dict[str, str] = {}
_VERTEX_REFRESH_THREAD_STARTED = False
_VERTEX_USER_PROJECT: Optional[str] = None

_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_PROMPT_PATH = _BASE_DIR / "prompts" / "instructions.md"
_DEFAULT_DESCRIPTION_PATH = _BASE_DIR / "prompts" / "description.txt"
_BASE_URL = os.environ.get("BASE_URL", "https://weaviate-openai-app-sdk.onrender.com")


def _build_vertex_header_map(token: str) -> Dict[str, str]:
    """
    Crea gli header per far sì che Weaviate possa chiamare text2vec-google
    usando il token OAuth2 ottenuto dalla service account.
    """
    headers: Dict[str, str] = {
        "X-Goog-Vertex-Api-Key": token,
    }
    # Includi sempre project_id se disponibile (per billing/quota esplicita)
    project_id = _VERTEX_USER_PROJECT or _discover_gcp_project()
    if project_id:
        headers["X-Goog-User-Project"] = project_id
    return headers


def _discover_gcp_project() -> Optional[str]:
    gac_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if gac_json:
        try:
            data = json.loads(gac_json)
            if isinstance(data, dict) and data.get("project_id"):
                return data["project_id"]
        except Exception:
            pass

    gac_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_path and os.path.exists(gac_path):
        try:
            with open(gac_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("project_id"):
                return data["project_id"]
        except Exception:
            pass

    try:
        import google.auth

        creds, proj = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        if proj:
            return proj
    except Exception:
        pass
    return None


def _get_weaviate_url() -> str:
    url = os.environ.get("WEAVIATE_CLUSTER_URL") or os.environ.get("WEAVIATE_URL")
    if not url:
        raise RuntimeError("Please set WEAVIATE_URL or WEAVIATE_CLUSTER_URL.")
    return url


def _get_weaviate_api_key() -> str:
    api_key = os.environ.get("WEAVIATE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set WEAVIATE_API_KEY.")
    return api_key


def _resolve_service_account_path() -> Optional[str]:
    gac_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_path and os.path.exists(gac_path):
        _load_vertex_user_project(gac_path)
        return gac_path

    candidates = [
        os.environ.get("VERTEX_SA_PATH"),
        "/etc/secrets/weaviate-sa.json",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = candidate
            _load_vertex_user_project(candidate)
            return candidate
    return None


def _load_vertex_user_project(path: str) -> None:
    global _VERTEX_USER_PROJECT
    if _VERTEX_USER_PROJECT:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _VERTEX_USER_PROJECT = data.get("project_id")
        if not _VERTEX_USER_PROJECT and data.get("quota_project_id"):
            _VERTEX_USER_PROJECT = data["quota_project_id"]
        if _VERTEX_USER_PROJECT:
            try:
                print(
                    f"[vertex-oauth] detected service account project: {_VERTEX_USER_PROJECT}"
                )
            except (ValueError, OSError):
                pass
        else:
            try:
                print(
                    "[vertex-oauth] warning: project_id not found in service account JSON"
                )
            except (ValueError, OSError):
                pass
    except Exception as exc:
        try:
            print(f"[vertex-oauth] unable to read project id from SA: {exc}")
        except (ValueError, OSError):
            pass


def _sync_refresh_vertex_token() -> bool:
    """
    Refresh sincrono del token OAuth2 Vertex dalla service account.
    Restituisce True se il refresh ha successo, False altrimenti.
    """
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
    except Exception as exc:
        print(f"[vertex-oauth] sync refresh unavailable: {exc}")
        return False

    cred_path = _resolve_service_account_path()
    if not cred_path or not os.path.exists(cred_path):
        print(f"[vertex-oauth] service account path not found: {cred_path}")
        return False
    
    try:
        creds = service_account.Credentials.from_service_account_file(
            cred_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        creds.refresh(Request())
    except Exception as exc:
        print(f"[vertex-oauth] sync refresh error: {exc}")
        return False

    token = creds.token
    if not token:
        print("[vertex-oauth] token is None after refresh")
        return False
    
    global _VERTEX_HEADERS
    _VERTEX_HEADERS = _build_vertex_header_map(token)
    print(f"[vertex-oauth] sync token refresh for text2vec-google (prefix: {token[:10]}...)")
    # Aggiorna anche le variabili d'ambiente per Weaviate vectorizer
    os.environ["GOOGLE_APIKEY"] = token
    os.environ["PALM_APIKEY"] = token
    return True


def _connect():
    """
    Connessione a Weaviate Cloud usando:
    - API key del cluster (WEAVIATE_API_KEY)
    - token OAuth2 di Vertex messo in X-Goog-Vertex-Api-Key per text2vec-google
    
    Questo replica il pattern del codice di esempio:
    il token viene generato dalla service account e passato a Weaviate
    come "API key" per text2vec-google.
    """
    url = _get_weaviate_url()
    key = _get_weaviate_api_key()
    _resolve_service_account_path()

    headers: Dict[str, str] = {}
    
    # OpenAI (se serve per text2vec-openai)
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
    if openai_key:
        headers["X-OpenAI-Api-Key"] = openai_key

    # 1) Refresh esplicito del token OAuth2 dalla service account (come nel codice di esempio)
    vertex_token = (
        os.environ.get("VERTEX_APIKEY")
        or os.environ.get("VERTEX_BEARER_TOKEN")
    )
    
    # Se non c'è una chiave statica/bearer, usa OAuth con refresh esplicito
    if not vertex_token:
        # Assicurati che _VERTEX_HEADERS contenga il token (refresh se necessario)
        if not ("_VERTEX_HEADERS" in globals() and _VERTEX_HEADERS and _VERTEX_HEADERS.get("X-Goog-Vertex-Api-Key")):
            # Refresh esplicito del token (come nel codice di esempio)
            if not _sync_refresh_vertex_token():
                print("[vertex-oauth] WARNING: failed to refresh Vertex token")
        # Prendi il token da _VERTEX_HEADERS
        if "_VERTEX_HEADERS" in globals() and _VERTEX_HEADERS:
            vertex_token = _VERTEX_HEADERS.get("X-Goog-Vertex-Api-Key")
    
    # 2) Build header HTTP per le chiamate REST (come nel codice di esempio)
    if vertex_token:
        headers.update(_build_vertex_header_map(vertex_token))
        # Aggiorna anche le variabili d'ambiente per Weaviate vectorizer (fallback)
        os.environ["GOOGLE_APIKEY"] = vertex_token
        os.environ["PALM_APIKEY"] = vertex_token
        print(f"[vertex-oauth] using Vertex token for text2vec-google (prefix: {vertex_token[:10]}...)")
    else:
        print("[vertex-oauth] WARNING: no Vertex token available for connection")

    # 3) Crea client Weaviate con header
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(key),
        headers=headers or None,
    )

    # 4) Metadata gRPC (per sicurezza, come nel codice di esempio)
    if vertex_token:
        try:
            conn = getattr(client, "_connection", None)
            if conn is not None:
                meta_list = [
                    ("x-goog-vertex-api-key", vertex_token),
                ]
                project_id = _VERTEX_USER_PROJECT or _discover_gcp_project()
                if project_id:
                    meta_list.append(("x-goog-user-project", project_id))
                
                # Prova a settare vari modalità (come nel tuo serve.py e nel codice di esempio)
                try:
                    setattr(conn, "grpc_metadata", meta_list)
                except Exception:
                    pass
                try:
                    setattr(conn, "_grpc_metadata", meta_list)
                except Exception:
                    pass
                if hasattr(conn, "set_grpc_metadata"):
                    try:
                        conn.set_grpc_metadata(meta_list)
                    except Exception:
                        pass
                debug_meta = getattr(conn, "grpc_metadata", None)
                print(f"[vertex-oauth] gRPC metadata set for text2vec-google: {debug_meta}")
        except Exception as e:
            print(f"[weaviate] warning: cannot set gRPC metadata headers: {e}")

    return client


def _update_client_grpc_metadata(client):
    """
    Aggiorna i metadata gRPC del client con le credenziali Vertex più recenti.
    Segue lo stesso pattern di _connect() per text2vec-google.
    """
    try:
        # Assicuriamoci che _VERTEX_HEADERS contenga il token più recente
        if not ("_VERTEX_HEADERS" in globals() and _VERTEX_HEADERS and _VERTEX_HEADERS.get("X-Goog-Vertex-Api-Key")):
            # Se _VERTEX_HEADERS è vuoto o non contiene il token, facciamo un refresh
            if not _sync_refresh_vertex_token():
                return
        
        # Prendi il token da _VERTEX_HEADERS
        vertex_token = None
        if "_VERTEX_HEADERS" in globals() and _VERTEX_HEADERS:
            vertex_token = _VERTEX_HEADERS.get("X-Goog-Vertex-Api-Key")
        
        if not vertex_token:
            return
        
        # Aggiorna i metadata gRPC del client (come nel codice di esempio)
        conn = getattr(client, "_connection", None)
        if conn is not None:
            meta_list = [
                ("x-goog-vertex-api-key", vertex_token),
            ]
            project_id = _VERTEX_USER_PROJECT or _discover_gcp_project()
            if project_id:
                meta_list.append(("x-goog-user-project", project_id))
            
            # Prova a settare vari modalità (come nel codice di esempio)
            try:
                setattr(conn, "grpc_metadata", meta_list)
            except Exception:
                pass
            try:
                setattr(conn, "_grpc_metadata", meta_list)
            except Exception:
                pass
            if hasattr(conn, "set_grpc_metadata"):
                try:
                    conn.set_grpc_metadata(meta_list)
                except Exception:
                    pass
            
            # Aggiorna anche le variabili d'ambiente per Weaviate vectorizer
            os.environ["GOOGLE_APIKEY"] = vertex_token
            os.environ["PALM_APIKEY"] = vertex_token
    except Exception as e:
        print(f"[vertex-oauth] warning: cannot update gRPC metadata: {e}")


def _load_text_source(env_keys, file_path):
    if isinstance(env_keys, str):
        env_keys = [env_keys]
    path = Path(file_path) if file_path else None
    if path and path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as exc:
            print(f"[mcp] warning: cannot read instructions file '{path}': {exc}")
    for key in env_keys:
        val = os.environ.get(key)
        if val:
            return val.strip()
    return None


_MCP_SERVER_NAME = os.environ.get("MCP_SERVER_NAME", "weaviate-mcp-http")
_MCP_INSTRUCTIONS_FILE = os.environ.get("MCP_PROMPT_FILE") or os.environ.get(
    "MCP_INSTRUCTIONS_FILE"
)
if not _MCP_INSTRUCTIONS_FILE and _DEFAULT_PROMPT_PATH.exists():
    _MCP_INSTRUCTIONS_FILE = str(_DEFAULT_PROMPT_PATH)
_MCP_DESCRIPTION_FILE = os.environ.get("MCP_DESCRIPTION_FILE")
if not _MCP_DESCRIPTION_FILE and _DEFAULT_DESCRIPTION_PATH.exists():
    _MCP_DESCRIPTION_FILE = str(_DEFAULT_DESCRIPTION_PATH)

_MCP_INSTRUCTIONS = _load_text_source(
    ["MCP_PROMPT", "MCP_INSTRUCTIONS"], _MCP_INSTRUCTIONS_FILE
)
_MCP_DESCRIPTION = _load_text_source("MCP_DESCRIPTION", _MCP_DESCRIPTION_FILE)

# Porta e host per FastMCP / uvicorn (per Render)
SERVER_PORT = int(os.environ.get("PORT", "10000"))
os.environ.setdefault("FASTMCP_PORT", str(SERVER_PORT))
os.environ.setdefault("FASTMCP_HOST", "0.0.0.0")

# Host esterno esposto da Render (se disponibile)
render_host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")

allowed_hosts = [
    "localhost",
    "127.0.0.1:*",  # utile per sviluppo locale
]

# Aggiungi host di Render, se definito
if render_host:
    allowed_hosts.append(render_host)
    allowed_hosts.append(f"{render_host}:*")
else:
    # fallback hard-coded per il tuo servizio attuale
    allowed_hosts.append("weaviate-text2vec-mcp.onrender.com")
    allowed_hosts.append("weaviate-text2vec-mcp.onrender.com:*")

transport_security = TransportSecuritySettings(
    # Manteniamo la protezione DNS rebinding ma permettiamo il tuo dominio
    enable_dns_rebinding_protection=True,
    allowed_hosts=allowed_hosts,
    # Lasciamo vuoto allowed_origins per evitare rogne con l'header Origin
    allowed_origins=[],
)

# Non passiamo host/port direttamente, lasciamo che FastMCP usi le env FASTMCP_*
mcp = FastMCP(
    _MCP_SERVER_NAME,
    stateless_http=True,
    transport_security=transport_security,
)


def _apply_mcp_metadata():
    try:
        if hasattr(mcp, "set_server_info"):
            server_info: Dict[str, Any] = {}
            if _MCP_DESCRIPTION:
                server_info["description"] = _MCP_DESCRIPTION
            if _MCP_INSTRUCTIONS:
                server_info["instructions"] = _MCP_INSTRUCTIONS
            if server_info:
                mcp.set_server_info(**server_info)
    except Exception:
        pass


_apply_mcp_metadata()


@mcp.custom_route("/health", methods=["GET"])
async def health(_request):
    return JSONResponse({"status": "ok", "service": "weaviate-mcp-http"})


@mcp.tool()
def get_instructions() -> Dict[str, Any]:
    return {
        "instructions": _MCP_INSTRUCTIONS,
        "description": _MCP_DESCRIPTION,
        "server_name": _MCP_SERVER_NAME,
        "prompt_file": _MCP_INSTRUCTIONS_FILE,
        "description_file": _MCP_DESCRIPTION_FILE,
    }


@mcp.tool()
def reload_instructions() -> Dict[str, Any]:
    global _MCP_INSTRUCTIONS, _MCP_DESCRIPTION, _MCP_INSTRUCTIONS_FILE, _MCP_DESCRIPTION_FILE
    _MCP_INSTRUCTIONS_FILE = os.environ.get("MCP_PROMPT_FILE") or os.environ.get(
        "MCP_INSTRUCTIONS_FILE"
    )
    if not _MCP_INSTRUCTIONS_FILE and _DEFAULT_PROMPT_PATH.exists():
        _MCP_INSTRUCTIONS_FILE = str(_DEFAULT_PROMPT_PATH)
    _MCP_DESCRIPTION_FILE = os.environ.get("MCP_DESCRIPTION_FILE")
    if not _MCP_DESCRIPTION_FILE and _DEFAULT_DESCRIPTION_PATH.exists():
        _MCP_DESCRIPTION_FILE = str(_DEFAULT_DESCRIPTION_PATH)
    _MCP_INSTRUCTIONS = _load_text_source(
        ["MCP_PROMPT", "MCP_INSTRUCTIONS"], _MCP_INSTRUCTIONS_FILE
    )
    _MCP_DESCRIPTION = _load_text_source("MCP_DESCRIPTION", _MCP_DESCRIPTION_FILE)
    _apply_mcp_metadata()
    return get_instructions()


@mcp.tool()
def get_config() -> Dict[str, Any]:
    return {
        "weaviate_url": os.environ.get("WEAVIATE_CLUSTER_URL")
        or os.environ.get("WEAVIATE_URL"),
        "weaviate_api_key_set": bool(os.environ.get("WEAVIATE_API_KEY")),
        "openai_api_key_set": bool(
            os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
        ),
        "cohere_api_key_set": bool(os.environ.get("COHERE_API_KEY")),
    }


@mcp.tool()
def check_connection() -> Dict[str, Any]:
    client = _connect()
    try:
        ready = client.is_ready()
        return {"ready": bool(ready)}
    finally:
        client.close()


@mcp.tool()
def list_collections() -> List[str]:
    client = _connect()
    try:
        colls = client.collections.list_all()
        if isinstance(colls, dict):
            names = list(colls.keys())
        else:
            try:
                names = [getattr(c, "name", str(c)) for c in colls]
            except Exception:
                names = list(colls)
        return sorted(set(names))
    finally:
        client.close()


@mcp.tool()
def get_schema(collection: str) -> Dict[str, Any]:
    client = _connect()
    try:
        coll = client.collections.get(collection)
        if coll is None:
            return {"error": f"Collection '{collection}' not found"}
        try:
            cfg = coll.config.get()
        except Exception:
            try:
                cfg = coll.config.get_class()
            except Exception:
                cfg = {"info": "config API not available in this client version"}
        return {"collection": collection, "config": cfg}
    finally:
        client.close()


@mcp.tool()
def keyword_search(collection: str, query: str, limit: int = 10) -> Dict[str, Any]:
    client = _connect()
    try:
        coll = client.collections.get(collection)
        if coll is None:
            return {"error": f"Collection '{collection}' not found"}
        resp = coll.query.bm25(
            query=query,
            return_metadata=MetadataQuery(score=True),
            limit=limit,
        )
        out = []
        for o in getattr(resp, "objects", []) or []:
            out.append(
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", {}),
                    "bm25_score": getattr(getattr(o, "metadata", None), "score", None),
                }
            )
        return {"count": len(out), "results": out}
    finally:
        client.close()


@mcp.tool()
def semantic_search(collection: str, query: str, limit: int = 10) -> Dict[str, Any]:
    client = _connect()
    try:
        coll = client.collections.get(collection)
        if coll is None:
            return {"error": f"Collection '{collection}' not found"}
        resp = coll.query.near_text(
            query=query,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )
        out = []
        for o in getattr(resp, "objects", []) or []:
            out.append(
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", {}),
                    "distance": getattr(getattr(o, "metadata", None), "distance", None),
                }
            )
        return {"count": len(out), "results": out}
    finally:
        client.close()


@mcp.tool()
def hybrid_search(
    collection: str,
    query: str,
    limit: int = 10,
    alpha: float = 0.2,
    query_properties: Optional[Any] = None,
) -> Dict[str, Any]:
    if collection and collection != "WindBilance":
        print(
            f"[hybrid_search] warning: collection '{collection}' requested, but using 'WindBilance' as per instructions"
        )
        collection = "WindBilance"

    if query_properties and isinstance(query_properties, str):
        try:
            query_properties = json.loads(query_properties)
        except (json.JSONDecodeError, TypeError):
            pass

    client = _connect()
    try:
        coll = client.collections.get(collection)
        if coll is None:
            return {"error": f"Collection '{collection}' not found"}

        # Aggiorna i metadata gRPC prima della query per assicurarci che siano aggiornati
        _update_client_grpc_metadata(client)

        hybrid_params = {
            "query": query,
            "alpha": alpha,
            "limit": limit,
            "return_properties": ["name", "source_pdf", "page_index", "mediaType"],
            "return_metadata": MetadataQuery(score=True, distance=True),
        }
        if query_properties:
            hybrid_params["query_properties"] = query_properties
        resp = coll.query.hybrid(**hybrid_params)

        # Log dei risultati nel formato Colab
        print("[DEBUG] Risultati hybrid search:")
        for o in getattr(resp, "objects", []) or []:
            name = getattr(o, "properties", {}).get("name", "N/A")
            md = getattr(o, "metadata", None)
            score = getattr(md, "score", None)
            if score is not None:
                print(f"{name}  score={score:.4f}")
            else:
                print(f"{name}  score=N/A")

        out = []
        for o in getattr(resp, "objects", []) or []:
            md = getattr(o, "metadata", None)
            score = getattr(md, "score", None)
            distance = getattr(md, "distance", None)
            out.append(
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", {}),
                    "bm25_score": score,
                    "distance": distance,
                }
            )
        return {"count": len(out), "results": out}
    finally:
        client.close()


def _ensure_gcp_adc():
    gac_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    gac_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_json and not gac_path:
        tmp_path = "/app/gcp_credentials.json"
        with open(tmp_path, "w", encoding="utf-8") as f2:
            f2.write(gac_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_path
    _resolve_service_account_path()
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        _load_vertex_user_project(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])


@mcp.tool()
def diagnose_vertex() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["project_id"] = _discover_gcp_project()
    info["oauth_enabled"] = os.environ.get("VERTEX_USE_OAUTH", "").lower() in (
        "1",
        "true",
        "yes",
    )
    info["headers_active"] = bool(_VERTEX_HEADERS) if "_VERTEX_HEADERS" in globals() else False
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request

        SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
        gac_path = _resolve_service_account_path()
        token_preview = None
        expiry = None
        if gac_path and os.path.exists(gac_path):
            creds = service_account.Credentials.from_service_account_file(
                gac_path, scopes=SCOPES
            )
            creds.refresh(Request())
            token_preview = (creds.token[:12] + "...") if creds.token else None
            expiry = getattr(creds, "expiry", None)
        info["token_sample"] = token_preview
        info["token_expiry"] = str(expiry) if expiry else None
    except Exception as e:
        info["token_error"] = str(e)
    return info


# Registry dei tool normali che vuoi esporre alla App
TOOL_REGISTRY: Dict[str, Any] = {
    "get_instructions": get_instructions,
    "reload_instructions": reload_instructions,
    "get_config": get_config,
    "check_connection": check_connection,
    "list_collections": list_collections,
    "get_schema": get_schema,
    "keyword_search": keyword_search,
    "semantic_search": semantic_search,
    "hybrid_search": hybrid_search,
    "diagnose_vertex": diagnose_vertex,
}

# Tool nascosti (non esposti all'LLM ma ancora disponibili internamente)
_HIDDEN_TOOLS: set[str] = {
    "semantic_search",
    "keyword_search",
    "diagnose_vertex",
}


# ==== Vertex OAuth Token Refresher (optional) ===============================
def _write_adc_from_json_env():
    gac_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    gac_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_json and not gac_path:
        tmp_path = "/app/gcp_credentials.json"
        with open(tmp_path, "w", encoding="utf-8") as f2:
            f2.write(gac_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_path
    _resolve_service_account_path()


def _refresh_vertex_oauth_loop():
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    import datetime
    import time

    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    cred_path = _resolve_service_account_path()
    if not cred_path or not os.path.exists(cred_path):
        print("[vertex-oauth] GOOGLE_APPLICATION_CREDENTIALS missing; token refresher disabled")
        return
    creds = service_account.Credentials.from_service_account_file(
        cred_path, scopes=SCOPES
    )
    global _VERTEX_HEADERS
    while True:
        try:
            creds.refresh(Request())
            token = creds.token
            _VERTEX_HEADERS = _build_vertex_header_map(token)
            # Aggiorna anche le variabili d'ambiente per Weaviate vectorizer
            os.environ["GOOGLE_APIKEY"] = token
            os.environ["PALM_APIKEY"] = token
            token_preview = token[:10] if token else None
            print(f"[vertex-oauth] 🔄 Vertex token refreshed (prefix: {token_preview}...)")
            sleep_s = 55 * 60
            if creds.expiry:
                from datetime import timezone
                now = datetime.datetime.now(timezone.utc).replace(tzinfo=creds.expiry.tzinfo)
                delta = (creds.expiry - now).total_seconds() - 300
                if delta > 300:
                    sleep_s = int(delta)
            time.sleep(sleep_s)
        except Exception as e:
            print(f"[vertex-oauth] refresh error: {e}")
            time.sleep(60)


def _maybe_start_vertex_oauth_refresher():
    global _VERTEX_REFRESH_THREAD_STARTED
    if _VERTEX_REFRESH_THREAD_STARTED:
        return
    if os.environ.get("VERTEX_USE_OAUTH", "").lower() not in ("1", "true", "yes"):
        return
    _write_adc_from_json_env()
    sa_path = _resolve_service_account_path()
    if not sa_path:
        print("[vertex-oauth] service account path not found; refresher not started")
        return
    import threading

    t = threading.Thread(target=_refresh_vertex_oauth_loop, daemon=True)
    t.start()
    _VERTEX_REFRESH_THREAD_STARTED = True


_maybe_start_vertex_oauth_refresher()

# --- Alias /mcp senza slash finale, se serve --------------------------------
try:
    from starlette.routing import Route

    _starlette_app = getattr(mcp, "app", None) or getattr(mcp, "_app", None)

    if _starlette_app is not None:

        async def _mcp_alias(request):
            scope = dict(request.scope)
            scope["path"] = "/mcp/"
            scope["raw_path"] = b"/mcp/"
            return await _starlette_app(scope, request.receive, request.send)

        _starlette_app.router.routes.insert(
            0,
            Route(
                "/mcp",
                endpoint=_mcp_alias,
                methods=["GET", "HEAD", "POST", "OPTIONS"],
            ),
        )
except Exception as _route_err:
    print("[mcp] warning: cannot register MCP alias route:", _route_err)

@mcp._mcp_server.list_tools()
async def _list_tools() -> List[types.Tool]:
    """Espone tutti i tool normali a ChatGPT."""
    tools: List[types.Tool] = []

    # Tutti i tool normali (escludendo quelli nascosti)
    for name in TOOL_REGISTRY.keys():
        # Salta i tool nascosti
        if name in _HIDDEN_TOOLS:
            continue
        # Schema di default: argomenti liberi
        input_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": True,
        }

        tool_title = name
        tool_description = name
        annotations = {
            "destructiveHint": False,
            "openWorldHint": True,
            "readOnlyHint": False,
        }

        # ✅ Schema specifico per hybrid_search con istruzioni incluse
        if name == "hybrid_search":
            input_schema = {
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Nome della collection (sempre 'WindBilance' per questo assistente)",
                    },
                    "query": {
                        "type": "string",
                        "description": "Query di ricerca testuale",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Numero massimo di risultati da restituire",
                        "default": 10,
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Peso della ricerca vettoriale (0.0 = solo keyword, 1.0 = solo vettoriale)",
                        "default": 0.8,
                    },
                    "query_properties": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Proprietà su cui cercare (default: ['caption', 'name'])",
                    },
                    "return_properties": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Proprietà da restituire (default: ['name', 'source_pdf', 'page_index', 'mediaType'])",
                    },
                },
                "required": ["collection", "query"],
                "additionalProperties": False,
            }
            tool_title = "Ricerca ibrida (BM25 + vettoriale)"
            tool_description = (
                "Esegue una ricerca ibrida combinando ricerca keyword (BM25) e ricerca vettoriale. "
                "Tool principale per cercare nella collection WindBilance.\n\n"
                "ISTRUZIONI: Usa SEMPRE collection='WindBilance'. Usa query_properties=['caption','name'] e "
                "return_properties=['name','source_pdf','page_index','mediaType']. Mantieni alpha=0.8 e limit=10 "
                "salvo richieste diverse."
            )

        tools.append(
            types.Tool(
                name=name,
                title=tool_title,
                description=tool_description,
                inputSchema=input_schema,
                annotations=annotations,
            )
        )

    return tools


@mcp._mcp_server.list_resources()
async def _list_resources() -> List[types.Resource]:
    return []


@mcp._mcp_server.list_resource_templates()
async def _list_resource_templates() -> List[types.ResourceTemplate]:
    return []


async def _handle_read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
    return types.ServerResult(
        types.ReadResourceResult(
            contents=[],
            _meta={"error": f"Unknown resource: {req.params.uri}"},
        )
    )


async def _call_tool_request(req: types.CallToolRequest) -> types.ServerResult:
    name = req.params.name
    args = req.params.arguments or {}

    # LOG DI DEBUG: vediamo quali tool vengono chiamati
    print(f"[call_tool] name={name}, args={json.dumps(args, ensure_ascii=False)}")

    # Tool normali (quelli del registry)
    if name in TOOL_REGISTRY:
        fn = TOOL_REGISTRY[name]

        # Caso speciale: hybrid_search → ripuliamo gli argomenti (niente return_properties)
        if name == "hybrid_search":
            print("[call_tool] hybrid_search called with:", args)

            clean_args: Dict[str, Any] = {}

            # collection con default "WindBilance"
            clean_args["collection"] = args.get("collection") or "WindBilance"

            # query obbligatoria
            q = args.get("query")
            if not q:
                return types.ServerResult(
                    types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text",
                                text="Errore: parametro obbligatorio 'query' mancante per hybrid_search.",
                            )
                        ],
                        isError=True,
                    )
                )
            clean_args["query"] = q

            # parametri opzionali
            if "limit" in args:
                clean_args["limit"] = args["limit"]
            if "alpha" in args:
                clean_args["alpha"] = args["alpha"]
            if "query_properties" in args:
                clean_args["query_properties"] = args["query_properties"]

            # 🔴 QUI LA COSA IMPORTANTE:
            # sovrascriviamo args con la versione ripulita
            # (così return_properties e qualsiasi altro extra SPARISCONO)
            args = clean_args

        # Tutti gli altri tool normali rimangono come prima
        try:
            # Proviamo a passare gli argomenti così come sono
            result = fn(**args)
            # Se la funzione è async, await
            if hasattr(result, "__await__"):
                result = await result
        except TypeError as e:
            # Se la firma non combacia (ad es. tool senza parametri), riproviamo senza args
            try:
                result = fn()
                if hasattr(result, "__await__"):
                    result = await result
            except Exception as e2:
                return types.ServerResult(
                    types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text",
                                text=f"Errore chiamando tool {name}: {e2}",
                            )
                        ],
                        isError=True,
                    )
                )
        except Exception as e:
            return types.ServerResult(
                types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=f"Errore chiamando tool {name}: {e}",
                        )
                    ],
                    isError=True,
                )
            )

        text_msg = f"Risultato del tool {name} disponibile in structuredContent."

        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=text_msg,
                    )
                ],
                structuredContent=(
                    result if isinstance(result, dict) else {"result": result}
                ),
            )
        )

    # 3) Tool sconosciuto
    return types.ServerResult(
        types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Unknown tool: {name}",
                )
            ],
            isError=True,
        )
    )


# Registra i request handler sul server MCP
mcp._mcp_server.request_handlers[types.CallToolRequest] = _call_tool_request
mcp._mcp_server.request_handlers[types.ReadResourceRequest] = _handle_read_resource


# ==== Esponi l'app ASGI per uvicorn (per uso diretto nello start command) ====
# Puoi usare: uvicorn serve:app --host 0.0.0.0 --port $PORT
# Come nell'esempio Pizzaz, usiamo semplicemente mcp.streamable_http_app()
try:
    app = mcp.streamable_http_app()
    if app is None:
        raise ValueError("streamable_http_app() returned None")
    print("[mcp] app obtained via streamable_http_app()")
except Exception as e:
    print(f"[mcp] error getting app via streamable_http_app(): {e}")
    # Fallback: prova a ottenere l'app in altri modi
    from starlette.applications import Starlette
    app = None
    for attr_name in ["app", "_app", "asgi_app", "_asgi_app"]:
        app = getattr(mcp, attr_name, None)
        if app and isinstance(app, Starlette):
            print(f"[mcp] found app via mcp.{attr_name} (fallback)")
            break
    if app is None:
        raise RuntimeError("Cannot get FastMCP app - streamable_http_app() failed and no app found")

# Aggiungi CORS middleware se disponibile (opzionale)
try:
    from starlette.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
except Exception:
    pass

# ==== main: avvia il server con uvicorn ==================
if __name__ == "__main__":
    import uvicorn
    
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "10000"))
    
    # Usa direttamente l'oggetto app che hai creato sopra
    uvicorn.run(app, host=host, port=port)

