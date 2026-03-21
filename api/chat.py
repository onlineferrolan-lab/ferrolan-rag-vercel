"""
FERROLAN RAG - Vercel Serverless Function
==========================================
Endpoint /api/chat - Recibe pregunta, busca en Pinecone, responde con Claude.
Usa Pinecone Inference para embeddings (multilingual-e5-large, 1024 dims).
"""

import json
import os
import re
import time
import hashlib
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler

import requests as http_requests
from anthropic import Anthropic

# ── Configuracion ──────────────────────────────────────────
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_HOST = "https://ferrolan-rag-qtvdakx.svc.aped-4627-b74a.pinecone.io"
PINECONE_INFERENCE_URL = "https://api.pinecone.io/embed"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
PRESTASHOP_API_KEY = os.environ.get("PRESTASHOP_API_KEY", "")

SHOP_URL = "https://ferrolan.es"
API_BASE = f"{SHOP_URL}/api"

# ── Memoria conversacional (en memoria, por instancia serverless) ──
MAX_SESSIONS = 200          # Limitar memoria por instancia
MAX_HISTORY_TURNS = 5       # Últimos 5 intercambios (user+assistant)
SESSION_TTL = 1800          # 30 min sin actividad → se borra

class ConversationStore:
    """Almacén LRU de historiales de conversación con TTL."""
    def __init__(self, max_size=MAX_SESSIONS):
        self._store = OrderedDict()
        self._max_size = max_size

    def get(self, session_id):
        if session_id in self._store:
            entry = self._store[session_id]
            if time.time() - entry["last_active"] > SESSION_TTL:
                del self._store[session_id]
                return []
            self._store.move_to_end(session_id)
            entry["last_active"] = time.time()
            return entry["messages"]
        return []

    def add_turn(self, session_id, user_msg, assistant_msg):
        if session_id not in self._store:
            if len(self._store) >= self._max_size:
                self._store.popitem(last=False)
            self._store[session_id] = {"messages": [], "last_active": time.time()}
        entry = self._store[session_id]
        entry["messages"].append({"role": "user", "content": user_msg})
        entry["messages"].append({"role": "assistant", "content": assistant_msg})
        # Mantener solo los últimos N turnos (N*2 mensajes)
        if len(entry["messages"]) > MAX_HISTORY_TURNS * 2:
            entry["messages"] = entry["messages"][-(MAX_HISTORY_TURNS * 2):]
        entry["last_active"] = time.time()
        self._store.move_to_end(session_id)

conversation_store = ConversationStore()


# ── Observabilidad: métricas por instancia ────────────────
class QueryMetrics:
    """Registra métricas de cada query para análisis de calidad."""
    def __init__(self, max_entries=500):
        self._entries = []
        self._max = max_entries
        self._counters = {
            "total": 0, "off_topic": 0, "no_results": 0,
            "with_products": 0, "with_unverified": 0,
            "expanded_queries": 0, "reranked": 0,
        }

    def log(self, entry):
        self._counters["total"] += 1
        if entry.get("off_topic"):
            self._counters["off_topic"] += 1
        if entry.get("no_results"):
            self._counters["no_results"] += 1
        if entry.get("has_products"):
            self._counters["with_products"] += 1
        if entry.get("has_unverified"):
            self._counters["with_unverified"] += 1
        if entry.get("query_expanded"):
            self._counters["expanded_queries"] += 1
        if entry.get("reranked"):
            self._counters["reranked"] += 1
        self._entries.append(entry)
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max:]

    def summary(self):
        if not self._entries:
            return {"counters": self._counters, "recent": [], "avg_times": {}}
        recent = self._entries[-20:]
        times = [e.get("tiempo_total", 0) for e in self._entries if e.get("tiempo_total")]
        search_times = [e.get("tiempo_busqueda", 0) for e in self._entries if e.get("tiempo_busqueda")]
        scores = [e.get("top_score", 0) for e in self._entries if e.get("top_score")]
        return {
            "counters": self._counters,
            "avg_times": {
                "total": round(sum(times) / len(times), 3) if times else 0,
                "search": round(sum(search_times) / len(search_times), 3) if search_times else 0,
            },
            "avg_top_score": round(sum(scores) / len(scores), 3) if scores else 0,
            "recent": [{
                "query": e.get("query", "")[:80],
                "dominios": e.get("dominios", []),
                "top_score": e.get("top_score", 0),
                "tiempo_total": e.get("tiempo_total", 0),
                "query_expanded": e.get("query_expanded", False),
                "num_results": e.get("num_results", 0),
            } for e in recent],
        }

query_metrics = QueryMetrics()


# ── System Prompt ──────────────────────────────────────────
SYSTEM_PROMPT = """Eres el asistente virtual de FERROLAN, una empresa especializada en ceramica, griferia y materiales de construccion y reforma.

Tu trabajo es ayudar a los clientes a elegir los productos adecuados para sus proyectos, resolver dudas tecnicas y orientar sobre colocacion, mantenimiento y estilos.

REGLAS DE COMPORTAMIENTO:
1. Responde SOLO con informacion de los documentos que se te proporcionan como contexto. Si no tienes informacion suficiente, dilo claramente: "No dispongo de informacion suficiente sobre este tema. Te recomiendo consultarlo con nuestro equipo en tienda."
2. NUNCA inventes datos tecnicos (PEI, clases antideslizante, medidas, precios). Si no aparecen en el contexto, no los menciones.
3. Usa un tono profesional pero cercano, como un asesor ceramico experimentado.
4. Cuando sea relevante, sugiere al cliente visitar la tienda de Ferrolan para ver las piezas en persona o consultar la web ferrolan.es.
5. Si el cliente pregunta sobre griferia pero no tienes contexto de griferia, indica que estais ampliando esa seccion.
6. Cita las fuentes al final de cada respuesta con el formato: [Fuente: titulo del documento]
7. Si la informacion viene de un documento NO verificado, anade al final: "(Nota: esta informacion esta pendiente de revision por nuestro equipo tecnico)"
8. Responde siempre en espanol.
9. Se conciso: respuestas claras de 2-4 parrafos maximo, salvo que el cliente pida detalle.
10. IMPORTANTE: Solo puedes responder sobre temas relacionados con ceramica, azulejos, griferia, bano, cocina, construccion, reformas y los productos/servicios de Ferrolan. Si el usuario pregunta sobre cualquier otro tema NO relacionado, responde SIEMPRE: "Soy el asistente de Ferrolan y solo puedo ayudarte con temas relacionados con ceramica, griferia, reformas y nuestros productos."
11. Cuando muestres productos del catalogo de Ferrolan, incluye el enlace a la web si esta disponible en el contexto.

FORMATO DE RESPUESTA:
- Usa parrafos cortos y listas con guiones cuando sea util.
- Pon los datos tecnicos clave en negrita (ej: **PEI 4**, **clase C2**, **formato 60x60**).
- Si hay varias opciones, presentalas como lista comparativa.
- Cuando muestres productos, incluye: nombre, formato, precio y enlace si disponible.
"""


# ── Guardia de tema ────────────────────────────────────────
TOPIC_KEYWORDS = [
    "azulejo", "baldosa", "ceramica", "cerámica", "porcelanico", "porcelánico",
    "gres", "pasta blanca", "mosaico", "slab", "pavimento", "revestimiento",
    "suelo", "pared", "loseta", "formato", "60x60", "60x120", "30x60",
    "efecto", "marmol", "mármol", "piedra", "madera", "cemento", "hidraulico",
    "terrazo", "estilo", "tendencia", "color", "decorado",
    "bano", "baño", "cocina", "salon", "salón", "terraza", "exterior",
    "ducha", "piscina", "fachada",
    "pei", "antideslizante", "resistencia", "calibre", "espesor",
    "colocar", "colocacion", "colocación", "instalar", "adhesivo", "junta",
    "limpiar", "limpieza", "mantenimiento", "mancha",
    "grifo", "griferia", "grifería", "monomando", "termostatico",
    "precio", "presupuesto", "marca", "comprar", "tienda",
    "problema", "desprendimiento", "fisura", "grieta",
    "reforma", "obra", "construccion", "construcción",
    "ferrolan", "catalogo", "catálogo", "producto", "coleccion", "colección",
    "horario", "direccion", "dirección", "contacto", "envio", "envío",
    "devolucion", "devolución", "devolver", "garantia", "garantía",
    "tienda", "tiendas", "donde", "dónde", "visitar", "exposicion", "exposición",
    "parking", "aparcamiento", "telefono", "teléfono", "email", "correo",
    "urgell", "eixample", "santa coloma", "rubi", "rubí", "badalona", "clot",
    "meridiana", "outlet fondo", "outlet central", "sede",
    "cajas", "metros", "m2", "cantidad", "cuantas", "cuántas", "sobra", "sobran",
    "pedido", "fabrica", "fábrica", "stock", "outlet",
    "parquet", "laminado", "vinilico", "vinílico", "tarima",
    "mampara", "lavabo", "sanitario", "inodoro", "mueble de baño",
    "plato de ducha", "bañera", "banera",
]

GREETINGS = [
    "hola", "buenas", "buenos dias", "buenos días", "buenas tardes",
    "buenas noches", "hey", "gracias", "adios", "adiós", "vale", "ok",
    "si", "sí", "no", "de acuerdo", "perfecto", "genial", "ayuda",
]

OFF_TOPIC_RESPONSE = (
    "Soy el asistente de Ferrolan y solo puedo ayudarte con temas relacionados "
    "con ceramica, griferia, reformas y nuestros productos. "
    "Si tienes alguna duda sobre azulejos, banos, cocinas o materiales de construccion, "
    "estare encantado de ayudarte."
)


def is_on_topic(query):
    query_lower = query.lower().strip()
    for g in GREETINGS:
        if query_lower == g or query_lower.startswith(g + " ") or query_lower.startswith(g + ","):
            return True
    if len(query_lower.split()) < 3:
        return True
    for kw in TOPIC_KEYWORDS:
        if kw in query_lower:
            return True
    return False


# ── Router semántico ──────────────────────────────────────
AVAILABLE_DOMAINS = [
    "productos",      # tipos de cerámica: porcelánico, gres, pasta blanca, mosaico, slab, 20mm
    "estetica",       # efectos (mármol, piedra, madera, cemento), estilos, tendencias, colores
    "zonas",          # estancias: baño, cocina, terraza, exterior, piscina, fachada, salón
    "tecnico",        # PEI, antideslizante, formatos, espesores, rectificado, acabados
    "colocacion",     # cemento cola, adhesivos, juntas, plots, patrones de colocación
    "mantenimiento",  # limpieza, manchas, mantenimiento periódico
    "comercial",      # precios, marcas, pedidos, stock, cálculo de material, devoluciones
    "problemas",      # fisuras, desprendimientos, eflorescencias, cejas, errores
    "griferia",       # grifos, monomandos, termostáticos, duchas, rociadores
    "general",        # tiendas Ferrolan, horarios, direcciones, contacto
]

DOMAIN_DESCRIPTIONS = "\n".join([
    f"- {d}" for d in AVAILABLE_DOMAINS
])

ROUTER_PROMPT = f"""Eres un clasificador de consultas para Ferrolan (tienda de cerámica y grifería).
Dada una consulta de cliente, clasifícala en 1 a 3 de estos dominios:

{DOMAIN_DESCRIPTIONS}

Reglas:
- Elige los dominios cuyo contenido sea NECESARIO para responder bien.
- Si la consulta implica elegir producto para una zona, incluye "zonas" Y "productos".
- Si implica características técnicas (antideslizante, resistencia, PEI), incluye "tecnico".
- Si pregunta por tiendas, horarios o contacto, usa "general".
- Responde SOLO con los nombres de dominio separados por coma, sin explicación."""


def route_query_semantic(query):
    """Clasificación semántica con Haiku — rápido y preciso."""
    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=60,
            temperature=0,
            messages=[{"role": "user", "content": f"Consulta: \"{query}\""}],
            system=ROUTER_PROMPT,
        )
        raw = response.content[0].text.strip().lower()
        # Parsear dominios de la respuesta
        domains = [d.strip() for d in raw.split(",")]
        # Filtrar solo dominios válidos
        valid = [d for d in domains if d in AVAILABLE_DOMAINS]
        if valid:
            if "general" not in valid:
                valid.append("general")
            return valid[:4]
    except Exception:
        pass
    # Fallback al routing por keywords si falla Haiku
    return route_query_keywords(query)


def route_query_keywords(query):
    """Fallback: routing por keywords (método original)."""
    query_lower = query.lower()
    rules = {
        "productos": ["porcelanico", "porcelánico", "pasta blanca", "gres", "20mm", "mosaico", "slab", "tipo de azulejo"],
        "estetica": ["efecto", "marmol", "mármol", "piedra", "cemento", "madera", "hidraulico", "estilo", "tendencia", "combinar", "color", "decorar", "se lleva", "moda", "actual"],
        "zonas": ["baño", "bano", "ducha", "cocina", "salon", "salón", "terraza", "exterior", "piscina", "fachada"],
        "tecnico": ["pei", "antideslizante", "clase r", "clase c", "resistencia", "formato", "espesor", "rectificado", "tecnico", "acabado", "canto", "biselado"],
        "colocacion": ["colocar", "colocacion", "colocación", "instalar", "cemento cola", "adhesivo", "junta", "plots", "calzos", "cuñas", "rejuntado", "mortero de junta", "patron", "patrón", "espiga"],
        "mantenimiento": ["limpiar", "limpieza", "mantenimiento", "mancha", "primera limpieza", "oxido"],
        "comercial": ["marca", "precio", "presupuesto", "pedido", "stock", "oferta", "outlet", "devolucion", "devolución", "devolver", "sobra", "sobran", "cajas", "cantidad", "cuantas", "cuántas", "metros", "m2", "desperdicio", "merma", "calculo", "cálculo", "fabrica", "fábrica"],
        "problemas": ["problema", "ceja", "desprendimiento", "fisura", "grieta", "levantado", "eflorescencia"],
        "griferia": ["grifo", "griferia", "grifería", "monomando", "termostatico", "rociador"],
        "general": ["tienda", "tiendas", "horario", "horarios", "direccion", "dirección", "contacto", "telefono", "teléfono", "email", "visitar", "parking", "aparcamiento", "exposicion", "exposición", "donde", "dónde", "urgell", "eixample", "santa coloma", "rubi", "rubí", "badalona", "clot", "meridiana", "outlet fondo", "outlet central", "sede", "ferrolan"],
    }
    selected = []
    for dominio, keywords in rules.items():
        for kw in keywords:
            if kw in query_lower:
                if dominio not in selected:
                    selected.append(dominio)
                break
    if not selected:
        selected = ["productos", "zonas", "tecnico"]
    if "general" not in selected:
        selected.append("general")
    return selected[:4]


# ── Query Expansion (reformulación con historial) ────────
def expand_query(query, chat_history):
    """Reformula una pregunta ambigua usando el historial conversacional.
    Ejemplo: historial habla de porcelánico para cocina, usuario dice
    '¿y en formato grande?' → 'porcelánico en formato grande para cocina'
    """
    if not chat_history:
        return query

    # Solo usar los últimos 2 turnos (4 mensajes) para contexto
    recent = chat_history[-(2 * 2):]
    history_text = "\n".join([
        f"{'Cliente' if m['role'] == 'user' else 'Asistente'}: {m['content'][:200]}"
        for m in recent
    ])

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            temperature=0,
            system="""Eres un reformulador de consultas para un buscador de cerámica y grifería.
Dada una conversación previa y una nueva pregunta del cliente, reformula la pregunta para que sea AUTOCONTENIDA (incluya todo el contexto necesario de la conversación).
Si la pregunta ya es clara y autocontenida, devuélvela tal cual.
Responde SOLO con la pregunta reformulada, sin explicación.""",
            messages=[{"role": "user", "content": f"Conversación previa:\n{history_text}\n\nNueva pregunta: \"{query}\""}],
        )
        expanded = response.content[0].text.strip()
        # Sanity check: no devolver algo absurdo
        if expanded and 3 < len(expanded) < 500:
            return expanded
    except Exception:
        pass
    return query


# ── Embeddings via Pinecone Inference REST API ────────────
def get_embedding(text):
    """Genera embedding usando Pinecone Inference REST (multilingual-e5-large, 1024 dims)."""
    response = http_requests.post(
        PINECONE_INFERENCE_URL,
        headers={
            "Api-Key": PINECONE_API_KEY,
            "Content-Type": "application/json",
            "X-Pinecone-Api-Version": "2025-10",
        },
        json={
            "model": "multilingual-e5-large",
            "inputs": [{"text": text}],
            "parameters": {"input_type": "query"},
        },
        timeout=15,
    )
    if response.status_code != 200:
        raise Exception(f"Pinecone Inference error {response.status_code}: {response.text[:200]}")
    return response.json()["data"][0]["values"]


# ── Pinecone search ───────────────────────────────────────
def search_pinecone(embedding, dominios, top_k=8):
    headers = {
        "Api-Key": PINECONE_API_KEY,
        "Content-Type": "application/json",
    }

    # Build filter
    if len(dominios) == 1:
        filter_obj = {"dominio": {"$eq": dominios[0]}}
    else:
        filter_obj = {"dominio": {"$in": dominios}}

    # Search with domain filter
    body = {
        "vector": embedding,
        "topK": top_k,
        "includeMetadata": True,
        "filter": filter_obj,
    }

    r = http_requests.post(
        f"{PINECONE_HOST}/query",
        headers=headers,
        json=body,
        timeout=10,
    )

    results = []
    if r.status_code == 200:
        matches = r.json().get("matches", [])
        for m in matches:
            results.append({
                "text": m["metadata"].get("text", ""),
                "metadata": m["metadata"],
                "score": m["score"],
                "source": "semantic",
            })

    # Also search without filter (global) to catch misrouted queries
    body_global = {
        "vector": embedding,
        "topK": 4,
        "includeMetadata": True,
    }

    r2 = http_requests.post(
        f"{PINECONE_HOST}/query",
        headers=headers,
        json=body_global,
        timeout=10,
    )

    seen_ids = {r["metadata"].get("doc_id", "") + r["metadata"].get("seccion", "")[:30] for r in results}

    if r2.status_code == 200:
        for m in r2.json().get("matches", []):
            key = m["metadata"].get("doc_id", "") + m["metadata"].get("seccion", "")[:30]
            if key not in seen_ids and m["score"] > 0.5:
                results.append({
                    "text": m["metadata"].get("text", ""),
                    "metadata": m["metadata"],
                    "score": m["score"] * 0.9,
                    "source": "global",
                })
                seen_ids.add(key)

    results.sort(key=lambda r: -r["score"])
    return results[:top_k]


# ── Reranking via Pinecone Inference ─────────────────────
def rerank_results(query, results, top_n=8):
    """Reordena resultados usando Pinecone Rerank para mayor precisión."""
    if not results or len(results) <= 1:
        return results

    documents = []
    for r in results:
        titulo = r["metadata"].get("titulo_documento", "")
        seccion = r["metadata"].get("seccion", "")
        text = r.get("text", "")
        documents.append(f"[{titulo}] ({seccion}) {text}")

    try:
        response = http_requests.post(
            "https://api.pinecone.io/rerank",
            headers={
                "Api-Key": PINECONE_API_KEY,
                "Content-Type": "application/json",
                "X-Pinecone-Api-Version": "2025-10",
            },
            json={
                "model": "bge-reranker-v2-m3",
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
            },
            timeout=10,
        )
        if response.status_code != 200:
            return results[:top_n]

        reranked_data = response.json().get("data", [])
        reranked = []
        for item in reranked_data:
            idx = item["index"]
            r = results[idx].copy()
            r["rerank_score"] = item["score"]
            reranked.append(r)
        return reranked

    except Exception:
        return results[:top_n]


# ── PrestaShop search ──────────────────────────────────────
STOP_WORDS = [
    "azulejo", "azulejos", "baldosa", "baldosas", "ceramica", "cerámica",
    "teneis", "tenéis", "quiero", "busco", "necesito", "recomiendas",
    "hay", "tienen", "venden", "me", "un", "una", "de", "para", "con",
    "que", "el", "la", "los", "las", "en", "del", "al", "por", "como",
    "cuales", "cuáles", "mejor", "mejores", "recomendacion", "estilo",
]

PRODUCT_SEARCH_KEYWORDS = [
    "teneis", "tenéis", "tienen", "hay", "venden", "busco", "quiero",
    "comprar", "precio", "catalogo", "catálogo", "disponible", "stock",
    "referencia", "serie", "coleccion", "colección", "modelo", "recomiendas",
    "recomienda", "recomendacion", "sugieres", "opciones",
]

# Mapeo de palabras del usuario -> category IDs en PrestaShop
# Estructura de categorias: Azulejos(16) > Efecto(18), Estilo(17), Zona(19), Tipo(20)
CATEGORY_MAP = {
    # Efectos (subcategorias de 18)
    "cemento":       [31],
    "marmol":        [36],
    "mármol":        [36],
    "piedra":        [37],
    "madera":        [34],
    "hidraulico":    [32],
    "hidráulico":    [32],
    "terrazo":       [38],
    "monocolor":     [35],
    "barro":         [30],
    "oxido":         [41],
    "óxido":         [41],
    "textil":        [39],
    "ladrillo":      [33],
    "botanico":      [40],
    "botánico":      [40],
    # Estilos (subcategorias de 17)
    "nordico":       [27],
    "nórdico":       [27],
    "industrial":    [25],
    "vintage":       [29],
    "clasico":       [22],
    "clásico":       [22],
    "contemporaneo": [24],
    "contemporáneo": [24],
    "mediterraneo":  [26],
    "mediterráneo":  [26],
    "rustico":       [28],
    "rústico":       [28],
    # Zonas de uso (subcategorias de 19)
    "bano":          [42],
    "baño":          [42],
    "cocina":        [43],
    "exterior":      [46],
    "terraza":       [46],
    "piscina":       [49],
    "fachada":       [47],
    "comercio":      [44],
    "comercial":     [44],
    # Tipo producto (subcategorias de 20)
    "porcelanico":   [19540],
    "porcelánico":   [19540],
    "gres":          [54, 57],
    "revestimiento": [19558],
    "mosaico":       [52],
    "slab":          [55],
    "20mm":          [51],
    "gran formato":  [53],
}


def wants_product_search(query):
    query_lower = query.lower()
    for kw in PRODUCT_SEARCH_KEYWORDS:
        if kw in query_lower:
            return True
    if re.search(r'\d{2,3}\s*[xX]\s*\d{2,3}', query):
        return True
    for kw in CATEGORY_MAP:
        if kw in query_lower:
            return True
    return False


def _ps_api_get(endpoint, params, auth):
    """Helper for PrestaShop API calls."""
    params["output_format"] = "JSON"
    params["language"] = "1"
    try:
        r = http_requests.get(f"{API_BASE}/{endpoint}", params=params, auth=auth, timeout=12)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _get_category_product_ids(category_id, auth, max_ids=50):
    """Obtiene IDs de productos de una categoria via la API de categorias."""
    data = _ps_api_get(f"categories/{category_id}", {}, auth)
    if not data:
        return []
    cat = data.get("category", {})
    assoc = cat.get("associations", {})
    products = assoc.get("products", [])
    ids = [str(p.get("id", "")) for p in products if p.get("id")]
    return ids[:max_ids]


def search_prestashop(query, limit=5):
    if not PRESTASHOP_API_KEY:
        return None

    auth = (PRESTASHOP_API_KEY, "")
    query_lower = query.lower()

    # Parse formato (NNxNN)
    formato_match = re.search(r'(\d{2,3})\s*[xX]\s*(\d{2,3})', query)
    formato = None
    if formato_match:
        w, h = int(formato_match.group(1)), int(formato_match.group(2))
        formato = f"{min(w,h)}X{max(w,h)}"

    # Find matching categories from query
    matched_categories = []
    for keyword, cat_ids in CATEGORY_MAP.items():
        if keyword in query_lower:
            matched_categories.extend(cat_ids)

    if not matched_categories:
        return None

    # Get product IDs from each category and intersect
    category_product_sets = []
    for cat_id in matched_categories:
        ids = _get_category_product_ids(cat_id, auth, max_ids=200)
        if ids:
            category_product_sets.append(set(ids))

    if not category_product_sets:
        return None

    # Intersect all category sets (product must be in ALL matched categories)
    candidate_ids = category_product_sets[0]
    for s in category_product_sets[1:]:
        candidate_ids = candidate_ids & s

    if not candidate_ids:
        # If intersection is empty, use union of first category (most specific)
        candidate_ids = category_product_sets[0]

    # Get product details for a sample
    sample_ids = list(candidate_ids)[:30]
    id_filter = "|".join(sample_ids)
    data = _ps_api_get("products", {
        "display": "[id,name,price,reference,link_rewrite]",
        "filter[id]": f"[{id_filter}]",
    }, auth)

    products = []
    if data:
        for key in data:
            if isinstance(data[key], list):
                products = data[key]
                break

    # Filter out products without name
    products = [p for p in products if p.get("name")]

    if not products:
        return None

    # Score by formato match in name
    scored = []
    for p in products:
        score = 1
        name_lower = p.get("name", "").lower()
        if formato:
            # Check if formato appears in product name
            fmt_lower = formato.lower()
            name_clean = name_lower.replace(" ", "")
            if fmt_lower in name_clean or fmt_lower.replace("x", "x") in name_clean:
                score += 10
        scored.append((score, p))

    scored.sort(key=lambda x: -x[0])
    top_products = [p for s, p in scored[:limit]]

    lines = [f"\n--- PRODUCTOS DEL CATALOGO FERROLAN.ES ({len(top_products)} resultados) ---\n"]
    for i, p in enumerate(top_products):
        name = p.get("name", "")
        ref = p.get("reference", "")
        try:
            price = round(float(p.get("price", 0)) * 1.21, 2)
        except (ValueError, TypeError):
            price = 0

        link = p.get("link_rewrite", "")
        url = f"{SHOP_URL}/{link}" if link else ""

        line = f"{i+1}. {name}"
        if ref:
            line += f" (Ref: {ref})"
        if price > 0:
            line += f" - {price:.2f} EUR/m2 IVA incl."
        if url:
            line += f"\n   URL: {url}"
        lines.append(line)

    lines.append("\nNota: los precios son orientativos. Consultar disponibilidad y precio final en tienda o web.")
    return "\n".join(lines)


# ── Build context ──────────────────────────────────────────
def build_context(results, product_context=None):
    context_parts = []
    sources = []
    has_unverified = False

    for i, r in enumerate(results):
        meta = r["metadata"]
        verificado = meta.get("verificado", False)
        titulo = meta.get("titulo_documento", "Sin titulo")
        seccion = meta.get("seccion", "")

        if not verificado:
            has_unverified = True

        header = f"[Documento {i+1}: {titulo}]"
        if seccion:
            header += f" (Seccion: {seccion})"
        if not verificado:
            header += " [NO VERIFICADO]"

        context_parts.append(f"{header}\n{r['text']}\n")
        if titulo not in sources:
            sources.append(titulo)

    context = "\n---\n".join(context_parts)
    if product_context:
        context += f"\n\n{product_context}"
        sources.append("Catalogo ferrolan.es")

    return context, sources, has_unverified


# ── Call Claude ────────────────────────────────────────────
def call_claude(query, context, sources, has_unverified, chat_history=None):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    user_message = f"""Contexto de documentos de Ferrolan:

{context}

---

Pregunta del cliente: {query}

Responde usando SOLO la informacion del contexto proporcionado. Cita las fuentes al final."""

    # Construir mensajes con historial previo
    messages = []
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        temperature=0.1,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    answer = response.content[0].text
    if has_unverified:
        answer += "\n\n_(Nota: parte de esta informacion esta pendiente de revision por nuestro equipo tecnico.)_"

    return answer


# ── Vercel handler ─────────────────────────────────────────
class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_bytes = self.rfile.read(content_length)
            # Ensure UTF-8 decoding (handle Latin-1 fallback)
            try:
                raw_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raw_text = raw_bytes.decode("latin-1")
            body = json.loads(raw_text)
            query = body.get("query", "").strip()
            session_id = body.get("session_id", "")

            if not query or len(query) < 3:
                self._json_response(400, {"error": "Query must be at least 3 characters"})
                return

            if len(query) > 500:
                self._json_response(400, {"error": "Query too long (max 500 chars)"})
                return

            t_total = time.time()

            # Topic guard
            if not is_on_topic(query):
                query_metrics.log({"query": query, "off_topic": True, "tiempo_total": round(time.time() - t_total, 3)})
                self._json_response(200, {
                    "answer": OFF_TOPIC_RESPONSE,
                    "sources": [],
                    "dominios_consultados": [],
                    "tiempo_busqueda": 0,
                    "tiempo_total": round(time.time() - t_total, 3),
                    "has_unverified": False,
                    "session_id": session_id or "vercel",
                    "cached": False,
                })
                return

            # Query expansion: reformular si hay historial conversacional
            chat_history = conversation_store.get(session_id) if session_id else []
            search_query = expand_query(query, chat_history) if chat_history else query

            # Route query (semántico con Haiku, fallback a keywords)
            dominios = route_query_semantic(search_query)

            # Get embedding (con la query expandida para mejor búsqueda)
            t_search = time.time()
            embedding = get_embedding(search_query)

            # Search Pinecone (topK=12 para tener más candidatos antes del reranking)
            results = search_pinecone(embedding, dominios, top_k=12)

            # Reranking: reordena por relevancia real y devuelve top 8
            if results:
                results = rerank_results(query, results, top_n=8)

            search_time = time.time() - t_search

            # Search PrestaShop
            product_context = None
            if wants_product_search(query):
                try:
                    product_context = search_prestashop(query)
                except Exception:
                    pass

            if not results and not product_context:
                query_metrics.log({"query": query, "no_results": True, "dominios": dominios, "tiempo_busqueda": round(search_time, 3), "tiempo_total": round(time.time() - t_total, 3), "query_expanded": search_query != query})
                self._json_response(200, {
                    "answer": "Lo siento, no he encontrado informacion relevante sobre eso. Te recomiendo consultarlo con nuestro equipo en tienda.",
                    "sources": [],
                    "dominios_consultados": dominios,
                    "tiempo_busqueda": round(search_time, 3),
                    "tiempo_total": round(time.time() - t_total, 3),
                    "has_unverified": False,
                    "session_id": session_id or "vercel",
                    "cached": False,
                })
                return

            # Build context and call LLM
            context, source_names, has_unverified = build_context(results, product_context)
            answer = call_claude(query, context, source_names, has_unverified, chat_history)

            # Guardar turno en historial
            if session_id:
                conversation_store.add_turn(session_id, query, answer)

            # Format sources
            sources_out = []
            seen_titles = set()
            for r in results:
                titulo = r["metadata"].get("titulo_documento", "")
                if titulo and titulo not in seen_titles:
                    sources_out.append({
                        "titulo": titulo,
                        "dominio": r["metadata"].get("dominio", ""),
                        "verificado": bool(r["metadata"].get("verificado", False)),
                        "score": round(r["score"], 3),
                    })
                    seen_titles.add(titulo)

            total_time = round(time.time() - t_total, 3)
            top_score = round(results[0]["score"], 3) if results else 0

            # Registrar métricas
            query_metrics.log({
                "query": query,
                "search_query": search_query if search_query != query else None,
                "query_expanded": search_query != query,
                "dominios": dominios,
                "num_results": len(results),
                "top_score": top_score,
                "has_products": product_context is not None,
                "has_unverified": has_unverified,
                "reranked": bool(results and "rerank_score" in results[0]),
                "tiempo_busqueda": round(search_time, 3),
                "tiempo_total": total_time,
            })

            self._json_response(200, {
                "answer": answer,
                "sources": sources_out[:5],
                "dominios_consultados": dominios,
                "tiempo_busqueda": round(search_time, 3),
                "tiempo_total": total_time,
                "has_unverified": has_unverified,
                "session_id": session_id or "vercel",
                "cached": False,
            })

        except Exception as e:
            self._json_response(500, {"error": f"Error interno: {str(e)[:200]}"})

    def do_GET(self):
        """GET /api/chat → devuelve métricas de observabilidad."""
        self._json_response(200, query_metrics.summary())

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _json_response(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _cors_headers(self):
        origin = self.headers.get("Origin", "")
        allowed = [
            "https://ferrolan.com", "https://www.ferrolan.com",
            "https://ferrolan.es", "https://www.ferrolan.es",
            "http://localhost:3000", "http://localhost:8000",
        ]
        # Also allow the Vercel deployment domain
        if origin in allowed or origin.endswith(".vercel.app"):
            self.send_header("Access-Control-Allow-Origin", origin)
        else:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
