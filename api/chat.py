"""
FERROLAN RAG - Vercel Serverless Function
==========================================
Endpoint /api/chat - Recibe pregunta, busca en Pinecone, responde con Claude.
Usa HuggingFace Inference API para embeddings (sin dependencias pesadas).
"""

import json
import os
import re
import time
import hashlib
from http.server import BaseHTTPRequestHandler

import requests as http_requests
from anthropic import Anthropic

# ── Configuracion ──────────────────────────────────────────
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_HOST = "https://ferrolan-rag-qtvdakx.svc.aped-4627-b74a.pinecone.io"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
PRESTASHOP_API_KEY = os.environ.get("PRESTASHOP_API_KEY", "")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_INFERENCE_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"

SHOP_URL = "https://ferrolan.es"
API_BASE = f"{SHOP_URL}/api"


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
    "horario", "direccion", "contacto", "envio", "devolucion", "garantia",
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


# ── Router ─────────────────────────────────────────────────
def route_query(query):
    query_lower = query.lower()
    rules = {
        "productos": ["porcelanico", "porcelánico", "pasta blanca", "gres", "20mm", "mosaico", "slab", "tipo de azulejo"],
        "estetica": ["efecto", "marmol", "mármol", "piedra", "cemento", "madera", "hidraulico", "estilo", "tendencia", "combinar", "color", "decorar"],
        "zonas": ["baño", "bano", "ducha", "cocina", "salon", "salón", "terraza", "exterior", "piscina", "fachada"],
        "tecnico": ["pei", "antideslizante", "clase r", "clase c", "resistencia", "formato", "espesor", "rectificado", "tecnico"],
        "colocacion": ["colocar", "colocacion", "colocación", "instalar", "cemento cola", "adhesivo", "junta", "plots"],
        "mantenimiento": ["limpiar", "limpieza", "mantenimiento", "mancha", "primera limpieza", "oxido"],
        "comercial": ["marca", "precio", "presupuesto", "pedido", "stock", "oferta", "outlet"],
        "problemas": ["problema", "ceja", "desprendimiento", "fisura", "grieta", "levantado", "eflorescencia"],
        "griferia": ["grifo", "griferia", "grifería", "monomando", "termostatico", "rociador"],
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
    return selected[:3]


# ── Embeddings via HuggingFace Inference API ───────────────
def get_embedding(text):
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    response = http_requests.post(
        HF_INFERENCE_URL,
        headers=headers,
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=30,
    )
    if response.status_code != 200:
        raise Exception(f"HF Inference error {response.status_code}: {response.text[:200]}")

    result = response.json()
    # HF returns [[...]] for single input
    if isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], list):
            return result[0]
        return result
    raise Exception(f"Unexpected HF response format: {type(result)}")


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


# ── PrestaShop search ──────────────────────────────────────
STOP_WORDS = [
    "azulejo", "azulejos", "baldosa", "baldosas", "ceramica", "cerámica",
    "teneis", "tenéis", "quiero", "busco", "necesito", "recomiendas",
    "hay", "tienen", "venden", "me", "un", "una", "de", "para", "con",
    "que", "el", "la", "los", "las", "en", "del", "al", "por", "como",
]

PRODUCT_SEARCH_KEYWORDS = [
    "teneis", "tenéis", "tienen", "hay", "venden", "busco", "quiero",
    "comprar", "precio", "catalogo", "catálogo", "disponible", "stock",
    "referencia", "serie", "coleccion", "colección", "modelo", "recomiendas",
]


def wants_product_search(query):
    query_lower = query.lower()
    for kw in PRODUCT_SEARCH_KEYWORDS:
        if kw in query_lower:
            return True
    if re.search(r'\d{2,3}\s*[xX]\s*\d{2,3}', query):
        return True
    return False


def search_prestashop(query, limit=5):
    if not PRESTASHOP_API_KEY:
        return None

    # Parse query
    formato_match = re.search(r'(\d{2,3})\s*[xX]\s*(\d{2,3})', query)
    formato = None
    query_clean = query
    if formato_match:
        w, h = int(formato_match.group(1)), int(formato_match.group(2))
        formato = f"{min(w,h)}X{max(w,h)}"
        query_clean = query.replace(formato_match.group(0), "").strip()

    words = query_clean.lower().split()
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]

    auth = (PRESTASHOP_API_KEY, "")
    all_products = []

    for kw in keywords[:3]:
        try:
            r = http_requests.get(
                f"{API_BASE}/products",
                params={
                    "output_format": "JSON",
                    "display": "[id,name,price,reference,link_rewrite]",
                    "filter[name]": f"%[{kw.upper()}]%",
                    "limit": "10",
                },
                auth=auth,
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                for key in data:
                    if isinstance(data[key], list):
                        all_products.extend(data[key])
        except Exception:
            pass

    if formato:
        try:
            r = http_requests.get(
                f"{API_BASE}/products",
                params={
                    "output_format": "JSON",
                    "display": "[id,name,price,reference,link_rewrite]",
                    "filter[name]": f"%[{formato}]%",
                    "limit": "10",
                },
                auth=auth,
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                for key in data:
                    if isinstance(data[key], list):
                        all_products.extend(data[key])
        except Exception:
            pass

    # Deduplicate and format
    seen = set()
    unique = []
    for p in all_products:
        pid = str(p.get("id", ""))
        if pid and pid not in seen and p.get("name"):
            seen.add(pid)
            unique.append(p)

    if not unique:
        return None

    # Score by relevance
    if keywords:
        scored = []
        for p in unique:
            name_lower = p.get("name", "").lower()
            score = sum(1 for kw in keywords if kw in name_lower)
            if formato and formato in p.get("name", "").upper().replace(" ", ""):
                score += 2
            scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        unique = [p for s, p in scored if s > 0]

    lines = [f"\n--- PRODUCTOS DEL CATALOGO FERROLAN.ES ({min(len(unique), limit)} resultados) ---\n"]
    for i, p in enumerate(unique[:limit]):
        name = p.get("name", "")
        ref = p.get("reference", "")
        try:
            price = round(float(p.get("price", 0)) * 1.21, 2)
        except (ValueError, TypeError):
            price = 0

        link = p.get("link_rewrite", "")
        url = f"{SHOP_URL}/{link}-{ref}" if link and ref else ""

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
def call_claude(query, context, sources, has_unverified):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    user_message = f"""Contexto de documentos de Ferrolan:

{context}

---

Pregunta del cliente: {query}

Responde usando SOLO la informacion del contexto proporcionado. Cita las fuentes al final."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
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
            body = json.loads(self.rfile.read(content_length))
            query = body.get("query", "").strip()

            if not query or len(query) < 3:
                self._json_response(400, {"error": "Query must be at least 3 characters"})
                return

            if len(query) > 500:
                self._json_response(400, {"error": "Query too long (max 500 chars)"})
                return

            t_total = time.time()

            # Topic guard
            if not is_on_topic(query):
                self._json_response(200, {
                    "answer": OFF_TOPIC_RESPONSE,
                    "sources": [],
                    "dominios_consultados": [],
                    "tiempo_busqueda": 0,
                    "tiempo_total": round(time.time() - t_total, 3),
                    "has_unverified": False,
                    "session_id": "vercel",
                    "cached": False,
                })
                return

            # Route query
            dominios = route_query(query)

            # Get embedding
            t_search = time.time()
            embedding = get_embedding(query)

            # Search Pinecone
            results = search_pinecone(embedding, dominios)
            search_time = time.time() - t_search

            # Search PrestaShop
            product_context = None
            if wants_product_search(query):
                try:
                    product_context = search_prestashop(query)
                except Exception:
                    pass

            if not results and not product_context:
                self._json_response(200, {
                    "answer": "Lo siento, no he encontrado informacion relevante sobre eso. Te recomiendo consultarlo con nuestro equipo en tienda.",
                    "sources": [],
                    "dominios_consultados": dominios,
                    "tiempo_busqueda": round(search_time, 3),
                    "tiempo_total": round(time.time() - t_total, 3),
                    "has_unverified": False,
                    "session_id": "vercel",
                    "cached": False,
                })
                return

            # Build context and call LLM
            context, source_names, has_unverified = build_context(results, product_context)
            answer = call_claude(query, context, source_names, has_unverified)

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

            self._json_response(200, {
                "answer": answer,
                "sources": sources_out[:5],
                "dominios_consultados": dominios,
                "tiempo_busqueda": round(search_time, 3),
                "tiempo_total": round(time.time() - t_total, 3),
                "has_unverified": has_unverified,
                "session_id": "vercel",
                "cached": False,
            })

        except Exception as e:
            self._json_response(500, {"error": f"Error interno: {str(e)[:200]}"})

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
