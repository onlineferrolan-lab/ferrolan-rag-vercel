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

# Mapeo de palabras del usuario -> feature_value IDs en PrestaShop
# Feature 86 = Efecto, 99 = Efecto Filtro, 100 = Estilo Filtro
# Feature 26 = Zona de Uso, 20 = Tipo Producto, 4 = Color
FEATURE_MAP = {
    # Efectos (feature 99 - Efecto Filtro)
    "cemento":     {"feature_id": 99, "value_ids": [4467]},
    "marmol":      {"feature_id": 99, "value_ids": [4465]},
    "mármol":      {"feature_id": 99, "value_ids": [4465]},
    "piedra":      {"feature_id": 99, "value_ids": [4413]},
    "madera":      {"feature_id": 99, "value_ids": [4471]},
    "hidraulico":  {"feature_id": 99, "value_ids": [4468]},
    "hidráulico":  {"feature_id": 99, "value_ids": [4468]},
    "terrazo":     {"feature_id": 99, "value_ids": [4477]},
    "monocolor":   {"feature_id": 99, "value_ids": [4469]},
    "barro":       {"feature_id": 99, "value_ids": [4473]},
    "oxido":       {"feature_id": 99, "value_ids": [4476]},
    "óxido":       {"feature_id": 99, "value_ids": [4476]},
    "textil":      {"feature_id": 99, "value_ids": [4474]},
    "ladrillo":    {"feature_id": 99, "value_ids": [4470]},
    "botanico":    {"feature_id": 99, "value_ids": [4475]},
    # Estilos (feature 100 - Estilo Filtro)
    "nordico":     {"feature_id": 100, "value_ids": [4463]},
    "nórdico":     {"feature_id": 100, "value_ids": [4463]},
    "industrial":  {"feature_id": 100, "value_ids": [4472]},
    "vintage":     {"feature_id": 100, "value_ids": [4464]},
    "clasico":     {"feature_id": 100, "value_ids": [4460]},
    "clásico":     {"feature_id": 100, "value_ids": [4460]},
    "contemporaneo": {"feature_id": 100, "value_ids": [4461]},
    "contemporáneo": {"feature_id": 100, "value_ids": [4461]},
    "mediterraneo": {"feature_id": 100, "value_ids": [4462]},
    "mediterráneo": {"feature_id": 100, "value_ids": [4462]},
    "rustico":     {"feature_id": 100, "value_ids": [4466]},
    "rústico":     {"feature_id": 100, "value_ids": [4466]},
    # Zonas de uso (feature 26)
    "bano":        {"feature_id": 26, "value_ids": [161]},
    "baño":        {"feature_id": 26, "value_ids": [161]},
    "cocina":      {"feature_id": 26, "value_ids": [162]},
    "exterior":    {"feature_id": 26, "value_ids": [167]},
    "terraza":     {"feature_id": 26, "value_ids": [167]},
    "piscina":     {"feature_id": 26, "value_ids": [298]},
    "fachada":     {"feature_id": 26, "value_ids": [173]},
    "comercio":    {"feature_id": 26, "value_ids": [163]},
    "comercial":   {"feature_id": 26, "value_ids": [163]},
    # Tipo producto (feature 20)
    "porcelanico": {"feature_id": 20, "value_ids": [78]},
    "porcelánico": {"feature_id": 20, "value_ids": [78]},
    "gres":        {"feature_id": 20, "value_ids": [31, 82]},
    "revestimiento": {"feature_id": 20, "value_ids": [61]},
    "mosaico":     {"feature_id": 20, "value_ids": [53]},
    "slab":        {"feature_id": 20, "value_ids": [853]},
    "20mm":        {"feature_id": 20, "value_ids": [202]},
    "gran formato": {"feature_id": 20, "value_ids": [235]},
    "parquet":     {"feature_id": 20, "value_ids": [1618]},
    "laminado":    {"feature_id": 20, "value_ids": [1618]},
    # Colores (feature 4)
    "gris":        {"feature_id": 4, "value_ids": [13, 45, 49]},
    "blanco":      {"feature_id": 4, "value_ids": [51]},
    "negro":       {"feature_id": 4, "value_ids": [70]},
    "beige":       {"feature_id": 4, "value_ids": [48]},
    "crema":       {"feature_id": 4, "value_ids": [44]},
    "marron":      {"feature_id": 4, "value_ids": [37]},
    "marrón":      {"feature_id": 4, "value_ids": [37]},
    "azul":        {"feature_id": 4, "value_ids": [83]},
    "verde":       {"feature_id": 4, "value_ids": [84]},
    "arena":       {"feature_id": 4, "value_ids": [46]},
    # Acabados (feature 16)
    "mate":        {"feature_id": 16, "value_ids": [27]},
    "brillo":      {"feature_id": 16, "value_ids": [89]},
    "pulido":      {"feature_id": 16, "value_ids": [242]},
    "antideslizante": {"feature_id": 16, "value_ids": [104]},
}


def wants_product_search(query):
    query_lower = query.lower()
    for kw in PRODUCT_SEARCH_KEYWORDS:
        if kw in query_lower:
            return True
    if re.search(r'\d{2,3}\s*[xX]\s*\d{2,3}', query):
        return True
    # Also trigger if query matches feature keywords (efecto, zona, etc.)
    for kw in FEATURE_MAP:
        if kw in query_lower:
            return True
    return False


def _ps_api_get(endpoint, params, auth):
    """Helper for PrestaShop API calls."""
    params["output_format"] = "JSON"
    try:
        r = http_requests.get(f"{API_BASE}/{endpoint}", params=params, auth=auth, timeout=12)
        if r.status_code != 200:
            return []
        data = r.json()
        for key in data:
            if isinstance(data[key], list):
                return data[key]
        return []
    except Exception:
        return []


def _get_product_ids_by_features(query_lower, auth):
    """Busca product IDs que tengan las features matching de la query."""
    matched_features = {}  # feature_id -> [value_ids]

    for keyword, feature_info in FEATURE_MAP.items():
        if keyword in query_lower:
            fid = feature_info["feature_id"]
            if fid not in matched_features:
                matched_features[fid] = []
            matched_features[fid].extend(feature_info["value_ids"])

    if not matched_features:
        return []

    # For each matched feature, get products with those values
    # We use product_feature_values to find which products have these values
    candidate_ids = None

    for feature_id, value_ids in matched_features.items():
        # Search products that have any of these feature values
        products_with_feature = set()
        for vid in value_ids:
            products = _ps_api_get("products", {
                "display": "[id]",
                "filter[id_product_feature]": f"{feature_id}|{vid}",
                "limit": "50",
            }, auth)
            for p in products:
                products_with_feature.add(str(p.get("id", "")))

        # If direct filter doesn't work, search by feature value in product name
        if not products_with_feature:
            # Get the value text
            values = _ps_api_get("product_feature_values", {
                "display": "[id,value]",
                "filter[id]": "|".join(str(v) for v in value_ids),
            }, auth)
            for val_obj in values:
                val_text = val_obj.get("value", "")
                if isinstance(val_text, list):
                    val_text = val_text[0].get("value", "") if val_text else ""
                elif isinstance(val_text, dict):
                    val_text = list(val_text.values())[0] if val_text else ""
                if val_text and len(val_text) > 2:
                    prods = _ps_api_get("products", {
                        "display": "[id]",
                        "filter[name]": f"%{val_text}%",
                        "limit": "30",
                    }, auth)
                    for p in prods:
                        products_with_feature.add(str(p.get("id", "")))

        if products_with_feature:
            if candidate_ids is None:
                candidate_ids = products_with_feature
            else:
                # Intersect: products must match ALL features
                candidate_ids = candidate_ids & products_with_feature

    return list(candidate_ids or [])


def search_prestashop(query, limit=5):
    if not PRESTASHOP_API_KEY:
        return None

    auth = (PRESTASHOP_API_KEY, "")
    query_lower = query.lower()

    # Parse formato (NNxNN)
    formato_match = re.search(r'(\d{2,3})\s*[xX]\s*(\d{2,3})', query)
    formato = None
    query_clean = query
    if formato_match:
        w, h = int(formato_match.group(1)), int(formato_match.group(2))
        formato = f"{min(w,h)}X{max(w,h)}"
        query_clean = query.replace(formato_match.group(0), "").strip()

    words = query_clean.lower().split()
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]

    all_products = []

    # Strategy 1: Search by product features (efecto, zona, estilo, color)
    feature_product_ids = _get_product_ids_by_features(query_lower, auth)
    if feature_product_ids:
        # Get full product data for matched IDs (batch of max 20)
        id_filter = "|".join(feature_product_ids[:20])
        products = _ps_api_get("products", {
            "display": "[id,name,price,reference,link_rewrite,id_category_default]",
            "filter[id]": f"[{id_filter}]",
        }, auth)
        all_products.extend(products)

    # Strategy 2: Search by name keywords
    for kw in keywords[:3]:
        if len(kw) < 3:
            continue
        products = _ps_api_get("products", {
            "display": "[id,name,price,reference,link_rewrite,id_category_default]",
            "filter[name]": f"%{kw}%",
            "limit": "15",
        }, auth)
        all_products.extend(products)

    # Strategy 3: Search by formato
    if formato:
        products = _ps_api_get("products", {
            "display": "[id,name,price,reference,link_rewrite,id_category_default]",
            "filter[name]": f"%{formato}%",
            "limit": "15",
        }, auth)
        all_products.extend(products)

    # Deduplicate
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
    scored = []
    feature_ids_set = set(feature_product_ids)
    for p in unique:
        score = 0
        pid = str(p.get("id", ""))
        name_lower = p.get("name", "").lower()

        # Bonus if found via features (most relevant)
        if pid in feature_ids_set:
            score += 5

        # Keyword matches in name
        for kw in keywords:
            if kw in name_lower:
                score += 1

        # Formato match
        if formato and formato.lower().replace("x", "") in name_lower.replace("x", "").replace(" ", ""):
            score += 3

        scored.append((score, p))

    scored.sort(key=lambda x: -x[0])
    unique = [p for s, p in scored if s > 0]

    if not unique:
        return None

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
