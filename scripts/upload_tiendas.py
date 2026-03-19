"""
Script para subir información de tiendas Ferrolan a Pinecone.
Crea vectores con dominio "general" y doc_id "TIENDA-XX".
"""

import os
import json
import requests
import time

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_HOST = "https://ferrolan-rag-qtvdakx.svc.aped-4627-b74a.pinecone.io"
PINECONE_INFERENCE_URL = "https://api.pinecone.io/embed"

# ── Datos de tiendas ──────────────────────────────────────
TIENDAS = [
    {
        "doc_id": "TIENDA-01",
        "titulo": "Ferrolan Urgell - Barcelona Eixample",
        "seccion": "Información de tienda",
        "text": (
            "FERROLAN URGELL - BARCELONA EIXAMPLE\n"
            "Dirección: Carrer del Comte d'Urgell, 172, 08036 Barcelona (barrio de L'Eixample)\n"
            "Teléfono: 93 323 17 58\n"
            "Email: urgell@ferrolan.com\n"
            "Horario: Lunes a Viernes de 9:00 a 14:00 y de 16:00 a 20:00. Sábados de 9:00 a 14:00.\n"
            "Parking gratuito: Carrer del Comte d'Urgell, 154\n"
            "Metro: Próxima a Hospital Clínic\n\n"
            "Más de 600 m² de exposición con stock de más de 40.000 referencias de cerámica. "
            "Capacidad de suministro de más de 600 referencias el mismo día.\n"
            "Servicios: asesoramiento personalizado, transporte a domicilio, WiFi gratuito, "
            "vales descuento, estudio de cocina especializado, Tile Cube (experiencia inmersiva con realidad virtual).\n"
            "Especialización: azulejos, cocinas, baños, parquet y cerámicas."
        ),
    },
    {
        "doc_id": "TIENDA-02",
        "titulo": "Ferrolan Santa Coloma de Gramenet (Sede Central)",
        "seccion": "Información de tienda",
        "text": (
            "FERROLAN SANTA COLOMA DE GRAMENET - SEDE CENTRAL\n"
            "Dirección: Carretera de la Roca, km 5,6, 08924 Santa Coloma de Gramenet, Barcelona\n"
            "Teléfono: 93 391 90 11\n"
            "Email: santacoloma@ferrolan.com\n"
            "Horario: Lunes a Viernes de 7:30 a 13:00 y de 15:00 a 20:00. Sábados de 8:00 a 13:30.\n"
            "Aparcamiento gratuito.\n\n"
            "Sede central de Ferrolan con más de 2.000 m² de exposición y más de 14.000 m² de instalaciones. "
            "Más de 40 años de experiencia en el sector.\n"
            "Servicios: asesoramiento personalizado, transporte a domicilio, WiFi gratuito, "
            "vales descuento, amplio surtido en stock.\n"
            "Especialización: azulejos cerámicos, mobiliario de baño y cocina, parquets, materiales de construcción."
        ),
    },
    {
        "doc_id": "TIENDA-03",
        "titulo": "Ferrolan Rubí",
        "seccion": "Información de tienda",
        "text": (
            "FERROLAN RUBÍ\n"
            "Dirección: Carretera de Rubí-Terrassa, 125-127, 08191 Rubí, Barcelona\n"
            "Teléfono: 93 586 00 00\n"
            "Email: rubi@ferrolan.com\n"
            "Horario: Lunes a Viernes de 7:30 a 13:00 y de 15:00 a 20:00. Sábados de 8:00 a 13:30.\n"
            "Aparcamiento gratuito.\n\n"
            "Más de 5.000 m² de superficie con más de 100 marcas disponibles. "
            "Servicio logístico con camiones grúa. Más de 40 años en el sector.\n"
            "Servicios: asesoramiento personalizado, transporte a domicilio, WiFi gratuito, "
            "vales descuento, amplio surtido en stock, servicio logístico con camiones grúa.\n"
            "Especialización: construcción, reforma, bricolaje, baño, cocina y ferretería."
        ),
    },
    {
        "doc_id": "TIENDA-04",
        "titulo": "Ferrolan Badalona",
        "seccion": "Información de tienda",
        "text": (
            "FERROLAN BADALONA\n"
            "Dirección: Avinguda de la Cerdanya, s/n, 08915 Badalona, Barcelona\n"
            "Teléfono: 93 465 74 66\n"
            "Email: badalona@ferrolan.com\n"
            "Horario: Lunes a Viernes de 8:00 a 13:00 y de 15:00 a 20:00. Sábados cerrado.\n"
            "Aparcamiento gratuito.\n\n"
            "5.000 m² de superficie con más de 40.000 productos y 700 referencias en stock. "
            "Marcas internacionales y nacionales prestigiosas.\n"
            "Servicios: asesoramiento personalizado, transporte a domicilio, WiFi gratuito, "
            "vales descuento, amplio stock.\n"
            "Especialización: azulejos cerámicos, parquets, suelos porcelánicos, mobiliario de cocina y baño, "
            "ferretería, materiales de construcción, herramientas profesionales."
        ),
    },
    {
        "doc_id": "TIENDA-05",
        "titulo": "Ferrolan Barcelona Clot",
        "seccion": "Información de tienda",
        "text": (
            "FERROLAN BARCELONA - CLOT\n"
            "Dirección: Avinguda Meridiana, 182, 08026 Barcelona\n"
            "Teléfono: 93 349 01 50\n"
            "Email: clot@ferrolan.com\n"
            "Horario: Lunes a Viernes de 9:00 a 14:00 y de 16:00 a 20:00. Sábados de 9:00 a 14:00.\n"
            "Parking gratuito: Carrer de Mallorca, 632\n"
            "Metro: A 2 minutos de la estación de Clot.\n\n"
            "Más de 700 m² de exposición en dos plantas. "
            "Referencia para particulares, decoradores y diseñadores de interiores en Barcelona.\n"
            "Servicios: asesoramiento personalizado, transporte a domicilio, WiFi gratuito, "
            "vales descuento, amplio surtido en stock.\n"
            "Especialización: azulejos cerámicos, suelos de madera/vinilo/laminado, "
            "mobiliario de cocina y baño, accesorios de baño."
        ),
    },
    {
        "doc_id": "TIENDA-06",
        "titulo": "Ferrolan Outlet Fondo",
        "seccion": "Información de tienda",
        "text": (
            "FERROLAN OUTLET FONDO\n"
            "Dirección: Rambla Del Fondo, cantonada C/ Verdi, 08924 Santa Coloma de Gramenet\n"
            "Teléfono: 93 391 20 02\n"
            "Email: outletfondo@ferrolan.com\n"
            "Horario: Lunes a Viernes de 8:00 a 13:00 y de 16:00 a 19:30. Sábados cerrado.\n"
            "Aparcamiento gratuito.\n\n"
            "Más de 600 m² dedicados a baño, cocina y azulejos con precios especiales de outlet. "
            "Exposición y almacén en el mismo local.\n"
            "Servicios: precios especiales, transporte a domicilio, WiFi gratuito, "
            "vales descuento, amplio surtido en stock."
        ),
    },
    {
        "doc_id": "TIENDA-07",
        "titulo": "Ferrolan Outlet Central",
        "seccion": "Información de tienda",
        "text": (
            "FERROLAN OUTLET CENTRAL\n"
            "Dirección: Carretera de la Roca, km 5, 08924 Santa Coloma de Gramenet\n"
            "Teléfono: 93 391 53 54\n"
            "Email: outletcentral@ferrolan.com\n"
            "Horario: Lunes a Viernes de 8:00 a 13:00 y de 15:30 a 19:30. Sábados de 9:00 a 13:30.\n"
            "Aparcamiento gratuito.\n\n"
            "El primer outlet de cerámica. Más de 1.350 m² de instalaciones con precios reducidos. "
            "Amplio catálogo de azulejos de marcas reconocidas, grifería, muebles de baño y materiales de construcción.\n"
            "Servicios: precios reducidos, transporte a domicilio, WiFi gratuito, "
            "vales descuento, amplio stock, equipo de profesionales para asesoramiento."
        ),
    },
    {
        "doc_id": "TIENDA-00",
        "titulo": "Ferrolan - Red de Tiendas (Resumen General)",
        "seccion": "Información general de tiendas",
        "text": (
            "RED DE TIENDAS FERROLAN\n\n"
            "Ferrolan cuenta con 7 tiendas en el área metropolitana de Barcelona:\n\n"
            "1. FERROLAN URGELL (Barcelona Eixample) - C/ Comte d'Urgell 172 - Tel: 93 323 17 58\n"
            "   Horario: L-V 9:00-14:00 / 16:00-20:00, Sáb 9:00-14:00\n\n"
            "2. FERROLAN SANTA COLOMA (Sede Central) - Ctra. de la Roca km 5,6 - Tel: 93 391 90 11\n"
            "   Horario: L-V 7:30-13:00 / 15:00-20:00, Sáb 8:00-13:30\n\n"
            "3. FERROLAN RUBÍ - Ctra. Rubí-Terrassa 125-127 - Tel: 93 586 00 00\n"
            "   Horario: L-V 7:30-13:00 / 15:00-20:00, Sáb 8:00-13:30\n\n"
            "4. FERROLAN BADALONA - Av. de la Cerdanya s/n - Tel: 93 465 74 66\n"
            "   Horario: L-V 8:00-13:00 / 15:00-20:00, Sáb cerrado\n\n"
            "5. FERROLAN BARCELONA CLOT - Av. Meridiana 182 - Tel: 93 349 01 50\n"
            "   Horario: L-V 9:00-14:00 / 16:00-20:00, Sáb 9:00-14:00\n\n"
            "6. FERROLAN OUTLET FONDO - Rambla Del Fondo, cant. C/ Verdi - Tel: 93 391 20 02\n"
            "   Horario: L-V 8:00-13:00 / 16:00-19:30, Sáb cerrado\n\n"
            "7. FERROLAN OUTLET CENTRAL - Ctra. de la Roca km 5 - Tel: 93 391 53 54\n"
            "   Horario: L-V 8:00-13:00 / 15:30-19:30, Sáb 9:00-13:30\n\n"
            "Contacto general: online@ferrolan.com\n"
            "Web: ferrolan.es\n\n"
            "Todas las tiendas ofrecen: asesoramiento personalizado, transporte a domicilio, "
            "WiFi gratuito y aparcamiento gratuito. Las tiendas Outlet ofrecen precios especiales y reducidos."
        ),
    },
]


def get_embedding(text):
    """Genera embedding usando Pinecone Inference REST."""
    response = requests.post(
        PINECONE_INFERENCE_URL,
        headers={
            "Api-Key": PINECONE_API_KEY,
            "Content-Type": "application/json",
            "X-Pinecone-Api-Version": "2025-10",
        },
        json={
            "model": "multilingual-e5-large",
            "inputs": [{"text": text}],
            "parameters": {"input_type": "passage"},
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise Exception(f"Embedding error {response.status_code}: {response.text[:200]}")
    return response.json()["data"][0]["values"]


def upsert_vectors(vectors):
    """Sube vectores a Pinecone."""
    response = requests.post(
        f"{PINECONE_HOST}/vectors/upsert",
        headers={
            "Api-Key": PINECONE_API_KEY,
            "Content-Type": "application/json",
        },
        json={"vectors": vectors},
        timeout=30,
    )
    if response.status_code != 200:
        raise Exception(f"Upsert error {response.status_code}: {response.text[:200]}")
    return response.json()


def main():
    if not PINECONE_API_KEY:
        print("ERROR: Set PINECONE_API_KEY environment variable")
        return

    vectors = []
    for tienda in TIENDAS:
        print(f"Generating embedding for {tienda['doc_id']}: {tienda['titulo']}...")
        embedding = get_embedding(tienda["text"])

        vector = {
            "id": tienda["doc_id"].lower(),
            "values": embedding,
            "metadata": {
                "doc_id": tienda["doc_id"],
                "titulo_documento": tienda["titulo"],
                "seccion": tienda["seccion"],
                "dominio": "general",
                "verificado": True,
                "text": tienda["text"],
            },
        }
        vectors.append(vector)
        time.sleep(0.5)  # Rate limit

    print(f"\nUpserting {len(vectors)} vectors to Pinecone...")
    result = upsert_vectors(vectors)
    print(f"Result: {result}")
    print("Done! Store info uploaded to Pinecone.")


if __name__ == "__main__":
    main()
