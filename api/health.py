"""Health check endpoint."""

import json
import os
from http.server import BaseHTTPRequestHandler

import requests


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        pinecone_ok = False
        vector_count = 0

        try:
            api_key = os.environ.get("PINECONE_API_KEY", "")
            host = "https://ferrolan-rag-qtvdakx.svc.aped-4627-b74a.pinecone.io"
            r = requests.post(
                f"{host}/query",
                headers={"Api-Key": api_key, "Content-Type": "application/json"},
                json={"vector": [0.0] * 384, "topK": 1},
                timeout=5,
            )
            pinecone_ok = r.status_code == 200
            if pinecone_ok:
                vector_count = len(r.json().get("matches", []))
        except Exception:
            pass

        data = {
            "status": "ok" if pinecone_ok else "degraded",
            "pinecone_connected": pinecone_ok,
            "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "prestashop_key_set": bool(os.environ.get("PRESTASHOP_API_KEY")),
            "platform": "vercel",
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
