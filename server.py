import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
import pickle
import pandas as pd

with open("modelo_gravidade_simplificado2.pkl", "rb") as f:
    modelo = pickle.load(f)

mapa_dias = {
    "segunda-feira": 0,
    "terça-feira": 1,
    "quarta-feira": 2,
    "quinta-feira": 3,
    "sexta-feira": 4,
    "sábado": 5,
    "domingo": 6,
}


class PredictHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path != "/predict":
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
            return

        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data)

            hora = int(data["hora"])

            clima = data["clima"]

            nova_entrada = pd.DataFrame([{
                "hour": hora,
                "dia_semana": data["dia_semana"].lower(),
                "condicao_metereologica": clima
            }])
            print(f"Nova entrada: {nova_entrada}")

            pred = modelo.predict(nova_entrada)
            if pred[0] == 1:
                resultado = "alta"
            else:
                resultado = "baixa"
            self._set_headers()
            self.wfile.write(json.dumps({"predicao": str(resultado)}).encode())

        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())


PORT = 5000
print(f"🚀 Servidor rodando em http://localhost:{PORT}")
httpd = HTTPServer(("localhost", PORT), PredictHandler)
httpd.serve_forever()
