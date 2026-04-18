"""
RetNet Training Monitor — Local HTTP server.
Serves the visual knowledge graph dashboard + live training state.

Usage:  python monitor_server.py
Then open: http://localhost:8765
"""

import json, csv, os, webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

BASE = Path(__file__).parent
CHECKPOINTS = BASE / "checkpoints"
HTML_FILE   = BASE / "monitor.html"
PORT = 8765


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass   # silence default access log

    def _json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        # ── /api/state ──────────────────────────────────────────────────────
        if path == "/api/state":
            state_file = CHECKPOINTS / "training_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    data = json.load(f)
            else:
                data = {"step": 0, "total_steps": 50000, "loss": None, "ppl": None,
                        "best_val_ppl": None, "lr": "0", "gpu_gb": 0,
                        "speed": 0, "eta": "--", "phase": "Starting",
                        "val_history": [], "updated_at": ""}
            self._json(data)

        # ── /api/log ─────────────────────────────────────────────────────────
        elif path == "/api/log":
            log_file = CHECKPOINTS / "training_log.csv"
            rows = []
            if log_file.exists():
                with open(log_file, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append({
                            "step":       int(row["step"]),
                            "train_loss": float(row["train_loss"]),
                            "train_ppl":  float(row["train_ppl"]),
                            "val_ppl":    float(row["val_ppl"]) if row["val_ppl"] else None,
                            "lr":         float(row["lr"]),
                            "gpu_gb":     float(row["gpu_gb"]),
                        })
            self._json(rows)

        # ── Serve monitor.html ───────────────────────────────────────────────
        else:
            if HTML_FILE.exists():
                body = HTML_FILE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"monitor.html not found")


if __name__ == "__main__":
    server = HTTPServer(("localhost", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"\n  RetNet Training Monitor")
    print(f"  Open in browser: {url}")
    print(f"  Auto-refreshes every 5 seconds")
    print(f"  Press Ctrl+C to stop\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Monitor stopped.")
