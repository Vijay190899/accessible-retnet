"""
RetNet — Local Text Generation UI.

Loads the pretrained base model and serves a browser UI for
free-form text completion.

Usage:
  python app.py              # runs on http://localhost:5000
  python app.py --port 8080
"""

import sys
import io
import argparse
import warnings
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import torch
from flask import Flask, request, jsonify, send_from_directory
from retnet_model import RetNetLM, RetNetConfig

# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

_model     = None
_tokenizer = None
_device    = None


def load_model():
    global _model, _tokenizer, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import GPT2TokenizerFast
    _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    _tokenizer.pad_token = _tokenizer.eos_token

    candidates = [
        Path(__file__).parent / "checkpoints" / "best_model.pt",
        Path(__file__).parent / "checkpoints" / "final_model.pt",
        Path(__file__).parent / "checkpoints" / "latest.pt",
    ]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        print("[App] ERROR: No checkpoint found in checkpoints/")
        sys.exit(1)

    print(f"[App] Loading: {ckpt_path}")
    ckpt   = torch.load(ckpt_path, map_location=_device)
    config = RetNetConfig()
    if "config" in ckpt:
        for k, v in ckpt["config"].items():
            if hasattr(config, k): setattr(config, k, v)

    _model = RetNetLM(config).to(_device)
    _model.load_state_dict(ckpt["model"])
    _model.eval()

    print(f"[App] RetNet 124M loaded — Step={ckpt.get('step','?')} | Val PPL={ckpt.get('val_ppl','?')}")
    print(f"[App] Device: {_device}")


# ─────────────────────────────────────────────────────────────────────────────
# Flask
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="frontend", static_url_path="")


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data        = request.get_json()
    prompt      = data.get("prompt", "").strip()
    max_tokens  = int(data.get("max_tokens",  200))
    temperature = float(data.get("temperature", 0.8))
    top_p       = float(data.get("top_p",       0.9))
    rep_penalty = float(data.get("rep_penalty", 1.3))

    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    try:
        ids = torch.tensor(
            [_tokenizer.encode(prompt)], dtype=torch.long
        ).to(_device)

        with torch.no_grad():
            out = _model.generate_parallel(
                ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=rep_penalty,
                ngram_block=3,
            )

        new_ids    = out[0, ids.shape[1]:].tolist()
        completion = _tokenizer.decode(new_ids, skip_special_tokens=True)
        return jsonify({"completion": completion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(_device),
        "model_params": _model.num_parameters() if _model else 0,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    load_model()

    print(f"\n[App] Open http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)
