"""
RetNet SFT — Local Web Frontend.

Hosts a simple browser UI to test generation from the fine-tuned model.

Usage:
  python app.py                  # loads checkpoints_sft/best_sft.pt
  python app.py --port 5000
  python app.py --base            # load base pretrained model instead

Then open: http://localhost:5000
"""

import sys
import io
import argparse
import warnings
import math
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


def load_model(base_mode=False):
    global _model, _tokenizer, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import GPT2TokenizerFast
    _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    _tokenizer.pad_token = _tokenizer.eos_token

    if base_mode:
        candidates = [
            Path(__file__).parent / "checkpoints" / "best_model.pt",
            Path(__file__).parent / "checkpoints" / "final_model.pt",
        ]
        label = "base"
    else:
        candidates = [
            Path(__file__).parent / "checkpoints_sft" / "best_sft.pt",
            Path(__file__).parent / "checkpoints_sft" / "latest_sft.pt",
        ]
        label = "SFT"

    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        print(f"[App] ERROR: No {label} checkpoint found.")
        sys.exit(1)

    print(f"[App] Loading {label} checkpoint: {ckpt_path}")
    ckpt   = torch.load(ckpt_path, map_location=_device)
    config = RetNetConfig()
    if "config" in ckpt:
        for k, v in ckpt["config"].items():
            if hasattr(config, k): setattr(config, k, v)

    _model = RetNetLM(config).to(_device)
    _model.load_state_dict(ckpt["model"])
    _model.eval()

    val_ppl = ckpt.get("val_ppl", "?")
    step    = ckpt.get("step",    "?")
    print(f"[App] Loaded — Step={step} | Val PPL={val_ppl}")
    print(f"[App] Device: {_device}")


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="frontend", static_url_path="")

ALPACA_WITH_INPUT = """\
### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

ALPACA_NO_INPUT = """\
### Instruction:
{instruction}

### Response:
"""


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data        = request.get_json()
    instruction = data.get("instruction", "").strip()
    inp         = data.get("input", "").strip()
    max_tokens  = int(data.get("max_tokens",  200))
    temperature = float(data.get("temperature", 0.7))
    top_p       = float(data.get("top_p",       0.9))
    rep_penalty = float(data.get("rep_penalty", 1.3))

    if not instruction:
        return jsonify({"error": "Instruction cannot be empty."}), 400

    if inp:
        prompt = ALPACA_WITH_INPUT.format(instruction=instruction, input=inp)
    else:
        prompt = ALPACA_NO_INPUT.format(instruction=instruction)

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

        new_ids  = out[0, ids.shape[1]:].tolist()
        response = _tokenizer.decode(new_ids, skip_special_tokens=True)

        # Trim at section separator or double newline
        for sep in ["\n\n", "###"]:
            if sep in response:
                response = response[:response.index(sep)]

        return jsonify({"response": response.strip()})

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
    parser = argparse.ArgumentParser(description="RetNet Local Web UI")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--base", action="store_true", help="Load base model instead of SFT")
    args = parser.parse_args()

    load_model(base_mode=args.base)

    print(f"\n[App] Starting server on http://localhost:{args.port}")
    print(f"[App] Open your browser to http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)
