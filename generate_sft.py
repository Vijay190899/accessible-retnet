"""
RetNet SFT — Interactive Generation Script.

Loads the fine-tuned checkpoint (best_sft.pt) and generates responses
to instructions using the Alpaca prompt format.

Usage:
  python generate_sft.py --prompt "Explain what a neural network is"
  python generate_sft.py --prompt "Write a short poem about the ocean"
  python generate_sft.py --prompt "What is the capital of France?" --max-tokens 50
  python generate_sft.py --interactive    # chat-style loop
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
import torch.nn.functional as F
from retnet_model import RetNetLM, RetNetConfig

RESPONSE_TEMPLATE = "### Response:\n"

PROMPT_WITH_INPUT = """\
### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

PROMPT_NO_INPUT = """\
### Instruction:
{instruction}

### Response:
"""


def get_tokenizer():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


def load_model(device):
    sft_dir = Path(__file__).parent / "checkpoints_sft"
    for name in ["best_sft.pt", "latest_sft.pt"]:
        p = sft_dir / name
        if p.exists():
            ckpt_path = p
            break
    else:
        print("[Error] No SFT checkpoint found. Run finetune.py first.")
        sys.exit(1)

    print(f"[Generate] Loading: {ckpt_path}")
    ckpt   = torch.load(ckpt_path, map_location=device)
    config = RetNetConfig()
    if "config" in ckpt:
        for k, v in ckpt["config"].items():
            if hasattr(config, k): setattr(config, k, v)

    model = RetNetLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_ppl = ckpt.get("val_ppl", "?")
    step    = ckpt.get("step", "?")
    print(f"[Generate] Step={step} | Best Val PPL={val_ppl}")
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt_text, device,
             max_new_tokens=200, temperature=0.7, top_p=0.9,
             rep_penalty=1.3, ngram_block=3):

    ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long).to(device)
    out = model.generate_parallel(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=rep_penalty,
        ngram_block=ngram_block,
    )
    new_ids = out[0, ids.shape[1]:].tolist()
    text    = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Trim at double newline or EOS to avoid rambling
    for sep in ["\n\n", "###"]:
        if sep in text:
            text = text[:text.index(sep)]

    return text.strip()


def main(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = load_model(device)
    tokenizer = get_tokenizer()

    print(f"\nDevice: {device}")
    print("=" * 60)

    def run_prompt(instruction, input_text=""):
        if input_text.strip():
            prompt = PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text)
        else:
            prompt = PROMPT_NO_INPUT.format(instruction=instruction)

        print(f"\n[Instruction] {instruction}")
        if input_text.strip():
            print(f"[Input]       {input_text}")
        print("[Response]    ", end="", flush=True)

        response = generate(
            model, tokenizer, prompt, device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            rep_penalty=args.rep_penalty,
        )
        print(response)
        print("-" * 60)
        return response

    if args.interactive:
        print("Interactive mode — type an instruction, blank line to quit.\n")
        while True:
            try:
                instr = input("Instruction: ").strip()
                if not instr: break
                inp   = input("Input (optional, Enter to skip): ").strip()
                run_prompt(instr, inp)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExiting.")

    elif args.prompt:
        run_prompt(args.prompt, args.input or "")

    else:
        # Demo: a handful of built-in prompts
        demos = [
            ("Explain what a neural network is in simple terms.", ""),
            ("Write a short poem about the ocean.", ""),
            ("Summarize the following text in one sentence.",
             "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair."),
            ("What are three tips for staying productive while working from home?", ""),
            ("Translate the following sentence to French.", "Artificial intelligence is transforming the world."),
        ]
        print("Running demo prompts...\n")
        for instr, inp in demos:
            run_prompt(instr, inp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetNet SFT Generation")
    parser.add_argument("--prompt",      type=str, default=None, help="Instruction prompt")
    parser.add_argument("--input",       type=str, default="",   help="Optional input context")
    parser.add_argument("--interactive", action="store_true",    help="Interactive chat loop")
    parser.add_argument("--max-tokens",  type=int, default=200,  help="Max new tokens (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p",       type=float, default=0.9, help="Nucleus sampling p (default: 0.9)")
    parser.add_argument("--rep-penalty", type=float, default=1.3, help="Repetition penalty (default: 1.3)")
    args = parser.parse_args()

    if not (args.prompt or args.interactive):
        print("No prompt given — running built-in demo prompts.\n")

    main(args)
