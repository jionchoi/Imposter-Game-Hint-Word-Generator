import sys
import re
from pathlib import Path

sys.path.append("..")

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from data.load_data import PROMPT_TEMPLATE, normalize_category


MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "hint-generator" / "final"
EXPECTED_HINT_COUNT = 5


def parse_hints(text: str, expected_count: int = EXPECTED_HINT_COUNT) -> list[str]:
    parts = [part.strip().lower() for part in text.split(",")]
    cleaned = []
    seen = set()
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip(" .")
        if not part or part in seen:
            continue
        seen.add(part)
        cleaned.append(part)
        if len(cleaned) >= expected_count:
            break
    return cleaned


def score_hints(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    hints: list[str],
) -> list[tuple[str, float]]:
    if not hints:
        return []

    target_text = ", ".join(hints)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    labels = tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        logits = model(**inputs, labels=labels).logits

    token_probs = F.softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0]

    spans = []
    cursor = 0
    for i, hint in enumerate(hints):
        piece = hint if i == 0 else f", {hint}"
        piece_ids = tokenizer(piece, add_special_tokens=False).input_ids
        span_len = len(piece_ids)
        spans.append((cursor, cursor + span_len))
        cursor += span_len

    if cursor != labels.shape[1]:
        # Fallback if tokenizer segmentation differs unexpectedly.
        avg_weight = float(token_probs.mean().item())
        return [(hint, avg_weight) for hint in hints]

    weighted = []
    for hint, (start, end) in zip(hints, spans):
        span_probs = token_probs[start:end]
        weight = float(span_probs.mean().item())
        weighted.append((hint, weight))
    return weighted


def generate_hint(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    word: str,
    category: str,
    max_length: int,
    num_beams: int,
) -> str:
    prompt = PROMPT_TEMPLATE.format(category=normalize_category(category), word=word)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"prompt {prompt}")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        num_beams=num_beams,
        early_stopping=True,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        min_new_tokens=4,
        length_penalty=1.1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from: {MODEL_DIR}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    category = normalize_category(input("Category (default: animals): ").strip() or "animals")
    print("Enter words to test. Press Enter on empty line to exit.")
    while True:
        word = input("Word: ").strip()
        if not word:
            break
        hint = generate_hint(
            model=model,
            tokenizer=tokenizer,
            word=word,
            category=category,
            max_length=32,
            num_beams=4,
        )
        prompt = PROMPT_TEMPLATE.format(category=category, word=word)
        hints = parse_hints(hint)
        weighted_hints = score_hints(model=model, tokenizer=tokenizer, prompt=prompt, hints=hints)

        if not weighted_hints:
            print(f"{word} -> {hint}")
            continue

        best_hint, best_weight = max(weighted_hints, key=lambda item: item[1])
        weight_text = ", ".join(f"{h} ({w:.3f})" for h, w in weighted_hints)

        print(f"{word} -> {weight_text}")
        print(f"Best hint: {best_hint} ({best_weight:.3f})")


if __name__ == "__main__":
    main()
