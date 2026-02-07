import json
import re
from pathlib import Path
from datasets import Dataset


PROMPT_TEMPLATE = "generate 5 comma-separated hint words for {category}: {word}"
MAX_HINTS_PER_WORD = 5
_VALID_HINT_PATTERN = re.compile(r"^[a-z][a-z\s\-']{1,24}$")
CATEGORY_ALIASES = {
    "action": "actions",
    "actions": "actions",
    "animal": "animals",
    "animals": "animals",
    "food": "foods",
    "foods": "foods",
    "object": "objects",
    "objects": "objects",
    "place": "places",
    "places": "places",
    "profession": "professions",
    "professions": "professions",
    "sport": "sports",
    "sports": "sports",
}
DEFAULT_ALLOWED_CATEGORIES = set(CATEGORY_ALIASES.values())


def normalize_category(category: str) -> str:
    normalized = str(category).strip().lower()
    return CATEGORY_ALIASES.get(normalized, normalized)


def _clean_hints(raw_hints):
    cleaned = []
    seen = set()

    for hint in raw_hints or []:
        if hint is None:
            continue

        normalized = " ".join(str(hint).strip().lower().split())
        if not normalized:
            continue
        if not _VALID_HINT_PATTERN.fullmatch(normalized):
            continue
        if normalized in seen:
            continue

        seen.add(normalized)
        cleaned.append(normalized)

        if len(cleaned) >= MAX_HINTS_PER_WORD:
            break

    return cleaned


def load_hint_dataset(data_dir="datasets", allowed_categories=None):

    #Load all JSON hint datasets from a directory.

    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json")) #get all json dataset files

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    print(f"Found {len(json_files)} dataset files: {[f.name for f in json_files]}")

    all_examples = []
    dropped_words = 0
    skipped_files = 0
    active_categories = {
        normalize_category(category)
        for category in (allowed_categories or DEFAULT_ALLOWED_CATEGORIES)
    }

    for path in json_files:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        category = normalize_category(data["category"])
        if category not in active_categories:
            skipped_files += 1
            continue

        for item in data["words"]:
            word = item["word"]
            hints = _clean_hints(item.get("hints"))
            if not hints:
                dropped_words += 1
                continue

            #Format for T5: input prompt -> target output
            input_text = PROMPT_TEMPLATE.format(category=category, word=word)
            target_text = ", ".join(hints)

            all_examples.append({
                "input": input_text,
                "target": target_text,
                "category": category,
                "word": word
            })

    print(
        f"Prepared {len(all_examples)} examples "
        f"(dropped {dropped_words} words with invalid/empty hints, skipped {skipped_files} files by category filter)"
    )
    return Dataset.from_list(all_examples)
