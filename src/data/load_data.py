import json
from pathlib import Path
from datasets import Dataset


def load_hint_dataset(data_dir="datasets"):

    #Load all JSON hint datasets from a directory.

    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json")) #get all json dataset files

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    print(f"Found {len(json_files)} dataset files: {[f.name for f in json_files]}")

    all_examples = []

    for path in json_files:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        category = data["category"]

        for item in data["words"]:
            word = item["word"]
            hints = item["hints"]

            #Format for T5: input prompt -> target output
            input_text = f"generate hint for {category.lower()}: {word}"
            target_text = ", ".join(hints)

            all_examples.append({
                "input": input_text,
                "target": target_text,
                "category": category,
                "word": word
            })

    return Dataset.from_list(all_examples)
