"""Small lexical context state machine for the Wolof demo web app."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence


ALL_CATEGORY = "All"
DEFAULT_WEB_CATEGORIES = (
    ALL_CATEGORY,
    "education",
    "agriculture",
    "sante",
    "transport",
    "culture",
)
TOKEN_ALIASES = {
    "mathematik": "math",
    "mathematique": "math",
    "mathématique": "math",
    "mathematiques": "math",
    "mathématiques": "math",
    "mathematics": "math",
    "maths": "math",
    "mat": "math",
    "francais": "français",
    "français": "français",
    "anglais": "anglais",
    "wolof": "wolof",
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return [TOKEN_ALIASES.get(token, token) for token in tokens]


def is_all_category(category: str) -> bool:
    return category.strip().lower() in {"all", "*", "toutes", "tous"}


def load_training_rows(path: str | Path) -> List[Dict[str, str]]:
    data_path = Path(path)
    rows = []
    with data_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(validate_training_row(row, line_no))
    return rows


def validate_training_row(row: Dict[str, str], line_no: int) -> Dict[str, str]:
    required = ("instruction", "input", "output")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"Training row {line_no} is missing fields: {missing}")

    clean = {field: str(row.get(field, "")).strip() for field in required}
    if not clean["instruction"] or not clean["output"]:
        raise ValueError(f"Training row {line_no} has an empty instruction or output.")
    return clean


class CategoryContextStateMachine:
    """Retrieve in-category examples and build an augmented prompt.

    This is intentionally lightweight: no embeddings, no vector database, just
    category filtering plus lexical overlap. It is useful when compute and setup
    time are limited in class.
    """

    def __init__(
        self,
        rows: Iterable[Dict[str, str]],
        categories: Sequence[str] = DEFAULT_WEB_CATEGORIES,
    ) -> None:
        normalized_categories = []
        for category in categories:
            if category not in normalized_categories:
                normalized_categories.append(category)
        if ALL_CATEGORY not in normalized_categories:
            normalized_categories.insert(0, ALL_CATEGORY)

        self.categories = tuple(normalized_categories)
        self.all_rows = list(rows)
        self.rows_by_category: Dict[str, List[Dict[str, str]]] = {
            category: [] for category in self.categories if not is_all_category(category)
        }
        for row in self.all_rows:
            category = row["input"]
            if category in self.rows_by_category:
                self.rows_by_category[category].append(row)

    def available_categories(self) -> List[str]:
        visible = [ALL_CATEGORY]
        visible.extend(
            category
            for category in self.categories
            if not is_all_category(category) and self.rows_by_category.get(category)
        )
        return visible

    def retrieve(self, category: str, question: str, k: int = 4) -> List[Dict[str, str]]:
        if is_all_category(category):
            rows = self.all_rows
        elif category in self.rows_by_category:
            rows = self.rows_by_category[category]
        else:
            raise ValueError(f"Unknown category: {category}")

        question_tokens = Counter(tokenize(question))
        scored = []
        for row in rows:
            instruction_tokens = Counter(tokenize(row["instruction"]))
            output_tokens = Counter(tokenize(row["output"]))
            instruction_overlap = sum((question_tokens & instruction_tokens).values())
            output_overlap = sum((question_tokens & output_tokens).values())
            score = 2 * instruction_overlap + output_overlap
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = []
        seen_outputs = set()
        for score, row in scored:
            if score <= 0:
                continue
            output_key = row["output"].lower()
            if output_key in seen_outputs:
                continue
            selected.append(row)
            seen_outputs.add(output_key)
            if len(selected) >= k:
                break
        if selected:
            return selected
        return [row for _, row in scored[:k]]

    def build_context(self, examples: Sequence[Dict[str, str]]) -> str:
        chunks = []
        for index, row in enumerate(examples, start=1):
            chunks.append(
                "\n".join(
                    [
                        f"Example {index}",
                        f"Instruction: {row['instruction']}",
                        f"Expected Wolof answer: {row['output']}",
                    ]
                )
            )
        return "\n\n".join(chunks)

    def few_shot_messages(self, examples: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        messages = []
        for row in examples:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Context category: {row['input']}\n"
                        f"Instruction: {row['instruction']}"
                    ),
                }
            )
            messages.append({"role": "assistant", "content": row["output"]})
        return messages

    def augment_instruction(
        self,
        category: str,
        question: str,
        examples: Sequence[Dict[str, str]],
    ) -> str:
        context = self.build_context(examples)
        return (
            "Student question:\n"
            f"{question.strip()}\n\n"
            "In-domain examples from the training data:\n"
            f"{context}\n\n"
            "Task:\n"
            "- Answer the student question in clear Wolof.\n"
            "- Use the examples only as style and domain context.\n"
            "- Do not copy an unrelated example.\n"
            "- Do not output hidden reasoning or <think> tags.\n"
            f"- Keep the answer appropriate for the category: {category}."
        )
