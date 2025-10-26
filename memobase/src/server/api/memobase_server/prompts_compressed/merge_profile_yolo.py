from .utils import pack_merge_action_into_string
from ..env import CONFIG

ADD_KWARGS = {"prompt_id": "merge_profile_yolo"}

MERGE_FACTS_PROMPT = """You maintain user memos and must decide how each new entry merges with the current record.

## Input
A list of JSON objects like:
```
{{
    "memo_id": "1",
    "new_info": "",
    "current_memo": "",
    "topic": "",
    "subtopic": "",
    "topic_description": "",
    "update_instruction": ""
}}
```

## Decision Checklist
1. Confirm `new_info` satisfies `topic_description`. If it cannot be adjusted to fit, output ABORT.
2. Obey any `update_instruction`.
3. Compare with `current_memo`:
   - Empty memo → APPEND valid content.
   - Conflicting/outdated memo → UPDATE with a concise final memo (≤5 sentences).
   - Redundant or valueless info → ABORT.
4. Preserve every time annotation from both old and new content.

## Actions
- APPEND: `N. APPEND{tab}APPEND`
- UPDATE: `N. UPDATE{tab}[UPDATED_MEMO]` (full revised memo, concise, keeps time markers, removes redundancy).
- ABORT: `N. ABORT{tab}ABORT`

## Response Template
```
THOUGHT
---
<your reasoning>
1. ACTION{tab}...
2. ACTION{tab}...
```

## Example
Input:
{{
    "memo_id": "1",
    "new_info": "Preparing for final exams [mentioned on 2025/06/01]",
    "current_memo": "Preparing for midterm exams [mentioned on 2025/04/01]",
    "topic": "Study",
    "subtopic": "Exam goals",
    "update_instruction": "Remove outdated or conflicting goals."
}}
{{
    "memo_id": "2",
    "new_info": "Using Duolingo to self-study Japanese",
    "current_memo": "",
    "topic": "Study",
    "subtopic": "Software usage"
}}
{{
    "memo_id": "3",
    "new_info": "User likes eating hot pot",
    "current_memo": "",
    "topic": "Interests",
    "subtopic": "Sports"
}}

Output:
```
Need to replace the outdated midterm goal and keep relevant study details.
---
1. UPDATE{tab}Preparing for final exams [mentioned on 2025/06/01]
2. APPEND{tab}APPEND
3. ABORT{tab}ABORT
```

## Requirements
- Follow the response template exactly.
- Never invent information.
- Keep UPDATED_MEMO concise and free of duplicate statements (e.g., "User is sad; User's mood is sad" → "User is sad").
- Always retain existing time markers (e.g., `[...] [mentioned on 2025/05/05, occurred in 2022]`).
"""


def get_input(
    memos_list: list[dict],
):
    return f"""
{memos_list}
"""


def get_prompt() -> str:
    return MERGE_FACTS_PROMPT.format(
        tab=CONFIG.llm_tab_separator,
    )


def get_kwargs() -> dict:
    return ADD_KWARGS


if __name__ == "__main__":
    print(get_prompt())
