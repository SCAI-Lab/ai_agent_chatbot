from ..env import CONFIG

ADD_KWARGS = {
    "prompt_id": "summary_entry_chats",
}
SUMMARY_PROMPT = """You log user information, schedules, and events from a user–assistant chat.

## Task Checklist
- Capture every relevant user fact, plan, or event.
- Also apply: {additional_requirements}
- When a message carries a timestamp `[TIME]`, convert any relative wording into explicit dates in the output. If the exact date cannot be inferred, keep only the mention date.

## Reference
Topics to cover:
<topics>
{topics}
</topics>
Key attributes:
<attributes>
{attributes}
</attributes>

## Input
### Already Logged
Lines like `TOPIC{separator}SUBTOPIC{separator}CONTENT` (possibly truncated).
### Chats
Conversation lines in the form `[TIME] NAME: MESSAGE`.

## Output
Return Markdown bullets formatted as `- statement [mention YYYY/MM/DD, event ...] // TYPE`.
- Include the mention time for every entry and the event/plan time when it can be inferred.
- Keep sentences factual and concise.
Example:
```
- User's alias is Jack. [mention 2023/01/23] // info
- Jack plans to go to the gym. [mention 2023/01/23, event 2023/01/24] // schedule
```

Now perform your task.
"""


def pack_input(already_logged_str: str, chat_strs: str):
    return f"""### Already Logged
{already_logged_str}
### Input Chats
{chat_strs}
"""


def get_prompt(
    topic_examples: str, attribute_examples: str, additional_requirements: str = ""
) -> str:
    return SUMMARY_PROMPT.format(
        topics=topic_examples,
        attributes=attribute_examples,
        additional_requirements=additional_requirements,
        separator=CONFIG.llm_tab_separator,
    )


def get_kwargs() -> dict:
    return ADD_KWARGS


if __name__ == "__main__":
    print(get_prompt())
