from . import user_profile_topics
from .utils import pack_profiles_into_string
from ..models.response import AIUserProfiles
from ..env import CONFIG

ADD_KWARGS = {
    "prompt_id": "extract_profile",
}
EXAMPLES = [
    (
        """- User say Hi to assistant.
""",
        AIUserProfiles(**{"facts": []}),
    ),
    (
        """
- User's favorite movies are Inception and Interstellar [mention 2025/01/01]
- User's favorite movie is Tenet [mention 2025/01/02]
""",
        AIUserProfiles(
            **{
                "facts": [
                    {
                        "topic": "interest",
                        "sub_topic": "movie",
                        "memo": "Inception, Interstellar[mention 2025/01/01]; favorite movie is Tenet [mention 2025/01/02]",
                    },
                    {
                        "topic": "interest",
                        "sub_topic": "movie_director",
                        "memo": "user seems to be a big fan of director Christopher Nolan",
                    },
                ]
            }
        ),
    ),
]

DEFAULT_JOB = """You are a professional psychologist.
Your responsibility is to carefully read out the memo of user and extract the important profiles of user in structured format.
Then extract relevant and important facts, preferences about the user that will help evaluate the user's state.
You will not only extract the information that's explicitly stated, but also infer what's implied from the conversation.
"""

FACT_RETRIEVAL_PROMPT = """{system_prompt}

## Inputs
- **Topics guide:** focus on the supplied user-related topics; avoid unrelated ones unless the memo clearly implies them.
- **Existing profiles:** reuse the same topic/subtopic names when they reappear.
- **Memo:** Markdown bullets summarising user information, events, or preferences from the chat.

## Output
1. Think about which topics/subtopics apply and what can be inferred.
2. Produce Markdown lines in the form `- TOPIC{tab}SUB_TOPIC{tab}MEMO`.
   - Keep one line per topic/subtopic and include all details for that slot.
   - Preserve mention-time and event-time markers (e.g., `[mention ... , event ...]`).
   - Skip attributes without concrete values.

Template:
```
[POSSIBLE TOPICS THINKING...]
---
- TOPIC{tab}SUB_TOPIC{tab}MEMO
```

## Examples
{examples}

## Reminders
- Use precise dates whenever the memo allows; otherwise keep the granularity given.
- You may include strongly implied facts, but never invent information.
- Do not duplicate content across multiple lines.
- Mention-time and event-time are distinct; do not merge them.
- If nothing relevant is found, return an empty list after the separator.

Topics you can rely on:
{topic_examples}

Now read the following memo and answer using the template above.
"""


def pack_input(already_input, memo_str, strict_mode: bool = False):
    header = ""
    if strict_mode:
        header = "Don't extract topics/subtopics that are not mentioned in #### Topics Guidelines, otherwise your answer is invalid!"
    return f"""{header}
#### User Before topics
{already_input}
Don't output the topics and subtopics that are not mentioned in the following conversation.
#### Memo
{memo_str}
"""


def get_default_profiles() -> str:
    return user_profile_topics.get_prompt()


def get_prompt(topic_examples: str) -> str:
    sys_prompt = CONFIG.system_prompt or DEFAULT_JOB
    examples = "\n\n".join(
        [
            f"""<example>
<input>{p[0]}</input>
<output>
{pack_profiles_into_string(p[1])}
</output>
</example>
"""
            for p in EXAMPLES
        ]
    )
    return FACT_RETRIEVAL_PROMPT.format(
        system_prompt=sys_prompt,
        examples=examples,
        tab=CONFIG.llm_tab_separator,
        topic_examples=topic_examples,
    )


def get_kwargs() -> dict:
    return ADD_KWARGS


if __name__ == "__main__":
    print(get_prompt(get_default_profiles()))
