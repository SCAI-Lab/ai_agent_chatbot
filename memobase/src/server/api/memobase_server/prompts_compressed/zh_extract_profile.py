from . import zh_user_profile_topics
from ..models.response import AIUserProfiles
from ..env import CONFIG, LOG
from .utils import pack_profiles_into_string

ADD_KWARGS = {
    "prompt_id": "zh_extract_profile",
}

EXAMPLES = [
    (
        """- 用户向助手问好。
""",
        AIUserProfiles(**{"facts": []}),
    ),
    (
        """
- 用户最喜欢的电影是《盗梦空间》和《星际穿越》 [提及于2025/01/01]
- 用户最喜欢的电影是《信条》 [提及于2025/01/02]
""",
        AIUserProfiles(
            **{
                "facts": [
                    {
                        "topic": "兴趣爱好",
                        "sub_topic": "电影",
                        "memo": "《盗梦空间》、《星际穿越》[提及于2025/01/01]；最喜欢的是《信条》[提及于2025/01/02]",
                    },
                    {
                        "topic": "兴趣爱好",
                        "sub_topic": "电影导演",
                        "memo": "用户似乎是克里斯托弗·诺兰的忠实粉丝",
                    },
                ]
            }
        ),
    ),
]

DEFAULT_JOB = """你是一位专业的心理学家。
你的责任是仔细阅读用户的备忘录并以结构化的格式提取用户的重要信息。
然后提取相关且重要的事实、用户偏好，这些信息将有助于评估用户的状态。
你不仅要提取明确陈述的信息，还要推断对话中隐含的信息。
你要使用与用户输入相同的语言来记录这些事实。
"""

FACT_RETRIEVAL_PROMPT = """{system_prompt}

## 输入
- **主题指引：** 重点关注提供的用户相关主题，除非备忘录明确暗示，否则不要扩展到无关主题。
- **已有画像：** 若对话再次提到相同主题/子主题，请复用相同名称。
- **备忘录：** Markdown 列表，概括聊天中的用户信息、事件与偏好。

## 输出
1. 先思考可以提取或推断的主题/子主题。
2. 按 `- TOPIC{tab}SUB_TOPIC{tab}MEMO` 的格式逐条输出。
   - 一个主题/子主题只保留一行，写全该槽位的全部信息。
   - 保留提及时间与事件时间标记（如 `[提及于..., 发生于...]`）。
   - 没有明确取值的属性不要输出。

模板：
```
POSSIBLE TOPICS THINKING
---
- TOPIC{tab}SUB_TOPIC{tab}MEMO
```

## 示例
{examples}

## 注意
- 能推断具体日期时请明确写出；否则保持原有粒度。
- 允许写出明显推断出的事实，但禁止凭空编造。
- 避免重复记录同一内容。
- “提及时间”与“事件发生时间”是两类信息，需分别保留。
- 若没有可用信息，分隔线后可以输出空列表。

可使用的主题列表：
{topic_examples}

下面是备忘录，请根据上述模板输出，并使用与输入相同的语言记录事实。
"""


def pack_input(already_input, chat_strs, strict_mode: bool = False):
    header = ""
    if strict_mode:
        header = "不要提取#### 主题建议中没出现的主题/子主题， 否则你的回答是无效的！"
    return f"""{header}
#### 已有的主题
如果提取相关的主题/子主题，请考虑使用下面的主题/子主题命名:
{already_input}

#### 备忘录
请注意，不要输出任何关于备忘录中未提及的主题/子主题的信息:
{chat_strs}
"""


def get_default_profiles() -> str:
    return zh_user_profile_topics.get_prompt()


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
