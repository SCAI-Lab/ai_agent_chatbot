from ..env import CONFIG

ADD_KWARGS = {
    "prompt_id": "zh_summary_entry_chats",
}
SUMMARY_PROMPT = """你负责从用户与助手的对话中提取用户信息、日程与事件。

## 任务清单
- 捕捉所有与用户相关的事实、计划、事件。
- 额外要求：{additional_requirements}
- 若消息带有 `[TIME]`，需把相对时间换算为明确日期；无法推断具体日期时，仅保留提及日期。

## 参考
需覆盖的主题：
<topics>
{topics}
</topics>
可关注的属性：
<attributes>
{attributes}
</attributes>

## 输入
### 已记录
格式类似 `TOPIC{separator}SUBTOPIC{separator}CONTENT`（可能被截断）。
### 对话
形如 `[TIME] NAME: MESSAGE` 的对话行。

## 输出
使用 Markdown 列表：`- 描述 [提及 2023/01/23, 事件 ...] // TYPE`。
- 每条都写出提及时间；若能推断事件/计划发生时间，也请补充。
- 描述必须客观、简明。
示例：
```
- 用户昵称是 Jack。[提及 2023/01/23] // info
- Jack 计划去健身房。[提及 2023/01/23, 事件 2023/01/24] // schedule
```

现在开始执行任务。
"""


def pack_input(already_logged_str: str, chat_strs: str):
    return f"""### 已记录
{already_logged_str}

### 输入对话
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
