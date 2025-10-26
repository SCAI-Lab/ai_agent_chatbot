from .utils import pack_merge_action_into_string
from ..env import CONFIG

ADD_KWARGS = {"prompt_id": "zh_merge_profile_yolo"}

MERGE_FACTS_PROMPT = """你负责维护用户备忘录，需要判断每条补充信息与现有内容的合并方式。

## 输入
列表中的每个 JSON 对象形如：
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

## 决策准则
1. 先确认 `new_info` 是否符合 `topic_description`。若无法调整使其符合，直接 ABORT。
2. 若存在 `update_instruction`，必须遵守。
3. 对比 `current_memo`：
   - 为空 → 满足要求时 APPEND。
   - 有冲突或需改写 → UPDATE，并输出不超过 5 句的精炼备忘录。
   - 信息冗余或无价值 → ABORT。
4. 保留所有时间标注。

## 动作格式
- APPEND：`N. APPEND{tab}APPEND`
- UPDATE：`N. UPDATE{tab}[UPDATED_MEMO]`（写出完整更新后的内容，保留时间标记，避免重复表述）
- ABORT：`N. ABORT{tab}ABORT`

## 输出模版
```
THOUGHT
---
<你的思考>
1. ACTION{tab}...
2. ACTION{tab}...
```

## 示例
输入：
{{
    "memo_id": "1",
    "new_info": "准备期末考试[提及于2025/06/01]",
    "current_memo": "准备期中考试[提及于2025/04/01]",
    "topic": "学习",
    "subtopic": "考试目标",
    "update_instruction": "更新目标时删除过时或冲突内容"
}}
{{
    "memo_id": "2",
    "new_info": "使用多邻国自学日语",
    "current_memo": "",
    "topic": "学习",
    "subtopic": "使用软件"
}}
{{
    "memo_id": "3",
    "new_info": "用户喜欢吃火锅",
    "current_memo": "",
    "topic": "兴趣",
    "subtopic": "运动"
}}

输出：
```
目标需要从期中考试改为期末考试，其他信息按要求处理。
---
1. UPDATE{tab}准备期末考试[提及于2025/06/01]
2. APPEND{tab}APPEND
3. ABORT{tab}ABORT
```

## 要求
- 严格遵循上述格式。
- 不编造输入外的内容。
- UPDATE 结果要精炼，避免“User is sad; User's mood is sad”这类重复。
- 始终保留新旧内容中的时间标注（例如 `[...] [提及于2025/05/05, 发生于2022]`）。
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
