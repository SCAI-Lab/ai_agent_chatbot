"""LLM chat integration with Ollama."""
import copy
import time
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_STREAM,
    OLLAMA_TEMPERATURE,
    OLLAMA_MAX_TOKENS,
    DEFAULT_MAX_CONTEXT_SIZE,
    logger,
)
from .memory import (
    ensure_memobase_user,
    fetch_memobase_context,
    inject_memobase_context,
    prepare_recent_chats,
    string_to_uuid,
)


# Initialize OpenAI client for Ollama
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def chat(
    messages: List[Dict[str, str]],
    current_speaker: Optional[str] = None,
    close_session: bool = False,
    use_users: bool = True,
    stream: bool = OLLAMA_STREAM,
    model: str = OLLAMA_MODEL,
    debug: bool = False,
) -> Tuple[str, Optional[str]]:
    """Chat with LLM using MemoBase long-term memory enhancement.

    Args:
        messages: List of chat messages with role and content.
        current_speaker: Current speaker name for memory lookup.
        close_session: Whether to close the session (unused, for compatibility).
        use_users: Whether to use MemoBase user context.
        stream: Whether to stream the response.
        model: Model name to use.

    Returns:
        Tuple of (assistant_reply, user_uuid).
    """
    from .timing import _record_timing

    # Time: Prepare messages with MemoBase long-term memory
    prep_start = time.perf_counter()

    last_user_message = ""
    for item in reversed(messages):
        if item.get("role") == "user":
            last_user_message = item.get("content", "")
            break

    user_uuid: Optional[str] = None
    messages_for_llm = copy.deepcopy(messages)

    # Inject MemoBase context if using user memory
    if use_users and current_speaker:
        user_uuid = string_to_uuid(current_speaker)
        try:
            ensure_memobase_user(user_uuid)
        except Exception as exc:
            logger.error(f"Failed to ensure MemoBase user {current_speaker}: {exc}")
            user_uuid = None

        if user_uuid:
            try:
                chats_for_context = prepare_recent_chats(messages)
                context_text = fetch_memobase_context(
                    user_uuid,
                    DEFAULT_MAX_CONTEXT_SIZE,
                    chats=chats_for_context,
                )
                if debug and context_text:
                    print("\n[DEBUG] MemoBase context retrieved:")
                    print(context_text)
                    print("-" * 60)
            except Exception as exc:
                logger.error(f"Failed to fetch MemoBase context for {current_speaker}: {exc}")
                context_text = ""

            messages_for_llm = inject_memobase_context(messages_for_llm, context_text)

    _record_timing("llm_memory_fetch", time.perf_counter() - prep_start)

    if debug:
        print("\n[DEBUG] Final prompt sent to LLM (after memory injection):")
        for idx, message in enumerate(messages_for_llm, start=1):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            print(f"[DEBUG] Message {idx} ({role}):")
            print(content)
            print("-" * 60)

    # Send to Ollama and measure generation time
    inference_start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            messages=messages_for_llm,
            model=model,
            stream=stream,
            user=current_speaker if use_users else None,
            temperature=OLLAMA_TEMPERATURE,
            max_tokens=OLLAMA_MAX_TOKENS,
        )
    except Exception as exc:
        logger.error(f"LLM chat request failed: {exc}")
        return "", user_uuid

    assistant_reply = ""

    if stream:
        collected_chunks: list[str] = []
        first_token_received = False
        first_token_time = None

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            delta_text = getattr(delta, "content", None)
            if not delta_text:
                continue

            # Record time to first token
            if not first_token_received:
                first_token_time = time.perf_counter() - inference_start
                _record_timing('llm_first_token', first_token_time)
                first_token_received = True

            collected_chunks.append(delta_text)
            print(delta_text, end="", flush=True)

        print()
        assistant_reply = "".join(collected_chunks).strip()

        # Record total inference time
        total_inference_time = time.perf_counter() - inference_start
        _record_timing('llm_inference', total_inference_time)
    else:
        if response.choices:
            assistant_reply = (response.choices[0].message.content or "").strip()
            if assistant_reply:
                print(assistant_reply)

    return assistant_reply, user_uuid

