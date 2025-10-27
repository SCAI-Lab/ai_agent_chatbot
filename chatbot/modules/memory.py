"""MemoBase long-term memory integration."""
import copy
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from .config import (
    MEMOBASE_API_KEY,
    MEMOBASE_BASE_URL,
    MEMOBASE_TIMEOUT,
    DEFAULT_MEMORY_PROMPT,
    DEFAULT_MAX_CONTEXT_SIZE,
    MEMORY_CACHE_FILE,
    logger,
)


class MemoBaseAPIError(Exception):
    """Exception raised for MemoBase API errors."""
    pass


# Initialize session with authentication
memo_session = requests.Session()
memo_session.headers.update({"Authorization": f"Bearer {MEMOBASE_API_KEY}"})


def string_to_uuid(value: str, salt: str = "memobase_client") -> str:
    """Convert a string to a deterministic UUID.

    Args:
        value: Input string to convert.
        salt: Salt string for UUID generation.

    Returns:
        UUID string.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{value}{salt}"))


def _handle_memobase_response(response: requests.Response) -> Any:
    """Parse and validate MemoBase API response.

    Args:
        response: HTTP response object.

    Returns:
        Response data payload.

    Raises:
        MemoBaseAPIError: If response is invalid or contains an error.
    """
    try:
        payload = response.json()
    except ValueError as exc:
        raise MemoBaseAPIError("Invalid JSON response from MemoBase API") from exc
    if payload.get("errno") != 0:
        raise MemoBaseAPIError(payload.get("errmsg", "MemoBase API error"))
    return payload.get("data")


def memobase_request(
    method: str,
    path: str,
    *,
    params: Optional[dict] = None,
    json_payload: Optional[dict] = None,
    allow_404: bool = False,
) -> Any:
    """Make a request to the MemoBase API.

    Args:
        method: HTTP method (GET, POST, etc.).
        path: API endpoint path.
        params: Query parameters.
        json_payload: JSON request body.
        allow_404: If True, return None for 404 responses instead of raising.

    Returns:
        Response data or None if 404 and allow_404=True.

    Raises:
        MemoBaseAPIError: If the API returns an error.
        requests.HTTPError: If the HTTP request fails.
    """
    url = f"{MEMOBASE_BASE_URL}{path}"
    response = memo_session.request(
        method,
        url,
        params=params,
        json=json_payload,
        timeout=MEMOBASE_TIMEOUT,
    )
    if allow_404 and response.status_code == 404:
        return None
    response.raise_for_status()
    return _handle_memobase_response(response)


def ensure_memobase_user(user_uuid: str) -> None:
    """Ensure a user exists in MemoBase, creating if necessary.

    Args:
        user_uuid: UUID of the user.
    """
    if not user_uuid:
        return
    try:
        existing = memobase_request("GET", f"/users/{user_uuid}", allow_404=True)
    except MemoBaseAPIError as exc:
        message = str(exc).lower()
        # MemoBase returns errno!=0 with HTTP 200 when a user is missing; treat that as absent.
        if "not found" in message or "404" in message:
            existing = None
        else:
            raise
    if existing is not None:
        return
    memobase_request("POST", "/users", json_payload={"id": user_uuid, "data": None})


def fetch_memobase_context(
    user_uuid: str,
    max_token_size: int = DEFAULT_MAX_CONTEXT_SIZE,
    chats: Optional[list] = None,
) -> str:
    """Fetch long-term memory context for a user.

    Args:
        user_uuid: UUID of the user.
        max_token_size: Maximum tokens for context.
        chats: Recent chat messages for context retrieval.

    Returns:
        Context string from MemoBase.
    """
    params: dict[str, Any] = {"max_token_size": max(0, max_token_size)}
    if chats:
        params["chats_str"] = json.dumps(chats, ensure_ascii=False)
    data = memobase_request("GET", f"/users/context/{user_uuid}", params=params)
    if not data:
        return ""
    return data.get("context", "") or ""


def build_context_prompt(context: str, additional_prompt: str = DEFAULT_MEMORY_PROMPT) -> str:
    """Build a prompt with memory context.

    Args:
        context: Memory context string.
        additional_prompt: Additional instructions for using the memory.

    Returns:
        Formatted prompt string.
    """
    if not context:
        return ""

    prompt_template = """
--# LONG-TERM MEMORY #--
The following is the user's long-term memory retrieved from previous conversations:

{user_context}

{additional_memory_prompt}
--# END OF LONG-TERM MEMORY #--"""

    return prompt_template.format(
        user_context=context,
        additional_memory_prompt=additional_prompt,
    ).strip()


def inject_memobase_context(messages: List[Dict[str, str]], context: str) -> List[Dict[str, str]]:
    """Inject MemoBase context into message list.

    Args:
        messages: List of chat messages.
        context: Context string to inject.

    Returns:
        Modified message list with context injected.
    """
    if not context:
        return messages

    prompt_text = build_context_prompt(context)
    if not messages:
        return [{"role": "system", "content": prompt_text}]

    # Make a deep copy to avoid modifying the original
    messages = copy.deepcopy(messages)
    first_message = messages[0]

    if first_message.get("role") == "system":
        first_message["content"] = (first_message.get("content") or "") + "\n" + prompt_text
    else:
        messages.insert(0, {"role": "system", "content": prompt_text})

    return messages


def format_short_term_memory(history: List[Dict[str, str]]) -> str:
    """Format stored conversation turns for inclusion in the system prompt.

    Args:
        history: List of conversation messages.

    Returns:
        Formatted memory string with header.
    """
    if not history:
        return "--# SHORT-TERM MEMORY #--\nNo recent conversation history.\n--# END OF SHORT-TERM MEMORY #--"

    formatted = []
    for entry in history:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        formatted.append(f"{label}: {content}")

    if not formatted:
        return "--# SHORT-TERM MEMORY #--\nNo recent conversation history.\n--# END OF SHORT-TERM MEMORY #--"

    memory_content = "\n".join(formatted)
    return f"--# SHORT-TERM MEMORY #--\nThe following is the recent conversation history from this session:\n\n{memory_content}\n--# END OF SHORT-TERM MEMORY #--"


def prepare_recent_chats(messages: List[Dict[str, str]], max_items: int = 6) -> List[Dict[str, str]]:
    """Prepare recent chat messages for context retrieval.

    Args:
        messages: List of all messages.
        max_items: Maximum number of recent messages to include.

    Returns:
        Filtered list of recent user/assistant messages.
    """
    filtered = [
        {"role": msg.get("role", ""), "content": msg.get("content", "")}
        for msg in messages
        if msg.get("role") in {"user", "assistant"} and msg.get("content")
    ]
    return filtered[-max_items:]


def append_chat_to_cache(
    user_uuid: str,
    user_name: str,
    user_text: str,
    assistant_text: str,
    speech_duration: float,
    llm_duration: float,
    session_id: str = None,
) -> None:
    """Append a conversation round to the local cache file (session-based).

    Args:
        user_uuid: UUID of the user.
        user_name: Name of the user.
        user_text: User's message.
        assistant_text: Assistant's response.
        speech_duration: Time taken for speech processing.
        llm_duration: Time taken for LLM generation.
        session_id: Session ID (defaults to current date).
    """
    import os

    if not user_uuid or (not user_text and not assistant_text):
        return

    # Generate session ID if not provided (format: YYYY-MM-DD)
    if session_id is None:
        session_id = datetime.utcnow().strftime("%Y-%m-%d")

    speech_duration = round(speech_duration, 2)
    llm_duration = round(llm_duration, 2)
    total_duration = round(speech_duration + llm_duration, 2)

    entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timings": {
            "speech_to_text": speech_duration,
            "llm_generation": llm_duration,
            "total": total_duration,
        },
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
    }

    cache_dir = os.path.dirname(MEMORY_CACHE_FILE)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Load existing cache (session-based structure)
    cache_data: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(MEMORY_CACHE_FILE):
        try:
            with open(MEMORY_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cache_data = data
            else:
                logger.error("Unexpected cache format; resetting to session-based structure.")
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read cache file; creating new. %s", exc)

    # Initialize user if not exists
    if user_uuid not in cache_data:
        cache_data[user_uuid] = {
            "user_name": user_name,
            "sessions": {}
        }

    # Initialize session if not exists
    if session_id not in cache_data[user_uuid]["sessions"]:
        cache_data[user_uuid]["sessions"][session_id] = {
            "start_time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "conversations": []
        }

    # Append conversation to session
    cache_data[user_uuid]["sessions"][session_id]["conversations"].append(entry)

    # Save cache
    with open(MEMORY_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
