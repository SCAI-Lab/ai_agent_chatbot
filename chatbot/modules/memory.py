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


# In-memory cache to avoid repeated file I/O
_memory_cache: Optional[Dict[str, Dict[str, Any]]] = None
_cache_dirty: bool = False


def _load_cache_data() -> Dict[str, Dict[str, Any]]:
    """Load existing cache data from file.

    Returns:
        Cache data dictionary or empty dict if file doesn't exist or is invalid.
    """
    import os

    # Return in-memory cache if already loaded
    global _memory_cache
    if _memory_cache is not None:
        return _memory_cache

    if not os.path.exists(MEMORY_CACHE_FILE):
        _memory_cache = {}
        return _memory_cache

    try:
        with open(MEMORY_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _memory_cache = data
            return _memory_cache
        logger.error("Unexpected cache format; resetting to session-based structure.")
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read cache file; creating new. %s", exc)

    _memory_cache = {}
    return _memory_cache


def _save_cache_data(cache_data: Dict[str, Dict[str, Any]], force_write: bool = False) -> None:
    """Save cache data to file.

    Args:
        cache_data: Cache data to save.
        force_write: If True, write immediately; otherwise mark as dirty for later write.
    """
    import os

    global _memory_cache, _cache_dirty

    # Update in-memory cache
    _memory_cache = cache_data
    _cache_dirty = True

    # Only write to disk if forced (e.g., on shutdown)
    if force_write:
        cache_dir = os.path.dirname(MEMORY_CACHE_FILE)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        with open(MEMORY_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        _cache_dirty = False


def flush_cache_to_disk() -> None:
    """Force write the in-memory cache to disk if dirty.

    Call this on shutdown or periodically to persist changes.
    """
    global _memory_cache, _cache_dirty

    if _cache_dirty and _memory_cache is not None:
        _save_cache_data(_memory_cache, force_write=True)
        logger.info("Cache flushed to disk")


def get_recent_history(user_uuid: str, max_messages: int = 10) -> List[Dict[str, str]]:
    """Get recent conversation history from cache file.

    Args:
        user_uuid: UUID of the user.
        max_messages: Maximum number of messages to retrieve (user + assistant pairs).

    Returns:
        List of recent messages in format [{"role": "user", "content": "..."}, ...]
    """
    cache_data = _load_cache_data()

    if user_uuid not in cache_data:
        return []

    user_data = cache_data[user_uuid]
    sessions = user_data.get("sessions", {})

    # Collect all conversations from all sessions, with timestamps
    all_conversations = []
    for session_id, session_data in sessions.items():
        conversations = session_data.get("conversations", [])
        for conv in conversations:
            timestamp = conv.get("timestamp", "")
            messages = conv.get("messages", [])
            all_conversations.append({
                "timestamp": timestamp,
                "messages": messages
            })

    # Sort by timestamp (most recent first)
    all_conversations.sort(key=lambda x: x["timestamp"], reverse=True)

    # Extract messages from most recent conversations
    recent_messages = []
    for conv in all_conversations:
        messages = conv["messages"]
        # Add messages in reverse order (since we're going from newest to oldest)
        for msg in reversed(messages):
            recent_messages.insert(0, msg)
            if len(recent_messages) >= max_messages:
                break
        if len(recent_messages) >= max_messages:
            break

    # Return only the last max_messages
    return recent_messages[-max_messages:]


def append_message_to_cache(
    user_uuid: str,
    user_name: str,
    role: str,
    content: str,
    session_id: str = None,
) -> None:
    """Append a single message to the cache file immediately.

    Args:
        user_uuid: UUID of the user.
        user_name: Name of the user.
        role: Message role ("user" or "assistant").
        content: Message content.
        session_id: Session ID (defaults to current date).
    """
    if not user_uuid or not content:
        return

    # Generate session ID if not provided (format: YYYY-MM-DD)
    if session_id is None:
        session_id = datetime.utcnow().strftime("%Y-%m-%d")

    # Load existing cache
    cache_data = _load_cache_data()

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

    # Get or create current conversation entry
    conversations = cache_data[user_uuid]["sessions"][session_id]["conversations"]

    # Check if we need to create a new conversation entry or append to the last one
    if not conversations or "messages" not in conversations[-1]:
        # Create new conversation entry
        conversations.append({
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "messages": []
        })

    # Append message to the last conversation
    conversations[-1]["messages"].append({
        "role": role,
        "content": content
    })
    conversations[-1]["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Update in-memory cache (no disk write for performance)
    _save_cache_data(cache_data, force_write=False)


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
    if not user_uuid or (not user_text and not assistant_text):
        return

    # Generate session ID if not provided (format: YYYY-MM-DD)
    if session_id is None:
        session_id = datetime.utcnow().strftime("%Y-%m-%d")

    # Create conversation entry
    entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timings": {
            "speech_to_text": round(speech_duration, 2),
            "llm_generation": round(llm_duration, 2),
            "total": round(speech_duration + llm_duration, 2),
        },
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
    }

    # Load existing cache
    cache_data = _load_cache_data()

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

    # Update in-memory cache (no disk write for performance)
    _save_cache_data(cache_data, force_write=False)
