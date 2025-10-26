"""Sync cached conversations to MemoBase."""
import argparse
import json
import logging
import os
from typing import Any, Dict, List

from modules.config import MEMORY_CACHE_FILE
from modules.memory import ensure_memobase_user, memobase_request


# Configure logging for this script
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


MEMORY_CACHE_BATCH_SIZE = int(os.getenv("MEMORY_CACHE_BATCH_SIZE", "10"))


def insert_memobase_chat(user_uuid: str, user_text: str, assistant_text: str, sync: bool = False) -> None:
    """Insert a chat conversation into MemoBase.

    Args:
        user_uuid: UUID of the user.
        user_text: User's message.
        assistant_text: Assistant's response.
        sync: Whether to wait for processing to complete.
    """
    payload = {
        "blob_type": "chat",
        "fields": None,
        "blob_data": {
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        },
    }
    params = {"wait_process": "true" if sync else "false"}
    memobase_request("POST", f"/blobs/insert/{user_uuid}", params=params, json_payload=payload)


def flush_memobase_user(user_uuid: str, sync: bool = True) -> None:
    """Flush pending chat data for a user to MemoBase.

    Args:
        user_uuid: UUID of the user.
        sync: Whether to wait for processing to complete.
    """
    params = {"wait_process": "true" if sync else "false"}
    memobase_request("POST", f"/users/buffer/{user_uuid}/chat", params=params)


def load_cache_entries(cache_file: str) -> List[Dict[str, Any]]:
    """Load cache entries from JSON file (supports both old and new format).

    Args:
        cache_file: Path to cache file.

    Returns:
        List of cache entries in flat format for processing.
    """
    if not os.path.exists(cache_file):
        return []

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load cache entries: %s", exc)
        return []

    # Handle old format (list)
    if isinstance(data, list):
        logger.info("Old cache format detected.")
        return data

    # Handle new format (session-based dict)
    if not isinstance(data, dict):
        logger.error("Unexpected cache format.")
        return []

    # Flatten session-based structure into list of entries
    logger.info("New session-based cache format detected.")
    entries: List[Dict[str, Any]] = []
    for user_uuid, user_data in data.items():
        user_name = user_data.get("user_name", "unknown")
        sessions = user_data.get("sessions", {})

        for session_id, session_data in sessions.items():
            conversations = session_data.get("conversations", [])

            for conv in conversations:
                # Convert to flat format for compatibility with sync logic
                entry = {
                    "timestamp": conv.get("timestamp"),
                    "user_uuid": user_uuid,
                    "user_name": user_name,
                    "session_id": session_id,
                    "timings": conv.get("timings", {}),
                    "messages": conv.get("messages", []),
                }
                entries.append(entry)

    return entries


def persist_remaining_entries(cache_file: str, entries: List[Dict[str, Any]]) -> None:
    """Persist remaining entries back to cache file in session-based format.

    Args:
        cache_file: Path to cache file.
        entries: List of entries to persist.
    """
    if not entries:
        if os.path.exists(cache_file):
            os.remove(cache_file)
        return

    # Convert flat entries back to session-based structure
    session_based_data: Dict[str, Dict[str, Any]] = {}

    for entry in entries:
        user_uuid = entry.get("user_uuid")
        user_name = entry.get("user_name", "unknown")
        session_id = entry.get("session_id", "unknown")

        if not user_uuid:
            continue

        # Initialize user if not exists
        if user_uuid not in session_based_data:
            session_based_data[user_uuid] = {
                "user_name": user_name,
                "sessions": {}
            }

        # Initialize session if not exists
        if session_id not in session_based_data[user_uuid]["sessions"]:
            session_based_data[user_uuid]["sessions"][session_id] = {
                "start_time": entry.get("timestamp"),  # Use first conversation's timestamp
                "conversations": []
            }

        # Add conversation to session
        conversation = {
            "timestamp": entry.get("timestamp"),
            "timings": entry.get("timings", {}),
            "messages": entry.get("messages", []),
        }
        session_based_data[user_uuid]["sessions"][session_id]["conversations"].append(conversation)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(session_based_data, f, ensure_ascii=False, indent=2)


def process_cache(batch_size: int) -> None:
    """Process cached conversations and sync to MemoBase.

    Args:
        batch_size: Number of conversation rounds per batch.
    """
    entries = load_cache_entries(MEMORY_CACHE_FILE)

    if not entries:
        logger.info("Memory cache is empty; nothing to sync.")
        return

    logger.info("Loaded %d cached conversation rounds.", len(entries))

    try:
        while entries:
            batch = entries[:batch_size]
            user_uuid = batch[0].get("user_uuid")

            if not user_uuid:
                logger.warning("Skipping batch without user_uuid; discarding %d entries.", len(batch))
                entries = entries[len(batch):]
                persist_remaining_entries(MEMORY_CACHE_FILE, entries)
                continue

            ensure_memobase_user(user_uuid)

            for entry in batch:
                messages = entry.get("messages") or []
                user_text = ""
                assistant_text = ""

                for message in messages:
                    role = message.get("role")
                    content = message.get("content", "")
                    if role == "user":
                        user_text = content
                    elif role == "assistant":
                        assistant_text = content

                insert_memobase_chat(user_uuid, user_text, assistant_text, sync=False)

            flush_memobase_user(user_uuid, sync=True)
            logger.info("Synced %d rounds for user %s.", len(batch), user_uuid)

            entries = entries[len(batch):]
            persist_remaining_entries(MEMORY_CACHE_FILE, entries)

    except Exception as exc:
        logger.error("Sync interrupted: %s", exc)
        persist_remaining_entries(MEMORY_CACHE_FILE, entries)
        raise

    persist_remaining_entries(MEMORY_CACHE_FILE, [])
    logger.info("Memory cache sync complete; cache cleared.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sync cached conversations to MemoBase.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=MEMORY_CACHE_BATCH_SIZE,
        help="Number of conversation rounds per flush batch.",
    )
    args = parser.parse_args()
    process_cache(args.batch_size)


if __name__ == "__main__":
    main()
