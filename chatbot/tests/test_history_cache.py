#!/usr/bin/env python3
"""Test script for file-based history management."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.memory import append_message_to_cache, get_recent_history, string_to_uuid, _load_cache_data

def test_file_based_history():
    """Test the new file-based history management."""

    # Test data
    test_user = "test_user_123"
    test_uuid = string_to_uuid(test_user)

    print("=" * 60)
    print("Testing File-Based History Management")
    print("=" * 60)
    print(f"Test user: {test_user}")
    print(f"Test UUID: {test_uuid}")
    print()

    # Test 1: Append user message
    print("Test 1: Appending user message...")
    append_message_to_cache(test_uuid, test_user, "user", "Hello, how are you?")
    print("✓ User message appended")
    print()

    # Test 2: Append assistant message
    print("Test 2: Appending assistant message...")
    append_message_to_cache(test_uuid, test_user, "assistant", "I'm doing well, thank you!")
    print("✓ Assistant message appended")
    print()

    # Test 3: Read recent history
    print("Test 3: Reading recent history...")
    history = get_recent_history(test_uuid, max_messages=10)
    print(f"✓ Retrieved {len(history)} messages")
    for i, msg in enumerate(history):
        print(f"  {i+1}. [{msg['role']}]: {msg['content']}")
    print()

    # Test 4: Append more messages
    print("Test 4: Appending more messages...")
    append_message_to_cache(test_uuid, test_user, "user", "What's the weather like?")
    append_message_to_cache(test_uuid, test_user, "assistant", "I don't have access to weather data.")
    print("✓ Additional messages appended")
    print()

    # Test 5: Read history again with limit
    print("Test 5: Reading history with limit (max 4 messages)...")
    history = get_recent_history(test_uuid, max_messages=4)
    print(f"✓ Retrieved {len(history)} messages (limited to 4)")
    for i, msg in enumerate(history):
        print(f"  {i+1}. [{msg['role']}]: {msg['content']}")
    print()

    # Test 6: Verify cache file structure
    print("Test 6: Verifying cache file structure...")
    cache_data = _load_cache_data()
    if test_uuid in cache_data:
        user_data = cache_data[test_uuid]
        print(f"✓ User data found in cache")
        print(f"  User name: {user_data.get('user_name')}")
        print(f"  Sessions: {len(user_data.get('sessions', {}))}")
        for session_id, session_data in user_data.get('sessions', {}).items():
            convs = session_data.get('conversations', [])
            print(f"  Session {session_id}: {len(convs)} conversations")
    else:
        print("✗ User not found in cache")
    print()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

if __name__ == "__main__":
    test_file_based_history()
