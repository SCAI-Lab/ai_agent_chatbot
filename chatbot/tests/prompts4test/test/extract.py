import argparse
import os
import json
import random
import string
from rich.progress import track
from time import time
from memobase import MemoBaseClient, ChatBlob
from httpx import Client
from rich import print as pprint

# Fixed directory for reading chat data
CHAT_DATA_DIR = "mock_user"
DEFAULT_ROUNDS_PER_CHUNK = 5


def generate_random_user_id(length: int = 4) -> str:
    """Generate a random user ID for MemoBase storage."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


parser = argparse.ArgumentParser()
parser.add_argument(
    "--user-id",
    type=str,
    default=None,
    help="User ID for MemoBase storage (default: random 4-char string)"
)
parser.add_argument("-u", "--project_url", type=str, default="http://localhost:8019")
parser.add_argument("-t", "--project_token", type=str, default="secret")
parser.add_argument(
    "-r",
    "--rounds-per-chunk",
    type=int,
    default=DEFAULT_ROUNDS_PER_CHUNK,
    help="Number of user turns to include in each chunk.",
)
parser.add_argument(
    "--skip-profile",
    action="store_true",
    help="Skip fetching the user profile after inserting chats.",
)

args = parser.parse_args()

# Generate random user ID if not provided
USER_ID = args.user_id if args.user_id else generate_random_user_id(4)
PROJECT_URL = args.project_url
PROJECT_TOKEN = args.project_token
ROUNDS_PER_CHUNK = args.rounds_per_chunk
SKIP_PROFILE = args.skip_profile

if ROUNDS_PER_CHUNK <= 0:
    raise ValueError("rounds-per-chunk must be a positive integer")

print(f"MemoBase User ID: {USER_ID}")
print(f"Reading chat data from: chats/{CHAT_DATA_DIR}/")
print(f"Rounds per chunk: {ROUNDS_PER_CHUNK}")
print()

client = MemoBaseClient(
    project_url=PROJECT_URL,
    api_key=PROJECT_TOKEN,
    timeout=60000,
)
hclient = Client(
    base_url=PROJECT_URL, headers={"Authorization": f"Bearer {PROJECT_TOKEN}"}
)


def chunk_messages(messages: list[dict], rounds_per_chunk: int) -> list[list[dict]]:
    """Group messages so each chunk contains up to `rounds_per_chunk` user turns."""
    chunks: list[list[dict]] = []
    current_chunk: list[dict] = []
    user_turns = 0
    awaiting_round_close = False

    for message in messages:
        current_chunk.append(message)
        role = message.get("role")

        if role == "user":
            user_turns += 1
            if user_turns >= rounds_per_chunk:
                awaiting_round_close = True
        elif role == "assistant" and awaiting_round_close:
            chunks.append(current_chunk)
            current_chunk = []
            user_turns = 0
            awaiting_round_close = False

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# Read all chat data from the fixed CHAT_DATA_DIR
chat_dir_path = f"./chats/{CHAT_DATA_DIR}"
total_files = sorted(os.listdir(chat_dir_path))
total_files = [t for t in total_files if t.endswith(".json")]
sessions = []
for i in total_files:
    with open(f"{chat_dir_path}/{i}") as f:
        messages = json.load(f)
    sessions.append({"file": f"{chat_dir_path}/{i}", "messages": messages})
uid = client.add_user()
print("User ID is", uid)
u = client.get_user(uid)

total_sessions = len(sessions)
total_chunks_all = 0

# First pass: count total chunks
for session in sessions:
    messages = session["messages"]
    message_chunks = chunk_messages(messages, ROUNDS_PER_CHUNK)
    total_chunks_all += len(message_chunks)

chunk_counter = 0
total_start_time = time()

for session_idx, session in enumerate(sessions, start=1):
    messages = session["messages"]
    message_chunks = chunk_messages(messages, ROUNDS_PER_CHUNK)
    blobs = []

    print(f"\n[Session {session_idx}/{total_sessions}] File: {session['file']}")
    print(f"  Chunks in this session: {len(message_chunks)}")

    for idx, chunk in enumerate(message_chunks, start=1):
        chunk_counter += 1
        user_messages = sum(1 for msg in chunk if msg.get("role") == "user")
        assistant_messages = sum(1 for msg in chunk if msg.get("role") == "assistant")
        print(
            f"  Chunk {idx}: {user_messages} user turns, {assistant_messages} assistant replies "
            f"[Overall: {chunk_counter}/{total_chunks_all}]"
        )
        blobs.append(ChatBlob(messages=chunk))

    session_start = time()
    for index, blob in track(enumerate(blobs), total=len(blobs)):
        u.insert(blob, sync=True)
    u.flush(sync=True)
    print(f"  Session processing time: {time() - session_start:.2f}s")

total_time = time() - total_start_time
print(f"\n✓ All {total_sessions} sessions processed ({total_chunks_all} chunks total)")
print("Cost time(s)", total_time)

if not SKIP_PROFILE:
    prompts = [m.describe for m in u.profile()]
    print("* " + "\n* ".join(sorted(prompts)))
# pprint(hclient.get(f"/api/v1/users/event/{uid}").json()["data"]["events"])
