import argparse
import os
import json
from rich.progress import track
from time import time
from memobase import MemoBaseClient, ChatBlob
from httpx import Client
from rich import print as pprint

USER = "54"
DEFAULT_ROUNDS_PER_CHUNK = 5
parser = argparse.ArgumentParser()
parser.add_argument("--user", type=str, default=USER)
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
USER = args.user
PROJECT_URL = args.project_url
PROJECT_TOKEN = args.project_token
ROUNDS_PER_CHUNK = args.rounds_per_chunk
SKIP_PROFILE = args.skip_profile

if ROUNDS_PER_CHUNK <= 0:
    raise ValueError("rounds-per-chunk must be a positive integer")

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


total_files = sorted(os.listdir(f"./chats/{USER}"))
total_files = [t for t in total_files if t.endswith(".json")]
sessions = []
for i in total_files:
    with open(f"./chats/{USER}/{i}") as f:
        messages = json.load(f)
    sessions.append({"file": f"./chats/{USER}/{i}", "messages": messages})
uid = client.add_user()
print("User ID is", uid)
u = client.get_user(uid)

for session in sessions:
    messages = session["messages"]
    message_chunks = chunk_messages(messages, ROUNDS_PER_CHUNK)
    blobs = []
    print("File:", session["file"])
    print("  Chunks:")
    for idx, chunk in enumerate(message_chunks, start=1):
        user_messages = sum(1 for msg in chunk if msg.get("role") == "user")
        assistant_messages = sum(1 for msg in chunk if msg.get("role") == "assistant")
        print(
            f"  Chunk {idx}: {user_messages} user turns, {assistant_messages} assistant replies"
        )
        blobs.append(ChatBlob(messages=chunk))

    print("Total chats:", len(blobs))

    start = time()
    for index, blob in track(enumerate(blobs), total=len(blobs)):
        u.insert(blob, sync=True)
    u.flush(sync=True)
    print("Cost time(s)", time() - start)

    if not SKIP_PROFILE:
        prompts = [m.describe for m in u.profile()]
        print("* " + "\n* ".join(sorted(prompts)))
    # pprint(hclient.get(f"/api/v1/users/event/{uid}").json()["data"]["events"])
