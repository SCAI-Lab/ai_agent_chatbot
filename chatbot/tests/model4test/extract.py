import argparse
import os
import json
import random
import string
import sys
from time import time
from memobase import MemoBaseClient, ChatBlob
from httpx import Client
from rich.progress import track

# Generate random 4-character user ID
DEFAULT_USER = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
DEFAULT_ROUNDS_PER_CHUNK = 5
parser = argparse.ArgumentParser()
parser.add_argument("--user", type=str, default="mock_user", help="Directory name under ./chats/ to read data from")
parser.add_argument("--model-name", type=str, help="Model name for output file (e.g., llama, mistral, qwen). If not specified, uses random 4-char string")
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
parser.add_argument(
    "--warmup-model",
    type=str,
    default=None,
    help="Optional Ollama model name to warm up before benchmarking.",
)
parser.add_argument(
    "--ollama-base-url",
    type=str,
    default=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
    help="Base URL for the Ollama server (default: http://localhost:11434).",
)

args = parser.parse_args()
USER = args.user
MODEL_NAME = args.model_name if args.model_name else ''.join(random.choices(string.ascii_letters + string.digits, k=4))
PROJECT_URL = args.project_url
PROJECT_TOKEN = args.project_token
ROUNDS_PER_CHUNK = args.rounds_per_chunk
SKIP_PROFILE = args.skip_profile
WARMUP_MODEL = args.warmup_model
OLLAMA_BASE_URL = args.ollama_base_url.rstrip("/")

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

# Create output2 directory if it doesn't exist
OUTPUT_DIR = "./chats/output2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup output file redirection
output_file = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.txt")

# Redirect stdout to both console and file
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

output_log = open(output_file, 'w')
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, output_log)


def warm_up_model(model_name: str | None) -> None:
    """Call Ollama directly once so that model weights are loaded into cache."""
    if not model_name:
        return
    generate_url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model_name,
        "prompt": "Warm-up request for cache priming.",
        "stream": False,
    }
    try:
        with Client(timeout=60) as http_client:
            response = http_client.post(generate_url, json=payload)
            response.raise_for_status()
    except Exception as exc:
        pass  # 静默失败


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
u = client.get_user(uid)

warm_up_model(WARMUP_MODEL)

for session in sessions:
    messages = session["messages"]
    message_chunks = chunk_messages(messages, ROUNDS_PER_CHUNK)
    blobs = []

    # 只输出文件名（供 evaluate.py 识别 session）
    print("File:", session["file"])

    for idx, chunk in enumerate(message_chunks, start=1):
        blobs.append(ChatBlob(messages=chunk))

    start = time()
    for index, blob in track(enumerate(blobs), total=len(blobs), description="Processing", disable=True):
        u.insert(blob, sync=True)
    u.flush(sync=True)

    # 只输出时间（供 evaluate.py 读取）
    print("Cost time(s)", time() - start)

    # 只输出提取到的信息
    if not SKIP_PROFILE:
        prompts = [m.describe for m in u.profile()]
        for prompt in sorted(prompts):
            print("*", prompt)
    # pprint(hclient.get(f"/api/v1/users/event/{uid}").json()["data"]["events"])

# Close output file and restore stdout
sys.stdout = original_stdout
output_log.close()
print(f"\nOutput saved to: {output_file}")
