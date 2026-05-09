
from datetime import datetime

def log(state, message):
    state["logs"].append(
        f"[{datetime.now().isoformat()}] {message}"
    )
