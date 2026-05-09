
from datetime import datetime

def log_step(state, message):
    timestamp = datetime.now().isoformat()
    state["logs"].append(f"[{timestamp}] {message}")
    return state
