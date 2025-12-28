from collections import defaultdict
from typing import List, Dict

_CHAT_MEMORY: Dict[tuple, List[dict]] = defaultdict(list)
MAX_TURNS = 6

def get_memory(session_id: str, subject:str) -> List[dict]:
    return _CHAT_MEMORY[(session_id, subject)]

def append_memory(session_id: str, subject:str, role: str, content: str) -> None:
    key = (session_id, subject)
    _CHAT_MEMORY[key].append({"role": role, "content": content})

    _CHAT_MEMORY[key] = _CHAT_MEMORY[key][-MAX_TURNS*2:]  

def clear_memory(session_id: str, subject:str) -> None:
    _CHAT_MEMORY.pop((session_id, subject), None)