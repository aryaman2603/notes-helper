from ollama import chat

SYSTEM_PROMPT = """You are a helpful university-level tutor.
Answer using ONLY the provided notes.
If the notes do not contain the answer, say so clearly.
Explain concepts step by step in 3 to 4 paragraphs.
"""

def generate_answer(question: str, context: str, history: list) -> str:


    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    if context:
        messages.append({
            "role": "system",
            "content": f"Context\n{context}"
        })

    for msg in history:
        messages.append(msg)
    
    messages.append({
        "role": "user",
        "content": question
    })

    response = chat(
        model="llama3.1:8b",
        messages=messages
    )

    return response["message"]["content"]
