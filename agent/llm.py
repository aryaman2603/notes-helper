from ollama import chat

SYSTEM_PROMPT = """You are a helpful university-level tutor.
Answer using ONLY the provided notes.
If the notes do not contain the answer, say so clearly.
Explain concepts step by step.
"""

def generate_answer(question: str, context: str) -> str:
    response = chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
                Notes:
               {context}
               Question:
               {question}
               """
            }
        ]
    )

    return response["message"]["content"]
