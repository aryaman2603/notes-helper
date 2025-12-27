import asyncio
import json
import sys
import ollama

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

SYSTEM_PROMPT = """
You are a study assistant for a student.
You help the student understand concepts using their personal notes.
If you need information from the notes, use the available tools.
Explain concepts clearly and in an exam-oriented manner.
"""

async def ollama_chat_async(**kwargs):
    return await asyncio.to_thread(ollama.chat, **kwargs)

async def main():
    # -----------------------------
    # 1. Read query from CLI args
    # -----------------------------
    if len(sys.argv) < 2:
        print("Usage: python agent/client.py \"your question here\"")
        sys.exit(1)

    user_query = sys.argv[1]

    # -----------------------------
    # 2. MCP server parameters
    # -----------------------------
    print("1")
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "agent.min_server"]
    )
    print("2")

    # -----------------------------
    # 3. MCP client session
    # -----------------------------
    print("3")
    async with stdio_client(server_params) as (read, write):
        print("4")
        session = ClientSession(read, write)
        print("5")
        await session.initialize()
        print("6")

        tools = await session.list_tools()

        # -----------------------------
        # 4. First LLM call (may trigger tool)
        # -----------------------------
        response = await ollama_chat_async(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            tools=tools,
            tool_choice="auto"
        )

        # -----------------------------
        # 5. If tool was called
        # -----------------------------
        if response["message"].get("tool_calls"):
            for tool_call in response["message"]["tool_calls"]:
                tool_name = tool_call["name"]
                args = tool_call["arguments"]

                tool_result = await session.call_tool(tool_name, args)

                final_response = await ollama_chat_async(
                    model="llama3.1:8b",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_query},
                        response["message"],
                        {
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(tool_result)
                        }
                    ]
                )

                print("\nANSWER:\n")
                print(final_response["message"]["content"])
        else:
            # No tool needed
            print(response["message"]["content"])

if __name__ == "__main__":
    asyncio.run(main())


