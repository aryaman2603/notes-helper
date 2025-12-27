from mcp.server.fastmcp import FastMCP
from rag.retriever import retrieve

mcp = FastMCP("Notes helper MCP")

@mcp.tool()
def search_notes(query: str, k: int = 4) -> list[dict]:
    """Search notes for relevant information."""
    return retrieve(query, k)

if __name__ == "__main__":
    mcp.run()

