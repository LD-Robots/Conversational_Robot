# src/tools/websearch.py
"""Web search tool using DuckDuckGo."""

from typing import Optional
import logging

log = logging.getLogger(__name__)

# Tool definition for Groq function calling
WEB_SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current, real-time information. "
            "Use this when the user asks about recent news, current events, "
            "live prices, weather, sports scores, or anything that requires up-to-date information. "
            "Do NOT use for general knowledge questions that don't need current data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query in English (translate if needed for better results)"
                }
            },
            "required": ["query"]
        }
    }
}


def web_search(query: str, max_results: int = 3, region: str = "wt-wt") -> str:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        region: Region for search (wt-wt = worldwide)
    
    Returns:
        Formatted string with search results for LLM consumption
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        log.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return "Web search is not available. Please install duckduckgo-search package."
    
    log.info(f"🔍 Web search: \"{query}\"")
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, region=region))
        
        if not results:
            log.info("📊 Search results: 0 items")
            return f"No search results found for: {query}"
        
        log.info(f"📊 Search results: {len(results)} items")
        
        # Format results for LLM
        formatted = [f"Web search results for \"{query}\":\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", "No description")
            url = r.get("href", "")
            formatted.append(f"{i}. {title}\n   {body}\n   Source: {url}\n")
        
        return "\n".join(formatted)
    
    except Exception as e:
        log.error(f"Web search error: {e}")
        return f"Web search failed: {str(e)}"


# Registry of available tools - maps function name to callable
TOOLS_REGISTRY = {
    "web_search": web_search
}


def execute_tool(name: str, arguments: dict, config: Optional[dict] = None) -> str:
    """
    Execute a tool by name with given arguments.
    
    Args:
        name: Tool function name
        arguments: Dict of arguments for the tool
        config: Optional config (e.g., max_results from llm.yaml)
    
    Returns:
        Tool result as string
    """
    if name not in TOOLS_REGISTRY:
        return f"Unknown tool: {name}"
    
    func = TOOLS_REGISTRY[name]
    
    # Inject config values if applicable
    if name == "web_search" and config:
        max_results = config.get("websearch_max_results", 3)
        arguments["max_results"] = max_results
    
    try:
        return func(**arguments)
    except Exception as e:
        log.error(f"Tool execution error ({name}): {e}")
        return f"Tool error: {str(e)}"
