"""The Nebius cookbook starter, kept separate from the extended LangGraph app.

Run once with live keys before using the research workspace for a live demo.
Install its optional dependencies only in a separate cookbook environment.
"""

import json
import os
import sys
from pathlib import Path


def main():
    if not os.environ.get("NEBIUS_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        raise SystemExit(
            "Set NEBIUS_API_KEY and TAVILY_API_KEY before running the cookbook"
        )
    try:
        from deepagents import create_deep_agent
        from langchain_nebius import ChatNebius
        from langchain_tavily import TavilySearch
    except ImportError:
        raise SystemExit(
            "Install the optional cookbook requirements in a separate environment"
        ) from None
    model = ChatNebius(
        model=os.environ.get("NEBIUS_MODEL", "MiniMaxAI/MiniMax-M3"),
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url=os.environ.get(
            "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"
        ),
    )
    agent = create_deep_agent(
        model=model, tools=[TavilySearch(max_results=5, search_depth="advanced")]
    )
    question = (
        " ".join(sys.argv[1:])
        or "What is TORAX? Cite its original paper and official documentation."
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 30},
    )
    # Persist public tool decisions and tool outputs so a report alone is not
    # mistaken for evidence that retrieval actually happened. Private model
    # reasoning/additional_kwargs and credentials are deliberately omitted.
    trace = []
    for message in result["messages"]:
        content = message.content
        if isinstance(content, list):
            content = [item for item in content if isinstance(item, dict) and item.get("type") == "text"]
        trace.append({
            "type": message.type,
            "name": getattr(message, "name", None),
            "content": content,
            "tool_calls": getattr(message, "tool_calls", []),
            "usage": getattr(message, "usage_metadata", None),
        })
    trace_path = Path(__file__).resolve().parents[1] / ".dashboard" / "provider-verification" / "cookbook_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Deep research report:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
