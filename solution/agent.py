import json
import sys
from pathlib import Path

from .asr import ASR
from .llm import LLM


SYSTEM = """You are a tool-using voice assistant agent.
You receive a Vietnamese user request, user context, conversation history, and available tools.
You must output ONLY a JSON array of tool calls to execute next.
Each item: {"name": "<app.tool>", "arguments": {<json args>}}.
Use only tools from the provided catalog. Read state before writes when IDs are unknown.
If information is missing, call sys.ask_missing_info with {"tool_name": "...", "params": ["..."]}.
If no installed app can handle the request, return [].
Do not include explanations or markdown. Output JSON only."""


def load_tools(split, root):
    import json as _json
    root = Path(root)
    catalog = _json.load(open(root / split / f"{split}_tools_catalog.json", encoding="utf-8"))
    tools = {}
    for app in catalog["apps"]:
        for t in app["tools"]:
            tools[t["name"]] = t
    return catalog, tools


def tool_schema_text(installed_apps, catalog_tools):
    out = []
    for app_id, info in installed_apps.items():
        for tool_name in info.get("tools", []):
            spec = catalog_tools.get(tool_name)
            if spec:
                out.append(json.dumps(spec, ensure_ascii=False))
    # sys tool
    if "sys.ask_missing_info" not in catalog_tools:
        pass
    return "\n".join(out)


def transcribe_history(item, asr, root):
    root = Path(root)
    split = None
    # item doesn't carry split; caller passes via closure? We'll handle paths relative to root/split externally.
    return []


def run_sample(item, split, root, asr, llm, use_gold_text=False):
    root = Path(root)
    sys.path.insert(0, str(root / split))
    from zaic_tools import ToolEnv

    env = ToolEnv(split, item["id"], root=root)

    catalog, catalog_tools = load_tools(split, root)
    installed = item["context"].get("installed_apps", {})
    schemas = tool_schema_text(installed, catalog_tools)

    # transcribe user turns
    if use_gold_text:
        latest_text = item.get("audio_text", "")
    else:
        latest_text = asr.transcribe(root / split / item["audio"])

    history_transcripts = []
    for h in item.get("history", []):
        if h.get("role") == "user":
            if use_gold_text:
                history_transcripts.append(h.get("audio_text", ""))
            else:
                history_transcripts.append(asr.transcribe(root / split / h["audio"]))
        else:
            history_transcripts.append(None)

    messages = [{"role": "system", "content": SYSTEM}]
    context = {
        "current_time": item["context"].get("current_time"),
        "timezone": item["context"].get("timezone"),
        "location": item["context"].get("location"),
        "user_profile": item["context"].get("user_profile"),
        "installed_apps": list(installed.keys()),
    }
    messages.append({
        "role": "user",
        "content": "Context:\n" + json.dumps(context, ensure_ascii=False) +
        "\n\nAvailable tool schemas:\n" + schemas,
    })

    # replay history
    for idx, h in enumerate(item.get("history", [])):
        if h.get("role") == "user":
            txt = history_transcripts[idx] if idx < len(history_transcripts) else ""
            messages.append({"role": "user", "content": f"[User]: {txt}"})
        else:
            parts = []
            if h.get("text"):
                parts.append(f"Assistant: {h['text']}")
            if h.get("tool_calls"):
                parts.append("Tool calls: " + json.dumps(h["tool_calls"], ensure_ascii=False))
            if h.get("tool_results"):
                parts.append("Tool results: " + json.dumps(h["tool_results"], ensure_ascii=False))
            messages.append({"role": "assistant", "content": "\n".join(parts)})

    messages.append({"role": "user", "content": f"Latest user request: {latest_text}\nReturn next tool calls as JSON array, or [] if done."})

    for _ in range(8):
        raw = llm.generate(messages, max_new_tokens=512)
        calls = LLM.parse_calls(raw)
        if not calls:
            break
        results = []
        for c in calls:
            r = env.call(c["name"], c["arguments"])
            results.append(r)
        messages.append({"role": "assistant", "content": "Tool calls: " + json.dumps(calls, ensure_ascii=False)})
        messages.append({"role": "user", "content": "Tool results: " + json.dumps(results, ensure_ascii=False) + "\nReturn next tool calls as JSON array, or [] if done."})

    return env.get_tool_calls(), env.get_state_hash()
