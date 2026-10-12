"""One Model, Many Experts: an agent-level mixture-of-experts router on Qwen3-4B (Akamai Cloud vLLM)."""
import json
import os
import re
import sys
import time

from openai import OpenAI

BASE_URL = os.getenv("AKAMAI_BASE_URL", "http://localhost:8000/v1")  # vLLM on Akamai GPU (OpenAI-compatible)
API_KEY = os.getenv("AKAMAI_API_KEY", "EMPTY")
MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B")  # must match the name vLLM serves (GET /v1/models)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

EXPERTS = {
    "reasoner": {
        "emoji": "🧠", "think": True, "skill": "Reasoning (thinking mode)",
        "system": "You are a careful reasoning expert for math, logic and code. "
                  "Work it out, then give the final answer on its own line starting with 'Answer:'.",
    },
    "tool_agent": {
        "emoji": "🔧", "think": False, "skill": "Tool calling via MCP",
        "system": "You are a tool-using agent. Always use the available tools for time, arithmetic, "
                  "dice rolls and fetching URLs instead of guessing. Then answer briefly.",
    },
    "storyteller": {
        "emoji": "✍️", "think": False, "skill": "Creative writing",
        "system": "You are an award-winning creative writer. Write vivid, original pieces with "
                  "strong imagery. Respect any requested form, length or style exactly.",
    },
    "character": {
        "emoji": "🎭", "think": False, "skill": "Role-playing",
        "system": "You are a role-play actor. Fully embody the persona the user sets up (or invent a "
                  "fitting one), stay in character across turns, and keep replies immersive and short.",
    },
    "polyglot": {
        "emoji": "🌍", "think": False, "skill": "Multilingual (100+ languages)",
        "system": "You are a multilingual expert. Translate accurately and naturally, or reply in the "
                  "language the user writes in or asks for. For translations, label each language.",
    },
    "formatter": {
        "emoji": "📐", "think": False, "skill": "Strict instruction following",
        "system": "You follow formatting instructions exactly. When asked for JSON, output only valid "
                  "JSON with no extra text or code fences. Never add fields that were not requested.",
    },
}

ROUTER_PROMPT = """You are a router. Choose exactly one expert for the user's message.
- reasoner: math, logic puzzles, code, anything needing careful multi-step reasoning
- tool_agent: needs live data or actions: current time/date, exact calculation, dice roll, fetching a URL
- storyteller: poems, stories, slogans, creative writing
- character: role-play, "pretend you are", "act as", talking as a persona or continuing a role-play
- polyglot: translation, or a message written in / asking for a non-English language
- formatter: extract or convert information into strict JSON / lists / tables with exact rules
Reply with JSON only: {"expert": "<name>", "reason": "<max 12 words>"}"""


def split_thinking(msg):
    """Return (thinking, answer) whether vLLM uses a reasoning parser or leaves <think> in content."""
    thinking = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    content = msg.content or ""
    if "</think>" in content:
        t, content = content.split("</think>", 1)
        thinking = thinking or t.replace("<think>", "")
    return thinking.strip(), content.strip()


LAST_USAGE = {"tokens": 0, "latency": 0.0}


def chat(messages, think, max_tokens=2048):
    """One call to Qwen3-4B, using the model card's sampling settings for each mode."""
    sampling = {"temperature": 0.6, "top_p": 0.95} if think else {"temperature": 0.7, "top_p": 0.8}
    start = time.time()
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=max_tokens, **sampling,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": think}},
    )
    thinking, answer = split_thinking(resp.choices[0].message)
    latency = time.time() - start
    LAST_USAGE.update(tokens=getattr(resp.usage, "completion_tokens", 0) or 0, latency=latency)
    return thinking, answer, latency


def route(text):
    _, raw, _ = chat([{"role": "system", "content": ROUTER_PROMPT},
                      {"role": "user", "content": text}], think=False, max_tokens=80)
    try:
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception:
        data = {}
    expert = data.get("expert", "")
    if expert not in EXPERTS:
        return "storyteller", f"router fallback (got: {raw[:40]!r})"
    return expert, data.get("reason", "")


_tool_bot = None


def _get_tool_bot():
    global _tool_bot
    if _tool_bot is None:
        from qwen_agent.agents import Assistant
        llm_cfg = {
            "model": MODEL, "model_server": BASE_URL, "api_key": API_KEY,
            "generate_cfg": {
                "temperature": 0.7, "top_p": 0.8,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        }
        server = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
        tools = [{"mcpServers": {"hackathon_tools": {"command": sys.executable, "args": [server]}}}]
        _tool_bot = Assistant(llm=llm_cfg, function_list=tools, system_message=EXPERTS["tool_agent"]["system"])
    return _tool_bot


def run_tool_agent(messages):
    start = time.time()
    responses = []
    for responses in _get_tool_bot().run(messages=messages):
        pass
    log, answer = [], ""
    for m in responses:
        if m.get("function_call"):
            fc = m["function_call"]
            log.append(f"📤 **call** `{fc.get('name')}({fc.get('arguments')})`")
        elif m.get("role") == "function":
            log.append(f"📥 **{m.get('name')}** → `{str(m.get('content'))[:300]}`")
        elif m.get("role") == "assistant" and m.get("content"):
            answer = m["content"] if isinstance(m["content"], str) else str(m["content"])
    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1]
    return "\n\n".join(log), answer.strip(), time.time() - start


def handle(text, history=None):
    """Route a message to an expert and run it. history = prior [{'role','content'}] final answers only."""
    history = list(history or [])
    text = text.strip()

    # Manual override: "@character ..." forces an expert (handy for the demo)
    forced = re.match(r"@(\w+)\s+", text)
    if forced and forced.group(1) in EXPERTS:
        expert, reason, text = forced.group(1), "forced with @", text[forced.end():]
    else:
        expert, reason = route(text)

    # Thinking switch: /think or /no_think overrides the expert's default mode
    think = EXPERTS[expert]["think"]
    if "/no_think" in text:
        think, text = False, text.replace("/no_think", "").strip()
    elif "/think" in text:
        think, text = True, text.replace("/think", "").strip()

    turns = history[-8:] + [{"role": "user", "content": text}]  # no thinking content in history (model card)
    if expert == "tool_agent":
        tool_log, answer, latency = run_tool_agent(turns)
        thinking = ""
    else:
        tool_log = ""
        messages = [{"role": "system", "content": EXPERTS[expert]["system"]}] + turns
        thinking, answer, latency = chat(messages, think)
    tokens = LAST_USAGE["tokens"] if expert != "tool_agent" else 0

    return {"tokens": tokens, "tok_s": tokens / latency if latency and tokens else 0,"expert": expert, "reason": reason, "think": think, "thinking": thinking,
            "answer": answer, "latency": latency, "tool_log": tool_log, "text": text}


def compare_modes(text):
    """Same question, thinking ON vs OFF — shows Qwen3's mode switching within one model."""
    out = {}
    for think in (True, False):
        msgs = [{"role": "system", "content": EXPERTS["reasoner"]["system"]},
                {"role": "user", "content": text}]
        thinking, answer, latency = chat(msgs, think)
        out[think] = (thinking, answer, latency, LAST_USAGE["tokens"])
    return out


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What time is it in Tokyo, and what is 23*47?"
    r = handle(q)
    print(f"server: {BASE_URL}  model: {MODEL}")
    print(f"[{r['expert']}] think={r['think']} {r['latency']:.2f}s {r['tokens']} tok — {r['reason']}")
    if r["tool_log"]:
        print(r["tool_log"])
    print(r["answer"])
