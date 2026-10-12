# 🧩 One Model, Many Experts

An agent-level **mixture-of-experts router** on **Qwen3-4B**, served by **vLLM on Akamai Cloud**.

A fast router (thinking OFF) reads each message and picks one of six experts. Every expert is the *same* 4B model
on the *same* GPU — just a different system prompt, sampling config and thinking mode. Qwen3-4B itself is a dense
model; the "experts" live at the agent layer, so one small GPU behaves like a team of specialists.

| Expert | Qwen3 capability | Thinking |
|---|---|---|
| 🧠 reasoner | Math / logic / code reasoning | ON |
| 🔧 tool_agent | Tool calling via **MCP** (Qwen-Agent + local MCP server) | OFF |
| ✍️ storyteller | Creative writing | OFF |
| 🎭 character | Role-playing, multi-turn | OFF |
| 🌍 polyglot | 100+ languages, translation | OFF |
| 📐 formatter | Strict instruction following (JSON) | OFF |

**Mode switching:** add `/think` or `/no_think` to any message, or click **Compare thinking ON vs OFF** to run the
same question both ways and see the latency difference — same weights, one flag.
Force an expert with `@character ...`, `@polyglot ...`, etc.

## MCP tools (`mcp_server.py`)
`get_current_time`, `calculate` (safe arithmetic), `roll_dice`, `fetch_url`

## Run
```bash
pip install -r requirements.txt
export AKAMAI_BASE_URL="http://<your-akamai-vllm-host>:8000/v1"
export MODEL_NAME="Qwen/Qwen3-4B"        # whatever GET /v1/models returns
python agent.py "What time is it in Tokyo and what is 23*47?"   # CLI smoke test
python app.py                                                   # Gradio UI
```

vLLM tool calling needs the server started with:
```bash
vllm serve Qwen/Qwen3-4B --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
```
(Older vLLM: `--enable-reasoning --reasoning-parser deepseek_r1`.) The agent also handles `<think>` left inside content.

Sampling follows the Qwen3 model card: thinking temp 0.6 / top_p 0.95, non-thinking temp 0.7 / top_p 0.8, top_k 20.
