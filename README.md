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
vLLM is already serving Qwen3-4B on the Akamai GPU at `localhost:8000` (OpenAI-compatible). The app talks to it
with the `openai` client — no extra config needed:
```bash
pip install -r requirements.txt
python agent.py "What time is it in Tokyo and what is 23*47?"   # CLI smoke test
python app.py                                                   # Gradio UI -> http://127.0.0.1:7860
```
Different host/model? `export AKAMAI_BASE_URL=http://<host>:8000/v1 MODEL_NAME=Qwen/Qwen3-4B`.

Tool calling goes through Qwen-Agent's own function-calling template, so vLLM needs no tool-parser flags.
Thinking is switched per request with `chat_template_kwargs.enable_thinking`; `<think>` blocks are parsed whether
or not vLLM runs a reasoning parser. Each answer shows latency, completion tokens and tokens/sec from vLLM.

Sampling follows the Qwen3 model card: thinking temp 0.6 / top_p 0.95, non-thinking temp 0.7 / top_p 0.8, top_k 20.
