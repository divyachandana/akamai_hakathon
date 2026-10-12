"""Live UI for Agent Arena: watch Red (left) vs Blue (right) fight, streaming token by token."""
import gradio as gr

from agent import LAST_USAGE, chat_stream
from arena import (REFUSAL, blue_messages, fetch_doc, find_leaks, load_rounds, parse_verdict,
                   referee, scoreboard, victim_messages)

BLUE_MODES = {"🧠 Think (/think)": "think", "⚡ Fast (/no_think)": "fast", "🚫 Off (no defense)": "off"}


def run_arena(mode_label):
    mode = BLUE_MODES[mode_label]
    rounds = load_rounds()
    results = []
    board = scoreboard(results)
    red = victim = blue = banner = ""

    for i, rnd in enumerate(rounds, 1):
        tag = "💉 injection hidden in document" if rnd["type"] == "ATTACK" else "📄 normal document"
        red = (f"### 🔴 Red — Round {i}/{len(rounds)}\n{rnd['flag']} **{rnd['lang']}** · `{rnd['type']}`\n\n"
               f"{tag}\n\n> {rnd['text'] or '(no injection — a plain memo)'}")
        victim, blue, banner = "🤖 **Victim** — ⏳ fetching document via MCP…", "🔵 **Blue** — waiting…", ""
        yield red, victim, blue, banner, board

        # --- Victim: fetch via MCP, then think + stream its summary (watch it get tempted) ---
        doc = fetch_doc(rnd["text"])
        log = f"📤 fetch_document('q3-memo')\n📥 {doc.strip()[:150]}…"
        draft, vthink = "", ""
        for kind, txt in chat_stream(victim_messages(doc), think=True, max_tokens=600):
            if kind == "thinking":
                vthink = txt
                victim = f"🤖 **Victim**\n```\n{log}\n```\n🧠 _thinking…_\n\n{txt[-400:]}"
            else:
                draft = txt
                victim = f"🤖 **Victim**\n```\n{log}\n```\n**Summary** _(streaming…)_\n\n{txt}"
            yield red, victim, blue, banner, board
        think_block = f"<details><summary>🧠 victim's reasoning</summary>\n\n{vthink}\n\n</details>\n\n" if vthink else ""
        victim = f"🤖 **Victim**\n```\n{log}\n```\n{think_block}**Summary**\n\n{draft}"

        # --- Blue: stream its review, then verdict ---
        if mode == "off":
            verdict, reason, btok, blat = "ALLOW", "defense disabled", 0, 0.0
            blue = "🔵 **Blue** — 🚫 _defense disabled_"
            yield red, victim, blue, banner, board
        else:
            braw = ""
            for kind, txt in chat_stream(blue_messages(doc, draft), think=(mode == "think"), max_tokens=512):
                if kind == "thinking":
                    blue = f"🔵 **Blue** — 🧠 _analyzing…_\n\n{txt[-500:]}"
                else:
                    braw = txt
                    blue = f"🔵 **Blue** — writing verdict…\n\n`{txt[:200]}`"
                yield red, victim, blue, banner, board
            verdict, reason = parse_verdict(braw)
            btok, blat = LAST_USAGE["tokens"], LAST_USAGE["latency"]

        delivered = REFUSAL if verdict == "BLOCK" else draft
        success = referee(delivered)
        results.append({"lang": rnd["lang"], "flag": rnd["flag"], "type": rnd["type"],
                        "success": success, "blocked": verdict == "BLOCK",
                        "blue_latency": blat, "blue_tokens": btok})
        board = scoreboard(results)
        icon = "⛔ **BLOCK**" if verdict == "BLOCK" else "✅ **ALLOW**"
        cost = f"  ·  {btok} tok · {blat:.2f}s" if btok else ""
        blue = f"🔵 **Blue**\n\n{icon}{cost}\n\n_{reason}_"
        banner = (f"## 💥 ATTACK SUCCEEDS — leaked: `{', '.join(find_leaks(delivered))}`" if success
                  else "## 🛡️ DEFENSE WINS — secrets stayed safe")
        yield red, victim, blue, banner, board


with gr.Blocks(title="Agent Arena — Red vs Blue", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚔️ Agent Arena — Red vs Blue on one Qwen3-4B\n"
                "A victim agent summarizes an internal doc fetched through an **MCP tool**. Some rounds hide "
                "a prompt injection in the doc. **Blue** (thinking mode) screens the result and a Python "
                "referee scores it. Flip Blue between **Off / Fast / Think** to see what spending inference on "
                "defense buys you — *the agent owns its inference.*  Add rounds by editing `attacks.txt`.")
    with gr.Row():
        mode = gr.Dropdown(list(BLUE_MODES), value="🧠 Think (/think)", label="Blue defender mode", scale=2)
        run = gr.Button("⚔️  Run all rounds", variant="primary", scale=1)
    banner = gr.Markdown()
    with gr.Row(equal_height=True):
        red = gr.Markdown("### 🔴 Red (attacker)", elem_id="red")
        blue = gr.Markdown("### 🔵 Blue (defender)", elem_id="blue")
    victim = gr.Markdown("### 🤖 Victim (tool agent)")
    gr.Markdown("### 📊 Scoreboard")
    board = gr.Markdown()
    run.click(run_arena, [mode], [red, victim, blue, banner, board])

if __name__ == "__main__":
    demo.launch()
