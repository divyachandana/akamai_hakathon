"""Live UI for Agent Arena: watch Red vs Blue fight, round by round, with a scoreboard."""
import gradio as gr

from arena import load_rounds, play_round, scoreboard

BLUE_MODES = {"🧠 Think (/think)": "think", "⚡ Fast (/no_think)": "fast", "🚫 Off (no defense)": "off"}


def run_arena(mode_label):
    mode = BLUE_MODES[mode_label]
    rounds = load_rounds()
    results = []
    red = victim = blue = banner = ""
    board = scoreboard(results)
    for i, rnd in enumerate(rounds, 1):
        red = (f"### Round {i}/{len(rounds)} — {rnd['flag']} {rnd['lang']} · `{rnd['type']}`\n\n"
               f"{rnd['text'] or '_(normal memo, no injection)_'}")
        victim, blue, banner = "⏳ reading document…", "", ""
        yield red, victim, blue, banner, board

        r = play_round(rnd, mode)
        victim = f"**Tool calls**\n```\n{r['tool_log']}\n```\n**Draft summary**\n\n{r['draft'][:600]}"
        blue = "🧠 reviewing…" if mode != "off" else "_(defense disabled)_"
        yield red, victim, blue, banner, board

        icon = "⛔ **BLOCK**" if r["verdict"] == "BLOCK" else "✅ **ALLOW**"
        cost = f" · {r['blue_tokens']} tok · {r['blue_latency']:.2f}s" if r["blue_tokens"] else ""
        blue = f"{icon}{cost}\n\n_{r['reason']}_"
        results.append(r)
        board = scoreboard(results)
        if r["success"]:
            banner = f"## 💥 ATTACK SUCCEEDS — canary leaked: `{__import__('arena').CANARY}`"
        else:
            banner = "## 🛡️ DEFENSE WINS — secret stayed safe"
        yield red, victim, blue, banner, board


with gr.Blocks(title="Agent Arena — Red vs Blue") as demo:
    gr.Markdown("# ⚔️ Agent Arena — Red vs Blue on one Qwen3-4B\n"
                "A victim agent summarizes an internal doc using **MCP tools**. Some rounds hide an "
                "injection in the doc. A **Blue defender** (thinking mode) screens the result; a Python "
                "referee scores it. Switch Blue between **Off / Fast / Think** to see what spending more "
                "inference on security buys you — *the agent owns its inference.* Edit `attacks.txt` to add rounds.")
    with gr.Row():
        mode = gr.Dropdown(list(BLUE_MODES), value="🧠 Think (/think)", label="Blue defender mode", scale=2)
        run = gr.Button("⚔️  Run all rounds", variant="primary", scale=1)
    banner = gr.Markdown()
    with gr.Row():
        red = gr.Markdown("### 🔴 Red (attacker)")
        victim = gr.Markdown("### 🤖 Victim (tool agent)")
        blue = gr.Markdown("### 🔵 Blue (defender)")
    gr.Markdown("### 📊 Scoreboard")
    board = gr.Markdown()
    run.click(run_arena, [mode], [red, victim, blue, banner, board])

if __name__ == "__main__":
    demo.launch()
