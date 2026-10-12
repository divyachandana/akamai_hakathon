"""Gradio UI for the One Model, Many Experts agent."""
import gradio as gr

from agent import EXPERTS, compare_modes, handle

EXAMPLES = [
    "A bat and a ball cost $1.10 in total. The bat costs $1 more than the ball. How much is the ball?",
    "What time is it in Tokyo right now, and what is 1234 * 5678?",
    "Roll two 20-sided dice for my attack.",
    "Write a 6-line poem about GPUs humming in the San Francisco fog.",
    "@character You are a grumpy pirate captain. I'm the new crew member — what's my first job?",
    "Translate 'Agents that own their inference' into Spanish, Hindi, Japanese and French.",
    "¿Puedes explicarme qué es la inferencia de un modelo en dos frases?",
    'Extract as JSON with keys event, date, venue: "HackerSquad x Akamai build day, Oct 12 2026, AWS Builder Loft".',
]


def respond(text, history, transcript):
    if not text or not text.strip():
        return gr.update(), history, transcript, transcript, gr.update(), gr.update(), gr.update()
    try:
        r = handle(text, history)
    except Exception as e:
        err = f"⚠️ Error: `{e}`"
        return "", history, transcript, transcript, err, "", ""
    e = EXPERTS[r["expert"]]
    history = history + [{"role": "user", "content": r["text"]},
                         {"role": "assistant", "content": r["answer"]}]
    transcript = (transcript + f"\n\n**🧑 You:** {text}\n\n**{e['emoji']} {r['expert']}:** {r['answer']}\n\n---")
    mode = "🧠 thinking **ON**" if r["think"] else "⚡ thinking **OFF**"
    badge = (f"## {e['emoji']} {r['expert']}\n**Skill:** {e['skill']}  \n{mode} · ⏱️ **{r['latency']:.2f}s**"
             + (f" · 🔢 {r['tokens']} tokens · ⚡ {r['tok_s']:.0f} tok/s" if r["tokens"] else "") +
             f"  \n_Router: {r['reason']}_")
    thinking = r["thinking"] or "_(no thinking — fast mode)_"
    tools = r["tool_log"] or "_(no tool calls)_"
    return "", history, transcript, transcript, badge, thinking, tools


def compare(text):
    if not text or not text.strip():
        return "Type a question in the box first."
    out = compare_modes(text)
    t_on, a_on, l_on, k_on = out[True]
    _, a_off, l_off, k_off = out[False]
    return (f"### 🧠 Thinking ON — {l_on:.2f}s · {k_on} tokens ({len(t_on.split())} thinking words)\n{a_on}\n\n"
            f"### ⚡ Thinking OFF — {l_off:.2f}s · {k_off} tokens\n{a_off}\n\n"
            f"**Same model, same weights — one flag switched. Speedup: {l_on / max(l_off, 0.01):.1f}x**")


def reset():
    return [], "", "", "", "", "", ""


with gr.Blocks(title="Qwen3 Expert Router") as demo:
    gr.Markdown("# 🧩 One Model, Many Experts\n"
                "An agent-level mixture-of-experts router on **Qwen3-4B**, served by **vLLM on Akamai Cloud**. "
                "A fast router picks an expert per message; each expert is the same 4B model with its own "
                "prompt, sampling and thinking mode. Prefix `@expert` to force one, or add `/think` / `/no_think`.")
    history = gr.State([])
    transcript_state = gr.State("")
    with gr.Row():
        with gr.Column(scale=3):
            transcript = gr.Markdown()
            inp = gr.Textbox(label="Message", placeholder="Ask anything…", lines=2)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                cmp_btn = gr.Button("⚖️ Compare thinking ON vs OFF")
                clear = gr.Button("Reset")
            gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column(scale=2):
            badge = gr.Markdown("## Waiting for a message…")
            with gr.Accordion("🧠 Thinking trace", open=False):
                thinking = gr.Markdown()
            with gr.Accordion("🔧 MCP tool calls", open=True):
                tools = gr.Markdown()
            cmp_out = gr.Markdown()
            gr.Markdown("**Experts:** " + " · ".join(f"{v['emoji']} {k}" for k, v in EXPERTS.items()))

    outs = [inp, history, transcript_state, transcript, badge, thinking, tools]
    inp.submit(respond, [inp, history, transcript_state], outs)
    send.click(respond, [inp, history, transcript_state], outs)
    cmp_btn.click(compare, [inp], [cmp_out])
    clear.click(reset, None, [history, transcript_state, transcript, badge, thinking, tools, cmp_out])

if __name__ == "__main__":
    demo.launch()
