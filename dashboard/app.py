"""
dashboard/app.py — GemmaRoute Premium Observability Dashboard

Real-time visualization of the routing engine's performance:
  - Live cost savings counter (vs always-complex routing)
  - Routing distribution donut chart
  - Latency and quality metrics
  - Per-request routing log with model and tier details
  - Sidebar demo prompt panel for judge testing

Auto-refreshes every 5 seconds using streamlit-autorefresh.
"""
import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GemmaRoute — AI Routing Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Color palette ─────────────────────────────────────────────────────────────
TIER_COLORS = {
    "trivial": "#64748B",
    "simple":  "#10B981",
    "medium":  "#3B82F6",
    "complex": "#F59E0B",
}
TIER_EMOJI = {
    "trivial": "⬜",
    "simple":  "🟢",
    "medium":  "🔵",
    "complex": "🟡",
}

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero title gradient */
.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #8B5CF6 0%, #E8001C 55%, #F59E0B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.15rem;
}
.hero-sub {
    color: #94A3B8;
    font-size: 0.95rem;
    margin-top: 0;
    margin-bottom: 0.5rem;
}

/* Sponsor badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-amd   { background: rgba(232,0,28,0.12);  color: #E8001C; border: 1px solid rgba(232,0,28,0.3); }
.badge-gemma { background: rgba(139,92,246,0.12); color: #A78BFA; border: 1px solid rgba(139,92,246,0.3); }
.badge-fw    { background: rgba(59,130,246,0.12); color: #60A5FA; border: 1px solid rgba(59,130,246,0.3); }

/* Metric card polish */
div[data-testid="stMetric"] {
    background: #151929 !important;
    border: 1px solid #1E2A3A !important;
    border-radius: 12px !important;
    padding: 1rem 1.5rem !important;
}
div[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}

/* Section divider */
.section-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748B;
    margin-bottom: 0.4rem;
    margin-top: 0.2rem;
}

/* Sidebar prompt codes */
.stCode { border-radius: 6px !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Auto-refresh every 5 s ────────────────────────────────────────────────────
st_autorefresh(interval=5_000, key="dash_refresh")


# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=4)
def fetch_stats() -> dict | None:
    try:
        r = httpx.get(f"{BACKEND_URL}/stats", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=4)
def fetch_health() -> dict | None:
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
col_title, col_live = st.columns([5, 1])

with col_title:
    st.markdown('<p class="hero-title">⚡ GemmaRoute</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">3-Layer AMD-Native AI Routing Engine — '
        "cut your LLM bill by 60–80% without dropping quality</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="badge badge-amd">🔴 AMD ROCm</span>'
        '<span class="badge badge-gemma">🟣 Gemma 4</span>'
        '<span class="badge badge-fw">🔵 Fireworks AI</span>',
        unsafe_allow_html=True,
    )

stats = fetch_stats()

with col_live:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    if stats is not None:
        st.success("● Live", icon=None)
    else:
        st.error("● Offline")

st.markdown("---")

# ── Circuit Breaker Status Banner ──────────────────────────────────────────────
heath_data = fetch_health()
if heath_data:
    cb = heath_data.get("fireworks_circuit", {})
    cb_state    = cb.get("state", "UNKNOWN")
    cb_failures = cb.get("failure_count", 0)
    cb_thresh   = cb.get("failure_threshold", 3)
    cb_reset    = cb.get("seconds_until_reset", 0)

    if cb_state == "OPEN":
        st.error(
            f"### 🔴 Circuit Breaker OPEN — Fireworks AI Fallback Active\n"
            f"After {cb_thresh} consecutive cloud API failures, all traffic is being routed "
            f"to **local AMD Ollama** automatically. "
            f"Look for `[CB_FALLBACK]` tags in the routing log below.  \n"
            f"Auto-reset in **{cb_reset:.0f}s**."
        )
    elif cb_failures > 0:
        st.warning(
            f"⚠️ **Circuit Breaker WARNING** — Fireworks AI has had "
            f"{cb_failures}/{cb_thresh} consecutive failures. "
            "Will trip OPEN and fall back to Ollama on next failure."
        )
    else:
        st.success(
            "🟢 **All systems nominal** — Fireworks AI circuit CLOSED. "
            "Local AMD Ollama and cloud routing both healthy."
        )

# ── Offline guard ─────────────────────────────────────────────────────────────
if stats is None:
    st.warning(
        "⏳ **Waiting for backend...** Start the full stack with `docker compose up --build`  \n"
        "Then try: `curl http://localhost:8000/health`"
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("📊 Total Requests", value=stats.get("total_requests", 0))

with k2:
    avg_lat = stats.get("avg_latency_ms", 0.0)
    st.metric("⚡ Avg Latency", value=f"{avg_lat:.0f} ms")

with k3:
    total_cost = stats.get("total_cost_usd", 0.0)
    st.metric("💸 Total API Cost", value=f"${total_cost:.4f}")

with k4:
    saved = stats.get("total_saved_vs_always_complex_usd", 0.0)
    st.metric(
        "💰 Total Saved",
        value=f"${saved:.4f}",
        delta="vs always-complex routing",
    )

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS ROW 1 — Distribution + Cost Savings
# ══════════════════════════════════════════════════════════════════════════════
chart_l, chart_r = st.columns(2)

# ── Routing distribution donut ────────────────────────────────────────────────
with chart_l:
    st.markdown('<p class="section-label">🔀 Routing Distribution</p>', unsafe_allow_html=True)
    dist = stats.get("routing_distribution", {})
    if dist:
        labels = list(dist.keys())
        values = list(dist.values())
        colors = [TIER_COLORS.get(lbl, "#8B5CF6") for lbl in labels]

        fig_donut = go.Figure(
            go.Pie(
                labels=[lbl.capitalize() for lbl in labels],
                values=values,
                hole=0.58,
                marker=dict(colors=colors, line=dict(color="#0B0D17", width=2)),
                textinfo="label+percent",
                textfont=dict(color="#F1F5F9", size=13),
                hovertemplate="<b>%{label}</b><br>%{value} requests (%{percent})<extra></extra>",
            )
        )
        fig_donut.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(color="#94A3B8"),
            ),
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("No routing data yet — send some requests! 👉")

# ── Cost savings bar ──────────────────────────────────────────────────────────
with chart_r:
    st.markdown('<p class="section-label">💡 Cost: Actual vs Always-Complex Baseline</p>', unsafe_allow_html=True)

    total_cost   = stats.get("total_cost_usd", 0.0)
    total_saved  = stats.get("total_saved_vs_always_complex_usd", 0.0)
    baseline     = total_cost + total_saved   # what it would cost without routing
    savings_pct  = (total_saved / baseline * 100) if baseline > 0 else 0.0

    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=["Baseline<br>(Always Complex)"],
            y=[baseline],
            name="Without GemmaRoute",
            marker_color="#EF4444",
            text=[f"${baseline:.4f}"],
            textposition="outside",
            textfont=dict(color="#F1F5F9", size=13),
        )
    )
    fig_bar.add_trace(
        go.Bar(
            x=["Actual<br>(GemmaRoute)"],
            y=[total_cost],
            name="With GemmaRoute",
            marker_color="#10B981",
            text=[f"${total_cost:.4f}"],
            textposition="outside",
            textfont=dict(color="#F1F5F9", size=13),
        )
    )
    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20, l=10, r=10),
        height=280,
        showlegend=False,
        yaxis=dict(title="Cost (USD)", gridcolor="#1E2A3A", tickformat=".4f"),
        bargap=0.35,
        annotations=[
            dict(
                text=f"💰 <b>{savings_pct:.1f}%</b> Saved",
                xref="paper", yref="paper",
                x=0.5, y=1.1,
                showarrow=False,
                font=dict(color="#10B981", size=15, family="Inter"),
            )
        ],
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS ROW 2 — Quality gauge + Model breakdown
# ══════════════════════════════════════════════════════════════════════════════
q_col, m_col = st.columns(2)

with q_col:
    st.markdown('<p class="section-label">🎯 Average Quality Score (LLM-as-Judge)</p>', unsafe_allow_html=True)
    avg_q = stats.get("avg_quality_score", 0.0)

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_q,
            number=dict(font=dict(color="#F1F5F9", size=32), valueformat=".2f"),
            gauge=dict(
                axis=dict(
                    range=[0, 1],
                    tickcolor="#64748B",
                    tickfont=dict(color="#64748B"),
                    dtick=0.25,
                ),
                bar=dict(color="#8B5CF6", thickness=0.25),
                bgcolor="#151929",
                borderwidth=0,
                steps=[
                    dict(range=[0.00, 0.50], color="#1A1F2E"),
                    dict(range=[0.50, 0.75], color="#1A2535"),
                    dict(range=[0.75, 1.00], color="#162435"),
                ],
                threshold=dict(
                    line=dict(color="#10B981", width=3),
                    thickness=0.85,
                    value=0.75,
                ),
            ),
        )
    )
    fig_gauge.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(t=20, b=20, l=30, r=30),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    esc_rate = stats.get("escalation_rate", 0.0)
    col_thr, col_esc = st.columns(2)
    col_thr.caption(f"Quality threshold: **0.75**")
    col_esc.caption(f"Escalation rate: **{esc_rate:.1%}**")

with m_col:
    st.markdown('<p class="section-label">🤖 Model Usage (last 20 requests)</p>', unsafe_allow_html=True)
    recent = stats.get("recent_logs", [])
    if recent:
        model_counts: dict[str, int] = {}
        for log in recent:
            raw = log.get("model_used") or "unknown"
            short = raw.split("/")[-1] if "/" in raw else raw
            model_counts[short] = model_counts.get(short, 0) + 1

        model_df = (
            pd.DataFrame(
                {"Model": list(model_counts.keys()), "Requests": list(model_counts.values())}
            )
            .sort_values("Requests", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            model_df,
            use_container_width=True,
            hide_index=True,
            height=220,
            column_config={
                "Model":    st.column_config.TextColumn("Model", width="large"),
                "Requests": st.column_config.ProgressColumn(
                    "Requests",
                    format="%d",
                    min_value=0,
                    max_value=20,
                ),
            },
        )
    else:
        st.info("No model data yet.")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE ROUTING LOG
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<p class="section-label">📋 Live Routing Log — last 20 requests</p>', unsafe_allow_html=True)

if recent := stats.get("recent_logs"):
    df = pd.DataFrame(recent)

    df["Tier"] = df["final_tier"].map(
        lambda t: f"{TIER_EMOJI.get(t or '', '⚪')} {(t or '-').capitalize()}"
    )
    df["Latency"]   = df["latency_ms"].map(lambda x: f"{x:.0f} ms")
    df["Cost"]      = df["cost_usd"].map(lambda x: "$0.00" if x == 0 else f"${x:.6f}")
    df["Quality"]   = df["quality_score"].map(lambda x: f"{x:.0%}")
    df["Escalated"] = df["escalations"].map(lambda e: "⬆️ Yes" if e > 0 else "—")
    df["Model"]     = df["model_used"].map(
        lambda m: (m.split("/")[-1] if m and "/" in m else (m or "—"))
    )

    display_df = df[
        ["Tier", "prompt_preview", "Model", "Latency", "Quality", "Cost", "Escalated"]
    ].rename(columns={"prompt_preview": "Prompt"})

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=380,
        column_config={
            "Tier":      st.column_config.TextColumn("Tier",    width="small"),
            "Prompt":    st.column_config.TextColumn("Prompt",  width="large"),
            "Model":     st.column_config.TextColumn("Model",   width="medium"),
            "Latency":   st.column_config.TextColumn("Latency", width="small"),
            "Quality":   st.column_config.TextColumn("Quality", width="small"),
            "Cost":      st.column_config.TextColumn("Cost",    width="small"),
            "Escalated": st.column_config.TextColumn("⬆️",      width="small"),
        },
    )
else:
    st.info("No routing logs yet — start sending prompts via the sidebar! →")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Demo Prompt Panel
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎮 Demo Prompt Panel")
    st.markdown("*Click a prompt → copy → paste into curl or Swagger UI*")
    st.markdown("---")

    # Tier legend
    for tier, color in TIER_COLORS.items():
        emoji = TIER_EMOJI[tier]
        st.markdown(
            f'<span style="color:{color};">{emoji} **{tier.capitalize()}**</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown(f'<span style="color:{TIER_COLORS["trivial"]};">**⬜ Trivial** — heuristic, 0ms, free</span>', unsafe_allow_html=True)
    for p in [
        "What are your business hours?",
        "Hi there!",
        "Thanks for your help!",
    ]:
        st.code(p, language=None)

    st.markdown("---")
    st.markdown(f'<span style="color:{TIER_COLORS["simple"]};">**🟢 Simple** — Gemma 4B local, free</span>', unsafe_allow_html=True)
    for p in [
        "How do I track my order?",
        "Can I change my delivery address?",
        "What payment methods do you accept?",
        "Is gift wrapping available?",
    ]:
        st.code(p, language=None)

    st.markdown("---")
    st.markdown(f'<span style="color:{TIER_COLORS["medium"]};">**🔵 Medium** — Gemma 12B cloud</span>', unsafe_allow_html=True)
    for p in [
        "I want to initiate a return for order #8821.",
        "My account is locked and I can't reset my password.",
        "Can you explain your refund timeline and process?",
        "I was charged twice for the same order.",
    ]:
        st.code(p, language=None)

    st.markdown("---")
    st.markdown(f'<span style="color:{TIER_COLORS["complex"]};">**🟡 Complex** — Gemma 31B cloud</span>', unsafe_allow_html=True)
    for p in [
        "I need a full refund analysis and policy review for dispute case #4821 involving three separate orders.",
        "I believe I was incorrectly charged a restocking fee. I'd like to formally dispute this and understand my legal options.",
        "I'm a business customer with 50 accounts. Please provide your enterprise SLA and compliance documentation.",
    ]:
        st.code(p, language=None)

    st.markdown("---")
    st.markdown("**📡 Quick curl test:**")
    st.code(
        'curl -s -X POST http://localhost:8000/route \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"prompt": "YOUR PROMPT HERE"}\' | python -m json.tool',
        language="bash",
    )
    st.markdown("**📖 API Docs:** [localhost:8000/docs](http://localhost:8000/docs)")
