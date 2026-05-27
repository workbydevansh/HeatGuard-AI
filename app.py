from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.advisory import generate_all_audience_advisories
from src.competition import (
    competition_file_status,
    find_competition_dir,
    load_context_and_test,
    make_location_forecast,
)
from src.config import ASSETS_DIR, AUDIENCES, MODEL_PATH, RISK_LEVELS
from src.explainability import explain_prediction, get_feature_importance
from src.model import load_model_artifact
from src.utils import classify_wbt, confidence_from_rmse, reliability_label, risk_color


load_dotenv()

st.set_page_config(
    page_title="HeatGuard AI",
    page_icon=str(ASSETS_DIR / "heatguard_logo.svg"),
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_svg(name: str) -> str:
    path = ASSETS_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #020604;
            --surface: rgba(5, 18, 12, 0.74);
            --surface-strong: rgba(7, 18, 13, 0.92);
            --border: rgba(103, 243, 194, 0.16);
            --text: #F6FBF8;
            --muted: #9FB3AA;
            --cyan: #67F3C2;
            --emerald: #20D998;
            --amber: #F4B547;
            --red: #F36B6B;
        }
        .stApp {
            background:
                radial-gradient(circle at 50% 10%, rgba(103, 243, 194, 0.16), transparent 34%),
                radial-gradient(circle at 88% 86%, rgba(32, 217, 152, 0.13), transparent 18%),
                linear-gradient(180deg, #07170F 0%, #020604 56%, #000000 100%);
            color: var(--text);
        }
        .block-container { max-width: 1180px; padding-top: 2.8rem; padding-bottom: 2.4rem; }
        h1, h2, h3, p, div, span, label { letter-spacing: 0; }
        .hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 22px;
            border: 1px solid rgba(103, 243, 194, 0.18);
            background:
                radial-gradient(circle at 50% 0%, rgba(103, 243, 194, 0.12), transparent 45%),
                linear-gradient(135deg, rgba(7, 18, 13, 0.97), rgba(1, 8, 5, 0.86));
            border-radius: 8px;
            padding: 28px 30px;
            margin: 6px 0 18px;
            box-shadow: 0 24px 62px rgba(0, 0, 0, 0.42), inset 0 1px 0 rgba(103, 243, 194, 0.08);
            min-height: 176px;
        }
        .hero-main { display: flex; align-items: center; gap: 20px; min-width: 0; }
        .brand-logo {
            width: 58px;
            height: 58px;
            flex: 0 0 58px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            background: rgba(2, 6, 4, 0.58);
            border: 1px solid rgba(103, 243, 194, 0.22);
        }
        .brand-logo svg { width: 44px; height: 44px; }
        .hero-copy { min-width: 0; }
        .eyebrow { color: var(--cyan); font-size: 0.76rem; font-weight: 820; text-transform: uppercase; margin-bottom: 9px; }
        .hero h1 { margin: 0; color: #F6FBF8; font-size: clamp(2.15rem, 4vw, 3.55rem); line-height: 1.02; font-weight: 880; }
        .hero h2 { margin: 10px 0 0; color: var(--cyan); font-size: clamp(1rem, 1.6vw, 1.18rem); }
        .hero p { margin: 10px 0 0; color: var(--muted); max-width: 780px; line-height: 1.58; }
        .hero-badge {
            flex: 0 0 auto;
            border: 1px solid rgba(103, 243, 194, 0.28);
            background: rgba(103, 243, 194, 0.10);
            color: #B8FFE4;
            border-radius: 8px;
            padding: 10px 12px;
            font-weight: 800;
            white-space: nowrap;
        }
        .panel {
            border: 1px solid var(--border);
            background: linear-gradient(145deg, rgba(8, 24, 16, 0.84), rgba(2, 8, 5, 0.72));
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 14px;
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.30);
            backdrop-filter: blur(14px);
        }
        .section-title { color: #F6FBF8; font-size: 1.12rem; font-weight: 820; margin: 0 0 12px; }
        .muted { color: var(--muted); font-size: 0.92rem; line-height: 1.5; }
        .status-row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
        .chip {
            border: 1px solid rgba(103, 243, 194, 0.16);
            background: rgba(2, 8, 5, 0.56);
            border-radius: 8px;
            padding: 12px;
            min-height: 72px;
        }
        .chip-label { color: var(--muted); font-size: 0.74rem; text-transform: uppercase; font-weight: 760; }
        .chip-value { color: #F6FBF8; margin-top: 6px; font-weight: 800; overflow-wrap: anywhere; }
        .kpi {
            border: 1px solid var(--border);
            background: linear-gradient(145deg, rgba(8, 22, 15, 0.92), rgba(3, 9, 6, 0.94));
            border-radius: 8px;
            padding: 15px;
            height: 142px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            overflow: hidden;
        }
        .kpi-label { color: var(--muted); font-size: 0.76rem; text-transform: uppercase; font-weight: 780; }
        .kpi-value {
            margin-top: 8px;
            color: #F6FBF8;
            font-size: clamp(1.25rem, 2vw, 1.78rem);
            font-weight: 880;
            line-height: 1.08;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .kpi-note { margin-top: 8px; color: var(--muted); font-size: 0.84rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .explain-box {
            border-left: 3px solid var(--cyan);
            background: rgba(2, 8, 5, 0.62);
            border-radius: 8px;
            padding: 14px 15px;
            color: var(--text);
            line-height: 1.58;
        }
        .flow { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
        .step {
            flex: 1 1 145px;
            min-height: 58px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(103, 243, 194, 0.16);
            background: rgba(2, 8, 5, 0.58);
            border-radius: 8px;
            font-weight: 760;
            text-align: center;
        }
        .arrow { color: var(--cyan); font-weight: 900; }
        div[data-baseweb="select"] > div { background-color: rgba(3, 12, 8, 0.96); border-color: rgba(103, 243, 194, 0.22); border-radius: 8px; }
        .stSlider div[data-baseweb="slider"] div[role="slider"] {
            background-color: var(--cyan) !important;
            border-color: var(--cyan) !important;
            box-shadow: 0 0 0 5px rgba(103, 243, 194, 0.10) !important;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid rgba(103, 243, 194, 0.38);
            background: linear-gradient(135deg, #67F3C2, #20D998);
            color: #03100A;
            font-weight: 850;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            background: rgba(3, 12, 8, 0.88);
            border: 1px solid var(--border);
            padding: 9px 13px;
        }
        @media (max-width: 920px) {
            .hero { align-items: flex-start; flex-direction: column; }
            .hero-badge { white-space: normal; }
            .status-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 620px) {
            .block-container { padding-top: 1.6rem; }
            .hero { padding: 22px; }
            .hero-main { align-items: flex-start; }
            .status-row { grid-template-columns: 1fr; }
            .arrow { display: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def cached_artifact() -> dict | None:
    return load_model_artifact(MODEL_PATH)


@st.cache_data(show_spinner=False)
def cached_frames(folder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_context_and_test(Path(folder))


def render_hero() -> None:
    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-main">
            <div class="brand-logo">{load_svg("heatguard_logo.svg")}</div>
            <div class="hero-copy">
              <div class="eyebrow">IIIT Lucknow Climate Intelligence Challenge 2026</div>
              <h1>HeatGuard AI</h1>
              <h2>10-Day Wet-Bulb Heat Risk Forecasting &amp; Action Intelligence</h2>
              <p>Predicts dangerous wet-bulb heat, classifies public-health risk, and generates practical recommendations for heat action planning.</p>
            </div>
          </div>
          <div class="hero-badge">Track 1 WBT Forecasting</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def chip(label: str, value: str, color: str = "#F6FBF8") -> str:
    return (
        '<div class="chip">'
        f'<div class="chip-label">{html.escape(label)}</div>'
        f'<div class="chip-value" style="color:{color};">{html.escape(value)}</div>'
        "</div>"
    )


def metric_card(label: str, value: str, note: str, color: str = "#67F3C2") -> None:
    st.markdown(
        '<div class="kpi">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value" style="color:{color};">{html.escape(value)}</div>'
        f'<div class="kpi-note">{html.escape(note)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def layout(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,8,5,0.35)",
        font=dict(color="#F6FBF8"),
        margin=dict(l=20, r=20, t=44, b=28),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="rgba(159,179,170,0.16)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(159,179,170,0.16)", zeroline=False)
    return fig


def forecast_chart(forecast: pd.DataFrame) -> go.Figure:
    y_min = min(18.0, float(forecast["lower_bound"].min()) - 1.0)
    y_max = max(34.0, float(forecast["predicted_wbt"].max()) + 2.0)
    fig = go.Figure()
    zones = [
        (y_min, 24, "Low", "rgba(103,243,194,0.10)"),
        (24, 27, "Moderate", "rgba(94,234,212,0.11)"),
        (27, 30, "High", "rgba(245,158,11,0.14)"),
        (30, y_max, "Extreme", "rgba(239,68,68,0.15)"),
    ]
    for y0, y1, label, color in zones:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, annotation_text=label, annotation_position="left")

    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["predicted_wbt"],
            mode="lines+markers",
            name="Predicted WBT",
            line=dict(color="#67F3C2", width=4),
            marker=dict(size=9, color="#20D998", line=dict(width=2, color="#020604")),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(forecast["date"]) + list(forecast["date"])[::-1],
            y=list(forecast["upper_bound"]) + list(forecast["lower_bound"])[::-1],
            fill="toself",
            fillcolor="rgba(103,243,194,0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="RMSE band",
        )
    )
    peak = forecast.loc[forecast["predicted_wbt"].idxmax()]
    fig.add_annotation(
        x=peak["date"],
        y=peak["predicted_wbt"],
        text=f"Peak {peak['predicted_wbt']:.1f} C",
        showarrow=True,
        arrowcolor="#F4B547",
        bgcolor="rgba(3,12,8,0.94)",
        bordercolor="#F4B547",
    )
    fig.update_layout(title="10-Day Wet-Bulb Forecast")
    fig.update_yaxes(title="Wet-bulb temperature (C)", range=[y_min, y_max])
    return layout(fig, 420)


def gauge_chart(value: float) -> go.Figure:
    risk = classify_wbt(value)
    axis_min = min(16.0, value - 1.0)
    axis_max = max(34.0, value + 2.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": " C", "font": {"color": "#F6FBF8", "size": 30}},
            gauge={
                "axis": {"range": [axis_min, axis_max]},
                "bar": {"color": risk_color(risk), "thickness": 0.3},
                "steps": [
                    {"range": [axis_min, 24], "color": "rgba(103,243,194,0.22)"},
                    {"range": [24, 27], "color": "rgba(94,234,212,0.18)"},
                    {"range": [27, 30], "color": "rgba(245,158,11,0.26)"},
                    {"range": [30, axis_max], "color": "rgba(239,68,68,0.28)"},
                ],
            },
            title={"text": f"Peak Risk: {risk}"},
        )
    )
    return layout(fig, 320)


def risk_distribution_chart(forecast: pd.DataFrame) -> go.Figure:
    order = [level["name"] for level in RISK_LEVELS]
    counts = forecast["risk_category"].value_counts().reindex(order, fill_value=0)
    fig = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker=dict(color=[risk_color(label) for label in counts.index]),
        )
    )
    fig.update_layout(title="10-Day Risk Category Distribution")
    fig.update_yaxes(title="Days")
    return layout(fig, 300)


def history_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not history.empty and {"date", "WBT"}.issubset(history.columns):
        hist = history.copy()
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist["WBT"],
                mode="lines+markers",
                name="Recent WBT",
                line=dict(color="#67F3C2", width=3),
            )
        )
    fig.update_layout(title="Recent Context Trend")
    fig.update_yaxes(title="WBT (C)")
    return layout(fig, 300)


def main() -> None:
    inject_css()
    render_hero()

    competition_dir = find_competition_dir()
    status = competition_file_status(competition_dir)
    artifact = cached_artifact()
    advisory_status = "Gemini enabled" if os.getenv("GEMINI_API_KEY") else "Rule-based fallback"

    if not status.get("available") or artifact is None or competition_dir is None:
        st.error("Project is missing required data or trained model files.")
        return

    context_df, test_df = cached_frames(str(competition_dir))
    locations = sorted(test_df["location_id"].astype(str).unique().tolist())
    metrics = artifact.get("metrics", {}).get("primary_horizon_metrics", {})
    confidence = confidence_from_rmse(metrics.get("rmse"), is_demo=False)

    st.markdown('<div class="section-title">Forecast Setup</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        location_id = st.selectbox("Kaggle grid/location", locations, index=0)
    available_days = sorted(
        pd.to_numeric(test_df.loc[test_df["location_id"].astype(str) == location_id, "day_index"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    with c2:
        start_day = st.slider(
            "Forecast anchor day_index",
            min_value=min(available_days),
            max_value=max(available_days),
            value=min(available_days),
            step=1,
        )

    result = make_location_forecast(
        artifact,
        context_df,
        test_df,
        location_id=location_id,
        start_day_index=start_day,
        horizon=10,
    )
    forecast = result["forecast"]
    peak = result["peak"]
    importance = get_feature_importance(artifact, top_n=10)
    explanation = explain_prediction(
        importance,
        risk_level=str(peak["risk_category"]),
        predicted_wbt=float(peak["predicted_wbt"]),
    )

    kpis = st.columns(5)
    kpi_values = [
        ("Peak WBT", f"{peak['predicted_wbt']:.1f} C", f"Day {int(peak['day'])}", "#67F3C2"),
        ("Risk Level", str(peak["risk_category"]), "WBT threshold", risk_color(str(peak["risk_category"]))),
        ("Peak Date", str(peak["date"]), f"Anchor {result.get('anchor_row_id', '')}", "#F6FBF8"),
        ("Reliability", f"{confidence:.0f}% {reliability_label(confidence)}", "Validation-derived", "#20D998"),
        ("Horizon", "10 days", "Track 1 target", "#B8FFE4"),
    ]
    for col, item in zip(kpis, kpi_values):
        with col:
            metric_card(*item)

    left, right = st.columns([1.45, 0.85])
    with left:
        st.plotly_chart(forecast_chart(forecast), use_container_width=True)
    with right:
        st.plotly_chart(gauge_chart(float(peak["predicted_wbt"])), use_container_width=True)

    insight_left, insight_right = st.columns([1, 1])
    with insight_left:
        st.markdown(
            '<div class="panel"><div class="section-title">Risk Explanation</div>'
            f'<div class="explain-box">{html.escape(explanation)}</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(history_chart(result["history"]), use_container_width=True)
    with insight_right:
        st.plotly_chart(risk_distribution_chart(forecast), use_container_width=True)

    st.markdown('<div class="section-title">Heat-Risk Advisory</div>', unsafe_allow_html=True)
    if st.button("Generate Action Advisory", use_container_width=True):
        st.session_state["advisories"] = generate_all_audience_advisories(
            str(peak["risk_category"]),
            float(peak["predicted_wbt"]),
            int(peak["day"]),
            explanation,
            AUDIENCES,
        )
    advisories = st.session_state.get("advisories")
    if advisories:
        tabs = st.tabs(AUDIENCES)
        for tab, audience in zip(tabs, AUDIENCES):
            with tab:
                st.markdown(advisories.get(audience, "No advisory generated."))
    else:
        st.info("Click the button to generate citizen, hospital, school, worker, and authority advisories.")


if __name__ == "__main__":
    main()
