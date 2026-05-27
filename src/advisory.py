from __future__ import annotations

import os
from textwrap import dedent

from dotenv import load_dotenv

from .config import AUDIENCES


def _fallback_advisory(
    risk_level: str,
    predicted_wbt: float,
    days_until_peak: int,
    explanation: str,
    audience: str,
) -> str:
    risk_level = str(risk_level)
    shared = (
        f"Forecast peak: {predicted_wbt:.1f}°C wet-bulb in {days_until_peak} day(s). "
        f"Risk level: {risk_level}. Model explanation: {explanation}"
    )

    playbooks = {
        "Citizens": {
            "Low": ["Keep normal hydration habits.", "Check on elderly family members during hot afternoons."],
            "Moderate": ["Avoid unnecessary midday exposure.", "Drink water before feeling thirsty.", "Use shade, fans, or cooled rooms during peak heat."],
            "High": ["Shift outdoor errands to early morning or evening.", "Use oral rehydration if sweating heavily.", "Watch for dizziness, cramps, confusion, or unusually rapid heartbeat."],
            "Extreme": ["Avoid outdoor activity during peak hours.", "Move vulnerable people to cooled spaces.", "Seek medical help immediately for confusion, fainting, or stopped sweating."],
        },
        "Hospitals": {
            "Low": ["Maintain routine heat illness readiness.", "Check cooling and backup power status."],
            "Moderate": ["Prepare triage prompts for heat exhaustion.", "Ensure ORS, IV fluids, and cooling supplies are stocked."],
            "High": ["Increase emergency department heat-stress readiness.", "Pre-alert ambulance teams and outpatient units.", "Prioritize cooling access for high-risk wards."],
            "Extreme": ["Activate heat surge staffing plans.", "Coordinate with local authorities on cooling shelters.", "Track heatstroke cases and bed capacity at short intervals."],
        },
        "Schools/colleges": {
            "Low": ["Continue routine hydration access.", "Keep shaded rest areas available."],
            "Moderate": ["Move sports practice away from midday heat.", "Add hydration breaks during assemblies and outdoor activities."],
            "High": ["Suspend strenuous outdoor activity.", "Use shaded or indoor spaces for queues and events.", "Monitor students with asthma or chronic illness."],
            "Extreme": ["Cancel outdoor sports and assemblies.", "Consider remote or shortened schedules if cooling is insufficient.", "Prepare transport and parent communication plans."],
        },
        "Outdoor workers": {
            "Low": ["Use normal sun protection.", "Keep drinking water close to the work area."],
            "Moderate": ["Use buddy checks.", "Schedule heavy work earlier in the day.", "Take short shaded breaks."],
            "High": ["Adopt work-rest cycles.", "Increase supervised breaks.", "Stop work for dizziness, cramps, or confusion."],
            "Extreme": ["Avoid heavy outdoor labor in peak hours.", "Require shaded cooling breaks.", "Escalate symptoms quickly to medical care."],
        },
        "Local authorities": {
            "Low": ["Maintain public heat messaging.", "Verify water points and cooling assets."],
            "Moderate": ["Issue preparedness messaging with uncertainty noted.", "Check school, hospital, and worker guidance channels."],
            "High": ["Prepare cooling centers and water distribution.", "Coordinate alerts for high-risk neighborhoods.", "Engage hospitals, employers, and schools."],
            "Extreme": ["Activate heat action plan measures.", "Extend cooling center hours.", "Prioritize outreach to elderly, homeless, and informal worker communities."],
        },
    }

    recommendations = playbooks.get(audience, playbooks["Citizens"]).get(risk_level, playbooks["Citizens"]["Moderate"])
    bullets = "\n".join(f"- {item}" for item in recommendations)
    return f"{shared}\n\nAction plan for {audience}:\n{bullets}\n- Treat this as decision support, not an official government alert."


def generate_heat_advisory(
    risk_level: str,
    predicted_wbt: float,
    days_until_peak: int,
    explanation: str,
    audience: str,
) -> str:
    """Generate actionable heat-risk advice with Gemini when configured."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    if not api_key:
        return _fallback_advisory(risk_level, predicted_wbt, days_until_peak, explanation, audience)

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = dedent(
            f"""
            You are HeatGuard AI, a climate-health decision support assistant.
            Create a concise practical action plan for: {audience}.

            Forecast:
            - Peak wet-bulb temperature: {predicted_wbt:.1f} °C
            - Risk level: {risk_level}
            - Days until peak risk: {days_until_peak}
            - Model explanation: {explanation}

            Requirements:
            - Do not invent official government alerts, closures, or emergency orders.
            - Mention uncertainty briefly.
            - Use 4 to 6 short bullets.
            - Make actions concrete and locally usable.
            - Do not include generic chatbot disclaimers.
            """
        ).strip()
        response = model.generate_content(prompt)
        text = getattr(response, "text", "").strip()
        if text:
            return text
    except Exception as exc:
        return (
            _fallback_advisory(risk_level, predicted_wbt, days_until_peak, explanation, audience)
            + f"\n\nGemini fallback note: {exc}"
        )

    return _fallback_advisory(risk_level, predicted_wbt, days_until_peak, explanation, audience)


def generate_all_audience_advisories(
    risk_level: str,
    predicted_wbt: float,
    days_until_peak: int,
    explanation: str,
    audiences: list[str] | None = None,
) -> dict[str, str]:
    selected = audiences or AUDIENCES
    return {
        audience: generate_heat_advisory(
            risk_level,
            predicted_wbt,
            days_until_peak,
            explanation,
            audience,
        )
        for audience in selected
    }

