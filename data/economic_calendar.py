"""Lightweight economic-calendar awareness for the trading agent.

Provides two things the agent needs:
1. ``get_todays_events()`` — what high-impact events are scheduled today
2. ``get_upcoming_events(days=3)`` — look-ahead for next N days

Data sources (in priority order):
- Static curated schedule of recurring US macro events for 2026
  (FOMC, NFP, CPI, PPI, Retail Sales, GDP, PCE, Jobless Claims)
- These dates are published well in advance by BLS / Fed / BEA

Why static?  External API calendars (Investing.com, ForexFactory) require
scraping with fragile parsing.  Major US macro events follow fixed schedules
that change < 2× per year.  A curated list is 100 % reliable and zero-latency.

Updating: add new dates to ``_EVENTS_2026`` at the start of each quarter
or whenever the Fed revises FOMC dates.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Impact levels ──
HIGH = "high"
MEDIUM = "medium"


class MacroEvent:
    """Single scheduled macro event."""

    __slots__ = ("date", "time_et", "name", "impact", "notes")

    def __init__(self, dt: date, time_et: str, name: str, impact: str = HIGH, notes: str = ""):
        self.date = dt
        self.time_et = time_et   # e.g. "08:30", "14:00", "pre-market"
        self.name = name
        self.impact = impact
        self.notes = notes

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "time_et": self.time_et,
            "name": self.name,
            "impact": self.impact,
            "notes": self.notes,
        }

    def __repr__(self):
        return f"MacroEvent({self.date} {self.time_et} {self.name})"


def _d(m: int, d: int, y: int = 2026) -> date:
    return date(y, m, d)


# ═══════════════════════════════════════════════════════════════════════════════
# 2026 US MACRO CALENDAR — curated from BLS/Fed/BEA public schedules
# ═══════════════════════════════════════════════════════════════════════════════
# Sources:
#   FOMC: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
#   BLS (NFP, CPI, PPI): https://www.bls.gov/schedule/news_release/empsit.htm
#   BEA (GDP, PCE): https://www.bea.gov/news/schedule
#
# Format: (date, time_ET, name, impact, notes)
_EVENTS_2026: List[MacroEvent] = [
    # ── JANUARY ──
    MacroEvent(_d(1, 3),  "08:30", "ISM Manufacturing PMI", HIGH, "Dec data"),
    MacroEvent(_d(1, 7),  "10:00", "JOLTS Job Openings", MEDIUM, "Nov data"),
    MacroEvent(_d(1, 8),  "14:00", "FOMC Minutes", HIGH, "Dec meeting minutes"),
    MacroEvent(_d(1, 10), "08:30", "Nonfarm Payrolls (NFP)", HIGH, "Dec jobs report"),
    MacroEvent(_d(1, 14), "08:30", "PPI", MEDIUM, "Dec producer prices"),
    MacroEvent(_d(1, 15), "08:30", "CPI", HIGH, "Dec consumer prices"),
    MacroEvent(_d(1, 16), "08:30", "Retail Sales", MEDIUM, "Dec retail"),
    MacroEvent(_d(1, 29), "14:00", "FOMC Decision", HIGH, "Rate decision"),
    MacroEvent(_d(1, 30), "08:30", "GDP (Advance)", HIGH, "Q4 advance estimate"),
    MacroEvent(_d(1, 31), "08:30", "PCE Price Index", HIGH, "Dec PCE — Fed's preferred inflation gauge"),
    # ── FEBRUARY ──
    MacroEvent(_d(2, 3),  "10:00", "ISM Manufacturing PMI", HIGH, "Jan data"),
    MacroEvent(_d(2, 5),  "08:30", "ADP Employment", MEDIUM, "Jan private payrolls"),
    MacroEvent(_d(2, 7),  "08:30", "Nonfarm Payrolls (NFP)", HIGH, "Jan jobs report"),
    MacroEvent(_d(2, 12), "08:30", "CPI", HIGH, "Jan consumer prices"),
    MacroEvent(_d(2, 13), "08:30", "PPI", MEDIUM, "Jan producer prices"),
    MacroEvent(_d(2, 14), "08:30", "Retail Sales", MEDIUM, "Jan retail"),
    MacroEvent(_d(2, 27), "08:30", "GDP (Second Estimate)", MEDIUM, "Q4 second estimate"),
    MacroEvent(_d(2, 28), "08:30", "PCE Price Index", HIGH, "Jan PCE"),
    # ── MARCH ──
    MacroEvent(_d(3, 6),  "08:30", "Nonfarm Payrolls (NFP)", HIGH, "Feb jobs report"),
    MacroEvent(_d(3, 11), "08:30", "CPI", HIGH, "Feb consumer prices"),
    MacroEvent(_d(3, 12), "08:30", "PPI", MEDIUM, "Feb producer prices"),
    MacroEvent(_d(3, 17), "08:30", "Retail Sales", MEDIUM, "Feb retail"),
    MacroEvent(_d(3, 18), "14:00", "FOMC Decision", HIGH, "Rate decision + projections (dot plot)"),
    MacroEvent(_d(3, 27), "08:30", "GDP (Third Estimate)", MEDIUM, "Q4 final"),
    MacroEvent(_d(3, 28), "08:30", "PCE Price Index", HIGH, "Feb PCE"),
    # ── APRIL ──
    MacroEvent(_d(4, 3),  "08:30", "Nonfarm Payrolls (NFP)", HIGH, "Mar jobs report"),
    MacroEvent(_d(4, 10), "08:30", "CPI", HIGH, "Mar consumer prices"),
    MacroEvent(_d(4, 11), "08:30", "PPI", MEDIUM, "Mar producer prices"),
    MacroEvent(_d(4, 15), "08:30", "Retail Sales", MEDIUM, "Mar retail"),
    MacroEvent(_d(4, 29), "08:30", "GDP (Advance)", HIGH, "Q1 advance estimate"),
    MacroEvent(_d(4, 30), "08:30", "PCE Price Index", HIGH, "Mar PCE"),
    # ── MAY ──
    MacroEvent(_d(5, 1),  "08:30", "Nonfarm Payrolls (NFP)", HIGH, "Apr jobs report — early due to calendar"),
    MacroEvent(_d(5, 6),  "14:00", "FOMC Decision", HIGH, "Rate decision"),
    MacroEvent(_d(5, 13), "08:30", "CPI", HIGH, "Apr consumer prices"),
    MacroEvent(_d(5, 14), "08:30", "PPI", MEDIUM, "Apr producer prices"),
    MacroEvent(_d(5, 15), "08:30", "Retail Sales", MEDIUM, "Apr retail"),
    MacroEvent(_d(5, 28), "08:30", "GDP (Second Estimate)", MEDIUM, "Q1 second estimate"),
    MacroEvent(_d(5, 29), "08:30", "PCE Price Index", HIGH, "Apr PCE"),
    # ── JUNE ──
    MacroEvent(_d(6, 5),  "08:30", "Nonfarm Payrolls (NFP)", HIGH, "May jobs report"),
    MacroEvent(_d(6, 10), "08:30", "CPI", HIGH, "May consumer prices"),
    MacroEvent(_d(6, 11), "08:30", "PPI", MEDIUM, "May producer prices"),
    MacroEvent(_d(6, 16), "08:30", "Retail Sales", MEDIUM, "May retail"),
    MacroEvent(_d(6, 17), "14:00", "FOMC Decision", HIGH, "Rate decision + projections (dot plot)"),
    MacroEvent(_d(6, 25), "08:30", "GDP (Third Estimate)", MEDIUM, "Q1 final"),
    MacroEvent(_d(6, 26), "08:30", "PCE Price Index", HIGH, "May PCE"),
    # ── JULY ──
    MacroEvent(_d(7, 2),  "08:30", "Nonfarm Payrolls (NFP)", HIGH, "Jun jobs report"),
    MacroEvent(_d(7, 10), "08:30", "CPI", HIGH, "Jun consumer prices"),
    MacroEvent(_d(7, 14), "08:30", "PPI", MEDIUM, "Jun producer prices"),
    MacroEvent(_d(7, 16), "08:30", "Retail Sales", MEDIUM, "Jun retail"),
    MacroEvent(_d(7, 29), "08:30", "GDP (Advance)", HIGH, "Q2 advance estimate"),
    MacroEvent(_d(7, 29), "14:00", "FOMC Decision", HIGH, "Rate decision"),
    MacroEvent(_d(7, 31), "08:30", "PCE Price Index", HIGH, "Jun PCE"),
    # ── AUGUST ──
    MacroEvent(_d(8, 7),  "08:30", "Nonfarm Payrolls (NFP)", HIGH, "Jul jobs report"),
    MacroEvent(_d(8, 12), "08:30", "CPI", HIGH, "Jul consumer prices"),
    MacroEvent(_d(8, 13), "08:30", "PPI", MEDIUM, "Jul producer prices"),
    MacroEvent(_d(8, 14), "08:30", "Retail Sales", MEDIUM, "Jul retail"),
    MacroEvent(_d(8, 27), "08:30", "GDP (Second Estimate)", MEDIUM, "Q2 second estimate"),
    MacroEvent(_d(8, 28), "08:30", "PCE Price Index", HIGH, "Jul PCE"),
    # ── SEPTEMBER ──
    MacroEvent(_d(9, 4),  "08:30", "Nonfarm Payrolls (NFP)", HIGH, "Aug jobs report"),
    MacroEvent(_d(9, 10), "08:30", "CPI", HIGH, "Aug consumer prices"),
    MacroEvent(_d(9, 11), "08:30", "PPI", MEDIUM, "Aug producer prices"),
    MacroEvent(_d(9, 16), "08:30", "Retail Sales", MEDIUM, "Aug retail"),
    MacroEvent(_d(9, 16), "14:00", "FOMC Decision", HIGH, "Rate decision + projections (dot plot)"),
    MacroEvent(_d(9, 25), "08:30", "GDP (Third Estimate)", MEDIUM, "Q2 final"),
    MacroEvent(_d(9, 26), "08:30", "PCE Price Index", HIGH, "Aug PCE"),
    # ── OCTOBER ──
    MacroEvent(_d(10, 2), "08:30", "Nonfarm Payrolls (NFP)", HIGH, "Sep jobs report"),
    MacroEvent(_d(10, 13), "08:30", "CPI", HIGH, "Sep consumer prices"),
    MacroEvent(_d(10, 14), "08:30", "PPI", MEDIUM, "Sep producer prices"),
    MacroEvent(_d(10, 15), "08:30", "Retail Sales", MEDIUM, "Sep retail"),
    MacroEvent(_d(10, 28), "08:30", "GDP (Advance)", HIGH, "Q3 advance estimate"),
    MacroEvent(_d(10, 28), "14:00", "FOMC Decision", HIGH, "Rate decision"),
    MacroEvent(_d(10, 30), "08:30", "PCE Price Index", HIGH, "Sep PCE"),
    # ── NOVEMBER ──
    MacroEvent(_d(11, 6), "08:30", "Nonfarm Payrolls (NFP)", HIGH, "Oct jobs report"),
    MacroEvent(_d(11, 12), "08:30", "CPI", HIGH, "Oct consumer prices"),
    MacroEvent(_d(11, 13), "08:30", "PPI", MEDIUM, "Oct producer prices"),
    MacroEvent(_d(11, 17), "08:30", "Retail Sales", MEDIUM, "Oct retail"),
    MacroEvent(_d(11, 25), "08:30", "GDP (Second Estimate)", MEDIUM, "Q3 second estimate"),
    MacroEvent(_d(11, 25), "08:30", "PCE Price Index", HIGH, "Oct PCE"),
    # ── DECEMBER ──
    MacroEvent(_d(12, 4), "08:30", "Nonfarm Payrolls (NFP)", HIGH, "Nov jobs report"),
    MacroEvent(_d(12, 9), "14:00", "FOMC Decision", HIGH, "Rate decision + projections (dot plot)"),
    MacroEvent(_d(12, 10), "08:30", "CPI", HIGH, "Nov consumer prices"),
    MacroEvent(_d(12, 11), "08:30", "PPI", MEDIUM, "Nov producer prices"),
    MacroEvent(_d(12, 16), "08:30", "Retail Sales", MEDIUM, "Nov retail"),
    MacroEvent(_d(12, 23), "08:30", "GDP (Third Estimate)", MEDIUM, "Q3 final"),
    MacroEvent(_d(12, 23), "08:30", "PCE Price Index", HIGH, "Nov PCE"),
    # ── Weekly recurring (every Thursday) ──
    # Jobless Claims – handled dynamically below, not in static list
]


def _is_thursday(d: date) -> bool:
    return d.weekday() == 3


def get_todays_events(today: Optional[date] = None) -> List[MacroEvent]:
    """Return all macro events scheduled for today (ET date)."""
    today = today or date.today()
    events = [e for e in _EVENTS_2026 if e.date == today]
    # Weekly: Initial Jobless Claims every Thursday at 08:30 ET
    if _is_thursday(today):
        events.append(MacroEvent(today, "08:30", "Initial Jobless Claims", MEDIUM, "Weekly"))
    return events


def get_upcoming_events(days: int = 3, today: Optional[date] = None) -> List[MacroEvent]:
    """Return macro events scheduled in the next *days* calendar days (inclusive today)."""
    today = today or date.today()
    end = today + timedelta(days=days)
    events = [e for e in _EVENTS_2026 if today <= e.date <= end]
    # Add weekly jobless claims Thursdays in range
    d = today
    while d <= end:
        if _is_thursday(d):
            events.append(MacroEvent(d, "08:30", "Initial Jobless Claims", MEDIUM, "Weekly"))
        d += timedelta(days=1)
    events.sort(key=lambda e: (e.date, e.time_et))
    return events


def get_high_impact_today(today: Optional[date] = None) -> List[MacroEvent]:
    """Return only HIGH impact events for today."""
    return [e for e in get_todays_events(today) if e.impact == HIGH]


def format_calendar_for_agent(today: Optional[date] = None) -> List[str]:
    """Format upcoming macro events as context lines for the agent prompt.

    Returns a list of strings to inject into the agent's live state view.
    Empty list if nothing notable upcoming.
    """
    today = today or date.today()
    today_events = get_todays_events(today)
    upcoming = get_upcoming_events(days=3, today=today + timedelta(days=1))  # next 3 days, excluding today

    if not today_events and not upcoming:
        return []

    lines = ["=== ECONOMIC CALENDAR ==="]

    if today_events:
        high_events = [e for e in today_events if e.impact == HIGH]
        med_events = [e for e in today_events if e.impact == MEDIUM]
        if high_events:
            lines.append("⚠️  HIGH-IMPACT TODAY:")
            for e in high_events:
                lines.append(f"  {e.time_et} ET — {e.name} ({e.notes})")
            lines.append("  → Expect volatility. Widen stops or reduce size before release.")
            lines.append("  → Avoid market orders 15 min before/after the release time.")
        if med_events:
            lines.append("📊 MEDIUM-IMPACT TODAY:")
            for e in med_events:
                lines.append(f"  {e.time_et} ET — {e.name} ({e.notes})")

    if upcoming:
        # Only show high-impact look-ahead
        high_upcoming = [e for e in upcoming if e.impact == HIGH]
        if high_upcoming:
            lines.append("📅 UPCOMING HIGH-IMPACT (next 3 days):")
            for e in high_upcoming[:5]:  # cap at 5
                lines.append(f"  {e.date.strftime('%a %b %d')} {e.time_et} ET — {e.name}")

    lines.append("")
    return lines
