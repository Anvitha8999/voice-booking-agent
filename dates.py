import datetime as dt
import re

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DAY = r"(\d{1,2})(?:st|nd|rd|th)?"


def _upcoming(today: dt.date, month: int, day: int) -> dt.date | None:
    for year in (today.year, today.year + 1):
        try:
            candidate = dt.date(year, month, day)
        except ValueError:
            return None
        if candidate >= today:
            return candidate
    return None


def resolve_date(text: str, today: dt.date) -> dt.date | None:
    """Turn a spoken date into a calendar date, or None if no date is mentioned.

    Policy: a bare weekday or "next <weekday>" means the next occurrence after today.
    The agent always reads the full date back, so the caller can correct it.
    """
    t = text.lower()

    if m := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t):
        try:
            return dt.date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
    if re.search(r"\btoday\b", t):
        return today
    if re.search(r"\btomorrow\b", t):
        return today + dt.timedelta(days=1)
    if m := re.search(rf"\b{_MONTH}\s+(?:the\s+)?{_DAY}\b", t):
        return _upcoming(today, MONTHS[m[1]], int(m[2]))
    if m := re.search(rf"\b{_DAY}\s+of\s+{_MONTH}\b", t):
        return _upcoming(today, MONTHS[m[2]], int(m[1]))
    for index, name in enumerate(WEEKDAYS):
        if re.search(rf"\b{name}\b", t):
            days_ahead = (index - today.weekday()) % 7 or 7
            return today + dt.timedelta(days=days_ahead)
    if m := re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\b", t):
        day = int(m[1])
        month, year = today.month, today.year
        if day < today.day:
            month, year = (1, year + 1) if month == 12 else (month + 1, year)
        try:
            return dt.date(year, month, day)
        except ValueError:
            return None
    return None