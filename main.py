import datetime as dt
import uuid

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Appointment Booking Tools")

# --- Fake "database" (in memory; resets on every restart) ---
OPEN_HOURS = ["09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"]
bookings: dict[tuple[dt.date, str], dict] = {}


# --- Request models: the tool's arguments, and Retell's wrapper around them ---
class CheckAvailabilityArgs(BaseModel):
    date: dt.date  # Pydantic parses "2026-10-01" and rejects "next tuesday"


class BookAppointmentArgs(BaseModel):
    date: dt.date
    time: str = Field(pattern=r"^\d{2}:\d{2}$", description="24h HH:MM")
    customer_name: str = Field(min_length=1, max_length=100)
    phone: str | None = Field(default=None, max_length=20)


class CheckAvailabilityRequest(BaseModel):
    name: str | None = None
    args: CheckAvailabilityArgs
    call: dict | None = None  # Retell call context; optional so we can test by hand


class BookAppointmentRequest(BaseModel):
    name: str | None = None
    args: BookAppointmentArgs
    call: dict | None = None


# --- Business logic helpers ---
def day_problem(day: dt.date) -> str | None:
    """Return a speakable reason the day can't be booked, or None if it's fine."""
    if day < dt.date.today():
        return "That date is in the past."
    if day.weekday() >= 5:
        return "We're closed on weekends."
    return None


def open_slots(day: dt.date) -> list[str]:
    return [t for t in OPEN_HOURS if (day, t) not in bookings]


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check-availability")
def check_availability(req: CheckAvailabilityRequest):
    day = req.args.date
    if problem := day_problem(day):
        return {"available": False, "slots": [], "message": problem}

    slots = open_slots(day)
    return {
        "date": day.isoformat(),
        "available": bool(slots),
        "slots": slots,
        "message": f"{len(slots)} open slots." if slots else "Fully booked that day.",
    }


@app.post("/book-appointment")
def book_appointment(req: BookAppointmentRequest):
    a = req.args
    if problem := day_problem(a.date):
        return {"success": False, "message": problem}
    if a.time not in OPEN_HOURS:
        return {"success": False, "message": f"{a.time} isn't a valid slot.",
                "valid_times": OPEN_HOURS}
    if (a.date, a.time) in bookings:
        return {"success": False, "message": "That slot is already taken.",
                "alternatives": open_slots(a.date)[:3]}

    confirmation_id = uuid.uuid4().hex[:6].upper()
    bookings[(a.date, a.time)] = {"name": a.customer_name, "phone": a.phone,
                                  "confirmation_id": confirmation_id}
    return {
        "success": True,
        "confirmation_id": confirmation_id,
        "message": f"Booked {a.customer_name} on {a.date.isoformat()} at {a.time}.",
    }