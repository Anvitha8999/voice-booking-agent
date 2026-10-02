import datetime as dt
import threading
import uuid

from pydantic import BaseModel, Field

OPEN_HOURS = ["09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"]

_bookings: dict[tuple[dt.date, str], dict] = {}
_lock = threading.Lock()


class CheckAvailabilityArgs(BaseModel):
    date: dt.date = Field(description="Appointment date in YYYY-MM-DD format.")


class BookAppointmentArgs(BaseModel):
    date: dt.date = Field(description="Appointment date in YYYY-MM-DD format.")
    time: str = Field(
        pattern=r"^\d{2}:\d{2}$",
        description="Start time in 24-hour HH:MM format, e.g. 14:00.",
    )
    customer_name: str = Field(
        min_length=1, max_length=100, description="The caller's full name."
    )
    phone: str | None = Field(
        default=None, max_length=20, description="The caller's phone number, if given."
    )


def _day_problem(day: dt.date) -> str | None:
    """Return a speakable reason the day can't be booked, or None."""
    if day < dt.date.today():
        return "That date is in the past."
    if day.weekday() >= 5:
        return "We're closed on weekends."
    return None


def _open_slots(day: dt.date) -> list[str]:
    return [t for t in OPEN_HOURS if (day, t) not in _bookings]


def check_availability(args: CheckAvailabilityArgs) -> dict:
    day = args.date
    if problem := _day_problem(day):
        return {"available": False, "slots": [], "message": problem}

    slots = _open_slots(day)
    return {
        "date": day.isoformat(),
        "available": bool(slots),
        "slots": slots,
        "message": f"{len(slots)} open slots." if slots else "Fully booked that day.",
    }


def book_appointment(args: BookAppointmentArgs) -> dict:
    name = args.customer_name.strip()

    with _lock:
        if problem := _day_problem(args.date):
            return {"success": False, "message": problem}
        if args.time not in OPEN_HOURS:
            return {
                "success": False,
                "message": f"{args.time} isn't a valid slot.",
                "valid_times": OPEN_HOURS,
            }

        existing = _bookings.get((args.date, args.time))
        if existing:
            if existing["customer_name"].lower() == name.lower():
                return {
                    "success": True,
                    "already_booked": True,
                    "confirmation_id": existing["confirmation_id"],
                    "message": f"{name} is already booked on {args.date.isoformat()} at {args.time}.",
                }
            return {
                "success": False,
                "message": "That slot is already taken.",
                "alternatives": _open_slots(args.date)[:3],
            }

        confirmation_id = uuid.uuid4().hex[:6].upper()
        _bookings[(args.date, args.time)] = {
            "customer_name": name,
            "phone": args.phone,
            "confirmation_id": confirmation_id,
        }

    return {
        "success": True,
        "confirmation_id": confirmation_id,
        "message": f"Booked {name} on {args.date.isoformat()} at {args.time}.",
    }
