from fastapi import FastAPI, Request
from fastapi.responses import Response

from agent import log
from booking import (
    BookAppointmentArgs,
    CheckAvailabilityArgs,
    book_appointment,
    check_availability,
)

app = FastAPI(title="Voice Booking Agent - Tool Server")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check-availability")
def check_availability_endpoint(args: CheckAvailabilityArgs):
    return check_availability(args)


@app.post("/book-appointment")
def book_appointment_endpoint(args: BookAppointmentArgs):
    return book_appointment(args)


@app.post("/voice")
async def voice(request: Request):
    form = dict(await request.form())
    log(f"twilio /voice webhook: {form}")

    caller = str(form.get("From", "unknown"))
    spoken_caller = " ".join(caller.lstrip("+"))
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Hello. Your call reached the booking agent server. You are calling from {spoken_caller}.</Say>
</Response>"""
    return Response(content=twiml, media_type="application/xml")