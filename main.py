from fastapi import FastAPI

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