import datetime as dt
import json
import sys
import time

import ollama
from pydantic import ValidationError

from booking import (
    BookAppointmentArgs,
    CheckAvailabilityArgs,
    book_appointment,
    check_availability,
)

MODEL = "qwen2.5:3b"
MAX_TOOL_ROUNDS = 4

TOOLS = {
    "check_availability": {
        "args_model": CheckAvailabilityArgs,
        "func": check_availability,
        "writes": False,
        "description": "Check open appointment times for one date. Call this before offering any times.",
    },
    "book_appointment": {
        "args_model": BookAppointmentArgs,
        "func": book_appointment,
        "writes": True,
        "description": "Book an appointment. Only call after the caller has confirmed the date, time, and their name.",
    },
}


def log(text: str) -> None:
    print(f"\033[2m  [{text}]\033[0m", file=sys.stderr)


def tool_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["args_model"].model_json_schema(),
            },
        }
        for name, spec in TOOLS.items()
    ]


def system_prompt(today: dt.date) -> str:
    lines = []
    for i in range(14):
        day = today + dt.timedelta(days=i)
        label = " (today)" if i == 0 else ""
        lines.append(f"- {day:%A %B %d}: {day.isoformat()}{label}")
    calendar = "\n".join(lines)

    return f"""You are a phone scheduling assistant that books appointments.
Today is {today:%A, %B %d, %Y}.
Use this table to turn spoken dates into YYYY-MM-DD. Never calculate dates yourself:
{calendar}

We are open Monday to Friday. Appointments start on the hour.

Steps:
1. Ask which day the caller wants and look it up in the table.
2. Call check_availability. Offer at most three of the returned times.
3. Ask for the caller's full name.
4. Repeat the day, time, and name, and ask them to confirm.
5. Only after they say yes, call book_appointment.
6. Read back the confirmation code.

Rules:
- Only offer times that a tool returned. Never guess availability.
- Send times to tools as 24-hour HH:MM. Say times naturally, like "two p.m."
- Reply in one or two short sentences with no lists or symbols, because your words will be spoken aloud."""


class Session:
    def __init__(self) -> None:
        self.messages: list = [{"role": "system", "content": system_prompt(dt.date.today())}]
        self.checked_dates: set[dt.date] = set()

    def run_tool(self, name: str, raw_args: dict, writes_this_turn: int) -> dict:
        spec = TOOLS.get(name)
        if spec is None:
            return {"error": f"Unknown tool '{name}'."}

        try:
            args = spec["args_model"].model_validate(raw_args)
        except ValidationError as e:
            details = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
            return {"error": "Invalid arguments.", "details": details}

        if name == "book_appointment":
            if args.date not in self.checked_dates:
                return {"error": "Call check_availability for this date before booking."}
            if writes_this_turn >= 1:
                return {"error": "Only one booking is allowed per turn."}

        result = spec["func"](args)
        if name == "check_availability":
            self.checked_dates.add(args.date)
        return result

    def reply(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        writes = 0

        for _ in range(MAX_TOOL_ROUNDS):
            start = time.perf_counter()
            response = ollama.chat(
                model=MODEL,
                messages=self.messages,
                tools=tool_schemas(),
                options={"temperature": 0},
            )
            log(f"llm {time.perf_counter() - start:.2f}s")

            message = response.message
            self.messages.append(message)

            if not message.tool_calls:
                return (message.content or "").strip()

            for call in message.tool_calls:
                name = call.function.name
                raw_args = call.function.arguments
                result = self.run_tool(name, raw_args, writes)
                if TOOLS.get(name, {}).get("writes") and result.get("success"):
                    writes += 1
                log(f"tool {name}({json.dumps(raw_args)}) -> {json.dumps(result)}")
                self.messages.append(
                    {"role": "tool", "content": json.dumps(result), "tool_name": name}
                )

        log("hit MAX_TOOL_ROUNDS")
        return "Sorry, I'm having trouble with that. Could you say it another way?"


def main() -> None:
    session = Session()
    print("Type to talk to the agent. Ctrl+C to quit.")
    try:
        while True:
            text = input("\nYou: ").strip()
            if text:
                print(f"Agent: {session.reply(text)}")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()