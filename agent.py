import datetime as dt
import json
import re
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
MAX_OFFERED_SLOTS = 3
KEEP_ALIVE = "30m"

YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|correct|confirm|confirmed|sure|go ahead|please do|that's right|sounds good)\b",
    re.IGNORECASE,
)
CLAIM_RE = re.compile(
    r"(i('ve| have) booked|you('re| are) (all )?booked|booked you|is booked for|"
    r"confirmation (code|number)|is confirmed|you're all set)",
    re.IGNORECASE,
)
CORRECTION = (
    "Correction: no booking has been made. Do not say an appointment is booked. "
    "If the caller has confirmed the day, time, and name, call book_appointment now. "
    "Otherwise, repeat the details and ask them to confirm."
)

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
        "description": "Book an appointment. Only call after the caller has said yes to the date, time, and name.",
    },
}


def log(text: str) -> None:
    print(f"\033[2m  [{text}]\033[0m", file=sys.stderr)


def spoken_time(hhmm: str) -> str:
    hour, minute = map(int, hhmm.split(":"))
    suffix = "a.m." if hour < 12 else "p.m."
    hour12 = hour % 12 or 12
    return f"{hour12} {suffix}" if minute == 0 else f"{hour12}:{minute:02d} {suffix}"


def present(name: str, result: dict) -> dict:
    """Reshape a tool result so a small model can speak it correctly."""
    out = dict(result)
    if name == "check_availability" and out.get("slots"):
        slots = out.pop("slots")
        out["offer_these_times"] = [
            {"time": t, "say": spoken_time(t)} for t in slots[:MAX_OFFERED_SLOTS]
        ]
        out["other_open_times"] = slots[MAX_OFFERED_SLOTS:]
    if out.get("alternatives"):
        out["alternatives"] = [{"time": t, "say": spoken_time(t)} for t in out["alternatives"]]
    if out.get("confirmation_id"):
        out["say_code"] = " ".join(out["confirmation_id"])
    return out


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
2. Call check_availability. Offer only the times in offer_these_times, using their "say" text.
3. Ask for the caller's full name.
4. Repeat the full date, time, and name, and ask them to confirm.
5. Only after they say yes, call book_appointment.
6. Read the confirmation code using say_code.

Rules:
- Ask only one question per reply.
- Never say an appointment is booked unless book_appointment returned success.
- Send times to tools as 24-hour HH:MM.
- Reply in one or two short sentences with no lists or symbols, because your words will be spoken aloud."""


def warm_up() -> None:
    start = time.perf_counter()
    ollama.generate(model=MODEL, prompt="", keep_alive=KEEP_ALIVE)
    log(f"model loaded in {time.perf_counter() - start:.2f}s")


class Session:
    def __init__(self) -> None:
        self.messages: list = [{"role": "system", "content": system_prompt(dt.date.today())}]
        self.checked_dates: set[dt.date] = set()
        self.confirmation_ids: set[str] = set()
        self.last_user_text = ""

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
            if not YES_RE.search(self.last_user_text):
                return {"error": "The caller has not confirmed yet. Repeat the date, time, and name, then ask them to confirm."}
            if writes_this_turn >= 1:
                return {"error": "Only one booking is allowed per turn."}

        result = spec["func"](args)
        if name == "check_availability":
            self.checked_dates.add(args.date)
        return result

    def reply(self, user_text: str) -> str:
        self.last_user_text = user_text
        self.messages.append({"role": "user", "content": user_text})
        writes = 0
        booked_this_turn = False

        for _ in range(MAX_TOOL_ROUNDS):
            start = time.perf_counter()
            response = ollama.chat(
                model=MODEL,
                messages=self.messages,
                tools=tool_schemas(),
                options={"temperature": 0},
                keep_alive=KEEP_ALIVE,
            )
            log(f"llm {time.perf_counter() - start:.2f}s")

            message = response.message
            self.messages.append(message)

            if not message.tool_calls:
                text = (message.content or "").strip()
                known_code = any(code in text for code in self.confirmation_ids)
                if CLAIM_RE.search(text) and not booked_this_turn and not known_code:
                    log(f"guardrail: blocked unbacked booking claim: {text!r}")
                    self.messages.append({"role": "system", "content": CORRECTION})
                    continue
                return text

            for call in message.tool_calls:
                name = call.function.name
                raw_args = call.function.arguments
                result = self.run_tool(name, raw_args, writes)
                if TOOLS.get(name, {}).get("writes") and result.get("success"):
                    writes += 1
                    booked_this_turn = True
                    self.confirmation_ids.add(result["confirmation_id"])
                shown = present(name, result)
                log(f"tool {name}({json.dumps(raw_args)}) -> {json.dumps(shown)}")
                self.messages.append(
                    {"role": "tool", "content": json.dumps(shown), "tool_name": name}
                )

        log("hit MAX_TOOL_ROUNDS")
        return "Sorry, I'm having trouble with that. Could you say it another way?"


def main() -> None:
    warm_up()
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