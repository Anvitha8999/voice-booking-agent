import datetime as dt
import json
import os
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
from dates import resolve_date

MODEL = os.environ.get("AGENT_MODEL", "qwen2.5:3b")
MAX_TOOL_ROUNDS = 3
MAX_OFFERED_SLOTS = 3
KEEP_ALIVE = "30m"
PLACEHOLDER_NAMES = {"user", "caller", "customer", "client", "guest", "unknown", "name"}

YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|correct|confirm|confirmed|sure|go ahead|please do|that's right|sounds good)\b",
    re.IGNORECASE,
)
CLAIM_RE = re.compile(
    r"(i('ve| have) (just )?booked|you('re| are) (all )?booked|booked you|"
    r"(has|have) been booked|(appointment|you) (is|are) now booked|"
    r"confirmation (code|number) is)",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[a-z']+")

CORRECTION = (
    "Correction: no booking has been made yet. Do not say it is booked. "
    "If the caller already said yes, call book_appointment now. "
    "Otherwise ask: Shall I book TIME on DATE for NAME?"
)
CONFIRM_FALLBACK = "Before I book anything, please confirm the day, the time, and your full name."
GENERIC_FALLBACK = "Sorry, I'm having trouble with that. Could you say it another way?"

TOOLS = {
    "check_availability": {
        "args_model": CheckAvailabilityArgs,
        "func": check_availability,
        "writes": False,
        "llm_visible": False,
        "description": "Check open appointment times for one date.",
    },
    "book_appointment": {
        "args_model": BookAppointmentArgs,
        "func": book_appointment,
        "writes": True,
        "llm_visible": True,
        "description": "Book an appointment. Only call after the caller has said yes to the date, time, and their full name.",
    },
}


def log(text: str) -> None:
    print(f"\033[2m  [{text}]\033[0m", file=sys.stderr)


def clean_for_speech(text: str) -> str:
    text = re.sub(r"^\s*[-*•#]+\s*", "", text, flags=re.MULTILINE)
    text = text.replace("*", "")
    return re.sub(r"\s+", " ", text).strip()


def spoken_time(hhmm: str) -> str:
    hour, minute = map(int, hhmm.split(":"))
    suffix = "a.m." if hour < 12 else "p.m."
    hour12 = hour % 12 or 12
    return f"{hour12} {suffix}" if minute == 0 else f"{hour12}:{minute:02d} {suffix}"


def spoken_date(iso: str) -> str:
    day = dt.date.fromisoformat(iso)
    return f"{day:%A, %B} {day.day}"


def present(name: str, result: dict) -> dict:
    """Reshape a tool result so a small model can speak it correctly."""
    out = dict(result)
    if out.get("date"):
        out["say_date"] = spoken_date(out["date"])
    if out.get("time"):
        out["say_time"] = spoken_time(out["time"])
    if "slots" in out:
        out.pop("message", None)
        slots = out.pop("slots")
        out["open_times"] = [spoken_time(t) for t in slots]
        out["open_times_24h"] = slots
        out["suggest_first"] = [spoken_time(t) for t in slots[:MAX_OFFERED_SLOTS]]
    if out.get("alternatives"):
        out["alternatives"] = [spoken_time(t) for t in out["alternatives"]]
    if out.get("confirmation_id"):
        out["say_code"] = " ".join(out["confirmation_id"])
    return out


def booking_confirmation(shown: dict) -> str:
    """Build the post-booking sentence from tool facts, not from the LLM."""
    opener = "You're already booked" if shown.get("already_booked") else "You're booked"
    return (
        f"{opener} for {shown['say_time']} on {shown['say_date']}. "
        f"Your confirmation code is {shown['say_code']}. Is there anything else I can help with?"
    )


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
        if spec["llm_visible"]
    ]


def system_prompt(today: dt.date) -> str:
    return f"""You are a phone scheduling assistant that books appointments.
Today is {today:%A, %B} {today.day}.
We are open Monday to Friday. Appointments start on the hour.

A caller's message may end with "(Booking system FACT: ...)". Facts are always correct.
- Only mention dates and times that appear in a FACT or a tool result, using their say text.
- If the caller asks about availability without naming a specific day, ask which day they want.

Steps:
1. Ask which day the caller wants.
2. When a FACT gives availability, offer the times in suggest_first in one sentence. If the caller asks about other times, answer from open_times, which lists every open time.
3. Once they pick a time, ask for their full name.
4. Ask exactly: "Shall I book TIME on DATE for NAME?"
5. Only after they say yes, call book_appointment.

Rules:
- Ask only one question per reply.
- If the caller's name is unclear, accept it as you heard it. Never ask for the name more than twice.
- Never say an appointment is booked unless book_appointment returned success.
- Send dates to tools as YYYY-MM-DD and times as 24-hour HH:MM, using open_times_24h.
- Reply in one or two short sentences. No lists, bullets, or symbols, because your words will be spoken aloud."""


def warm_up() -> None:
    start = time.perf_counter()
    ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt(dt.date.today())},
            {"role": "user", "content": "Hello"},
        ],
        tools=tool_schemas(),
        options={"temperature": 0, "num_predict": 1},
        keep_alive=KEEP_ALIVE,
    )
    log(f"model {MODEL} warmed up in {time.perf_counter() - start:.2f}s")


class Session:
    def __init__(self, caller_phone: str | None = None) -> None:
        self.caller_phone = caller_phone
        self.messages: list = [{"role": "system", "content": system_prompt(dt.date.today())}]
        self.checked_dates: set[dt.date] = set()
        self.confirmation_ids: set[str] = set()
        self.user_words: set[str] = set()
        self.last_user_text = ""

    def availability_fact(self, user_text: str) -> str | None:
        day = resolve_date(user_text, dt.date.today())
        if day is None:
            return None
        start = time.perf_counter()
        result = check_availability(CheckAvailabilityArgs(date=day))
        log(f"availability check took {(time.perf_counter() - start) * 1000:.2f} ms")
        self.checked_dates.add(day)
        shown = present("check_availability", {**result, "date": day.isoformat()})
        log(f"prefetch {day.isoformat()} -> {json.dumps(shown)}")
        return json.dumps(shown)

    def name_is_grounded(self, customer_name: str) -> bool:
        words = WORD_RE.findall(customer_name.lower())
        if not words or any(w in PLACEHOLDER_NAMES for w in words):
            return False
        return all(w in self.user_words for w in words)

    def run_tool(self, name: str, raw_args: dict, writes_this_turn: int) -> dict:
        spec = TOOLS.get(name)
        if spec is None or not spec["llm_visible"]:
            return {"error": f"Unknown tool '{name}'."}

        try:
            args = spec["args_model"].model_validate(raw_args)
        except ValidationError as e:
            details = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
            return {"error": "Invalid arguments.", "details": details}

        if name == "book_appointment":
            if args.date not in self.checked_dates:
                return {"error": "That date has not been checked. Ask the caller which day they want."}
            if not self.name_is_grounded(args.customer_name):
                return {"error": "Ask the caller for their full name before booking."}
            if not YES_RE.search(self.last_user_text):
                return {"error": "The caller has not said yes yet. Ask: Shall I book TIME on DATE for NAME?"}
            if writes_this_turn >= 1:
                return {"error": "Only one booking is allowed per turn."}
            if self.caller_phone:
                args = args.model_copy(update={"phone": self.caller_phone})

        start = time.perf_counter()
        result = spec["func"](args)
        log(f"tool {name} executed in {(time.perf_counter() - start) * 1000:.2f} ms")
        return result

    def say(self, text: str) -> str:
        """Record a code-generated reply in history so the model knows it was said."""
        self.messages.append({"role": "assistant", "content": text})
        return text

    def reply(self, user_text: str) -> str:
        self.last_user_text = user_text
        self.user_words.update(WORD_RE.findall(user_text.lower()))

        fact = self.availability_fact(user_text)
        content = user_text if fact is None else f"{user_text}\n\n(Booking system FACT: {fact})"
        self.messages.append({"role": "user", "content": content})

        writes = 0
        booked_this_turn = False
        blocked = 0

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
                text = clean_for_speech(message.content or "")
                if not text:
                    log("empty reply, retrying")
                    continue
                known_code = any(code in text for code in self.confirmation_ids)
                if CLAIM_RE.search(text) and not booked_this_turn and not known_code:
                    blocked += 1
                    log(f"guardrail: blocked unbacked booking claim: {text!r}")
                    if blocked >= 2:
                        return self.say(CONFIRM_FALLBACK)
                    self.messages.append({"role": "system", "content": CORRECTION})
                    continue
                return text

            confirmation = None
            for call in message.tool_calls:
                name = call.function.name
                raw_args = call.function.arguments
                result = self.run_tool(name, raw_args, writes)
                shown = present(name, result)
                if TOOLS.get(name, {}).get("writes") and result.get("success"):
                    writes += 1
                    booked_this_turn = True
                    self.confirmation_ids.add(result["confirmation_id"])
                    confirmation = booking_confirmation(shown)
                log(f"tool {name}({json.dumps(raw_args)}) -> {json.dumps(shown)}")
                self.messages.append(
                    {"role": "tool", "content": json.dumps(shown), "tool_name": name}
                )

            if confirmation:
                return self.say(confirmation)

        log("hit MAX_TOOL_ROUNDS")
        return self.say(GENERIC_FALLBACK)


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