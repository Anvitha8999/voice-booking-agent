import pytest

from agent import CLAIM_RE, clean_for_speech


@pytest.mark.parametrize(
    "text",
    [
        "I have booked you for 10 a.m. on Tuesday.",
        "Your appointment has been booked.",
        "You're all booked for Tuesday.",
        "Your confirmation code is 1 2 3 4 5.",
    ],
)
def test_booking_claims_are_detected(text):
    assert CLAIM_RE.search(text)


@pytest.mark.parametrize(
    "text",
    [
        "Shall I book 11 a.m. on Tuesday, October 6 for Anvitha Reddy?",
        "Which day would you like to come in?",
        "Monday is fully booked. Would Tuesday work?",
        "That slot is already taken. I have 9 a.m. or 11 a.m.",
    ],
)
def test_questions_and_facts_are_not_claims(text):
    assert not CLAIM_RE.search(text)


def test_clean_for_speech_removes_lists():
    raw = "Here they are:\n- 9 a.m.\n- 10 a.m.\n**Pick one**"
    assert clean_for_speech(raw) == "Here they are: 9 a.m. 10 a.m. Pick one"