import csv

QUESTIONS_FILE = "engineer_questions.csv"

with open(QUESTIONS_FILE, newline="", encoding="utf-8") as f:
    QUESTIONS = [row["engineer_question"] for row in csv.DictReader(f)]

REQUIRED_SPANS = [
    ["480 Nm"],
    ["340 litres", "reservoir alone holds 210 litres", "coolers retain approximately 60 litres"],
    ["up to 20 minutes", "pressure gauge reads zero"],
    ["4000 hours", "Borescope inspection of combustion section"],
    ["from 22 Nm to 28 Nm", "below HT400-2200", "every 500 hour interval", "alloy component"],
    ["at or above serial HT400-2200 retain the 22 Nm", "alloy component"],
    ["above 60 degrees C for approximately four hours"],
    [
        "Isolate and lock off electrical supply",
        "Confirm zero energy state",
        "approximately four hours after shutdown",
        "Two people are required for any lifting operation above 25 kg",
        "Support the liner before removing the final fixings",
        "Withdraw the liner squarely",
        "Replace on every removal",
    ],
    ["190 Nm", "Nickel anti-seize"],
    ["fuel skid", "HT-400-FS"],
    ["the generator", "HT-400-GEN"],
    [
        "four passes",
        "diametrically opposite",
        "within 30 minutes of the third",
        "480 Nm",
        "not recoverable in the field",
        "Isolate and lock off electrical supply",
    ],
    ["Entry requires a permit, a gas test, and a standby person"],
    [
        "do not exceed 2 bar differential",
        "Filter blocked",
        "Remove the retaining clips from each element",
        "Isolate and lock off electrical supply",
    ],
    ["Two people are required for any lifting operation above 25 kg"],
]