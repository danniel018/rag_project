import csv

QUESTIONS_FILE = "engineer_questions.csv"

with open(QUESTIONS_FILE, newline="", encoding="utf-8") as f:
    QUESTIONS = [row["engineer_question"] for row in csv.DictReader(f)]

REQUIRED_SPANS = [
    # Q01
    ["480 Nm"],
    # Q02
    [
        "340 litres",
        "reservoir alone holds 210 litres",
        "approximately 60 litres which will not drain through the reservoir point",
    ],
    # Q03
    ["up to **20 minutes**", "pressure gauge reads zero"],
    # Q04
    ["4000 hours", "Borescope inspection of combustion section"],
    # Q05
    [
        "28 Nm**. This supersedes",
        "serial numbers below HT400-2200",
        "witness-mark movement at every 500 hour",
        "alloy component",
    ],
    # Q06
    ["at or above serial HT400-2200 retain the 22 Nm", "alloy component"],
    # Q07
    ["60 degrees C for approximately **four hours**"],
    # Q08
    [
        "Isolate and lock off electrical supply",
        "Confirm zero energy state",
        "approximately **four hours** after shutdown",
        "Two people are required for any lifting operation above 25 kg",
        "Support the liner before removing the final fixings",
        "Withdraw the liner squarely",
        "Replace on every removal",
    ],
    # Q09
    ["190 Nm", "Nickel anti-seize"],
    # Q10
    ["fuel skid", "HT-400-FS"],
    # Q11
    ["the generator", "HT-400-GEN"],
    # Q12
    [
        "four passes",
        "diametrically opposite",
        "within 30 minutes of the third",
        "480 Nm",
        "not recoverable in the field",
        "Isolate and lock off electrical supply",
    ],
    # Q13
    ["permit, a gas test, and a standby person"],
    # Q14
    [
        "do not exceed 2 bar differential",
        "Filter blocked",
        "Remove the retaining clips from each element",
        "Isolate and lock off electrical supply",
    ],
    # Q15
    ["Two people are required for any lifting operation above 25 kg"],
]