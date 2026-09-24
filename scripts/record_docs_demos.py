#!/usr/bin/env -S uv run python
"""Records everything js/demos/{stream,doc-20,rules}.js need for static-mode playback:
- site/data/demos/{stream,doc20,rules}.json: the pools/questions/facts those modules read,
  plus (stream only) a "recorded" decide() result per pool item, embedded directly.
- site/data/replays.json: merged in under the same djb2 hashKey js/api.js::decide() uses
  for its live->replay fallback (doc-20 and rules call ctx.decide() directly and rely on
  this cache; stream doesn't need it but we record every call it makes anyway, per spec).
- site/data/demos/<name>.summary.json: tallies/timings for the report.

Requires site/server.py running on :8787 (uv run uvicorn --app-dir site server:app --port 8787).
Mirrors scripts/record_replays.py's hash_key/post_decide/self_test_hash_parity exactly -
imported from there rather than re-implemented.
"""
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "scripts"))
from typical.core import query_text  # noqa: E402
from record_replays import hash_key, post_decide, q, self_test_hash_parity  # noqa: E402

PRESETS = json.loads((ROOT / "site" / "data" / "presets.json").read_text())
DEMOS_DIR = ROOT / "site" / "data" / "demos"
DEMOS_DIR.mkdir(parents=True, exist_ok=True)

REPLAYS_PATH = ROOT / "site" / "data" / "replays.json"
replays = json.loads(REPLAYS_PATH.read_text()) if REPLAYS_PATH.exists() else {}


def record(state: str, queries: list) -> dict:
    """post_decide + stash under its hashKey; -> the raw server response."""
    res = post_decide(state, queries)
    replays[hash_key(state, queries)] = res
    return res


# ============================================================================================
# 6. ONE DOC, 20 QUESTIONS (demo-spec-v2.md section 6) -> data/demos/doc20.json
# ============================================================================================

DOC = (
    "Acme Employee Handbook (excerpt).\n"
    "Leave. Full-time employees accrue 1.5 days of paid leave per month, up to 18 days per year. "
    "Leave requests must be submitted at least 14 days in advance through the HR portal. Requests "
    "for dates in the annual audit week (the first week of March) are not approved. Unused leave "
    "up to 5 days carries over to the next year; the rest is forfeited. Sick leave does not "
    "require advance notice, but a doctor's note is required after 3 consecutive sick days.\n"
    "Refunds. Customers may request a full cash refund within 30 days of purchase if the item is "
    "unused and a receipt or order number is provided. Between 31 and 60 days after purchase, "
    "store credit is issued instead of cash. No refunds are issued for clearance items or gift "
    "cards. Refunds above $500 require approval from a team lead.\n"
    "On-call. The engineer on the primary rotation must acknowledge a page within 15 minutes. If "
    "the primary does not acknowledge, the page escalates to the secondary engineer after 15 "
    "minutes and to the engineering manager after 30 minutes. On-call shifts are one week long "
    "and pay a stipend of $300. Shifts may be swapped with 48 hours' notice if both engineers "
    "agree in writing.\n"
    "Expenses. Expense claims must be filed within 60 days of the expense with an itemised "
    "receipt. Meals are reimbursed up to $75 per day. Alcohol is never reimbursed. Flights must "
    "be booked in economy class unless the flight exceeds 6 hours, in which case premium economy "
    "is allowed. Claims over $1,000 require pre-approval from a director.\n"
    "Remote work. Employees may work remotely up to 3 days per week with manager approval. Fully "
    "remote arrangements require a director's sign-off."
)

# Noul grammar: the `fact`(BoolQ) family shape that scored 9/12 (doc20_probe2.py N2), not the
# simpler 8/12 grammar in doc20_probe.py.
def N(q_):
    return (
        f"Question: {q_}?\nAccording to the passage, is the answer to the question yes? Use only "
        "the information given.\nyes: the passage establishes that the question's answer is yes.  "
        "no: the passage contradicts it or does not settle it."
    )


NOUL = [
    ("can a full-time employee carry over 8 unused leave days to the next year", 0),
    ("does a sick leave of 4 consecutive days require a doctor's note", 1),
    ("is a leave request for the first week of March approved if it is submitted 20 days in advance", 0),
    ("can a customer get a cash refund 45 days after purchase with a receipt and an unused item", 0),
    ("is a full cash refund available 20 days after purchase for an unused item with a receipt", 1),
    ("does a $700 refund require a team lead's approval", 1),
    ("must the primary on-call engineer acknowledge a page within 15 minutes", 1),
    ("does an unacknowledged page reach the engineering manager after 30 minutes", 1),
    ("can two engineers swap on-call shifts with 24 hours' notice", 0),
    ("is a bottle of wine on a dinner receipt reimbursable", 0),
    ("is premium economy allowed on a 4-hour flight", 0),
    ("can an employee work remotely 3 days per week with manager approval", 1),
]


def C(q_):
    return f"According to the handbook, {q_}? Pick the single option the handbook supports."


CHOICE = [
    ("what does a customer receive for a return 40 days after purchase", ["cash refund", "store credit", "nothing"], "store credit"),
    ("who receives a page that has gone unacknowledged for 20 minutes", ["the primary engineer", "the secondary engineer", "the engineering manager"], "the secondary engineer"),
    ("who must pre-approve an expense claim of $1,500", ["a team lead", "a manager", "a director"], "a director"),
    ("how many days in advance must leave be requested", ["7 days", "14 days", "30 days"], "14 days"),
    ("what is the daily meal reimbursement limit", ["$50", "$75", "$100"], "$75"),
]

SCORE = [
    ("Rate the escalation level of a page that has gone unacknowledged for 35 minutes. Levels are ordered 0 < 1 < 2.\n0: still with the primary engineer  1: escalated to the secondary engineer  2: escalated to the engineering manager", ["0", "1", "2"], "2"),
    ("Rate the approval level required for a $600 refund. Levels are ordered 0 < 1 < 2.\n0: no approval needed  1: team lead approval  2: director approval", ["0", "1", "2"], "1"),
    ("Rate the annual paid leave a full-time employee can accrue. Levels are ordered low < medium < high.\nlow: fewer than 10 days  medium: 10 to 17 days  high: 18 days or more", ["low", "medium", "high"], "high"),
]


def build_doc20():
    questions = []
    for text, gold in NOUL:
        questions.append({"display": text, "type": "noul", "query": N(text), "labels": ["no", "yes"], "gold": "yes" if gold else "no"})
    for text, labels, gold in CHOICE:
        questions.append({"display": text, "type": "choice", "query": C(text), "labels": labels, "gold": gold})
    for text, labels, gold in SCORE:
        questions.append({"display": text.split("\n")[0], "type": "score", "query": text, "labels": labels, "gold": gold})
    return {"doc": DOC, "questions": questions}


def record_doc20():
    data = build_doc20()
    all_queries = [q(x["type"], x["query"], x["labels"]) for x in data["questions"]]
    t0 = time.time()
    single = record(DOC, all_queries[:1])
    single_ms = (time.time() - t0) * 1000
    t0 = time.time()
    full = record(DOC, all_queries)
    full_ms = (time.time() - t0) * 1000

    R = full["results"]
    hits = []
    for i, x in enumerate(data["questions"]):
        argmax = R[i]["argmax"]
        hits.append(argmax == x["gold"])
    acc_n = sum(hits[:12]) / 12
    acc_c = sum(hits[12:17]) / 5
    acc_s = sum(hits[17:]) / 3
    total = sum(hits)

    (DEMOS_DIR / "doc20.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "n": 20, "correct": total, "noul_acc": acc_n, "choice_acc": acc_c, "score_acc": acc_s,
        "ms_all20": full["ms"], "ms_single_wall_ms": single_ms, "ms_single_server": single["ms"],
        "wall_ms_all20": full_ms,
    }
    (DEMOS_DIR / "doc20.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"doc20: {total}/20 (noul {acc_n:.3f} choice {acc_c:.3f} score {acc_s:.3f}); all20 {full['ms']:.0f} ms, single {single['ms']:.0f} ms")
    return summary


# ============================================================================================
# R2 / RULES ARE THE PROGRAM -> data/demos/rules.json
# (the loan rubric itself stays in presets.json's "flip" - rules.js reads it via ctx.presets;
# this file only adds the 6 hand-written applicant fact-sets and the static caption.)
# ============================================================================================

FLIP = PRESETS["flip"]

APPLICANTS = [
    {
        "id": "a1-low-credit-verified",
        "label": "Low credit, income verified",
        "facts": [
            "The applicant's credit score is 590.",
            "The applicant's income was verified with pay stubs.",
            "No collateral was pledged.",
            "The applicant has no prior defaults.",
            "The applicant's debt-to-income ratio is 28%.",
        ],
    },
    {
        "id": "a2-low-credit-verified-collateral",
        "label": "Low credit, verified, collateral, one default",
        "facts": [
            "The applicant's credit score is 610.",
            "The applicant's income was verified with pay stubs.",
            "Collateral of $5,000 was pledged.",
            "The applicant has 1 prior default.",
            "The applicant's debt-to-income ratio is 41%.",
        ],
    },
    {
        "id": "a3-good-credit-verified",
        "label": "Good credit, income verified",
        "facts": [
            "The applicant's credit score is 705.",
            "The applicant's income was verified with pay stubs.",
            "No collateral was pledged.",
            "The applicant has no prior defaults.",
            "The applicant's debt-to-income ratio is 19%.",
        ],
    },
    {
        "id": "a4-midcredit-unverified",
        "label": "Sub-650 credit, income not verified",
        "facts": [
            "The applicant's credit score is 640.",
            "The applicant's income was not independently verified.",
            "No collateral was pledged.",
            "The applicant has no prior defaults.",
            "The applicant's debt-to-income ratio is 30%.",
        ],
    },
    {
        "id": "a5-okcredit-unverified-collateral",
        "label": "Ok credit, unverified, collateral pledged",
        "facts": [
            "The applicant's credit score is 680.",
            "The applicant's income was not independently verified.",
            "Collateral of $15,000 was pledged.",
            "The applicant has no prior defaults.",
            "The applicant's debt-to-income ratio is 33%.",
        ],
    },
    {
        "id": "a6-okcredit-unverified-defaults",
        "label": "Ok credit, unverified, two defaults",
        "facts": [
            "The applicant's credit score is 660.",
            "The applicant's income was not independently verified.",
            "No collateral was pledged.",
            "The applicant has 2 prior defaults.",
            "The applicant's debt-to-income ratio is 52%.",
        ],
    },
]
for a in APPLICANTS:
    a["state"] = " ".join(a["facts"])

RULES_CAPTION = (
    "Held-out DecisionMix v2 rule families and grammars: .836 / .898 accuracy (typical-small). "
    "Rubric flip (same facts, a different rule listed first) both-correct .718 / .743 - the model "
    "answers each order correctly more often than it answers both orders correctly, because the "
    "orders sometimes disagree."
)


def order_query(order):
    criteria = {label: FLIP["rules"][label] for label in order}
    return query_text({"type": "choice", "instructions": FLIP["question_prefix"], "criteria": criteria})


def record_rules():
    # the two canonical reorders (already in replays.json per content-spec, re-verified here)
    for order in FLIP["orders"]:
        record(FLIP["state"], [q("choice", order_query(order), order)])

    # 6 applicants x 2 orders, one decide() call per applicant (both orders as two queries on
    # one cached state - "decided in one batch").
    order_queries = [q("choice", order_query(order), order) for order in FLIP["orders"]]
    rows = []
    for a in APPLICANTS:
        res = record(a["state"], order_queries)
        deny_first, approve_first = res["results"]
        rows.append({"id": a["id"], "deny_first": deny_first["argmax"], "approve_first": approve_first["argmax"], "ms": res["ms"]})
        flip_flag = " <-- flips" if deny_first["argmax"] != approve_first["argmax"] else ""
        print(f"  {a['id']:36s} deny-first {deny_first['argmax']:24s} approve-first {approve_first['argmax']:24s}{flip_flag}")

    data = {
        "title": "Same facts, different rule - 6 applicants",
        "caption": RULES_CAPTION,
        "applicants": APPLICANTS,
    }
    (DEMOS_DIR / "rules.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    n_flips = sum(1 for r in rows if r["deny_first"] != r["approve_first"])
    summary = {"n": len(rows), "flips": n_flips, "rows": rows}
    (DEMOS_DIR / "rules.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"rules: {n_flips}/{len(rows)} applicants flip outcome between orders")
    return summary


# ============================================================================================
# HERO STREAM -> data/demos/stream.json
# 5 item types, pools copied verbatim from /tmp/probe2/{tickets,hero_probe*,ads_probe}.py.
# Every pool item is fully recorded (embedded "decisions"/"ms") so static mode never guesses.
# ============================================================================================

sys.path.insert(0, "/tmp/probe2")
from tickets import ROWS as TICKET_ROWS  # noqa: E402

TEAMS = {
    "billing": "charges, invoices, duplicate payments, refunds of money",
    "orders": "cancelling, changing, or tracking an order, subscription, or booking",
    "returns": "items that arrived damaged, defective, or wrong",
    "account": "login problems, passwords, account access",
    "general": "anything else",
}
Q_TEAM = ("Which team should own this message? Each option's criterion is listed; pick the single "
          "best match. A topic mentioned in passing is not a request.\n" + "  ".join(f"{k}: {v}" for k, v in TEAMS.items()))

WANTS = {
    "a refund": "money back for a charge or purchase",
    "a cancellation": "an order, subscription, booking, or account cancelled",
    "a replacement or repair": "a damaged or faulty item replaced or fixed",
    "account access": "to log in or reset a password",
    "information": "an answer to a question, no action on their account",
}
Q_WANTS = "What does the customer want? Each option's criterion is listed; pick the single best match.\n" + "  ".join(f"{k}: {v}" for k, v in WANTS.items())

TOOLS = [
    ("get_weather", "Fetches the current or forecast weather for a location", ["Will it rain in Lisbon tomorrow?", "What's the temperature in Denver right now?"]),
    ("send_email", "Sends an email to a given address", ["Email Marta the quarterly numbers.", "Send a note to Sam saying I'm running late."]),
    ("create_calendar_event", "Adds an event to the user's calendar", ["Put lunch with Lee on my calendar for Friday.", "Schedule a dentist visit next Tuesday at 3."]),
    ("set_timer", "Starts a countdown timer", ["Set a timer for 12 minutes.", "Start a 25-minute countdown."]),
    ("web_search", "Searches the web for a query", ["Look up who won the 40th Super Bowl.", "Search for reviews of the new Osaka museum."]),
    ("translate_text", "Translates text between languages", ["How do you say 'thank you' in Japanese?", "Translate this paragraph into French."]),
    ("get_directions", "Returns a route between two places", ["How do I get from Leeds to the airport?", "Directions to the nearest pharmacy."]),
    ("order_food", "Places a food delivery order", ["Order a large pepperoni pizza to my place.", "Get me pad thai from the usual spot."]),
    ("create_reminder", "Creates a reminder at a time or place", ["Remind me to call Dr. Okafor at 5.", "Remind me to buy milk when I'm at the store."]),
    ("track_package", "Reports the delivery status of a parcel", ["Where is my order #7?", "Has the package from the Patels shipped yet?"]),
    ("control_lights", "Turns smart lights on/off or dims them", ["Dim the living room lights to 40%.", "Turn off the bedroom lights."]),
    ("find_recipe", "Finds a recipe by dish or ingredients", ["How do I make shakshuka?", "Find a recipe using chickpeas and spinach."]),
]
CHITCHAT = [
    "Hey, good morning! Hope you're doing well.",
    "Ha, that made me laugh, thanks for sharing.",
    "Just wanted to say thanks for your help earlier.",
    "What a nice day it is today.",
    "No worries, I'll figure it out myself.",
]


def build_tool_items():
    rng = random.Random(5)
    items = []
    for t in TOOLS:
        for req in t[2]:
            others = rng.sample([x for x in TOOLS if x is not t], 5)
            opts = others + [t]
            rng.shuffle(opts)
            qtext = "Which tool applies? Each option's criterion is listed; pick the single best match. Small talk needs no tool.\n" + "  ".join(f"{o[0]}: {o[1]}" for o in opts)
            items.append({"text": f"User: {req}", "queries": [q("choice", qtext, [o[0] for o in opts])], "gold": t[0]})

    rng2 = random.Random(7)
    for req in CHITCHAT:
        opts = rng2.sample(TOOLS, 6)
        labels = [o[0] for o in opts] + ["none"]
        qtext = ("Which tool applies? Each option's criterion is listed; pick the single best match. Small talk needs no tool.\n"
                 + "  ".join(f"{o[0]}: {o[1]}" for o in opts)
                 + "  none: small talk, a greeting, or anything that does not need a tool")
        items.append({"text": f"User: {req}", "queries": [q("choice", qtext, labels)], "gold": "none"})
    return items


BRANDS = {
    "Nike": ("apparel and footwear", "swoosh logo, 'Just Do It', athletic shoes and sportswear"),
    "Adidas": ("apparel and footwear", "three stripes, 'Impossible Is Nothing', sneakers and football kit"),
    "Patagonia": ("apparel and footwear", "outdoor jackets and fleece, environmental activism, repair-not-replace"),
    "Lululemon": ("apparel and footwear", "yoga leggings, 'Align' pants, athleisure"),
    "Apple": ("consumer electronics", "iPhone, Mac, 'Think different', minimalist product shots"),
    "Samsung": ("consumer electronics", "Galaxy phones, foldable screens, QLED TVs"),
    "Coca-Cola": ("food and beverage", "red can, polar bears, 'Open Happiness', 'Taste the Feeling'"),
    "McDonald's": ("food and beverage", "golden arches, 'I'm lovin' it', Big Mac, McFlurry"),
    "Starbucks": ("food and beverage", "green siren logo, Pumpkin Spice Latte, Frappuccino"),
    "Red Bull": ("food and beverage", "'gives you wings', energy drink, extreme sports stunts"),
    "Tesla": ("automotive", "Model 3, Model Y, Autopilot, Supercharger, electric cars"),
    "Toyota": ("automotive", "Corolla, Camry, Prius hybrid, 'Let's Go Places'"),
    "IKEA": ("home and furniture", "flat-pack furniture, BILLY bookcase, Swedish meatballs, blue-and-yellow store"),
    "Spotify": ("media and apps", "music streaming, Wrapped, playlists, green circle logo"),
    "Netflix": ("media and apps", "streaming series, 'Tudum', red N logo, binge-watching"),
    "Duolingo": ("media and apps", "green owl, language lessons, streak reminders"),
    "Airbnb": ("travel and retail", "'Belong Anywhere', stays and experiences, hosts"),
    "Amazon": ("travel and retail", "Prime, Prime Day, one-day delivery, smile arrow logo"),
    "Peloton": ("fitness and personal care", "connected bike and treadmill, live classes, instructors"),
    "Old Spice": ("fitness and personal care", "deodorant and body wash, 'the man your man could smell like'"),
}
Q_IS_AD = ("Is this text an advertisement or not? Each option's criterion is listed; pick the single best match.\n"
           "advertisement: promotes a product, service, or brand so the reader will buy or use it  "
           "not_an_advertisement: news, a personal message, a recipe, a notice, a job post, a support request, or other text that does not sell anything")
LABELS_IS_AD = ["advertisement", "not_an_advertisement"]
Q_BRAND = ("Which brand is this observation advertising? Each option's criterion is listed; pick the single best match.\n"
           + "  ".join(f"{b}: {c}" for b, (_, c) in BRANDS.items()))
LABELS_BRAND = list(BRANDS)

AD_OBS = [
    ("Just Do It. The new Pegasus 41 is here. Run your run.", "Nike", 1),
    ("Billboard: a single white swoosh on black, no words. Below it, a runner mid-stride at dawn.", "Nike", 1),
    ("Impossible Is Nothing. Meet the Predator Elite. Three stripes, one goal.", "Adidas", 1),
    ("Instagram caption: three stripes on the pitch, three points on the table. #adidasfootball", "Adidas", 1),
    ("Don't buy this jacket unless you need it. If you do, we'll repair it for life. Worn Wear.", "Patagonia", 1),
    ("The Align pant, now in 12 new colours. Buttery soft. Made for the mat and everything after.", "Lululemon", 1),
    ("iPhone 17 Pro. Titanium. Shot on iPhone. Available Friday.", "Apple", 1),
    ("TV spot: a single laptop on a white table rotates slowly; the only text is 'Think different.'", "Apple", 1),
    ("Galaxy Z Fold7. Unfold your world. Pre-order now and get free Galaxy Buds.", "Samsung", 1),
    ("Open Happiness. Ice-cold, in the classic red can. Share a Coke with Sam.", "Coca-Cola", 1),
    ("Creative: two polar bears clinking glass bottles of cola under the northern lights.", "Coca-Cola", 1),
    ("ba da ba ba ba, I'm lovin' it. Two Big Macs for $6 this week only.", "McDonald's", 1),
    ("Billboard: golden arches at night with the words 'Open late.' No other copy.", "McDonald's", 1),
    ("The Pumpkin Spice Latte is back. Order ahead in the app and skip the line.", "Starbucks", 1),
    ("Creative: a green siren logo on a white cup, steam rising, 'Your fall ritual.'", "Starbucks", 1),
    ("Red Bull gives you wings. Watch the cliff-diving finals live this Saturday.", "Red Bull", 1),
    ("Model Y, now from $37,990. Zero emissions. Supercharging included for 12 months.", "Tesla", 1),
    ("Let's Go Places. The 2026 Corolla Hybrid, 50 mpg combined. Now at your local dealer.", "Toyota", 1),
    ("BILLY bookcase, $59. Flat-pack it, build it, fill it. Meatballs in the restaurant while you decide.", "IKEA", 1),
    ("Creative: a tiny blue-and-yellow store bag holds an entire living room; tagline 'It fits.'", "IKEA", 1),
    ("Your 2026 Wrapped is ready. See your top songs. Premium is $0 for 3 months.", "Spotify", 1),
    ("Tudum. The final season drops October 3. Only on the service with the red N.", "Netflix", 1),
    ("Instagram caption: the owl noticed you missed your lesson. 5 minutes a day keeps the streak alive.", "Duolingo", 1),
    ("Belong Anywhere. Book a treehouse, a lighthouse, or a room in someone's home this weekend.", "Airbnb", 1),
    ("Prime Day starts Tuesday. Two days of deals, one-day delivery, and the smile arrow on every box.", "Amazon", 1),
    ("The Bike+ with auto-resistance. Live classes every hour. Your instructor is waiting.", "Peloton", 1),
    ("Look at your man, now back to me. Sadly, he isn't me. But he could smell like me. Body wash, on sale.", "Old Spice", 1),
    ("Creative: a runner ties her laces; close-up on a swoosh; text 'Your only competition is yesterday.'", "Nike", 1),
    ("Creative: a red can sweats on a picnic table; text 'Taste the Feeling.'", "Coca-Cola", 1),
    ("A green owl in a party hat: 'Your 100-day streak is one lesson away.'", "Duolingo", 1),
    ("Creative: a treadmill in a sunlit apartment, an instructor on screen shouting encouragement. 'Bring the studio home.'", "Peloton", 1),
    ("Creative: a foldable phone opens like a book to show two apps side by side. 'Two screens. One device.'", "Samsung", 1),
    ("Reuters: Tesla recalls 120,000 vehicles over a seatbelt warning defect, regulator says.", None, 0),
    ("Tweet: the rain finally stopped in Portland, going for a walk before it starts again.", None, 0),
    ("Recipe: whisk two eggs with a pinch of salt, fold in chives, cook on low heat for three minutes.", None, 0),
    ("Support ticket: my Spotify app crashes every time I open the search tab on Android.", None, 0),
    ("Public notice: wildfire smoke advisory in effect until Thursday; limit outdoor exercise.", None, 0),
    ("Job posting: senior data engineer, remote, Python and dbt required, salary range posted.", None, 0),
    ("Abstract: we study the effect of caffeine on reaction time in 40 adults over two weeks.", None, 0),
    ("Text from a friend: running late, order me whatever you're having at the coffee place.", None, 0),
]

POLICY_ITEMS = [
    {"state": "The item was purchased 45 days ago. The item is unused. The customer is asking for a cash refund.",
     "query": "Should you issue a cash refund? Answer yes if the purchase was within the last 30 days; otherwise answer no.", "gold": "no"},
    {"state": "The item was purchased 45 days ago. The item is unused. The customer is asking for a cash refund.",
     "query": "Should you issue store credit? Answer yes if the purchase was more than 30 days ago and the item is unused; otherwise answer no.", "gold": "yes"},
    {"state": "The employee submitted a leave request for 3 days off, 20 days in advance.",
     "query": "Should you approve the leave request? Answer yes if it was submitted at least 14 days in advance; otherwise answer no.", "gold": "yes"},
    {"state": "The employee submitted a leave request for 3 days off, 5 days in advance.",
     "query": "Should you approve the leave request? Answer yes if it was submitted at least 14 days in advance; otherwise answer no.", "gold": "no"},
    {"state": "The user requested a password reset and verified their identity with a one-time code sent to their registered phone.",
     "query": "Should you reset the password? Answer yes if the user verified their identity with a one-time code sent to their registered phone; otherwise answer no.", "gold": "yes"},
    {"state": "The user requested a password reset and could not provide a one-time code or any other identity verification.",
     "query": "Should you reset the password? Answer yes if the user verified their identity with a one-time code sent to their registered phone; otherwise answer no.", "gold": "no"},
    {"state": "The employee filed an expense claim for $42 with an itemised receipt attached.",
     "query": "Should you approve the expense claim? Answer yes if the amount is $75 or less and an itemised receipt is attached; otherwise answer no.", "gold": "yes"},
    {"state": "The employee filed an expense claim for $180 with an itemised receipt attached.",
     "query": "Should you approve the expense claim? Answer yes if the amount is $75 or less and an itemised receipt is attached; otherwise answer no.", "gold": "no"},
]


def build_stream_pools():
    triage = PRESETS["triage"]
    triage_q = query_text(triage["question"])
    return {
        "support": {
            "queries": [q("choice", Q_TEAM, list(TEAMS)), q("choice", Q_WANTS, list(WANTS))],
            "items": [{"text": row} for row, _tag in TICKET_ROWS],
        },
        "alert": {
            "queries": [q("choice", triage_q, triage["labels"])],
            "items": [{"text": ex["state"]} for ex in triage["examples"]],
        },
        "tool": {"items": build_tool_items()},
        "ad": {
            "queries": [q("choice", Q_IS_AD, LABELS_IS_AD), q("choice", Q_BRAND, LABELS_BRAND)],
            "items": [{"text": text} for text, _brand, _isad in AD_OBS],
        },
        "policy": {
            "items": [{"text": it["state"], "queries": [q("noul", it["query"], ["no", "yes"])]} for it in POLICY_ITEMS],
        },
    }


def decisions_for(kind, res):
    """Server result(s) for one pool item -> ticker chip list [{label, p}], matching the
    exact shaping js/demos/stream.js does for a live decide() response - kept in lockstep by
    hand (small enough not to be worth a shared-language build step)."""
    R = res["results"]
    if kind == "support":
        team, wants = R
        return [
            {"label": team["argmax"], "p": team["probs"][team["argmax"]]},
            {"label": wants["argmax"], "p": wants["probs"][wants["argmax"]]},
        ]
    if kind in ("alert", "tool"):
        r = R[0]
        return [{"label": r["argmax"], "p": r["probs"][r["argmax"]]}]
    if kind == "ad":
        is_ad, brand = R
        out = [{"label": is_ad["argmax"], "p": is_ad["probs"][is_ad["argmax"]]}]
        if is_ad["argmax"] == "advertisement":
            out.append({"label": brand["argmax"], "p": brand["probs"][brand["argmax"]]})
        return out
    if kind == "policy":
        r = R[0]
        return [{"label": "yes", "p": r["probs"]["yes"]}]
    raise ValueError(kind)


def record_stream():
    pools = build_stream_pools()
    tallies = {}
    for kind, bucket in pools.items():
        recorded = []
        ms_list = []
        default_queries = bucket.get("queries")
        for item in bucket["items"]:
            queries = item.get("queries", default_queries)
            res = record(item["text"], queries)
            recorded.append({"decisions": decisions_for(kind, res), "ms": res["ms"]})
            ms_list.append(res["ms"])
        bucket["recorded"] = recorded
        tallies[kind] = {"n": len(recorded), "mean_ms": st.mean(ms_list) if ms_list else 0}
        print(f"stream/{kind}: {len(recorded)} items, mean {tallies[kind]['mean_ms']:.0f} ms")

    # tool-select accuracy (the one pool with a clean gold answer per demo-spec item 3)
    tool_items = pools["tool"]["items"]
    tool_hits = sum(1 for it, rec in zip(tool_items, pools["tool"]["recorded"]) if rec["decisions"][0]["label"] == it["gold"])
    tallies["tool"]["accuracy"] = tool_hits / len(tool_items)
    print(f"stream/tool accuracy: {tool_hits}/{len(tool_items)}")

    (DEMOS_DIR / "stream.json").write_text(json.dumps(pools, indent=2, ensure_ascii=False) + "\n")
    (DEMOS_DIR / "stream.summary.json").write_text(json.dumps(tallies, indent=2) + "\n")
    return tallies


def main():
    self_test_hash_parity()
    t0 = time.time()
    doc20_summary = record_doc20()
    rules_summary = record_rules()
    stream_summary = record_stream()

    # merge with whatever is on disk now -- other recorders write this file too
    on_disk = json.loads(REPLAYS_PATH.read_text()) if REPLAYS_PATH.exists() else {}
    on_disk.update(replays)
    REPLAYS_PATH.write_text(json.dumps(on_disk, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {REPLAYS_PATH} ({len(on_disk)} keys total)")
    print(f"done in {time.time() - t0:.0f}s")
    print(f"\ndoc20: {doc20_summary['correct']}/20  rules: {rules_summary['flips']}/{rules_summary['n']} flip  "
          f"stream/tool: {stream_summary['tool']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
