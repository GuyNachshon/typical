"""uv run --no-sync python inference/example.py"""
from typical import Typical

STATE = (
    "Ticket #4821: Customer says their package arrived damaged. They want a replacement "
    "shipped overnight, not a refund. Order was placed 3 days ago, still under warranty."
)

m = Typical.from_pretrained("OzLabs/typical-small", device="auto")

# K-way choice over a fixed label set.
print(m.choice(STATE, "What does the customer want?", ["refund", "replacement", "repair"]))

# Yes/no.
print(m.noul(STATE, "Is the order still under warranty?"))

# Ordinal levels + expected value.
print(m.score(STATE, "How urgent is this ticket?", ["0", "1", "2", "3"]))

# Full JevBench-style decide(), with the runtime block (latency, p_null, truncation flags).
probs, runtime = m.decide(
    STATE,
    {"type": "choice", "instructions": "What does the customer want?",
     "criteria": {"refund": "money back", "replacement": "a new item shipped"}},
    ["refund", "replacement"],
)
print(probs, runtime)
