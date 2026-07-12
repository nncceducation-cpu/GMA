"""Gate behaviour, printed as a table. Short clips are accepted but flagged."""
from pipeline.quality import (ABSOLUTE_MIN_DURATION_S, PROTOCOL_MIN_DURATION_S,
                              protocol_gate)

print("absolute floor: %.0fs   Prechtl standard: %.0fs\n"
      % (ABSOLUTE_MIN_DURATION_S, PROTOCOL_MIN_DURATION_S))
print("%8s %7s %11s  %s" % ("duration", "pass", "compliant", "message"))
for d in (3, 5, 20, 45, 60, 90, 400):
    r = protocol_gate(14.0, d)
    msg = (r["blocking"] + r["warnings"] + ["ok"])[0]
    print("%7.0fs %7s %11s  %s" % (d, r["pass"], r["protocol_compliant"], msg[:64]))

r = protocol_gate(4.0, 90)
print("\nage 4w / 90s -> pass=%s | %s" % (r["pass"], r["blocking"][0][:64]))
r = protocol_gate(30.0, 90)
print("age 30w / 90s -> pass=%s | %s" % (r["pass"], r["blocking"][0][:64]))
