"""Gate behaviour, printed as a table. Short clips are accepted but flagged."""
import sys
sys.path.insert(0, "/app")

from pipeline.quality import (ABSOLUTE_MIN_DURATION_S, PROTOCOL_MIN_DURATION_S,
                              protocol_gate)

print("absolute floor %.0fs | Prechtl standard %.0fs"
      % (ABSOLUTE_MIN_DURATION_S, PROTOCOL_MIN_DURATION_S))
print("%9s %6s %10s  %s" % ("duration", "pass", "compliant", "message"))
for d in (3, 5, 20, 45, 60, 90, 400):
    r = protocol_gate(14.0, d)
    msg = (r["blocking"] + r["warnings"] + ["ok"])[0]
    print("%8.0fs %6s %10s  %s" % (d, r["pass"], r["protocol_compliant"], msg[:62]))

for age in (4.0, 30.0):
    r = protocol_gate(age, 90)
    print("age %-5.0fw 90s  pass=%-5s  %s" % (age, r["pass"], r["blocking"][0][:62]))
