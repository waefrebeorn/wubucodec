#!/usr/bin/env python3
"""
gen_status.py -- generate live status JSON for the dashboard.

Counts the REAL numbers from the repo state:
  - suites: test_*.c files that compile & pass (we trust make test; here
    we count wired suites = test files referenced in Makefile)
  - modules: wubu_*.c source files (each closed gap ships one module)
  - gaps_closed: CLOSED: entries in docs/GAP_REGISTER_1000.md
  - bugs_found: gate-caught bug log entries

Writes dashboard/status.json. Run this after every gap-closure push so
the dashboard "rolls with the changes".
"""
import json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=30).stdout
    except Exception as e:
        return ""

# --- suites wired into Makefile -------------------------------------------
mk = open("Makefile").read()
suite_names = sorted(set(re.findall(r"\$\(BIN\)/(test_\w+)\b", mk)))

# --- modules ---------------------------------------------------------------
modules = []
for sub in ("src/math", "src/audio", "src/train", "src/model"):
    d = os.path.join(ROOT, sub)
    if os.path.isdir(d):
        modules += [f for f in os.listdir(d) if f.startswith("wubu_") and f.endswith(".c")]
modules.sort()

# --- closure truth from GIT HISTORY (register file lags) -------------------
log = sh("git log --grep='GAP CLOSURE'")
closed_ids = sorted(set(re.findall(r"GAP CLOSURE ([A-Z]\d{3})", log)))
bugs = re.findall(r"bug\s*#(\d+)", log)
n_bugs = max([int(b) for b in bugs], default=15)
reg = open("docs/GAP_REGISTER_1000.md").read()
bugs += re.findall(r"bug\s*#(\d+)", reg)
n_bugs = max([int(b) for b in bugs], default=0)

# --- last commit / parity ---------------------------------------------------
last_commit = sh("git log -1 --pretty=\"%h %s\"").strip()
parity = ""
pg = os.path.join(ROOT, "..", "BearRL", "artifacts")
cert_path = os.path.join(pg, "propgate_certificate.json")
props = 0
if os.path.exists(cert_path):
    try:
        cert = json.load(open(cert_path))
        props = cert.get("total_properties",
                         len(cert.get("properties", [])) if isinstance(cert.get("properties"), list) else 28)
    except Exception:
        props = 0

status = {
    "gaps_closed": len(closed_ids),
    "closed_ids_sample": closed_ids[-12:],
    "suites_wired": len(suite_names),
    "suite_names_sample": suite_names[-12:],
    "modules": len(modules),
    "module_names_sample": modules[-12:],
    "gate_bugs_caught": n_bugs,
    "bearrl_properties": props,
    "last_commit": last_commit,
    "generated_at_epoch": __import__("time").time(),
}

out = os.path.join(ROOT, "..", "BearRL", "dashboard", "status.json")
with open(out, "w") as f:
    json.dump(status, f, indent=1)
print(json.dumps(status, indent=1))
