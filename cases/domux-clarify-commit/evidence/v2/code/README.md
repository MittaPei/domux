# Exploratory v2 source archive

These files preserve the exact v2 bytes that were later superseded by the
post-v2 negated-selector hardening. Their SHA-256 values and sizes match the
corresponding entries in `../code_freeze.json`.

The other twelve files in the v2 content-addressed bundle remain unchanged in
the case root. `reproduce_v2.py` stages this archive over a temporary copy so
the recorded v2 result remains reproducible without fork-only Git history.
