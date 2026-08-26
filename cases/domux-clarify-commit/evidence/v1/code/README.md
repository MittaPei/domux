# Formal v1 source archive

These three files are the exact source bytes used by the immutable `v1-formal`
evaluation. They are archived here because an upstream squash merge may omit
the fork's intermediate commits. The v2 replay validates each file against the
corresponding SHA-256 value in `../manifest.json`; Git history is informative,
not required for reproduction.

The archive is provenance-only. The v2 replay executes the current policy and
the byte-identical v1 evaluator; it does not import or execute these archived
files.
