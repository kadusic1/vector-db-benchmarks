## General Workflow Rules

### 1. Verification Before Done
- Never mark a task complete without proving it works.
- Run lint, type check, and test commands after every implementation.
- Ask: "Would a senior engineer approve this?"

### 2. Demand Elegance
- For non-trivial changes, pause and ask: "is there a more elegant way?"
- Actively look for NumPy vectorization over `for` loops.
- Challenge your own work before presenting it.

### 3. Core Principles
- **Simplicity First**  -  make every change as simple as possible.
- **Root Causes**  -  no temporary fixes. Find and fix the root cause.
- **Minimal Impact**  -  only touch what is necessary. Avoid introducing
  bugs by changing unrelated code.

## Paper writing rules

- **Language** — written in Bosnian..
- **No em dashes (—), regular dashes (-) or colons (:).**
- **80-character line limit** — strictly enforced. Wrap long lines with
  `%` comments or `\begin{...}` blocks.
- **3rd person passive** — "It is shown", "The model was implemented",
  "The results are presented".
- **Abbreviations** — first use: full form + abbreviation in parentheses:
  `{Full Word Abbreviation} (FWA)`; thereafter just `FWA`.
  