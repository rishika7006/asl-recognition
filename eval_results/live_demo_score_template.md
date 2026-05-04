# Live demo qualitative evaluation — score sheet

Fill this in as you run `python app.py`, sign each class three times, and
tally how often each backend reports the correct label.

## Setup

| field | value |
|---|---|
| date |  |
| signer |  |
| camera |  (built-in / external / iPhone) |
| `ASL_CAMERA_INDEX` |  |
| lighting |  (e.g., overhead LED, warm desk lamp, mixed) |
| distance from camera |  (approx.) |
| `MIN_CONFIDENCE` |  (default 0.45 in `app.py`) |
| `CAMERA_STRIDE` |  (default 2 in `app.py`) |

## Signs (3 attempts each)

For each cell, record one of: **Y** (correct label shown), **N** (wrong label),
**—** (no prediction surfaced — confidence below threshold or no hand detected).

Optional: write the predicted label in parentheses if you want to track common
confusions (e.g. `N (drink)`).

| class | FB-1 | FB-2 | FB-3 | RAFT-1 | RAFT-2 | RAFT-3 | FB pass | RAFT pass | notes |
|---|---|---|---|---|---|---|---|---|---|
| book |  |  |  |  |  |  | _/3 | _/3 |  |
| drink |  |  |  |  |  |  | _/3 | _/3 |  |
| computer |  |  |  |  |  |  | _/3 | _/3 |  |
| before |  |  |  |  |  |  | _/3 | _/3 |  |
| go |  |  |  |  |  |  | _/3 | _/3 |  |
| chair |  |  |  |  |  |  | _/3 | _/3 |  |
| who |  |  |  |  |  |  | _/3 | _/3 |  |
| clothes |  |  |  |  |  |  | _/3 | _/3 |  |
| candy |  |  |  |  |  |  | _/3 | _/3 |  |
| cousin |  |  |  |  |  |  | _/3 | _/3 |  |
| **TOTAL** |  |  |  |  |  |  | **_/30** | **_/30** |  |

## Recording

The file `computer_demo.mov` in this folder is a recorded run of the
`computer` sign for the report. To capture more demos: use
QuickTime Screen Recording on the browser window
showing `http://localhost:5001`, or capture only the relevant region.

## Notes / observations

(Free-form: which signs felt easiest / hardest, when the two backends
agreed vs disagreed, lighting issues, hand-detection lock issues, etc.)

-
