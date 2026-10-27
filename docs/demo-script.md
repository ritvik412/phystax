# Demo script

## The 3-minute video

Judges watch this live in the final round, so it carries the project. One
sentence per beat; let the screen do the explaining.

| Time | On screen | Voice-over |
|---|---|---|
| 0:00–0:12 | A ball arcs across a cluttered tabletop scene; SAM's mask follows it. | "This universe has gravity, but not ours. Our AI has to work out its laws by watching." |
| 0:12–0:35 | The question card, then experiments appearing one by one with SAM tracks and confidence. | "It designs its own experiments. It never gets coordinates, only video, measured by SAM 3.1." |
| 0:35–0:55 | The agent's fitting code and its chosen law. | "It fits candidate laws in a sandbox and keeps the one the data supports." |
| 0:55–1:15 | The sealed prediction card: hash appears. | "Then it seals a prediction for conditions it has never tried. The hash is locked before the test exists." |
| 1:15–1:40 | The test plays; red marks land on (or miss) the ball. Grade appears. | "Now the universe runs the test. The grade comes from physics, not from another model." |
| 1:40–2:20 | The results table: oracle vs camera vs camera with confidence, by camera quality; the decomposition. | "Across N universes: seeing through a camera costs this much. SAM's confidence buys back this much (or doesn't). And here is how much was the instrument versus the scientist." |
| 2:20–2:40 | Click "Verify the ledger". | "Every run is a hash-chained ledger. Anyone can check the prediction came first." |
| 2:40–3:00 | Architecture slide: SAM 3.1, Muse Spark, Spark Open, Wonder, Muse Code. (No Muse Image: it is not available in India.) | "SAM 3.1 as a measuring instrument, Muse Spark as the scientist. Code and data are open." |

Record the live segments from **replays** of good runs, so the video never
depends on network luck. Show one honest failure somewhere; it makes the
successes credible.

## Live demo (if asked)

1. Have three finished runs ready in "Past runs" (one per family).
2. Start one fresh run on `power_drag`, `sam_conf`, camera level 2. While it
   runs (a few minutes), walk through a replay.
3. When it finishes, show its grade and verify its ledger.

## Questions to be ready for

- *Why not colour thresholding?* Show the `classical` column and a frame with
  decoys, the tether line, or the occluder.
- *Couldn't it just recall physics?* The exponents and gravity are randomized
  and non-textbook per seed; show two seeds with different laws.
- *Is the grader an LLM?* No. Show `grade.py`: it is arithmetic on positions.
- *Could it peek at the answer?* Tools are listed in `tools.py`; the law hash is
  published at the start; the ledger proves commit-before-test.
- *What did you build during the event?* All of it; the repo history and the
  ledgers' timestamps show it.
