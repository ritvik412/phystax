# PhysTax demo: speaker notes

Part A is what you SAY (3:00, about 470 words). Part B is what you need to
KNOW to say it with confidence. Read Part B once tonight; recite Part A on camera.

Screens you need open, in this order, in one clean browser window:
1. The Space (results site): https://huggingface.co/spaces/ritvik360/phystax
2. Local live page: http://127.0.0.1:8002/live (your 21.5% ball run result)
3. Local lab: http://127.0.0.1:8002/lab
4. The GitHub README (for the open-problem sources table)

---------------------------------------------------------------------------

## PART A: the spoken script (3:00)

### 0:00–0:15 · Hook  [Space, top of the page]
"[Your own sentence, e.g.: I wanted to know whether AI can actually do
experimental science, not just answer questions about it.]
Every team here uses SAM to measure the world. We measured SAM. And we asked:
if an AI scientist can only see the world through SAM, can it still discover
how the world works, and will it commit to predictions it can't take back?"

### 0:15–0:45 · The idea and the seal  [scroll: diagram → episode player → click
Experiment 1, then Sealed prediction, then the 'Sealed … verify' pill, then Grade]
"Here's the loop. A small universe with a hidden, made-up force law, so no model
can recall the answer. Muse Spark designs its own experiments. It only sees what
SAM 3.1 measures from the video: that blue trace. It writes fitting code, then it
seals a prediction: the law, its constants, and an eighty-percent range for every
point. That's hashed into a ledger before the test even exists. Then the universe
runs new starting conditions and grades it on what actually happens. No AI judges
another AI."

### 0:45–1:10 · Live, on real footage  [local /live: play the ball result, half speed]
"It works on real video too. I tossed a ball at my webcam and tapped it once.
That box is SAM tracking it. Muse Spark saw only the first half of the flight,
decided it was free flight, and sealed the second half before reading it. Orange
is what it predicted, grey is what the ball did. It got the law right; the error
is about twenty percent of the path. Honest, and not yet a pass."

### 1:10–1:20 · The lab  [local /lab: show the controls, click Run once]
"And this is the lab: pick a universe, a camera quality, a scientist and a seed,
and run the whole sealed experiment yourself."

### 1:20–1:45 · What we found  [Space: decomposition equation and bars, then credit bars]
"Because we know the truth, every mistake splits exactly into three parts: the
camera, the experiments, and the reasoning. Through SAM, the camera costs about
three percent of the error. The rest is reasoning: the agent chooses a cleaner
law than its own data supports. But its law still carries real physics. Give it
to a fresh forecaster and its error drops from fifty percent to nineteen."

### 1:45–2:05 · SAM as an instrument  [degradation chart, naming table, coverage rows]
"We measured SAM itself, frame by frame. It's sub-pixel accurate in the middle of
the image and five to eight times worse at the edges. Name the object, and it's
found only a third of the time; tap it, and it never misses. And because its
errors are correlated, a textbook fit becomes badly overconfident."

### 2:05–2:40 · Open problems  [Open-problem section: equations and table; flash the README table]
"Meta's research team published open problems. One: fit an ellipsoid through n
random points in d dimensions. A paper by Arslan and Muse Spark proves it works
below d squared over four and fails above, but leaves the edge itself open. We
measured that edge, and before computing larger dimensions we sealed four
predictions. My own conjecture was wrong all three times. Muse Spark's landed
inside the measured range every time. Evidence, not a proof. The second problem
is from CMU: does an AI's explanation actually help someone predict? Here it does.
Both papers are linked in the README."

### 2:40–2:50 · Real lab footage  [pendulum charts and table]
"On seven real pendulum videos from the IRIS benchmark, the damping is never the
exponential shape that benchmark assumes."

### 2:50–3:00 · Close  [back to the top]
"Labs, robots and world models increasingly measure through vision models. This
shows where those errors go, and holds AI scientists to predictions they can't
take back. The camera isn't the bottleneck. Reasoning is."

---------------------------------------------------------------------------

## PART B: what you need to know

### B1. The one-paragraph version
An AI scientist (Muse Spark) studies a universe whose physical law is hidden. It
can only observe through a camera measured by SAM 3.1. It designs experiments,
fits candidate laws with code, and then commits to a numeric prediction for new
conditions. The commitment is cryptographically sealed before the test exists.
Reality (a deterministic simulator, or a real video) grades it. Because we know
the truth, we can say exactly how much of every mistake came from the camera,
from the choice of experiments, and from the reasoning.

### B2. Why a hidden, made-up law?
Real objects follow textbook physics, so a language model can just recall
"F = ma" or "gravity is 9.8". Our universes use laws nobody has seen:
- Orbit: a marble pulled toward a planet with a = −k / r^α, where α is a random
  number between 1.3 and 3.1 (gravity would be exactly 2), or a "screened" pull
  that fades with distance.
- Projectile: a ball under gravity of an unknown strength, with air drag
  proportional to |v|^β, with β random between 1.15 and 2.6.
- Tethered block: a spring with an extra cubic term, and either viscous or dry
  friction.
The agent is never told which form is used, or even the parameter names.

### B3. What "sealed prediction" means (the core idea)
- Like a sealed envelope, or pre-registration in clinical trials: you write
  down your prediction before the experiment, so you can't adjust it afterwards.
- Technically: the prediction (law, constants, predicted positions, 80%
  intervals) is turned into text and run through SHA-256, a cryptographic hash.
  Any change, even one digit, gives a completely different hash.
- The hash goes into a ledger where each entry also contains the hash of the
  previous entry (a hash chain, the same idea as a blockchain). Changing any old
  entry breaks every later link, so tampering is detectable. "Verify the ledger"
  re-checks the chain.
- We also hash the hidden law at the very start, which proves we didn't change
  the universe after seeing the agent's answer.
- The test conditions are revealed only after the agent has finished its
  experiments, and the test video is rendered only after the seal. On the site
  you can see the exact seal time ("Sealed at 10:57:43 UTC, 7.8 s before the
  test clip existed").

### B4. How grading works (and why it can't be gamed)
- The prediction is a list of (x, y) positions at fixed times. We compare them to
  what the hidden law actually produces.
- Error = root-mean-square distance between predicted and actual positions,
  divided by the size of the path (nRMSE). Under 5% counts as a pass.
- The agent also gives an 80% range for each point. "Coverage" = how often the
  truth fell inside. Good scientists are right about 80% of the time, not 100%
  and not 20%. We score ranges with the interval score (Gneiting and Raftery), a
  "proper scoring rule": you can only minimise it by being honest about
  uncertainty.
- No language model is used for grading, ever.

### B5. The three-part error split (the "perception tax")
  total error = measurement + design + reasoning
- Measurement: how much worse a perfect fitter does with SAM's data than with
  exact coordinates (same experiments).
- Design: how much worse the perfect fitter does with the agent's chosen
  experiments than with a standard set.
- Reasoning: how much worse the agent does than the perfect fitter on the agent's
  own data. The "perfect fitter" (reference fitter) is told the true form of the
  law and only fits constants: an upper bound on what the data allow.
- These add up exactly, per episode. A test in the repo enforces that.
- Result (Muse Spark, 9 episodes through SAM box tracking): measurement 0.009,
  design 0.000, reasoning 0.226, total 0.273. So the camera is about 3% of the
  error. Example: it chose linear drag (exponent 1) when the truth was 1.21.

### B6. SAM 3.1 as a measuring instrument (432 tracks)
- Compared SAM's tracked positions with the simulator's exact positions, frame by
  frame, at four camera-quality levels (clean to noisy, blurred, low-res).
- Interior error with box prompts: 0.7, 1.5, 1.9, 2.1 pixels across levels. A
  classical tracker: 3.1, 6.5, 7.1, then collapses to 85 pixels.
- Frames where the object touches the image border have 5–8× the error (the
  mask is cut off, so its centre shifts). We detect and drop them using SAM's own
  box.
- Naming: specific words ("ball", "planet") were found in only 4 of 12 attempts;
  shape words ("circle") and a box or tap always worked.
- Confidence: box-prompt confidence tracks accuracy (rank correlation −0.61 over
  10,685 frames: confident frames are more accurate). Word-prompt confidence just
  rises with the frame number.
- Overconfidence: least squares assumes independent errors. SAM's errors persist
  from frame to frame, so a textbook fit's 80% ranges held the truth only
  17–25% of the time. Correcting for the correlation brings that to 46–58%:
  better, not solved.

### B7. The live mode (your webcam)
- Record 3–6 s, tap the object once on the first frame where it moves freely.
- SAM tracks it with a point prompt (the tap). Repeated webcam frames are
  removed (many webcams give ~15 real frames per second).
- The split is fixed at 50% in advance: the scientist sees the first half only.
  Fixed in advance so nobody can pick the split that looks best afterwards.
- It chooses one physical situation: free flight, constant velocity, uniform
  acceleration, or swinging. Laws are compared with BIC (Bayesian information
  criterion: goodness of fit with a penalty for extra parameters), and Muse Spark
  makes the final call.
- Your best run: free flight, 21.5% error. Why not a pass: about half a second of
  flight at webcam frame rates leaves few frames, and blur adds noise. Earlier
  runs failed for clear reasons (a tap while the object was still in the hand; a
  ball leaving the top of the frame). Each failure is explained, not hidden.

### B8. Predictive Credit (P2, CMU's protocol)
- Question: is the AI scientist's written law actually useful knowledge?
- A fresh Muse Spark forecaster, which never saw the experiments, predicts the
  same tests three ways: from the description only; plus the agent's sealed law;
  plus the true law fitted to the same data (the control).
- Median forecast error: 49.8% → 19.0% → 3.6%.
- Credit (how much the law helped): 20.4 points, 95% interval 3.3–28.4, from 16
  episodes; it helped in 75% of them. In the CMU paper, agents' explanations gave
  no confirmed gain; here they do.
- 9 of 57 forecaster calls returned no usable answer; reported, not hidden.

### B9. The open maths problem (P1), in plain words
- Take n random points in d dimensions (each coordinate a bell-curve random
  number). Is there an ellipsoid, centred at the origin, that passes exactly
  through all of them?
- Few points: yes, almost always. Many points: no. The switch happens around
  n = d²/4.
- Arslan with Muse Spark (Meta AI Research, 2026) proved the switch is sharp
  away from d²/4, and explicitly makes no claim exactly at the edge. Three other
  groups proved the same threshold around the same time.
- What we did: checked fits numerically with semidefinite programming (a convex
  optimisation that searches for the ellipsoid's matrix S; a fit exists if a
  positive-definite S satisfies x_i^T S x_i = 1 for every point), thousands of
  times, to find where the success rate crosses 50% (n50) and how wide the
  transition is.
- Calibrated on d = 8 to 24, then sealed predictions for d = 30, 36, 44 from:
  the plain formula d²/4; a fitted finite-size correction; Muse Spark; and our
  own conjecture d(d+1)/4 (from the "statistical dimension" of the cone of
  positive matrices).
- Measured n50: 227.5, 326.1, 490.0 (n50/d² ≈ 0.252–0.253, just above 1/4).
- Our conjecture predicted 232.5, 333, 495: outside the measured range all three
  times. Falsified, and we say so.
- Muse Spark predicted 227.9, 326.9, 486.9: inside all three times.
- About 57 minutes of compute on our server; all sealed in a ledger first.
- It is evidence about finite sizes, not a proof about the limit.

### B10. Real lab footage (IRIS benchmark, ECCV 2026)
- 7 real 4K, 60 fps pendulum videos. Period measured to 0.07–1.1%.
- The swing dies down in a shape that fits air drag (amplitude ∝ 1/(1 + κt)) or
  viscous plus dry friction, never pure exponential decay; exponential is
  rejected in all 7 (ΔBIC 118–338). IRIS derives its damping ground truth by
  assuming exponential decay.
- The full pendulum law (period grows with swing size) is preferred in 5 of 7;
  at 90° releases tracking degrades and the test fails (stated as a limitation).

### B11. Meta tools used
- SAM 3.1 (Meta Model API): video tracking with word prompts, box prompts and
  tap (point) prompts, with per-frame confidence; decoded with Meta's parser.
- Muse Spark 1.3: the scientist (tool calls, sandboxed code, sealed JSON), the
  live-mode chooser, the P1 forecaster and the P2 forecaster. About half a cent
  per full episode.
- Muse Glimmer: serving scripted, not run (GPU driver too old). Say so if asked.

### B12. Why it matters (future use and research gaps)
- World-model evaluation: GAUGE (2026) grades video generators by tracking them
  with SAM 3 and explicitly leaves the tracker's error budget to future work. We
  measure it.
- Robotics and lab automation measure through vision models; this tells them
  how far to trust those measurements.
- Every episode ends in a verifiable numeric reward, so it can become a training
  environment for scientist agents.
- Benchmarks can be audited: we found an assumption in IRIS's ground truth and
  an inconsistency in its cone metadata.

### B13. Likely judge questions, with short answers
- Why simulations? To hide laws no model could memorise, and to know the truth
  exactly. Real footage (IRIS, your webcam) checks it transfers.
- How do I know you didn't tweak results? Every prediction is hashed and chained
  before its test; anyone can re-verify. The live split is fixed at 50%.
- Is the open problem solved? No. It's sealed numerical evidence at finite size;
  the limit needs a proof.
- What's new vs NewtonBench and others? Perception in the loop with an exact
  error split, sealed predictions, a measured error budget for SAM, and real-video
  and open-problem extensions.
- Cost? About six dollars of API credits for everything reported.

---------------------------------------------------------------------------

## Recording checklist
- OBS: Window Capture of the browser, Mic/Aux on, Desktop Audio muted, 1920×1080,
  30 fps, Hybrid MP4.
- Clean window, no other tabs, bookmarks bar hidden, notifications off.
- Never show a terminal or the .env file.
- 2–3 full takes; keep the best; must be ≤ 3:00.
