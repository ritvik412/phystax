# Research audit (Oct 29, 2026)

An honest account of what PhysTax actually does, what the first real API run
showed, where it stands against prior work, and the experiments that would make
it a defensible study. Verified facts, interpretations and open questions are
labelled as such.

## 1. Current state

**Runs and is tested (20 offline tests):** hidden-law simulator with two
structurally different forms per system; domain-randomized H.264 renderer;
SAM 3.1 client following Meta's reference implementation (streaming, Files API,
official parser, cache replay); concept-prompt and box-prompt measurement;
classical tracker; reference fitter with Laplace prediction intervals;
hash-chained ledger; numeric grader with interval score and coverage; budget
caps; lab UI; Glimmer serving script with provenance logs.

**Verified against the real API (Day 1, $0.07 spent):**
- Muse Spark 1.3 contributor tier is available to this account. Tool calling
  with `tool_choice` left at "auto", the tool round trip, and strict
  `json_schema` output all work.
- SAM 3.1 video: 7.5–18.5 s per 3-second clip, billed exactly 90 frames
  ($0.018), confidence values returned when objects are found.
- **Concept prompts found nothing for "ball", "planet" and "marble".** "block"
  was tracked in 90/90 frames at 0.81 px RMSE, among 13 candidate instances
  (the background contains many rectangles). On those 90 frames, confidence
  correlated *positively* with error (Spearman +0.22): a single clip with tiny
  errors, so this is not evidence either way for H2, but it is a warning.

**Not yet verified:** a single agent episode on the real API; box-prompt mode;
SAM behaviour under degradation; anything about Glimmer.

## 2. What data the experiments use, and where it lives

There is **no external dataset**. Every video is generated, deterministically
from `(system, seed, camera level, experiment parameters)`. That is a design
choice (exact ground truth, unlimited fresh universes, no contamination), and
also the main threat to external validity (section 5).

Pipeline for one experiment:

1. `sim/laws.py`, `sim/world.py`: the seed picks the hidden form and constants.
2. `sim/engine.py`: RK4 with 8 substeps per frame; positions stored at 30 fps,
   in pixels, y up.
3. `sim/render.py`: every frame is drawn at 640×480: procedural background,
   same-coloured static decoys, shaded objects; camera level adds noise, motion
   blur, downscaling, flicker, an occluding pillar and dropped frames
   (`sim/degrade.py`). Encoded with libx264, yuv420p, CRF 18, `+faststart`.
4. All frames are sent: no frame sampling. One SAM request per noun phrase
   (concept mode) or one request per clip with a frame-0 box per object
   (box mode).
5. `perception/tracks.py`: each one-bit mask is decoded with the official
   parser; the object position is the mask centroid (box centre as fallback);
   y is flipped back to world coordinates; among several instances, the one
   that moves (or the largest, for fixed bodies) is kept.
6. The agent or reference fitter sees only these positions (plus confidence
   in `sam_conf`). The test clips are graded against simulator truth.

Where things are on disk (relative to the repo root):

| Artifact | Path |
|---|---|
| Day-1 smoke clips | `runs/smoke/<system>/e00.mp4` |
| Probe clips and contact sheets | `runs/probe/` (after `scripts/probe_sam.py --yes`) |
| Episode clips (UI / benchmark) | `runs/ui/<job>/e*.mp4`, `test*.mp4`; `runs/reference/...` |
| What the agent was shown | `<run>/sandbox/experiments.json` |
| Ledger, result | `<run>/ledger.jsonl`, `<run>/result.json` |
| Frame-by-frame sheets + CSV | `<run>/inspect/` (after `scripts/inspect_run.py <run>`) |
| Raw SAM event streams (replayable) | `.cache/sam_<hash>.json` |
| Spend | `runs/spend.jsonl` |

`docs/figures/day1_smoke_frames.png` reconstructs the four Day-1 clips
(identical generator and seeds; pixel-level identity across OpenCV versions is
not guaranteed). Red crosses mark simulator truth. It shows why concept prompts
failed: the objects are flat discs that look like dots, amid same-coloured
decoys, and the projectile leaves the frame at about t = 2 s.

## 3. Assessment of the Gemini Argon session

| Claim | Verdict |
|---|---|
| Total tax = agent(SAM) − agent(oracle), split into measurement + reasoning | Incomplete. Our definition baselines on the reference fitter with exact data, and has three terms; without the design term, agent experiment choice and fitting quality are confounded. |
| Confidence weighting "cuts error by 60%" | Unsupported: no data existed. Day 1 even showed a positive confidence–error correlation on the one clip measured. |
| SAM is prompted with "red ball" / "blue circle" | Wrong: the code prompted "ball", "planet", "marble", "block"; three of four found nothing. |
| Use mask centre of mass, not box centre | Already implemented (mask centroid, box fallback). |
| Glimmer runs in 17 GB via vLLM | Wrong: the quantized 17–20 GB builds are for llama.cpp-style servers; Meta's vLLM recipe budgets about 72 GB for bf16. |
| "Nobody has measured how SAM's centroids propagate through an ODE solver"; "first benchmark" | Overstated. GAUGE (2026) already extracts SAM 3 centroid trajectories to test physical laws; IRIS and WorldBench recover physical parameters from video. What remains open is narrower (section 4). |
| Contributor tier ≈ $0.004 per episode | Unmeasured; Day 1 used $0.0004 for five tiny calls; measure on one real episode before planning. |
| Map our systems onto NewtonBench's law families | Good idea: it makes results comparable to a published benchmark (section 5, E4). |
| Fisher-information experiment design | Sound idea, known technique (optimal experimental design). As written it double-counts confidence (multiplies by c and by Σ⁻¹); use Σ_t = σ(c_t)² from calibration. Useful as a design baseline. |
| GRPO reward minus the reference error | Interesting (a privileged-information baseline, as in asymmetric actor-critic), but RL training is out of scope this week. Future work. |

## 4. Prior art and the gap

| Work | What it does | Observes | Interacts | Relevance |
|---|---|---|---|---|
| NewtonBench (ICLR 2026) | 324 law-discovery tasks, shifted laws, LLM judge + RMSLE | numbers (optional noise) | yes | interactive discovery without perception |
| DiscoverPhysics, PhysGym, MaD Physics, CausaLab | interactive discovery in simulators | numbers | yes | same |
| Pixel2Phys (CVPR 2026) | equations from video, multi-agent | video | no | perception without intervention |
| GAUGE (2026) | real motion-capture benchmark for engines and video world models; SAM 3 centroid trajectories; separates law form from parameters | video (generated) | no | uses SAM 3 as an instrument, but names perception-error attribution as future work |
| IRIS (2026) | inverse recovery of dynamics from real monocular video | real video | no | closest passive analogue; candidate real-video validation |
| WorldBench, PISA, Physics-IQ, RigidBench | physics in generated video | video | no | evaluation of generators, not of measurement |
| SAM-EM | SAM 2 fine-tuned for quantitative particle tracking (electron microscopy) | microscopy video | no | SAM as a scientific tracker, domain-specific |

**Defensible gap:** several groups now use SAM 3 as a trajectory instrument and
recover physical parameters from video, while interactive-discovery benchmarks
use clean numbers. None that we found measures how much error the segmentation
instrument itself contributes to downstream physical inference, separates it
from experiment design and reasoning, or tests whether the segmenter's own
confidence can serve as measurement uncertainty. GAUGE explicitly flags
modelling perception uncertainty as needed. That is the claim; it must stay
this narrow.

## 5. Recommended hypotheses

### Primary: SAM 3.1 as a measurement instrument for dynamics (H-A)

- **Question.** How large and how structured is SAM 3.1's position error on
  moving objects, how does it depend on prompting mode, appearance and image
  degradation, and how much does it corrupt physical-parameter estimates?
- **Falsifiable statements.**
  - A1: box-prompted tracking yields lower centroid error and higher coverage
    than concept prompting on the same clips.
  - A2: per-frame confidence is negatively rank-correlated with centroid error
    (Spearman < 0 with a bootstrap CI excluding 0). If not, confidence must not
    be used as a weight.
  - A3: the measurement tax (reference-fitter error, SAM vs oracle) exceeds the
    tax from iid noise of matched magnitude, i.e. SAM errors are structured
    (dropouts, decoy capture, blur bias), not just noise.
- **Design.** Factorial: prompt mode {concept phrases, box} × appearance
  {procedural, photo sprites} × camera level {0–3} × system {3} × seeds. Same
  clips across modes. Repeat a subset of SAM calls to measure run-to-run
  variance.
- **Controls and baselines.** Exact coordinates; matched-σ noisy coordinates;
  classical tracker.
- **Metrics.** Detection rate, coverage, centroid RMSE and bias, confidence
  calibration bins, and downstream parameter error from the reference fitter.
- **Statistics.** Paired comparisons per clip; median with 95% bootstrap CI;
  report n.
- **Physics checks.**
  - Sampling: 30 fps is far above the slowest dynamics here (spring
    frequencies ≤ about 1 Hz).
  - Motion blur: frames average positions over the next 1/30 s, so the
    centroid lags by about half a frame. Quantify this bias, then correct the
    timestamps or render centred exposures.
  - Projection: the camera is orthographic and 2D, so this is a lower bound on
    real-camera error (no perspective, no depth).
- **Cost.** About $0.02 per request: a few dollars for the full factorial.
- **Contribution.** The first error budget for using a promptable segmenter
  as a dynamics instrument, directly usable by GAUGE/IRIS-style pipelines.

### Secondary: decomposed agent study (H-B)

Agents (Muse Spark, Wonder, self-hosted Glimmer) on the same universes under
oracle / noisy / SAM-box / SAM-concept channels, decomposed into measurement,
design and reasoning; calibration of committed intervals; pass@K. Report it as
a demonstration with honest n, not as the main result, unless the budget allows
enough seeds for intervals that exclude zero.

### Threats to validity and shortcuts

- **Synthetic renders.** Validate on real video: IRIS (to verify access), or a
  short own recording of a falling ball with a ruler.
- **Leaving the frame.** Out-of-frame motion makes concept and box modes look
  worse for reasons unrelated to segmentation. The agent is now told the frame
  size, and the probe reports coverage.
- **Tests interpolate.** Test conditions lie inside the experiment ranges, so
  they reward interpolation. Add a few extrapolation tests to separate
  form-discovery from curve-fitting.
- **Textbook priors.** Continuous non-textbook constants block recall, but the
  descriptions still suggest families. Check that agents do not default to the
  textbook form; per-form results show it.

## 6. Roadmap

| Exp | What | Est. cost |
|---|---|---|
| E0 | `probe_sam.py --yes`: phrases vs box on the Day-1 clips | ~$0.40 |
| E1 | appearance fix if needed (CC0 sprites cut with SAM image mode), re-probe | ~$0.30 |
| E2 | H-A factorial (`measure_perception`, both modes) + calibration | ~$5–8 |
| E3 | reference grid, all channels, 4 seeds × 3 systems × 3 levels (measurement tax) | SAM calls cached from E2 where clips match; ~$5 |
| E4 | NewtonBench comparison: same gravity law family, numeric vs camera observation | ~$5 |
| E5 | agent grid (H-B), Spark + Glimmer, smaller Wonder grid, repeats on one level | ~$20–40 |

Freeze `run_bench protocol` before E3 and commit it.

## 7. Immediate next steps

```bash
python scripts/probe_sam.py           # shows the plan and cost
python scripts/probe_sam.py --yes     # runs it, ~$0.40
python -m phystax.budget
```

Then look at `runs/probe/*.png` and `runs/probe/probe.json`. The result decides
E1 (whether we need realistic sprites) and which SAM mode the rest of the study
uses.

## Update, Oct 29: the competitor, and a real-world arm

### What [redacted] actually is (read from its repository)

[redacted] (`[redacted]`) is a polished phone app: plan → camera
pre-flight → predict → record → explain. SAM 3.1 tracks the object (text
concept, then a box re-prompt, with a tap/point prompt as fallback), an A4 sheet
gives a pixel-to-metre homography, deterministic code fits a **known textbook
law** (pendulum, free fall, spring) and a Muse Spark agent explains the result,
with a check that every number in the answer came from a tool.

Two facts matter for positioning:
- Its headline accuracy table (20/20 runs within 2σ) comes from **rendered
  clips**, as its own `docs/ACCURACY.md` states ("rendered scenes, not real
  footage"); real-phone results are described as reported separately and are
  not in the repository.
- It measures constants of laws it already assumes. It does not ask which law
  holds, does not design experiments, and does not attribute error between the
  instrument and the analysis.

So we should not copy it. Our lane is different and stays different:
**discovering which law holds, through a camera, and attributing the error.**

### The real-world arm: IRIS

IRIS (arXiv 2603.16432, CC BY 4.0) is a released benchmark of 220 real videos
at 4K/60 fps covering eight phenomena (dropping ball, falling ball, sliding
cone, pendulum, rotating cone, and three multi-body collisions), with
independently measured ground-truth parameters and uncertainties, a
checkerboard for metric scale, and fixed 7/1/2 splits. Its task is our task on
real footage: pick the governing equation from a candidate bank, then estimate
its parameters. Its published baselines are latent-space pipelines and VLM
equation routing (≈73% selection accuracy).

Plan (passive real video; the active-experiment part stays in simulation):
1. `scripts/iris_inspect.py`: list files and sizes, fetch metadata only.
2. Convert the test-split clips of the single-body phenomena to MP4, trim, and
   run `scripts/measure_clip.py` (SAM 3.1 concept and box prompts) →
   `phystax/real/fit.py` (candidate bank + BIC selection, pendulum circle fit).
3. Report parameter error against IRIS ground truth and equation-selection
   accuracy against IRIS's published baselines, using IRIS's own metrics.
4. Then let the Muse Spark agent do the selection from the measured tracks
   (instead of from raw frames, as IRIS's VLM baselines do).

Caveats to state up front: IRIS damping "ground truth" is itself fitted from
trajectories (flagged `measurement_type: fitted`); g is the textbook value; the
video format and size must be checked before committing to it (step 1).

### Appearance is now a controlled variable

`PHYSTAX_RENDER_STYLE=lab` (default) renders a bench scene: graph-paper board,
ruler, shaded objects with contact shadows, look-alike decoys. `abstract`
keeps the original test-pattern look. Running the SAM probe under both styles
measures how much detection and accuracy depend on appearance (H-A).
`docs/figures/lab_style_frames.png` shows the new style.

## Results, Oct 29: SAM 3.1 probe (32 requests, $0.54)

Same four tracked bodies, same experiments, rendered in two styles (lab and
abstract), each measured with several noun phrases and with a frame-0 box.
Raw rows: `docs/results/probe_abstract_2026-10-29.json` (abstract run) and the
terminal log of the lab run. One clip per body and style, so these are
observations to replicate, not conclusions.

**1. Concept prompts are lexically brittle; box prompts are not.**

| Phrase | lab style | abstract style |
|---|---|---|
| "ball" (ball) | found | not found |
| "planet" (planet) | found | not found |
| "block" (block) | not found | found (13 instances) |
| "sphere" | planet only | not found |
| "marble" | not found | not found |
| "circle", "dot", "rectangle" | found every time | found every time |
| frame-0 box | 4/4 bodies | 4/4 bodies |

Specific object nouns succeeded in 4 of 12 attempts and flipped with the
rendering; generic shape words succeeded every time; boxes always worked.
The same object in the same frames is detected or missed depending on the
word, which makes concept-prompted measurement fragile in a way pixel
accuracy numbers do not show.

**2. When found, accuracy is similar across modes (0.6–2.7 px RMSE).** Box
prompts were not more precise than the best concept prompt (lab: ball 1.83 vs
1.79 px, block 1.06 vs 0.78, planet 0.69 vs 0.63, marble 2.75 vs 1.87). The box
advantage is reliability, not precision. The original claim "box beats
concept" is therefore refined: box wins on detection, not on accuracy.

**3. The confidence/error sign differs by mode.** Spearman(confidence, error)
was positive for 16 of 17 concept-prompt tracks (higher confidence, larger
error) and negative for 6 of 8 box-prompt tracks. Before reading anything into
this: frames are autocorrelated (effective n is small), the rows share videos,
and both confidence and error may simply trend over the clip.
`scripts/analyze_probe.py` reports the moving-block bootstrap CI and the
partial correlation controlling for time and edge distance. Until that is
run, raw concept-mode confidence must not be used as a weight.

**4. Error tails come from the frame edge.** For the projectile and the
marble the 95th-percentile error is *below* the RMSE, i.e. a few frames carry
large errors. Those are the frames where the object is leaving the image and
its mask is cut off, which pulls the centroid inward. Coverage of 0.7–0.8 for
these bodies is the object leaving the frame, not SAM losing it.

**Why pixel RMSE is the wrong headline.** A constant offset changes no fitted
acceleration; drift and jitter do. `analyze_probe.py` splits each error into
bias, drift and jitter, and the reference fitter's measurement tax is the
downstream number that matters.

## Results, Oct 29 (evening): frame-level analysis of the probe

`scripts/analyze_probe.py` on the cached probe responses (no API calls):

- **Concept-mode confidence is a clock.** In all 16 concept-prompt tracks,
  confidence is a perfectly monotone function of frame index
  (Spearman with time = 1.00). Controlling for time, its association with error
  vanishes (partial ρ −0.05 to +0.10). The earlier positive correlation was
  that shared trend. Concept-mode confidence must not be used as a weight.
- **Box-mode confidence is not a clock** (Spearman with time −0.55 to +0.62) and
  points the expected way in 4 of 8 tracks after controls (partial ρ −0.43,
  −0.37, −0.31, −0.19), near zero in the rest; one bootstrap CI excludes zero.
  Weak evidence; the degradation study tests it where errors vary.
- **Interior accuracy is sub-pixel**: 0.33–0.86 px; a static object jitters by
  0.02–0.03 px; moving objects 0.6–0.7 px; constant offsets ≤ 0.8 px (harmless
  for fitted accelerations).
- **Border frames carry 5–8× the error** (1.9–3.2 px vs 0.3–0.6 px). The pipeline
  now flags frames whose SAM box touches the image border (SAM's own output,
  never ground truth) and drops them in all SAM channels by default
  (`PHYSTAX_DROP_TRUNCATED=0` to keep them).
- Concept channels now prompt with shape words ("circle", "rectangle").

`phystax/bench/measure_perception.py` replicates all of this across seeds and
camera levels, both prompt modes, and writes a box-mode calibration from
interior frames only.
