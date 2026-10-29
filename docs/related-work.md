# Related work: notes for the write-up

Reading notes on the two benchmarks closest to this project, with what each
means for PhysTax. Quotes are paraphrased; see the papers for exact wording.

## IRIS (Khanbayov, Barhdadi, Serpedin, Kurban; arXiv 2603.16432, ECCV 2026)

**What it is.** A real-video benchmark for recovering physical parameters
and identifying the governing equation from monocular video. 4K at 60 fps,
eight phenomena (dropping ball, falling ball, sliding cone, pendulum, rotating
cone; hitting cones, two moving pendulums, one static and one moving
pendulum), 10 takes per setting, fixed 7/1/2 splits. The paper reports 220
videos and 22 settings; the Hugging Face release has 240 videos and 24
settings (hitting cones has three settings there).

**Task.** Per clip: (i) choose the governing ODE from a bank of candidates,
(ii) estimate its parameters, unsupervised and without metadata. The bank:
damped linear second-order oscillator, first-order exponential decay,
Torricelli, constant acceleration, coupled oscillator.

**Physical models.** Dropping ball z'' = −g (fitted in the bank's linear
damped form); falling ball filmed from above with apparent radius
r(t) = r0 f / (h0 + g t²/2); sliding cone x'' = g(sin α − μ cos α); pendulum
θ'' + ζθ' + (g/L) sin θ = 0 (no small-angle approximation); rotating cone as a
damped torsional oscillator.

**Ground truth.** Geometric quantities are measured directly. Damping and
friction are *fitted* from trajectories (pendulum damping from an exponential
envelope A(t) = A0 exp(−ζt/2); friction from μ = tan α − a/(g cos α)) and
flagged as such. The released `parameters.json` contains only the directly
measured geometry (drop height, rope length, angles, camera distances), not
the fitted damping or friction values.

**Metrics.** Parameter MAE and its spread over the 2 test clips per setting;
equation-selection accuracy with confusion matrices; identifiability (gradient
norms, ODE residual); extrapolation beyond the training horizon.

**Baselines and results.** A latent-space pipeline (encoder plus a physics
step in latent space). IRIS found an integrator bug in it that blocked
gradients to the physical parameters; even corrected, pendulum length errors
on Delfys75 are metres (e.g. 3.80 m MAE for a 0.45 m pendulum). Sliding-cone
angle is recovered exactly. Equation selection: VLM describe-then-classify
72.92% on IRIS, VLM temporal reasoning 73.33% on Delfys75; a CNN reaches 99.30%
in distribution but does not generalize to unseen phenomena.

**What it means for us.**
- Comparable numbers we can produce on real footage: pendulum length and g
  (period is scale-free), sliding-cone incline angle, and equation-family
  selection from tracked trajectories instead of raw frames.
- The damping ground truth assumes exponential (viscous) decay. Whether the
  real decay is exponential, linear (dry friction) or quadratic (air drag) is a
  question their protocol cannot answer and ours can, from the amplitude
  envelope alone.
- Using setup metadata (drop height, camera distance) departs from IRIS's
  no-metadata task definition. Results that use it must say so and be
  reported separately.
- Our first look at a Dropping_ball clip shows a top-down camera (the ball
  shrinks as it falls), which matches the paper's description of the
  *falling ball* geometry. Check Appendix B.3 ("Dropping ball vs. falling ball:
  symbol definitions") before reporting drop results.

## GAUGE (Wang et al., Shanghai AI Lab; arXiv 2608.05948)

**What it is.** A real-world benchmark of 22 task families (rigid bodies,
rope, textiles, volumetric soft bodies), about 1,560 motion-capture trials
with millimetre accuracy, and calibrated physical parameters (friction,
restitution, stiffness, moduli). It evaluates physics engines (Isaac Sim,
Genesis, Newton) against the real trajectories, and video world models
(Cosmos3, Wan, Seedance, Genie 3) by tracking what they generate.

**How it measures generated video.** SAM 3 with a text prompt, plus negative
point prompts when ambiguous, tracks the object; the per-frame mask centroid
(stored as integer pixels) gives a 2-D trajectory; a known object size gives
the metric scale. No smoothing or interpolation; missing frames stay missing.

**Metrics for world models.** Dynamic error (consistency with Newton's second
law), R², Quadratic Form Improvement (a nested F statistic: does adding a t⁴
term to s = b + k t² reduce the residual?), fitted acceleration, pendulum
period, momentum-transfer efficiency.

**Main finding.** Law form and parameter accuracy are different capabilities:
generated pendulums fit a damped sinusoid with R² = 0.99 but have periods of
1.90 s against a real 1.06 s; the best-shaped free fall has an acceleration of
0.088 m/s² instead of 9.81.

**Stated limitation.** Perception and reconstruction uncertainty must be
modelled explicitly so that perception errors are not attributed to the
model being evaluated; this is left to future work.

**What it means for us.**
- Our measurement-tax study quantifies exactly the perception term GAUGE leaves
  open, for the same instrument (SAM 3 family) and the same observables
  (centroid trajectories, fitted accelerations and periods).
- Integer-pixel centroids add a uniform rounding error of about 0.29 px RMS,
  comparable to the interior SAM error we measure (0.3–0.9 px); our pipeline
  keeps sub-pixel centroids.
- We report GAUGE's QFI alongside our BIC model selection
  (`phystax/real/fit.py: qfi`) so law-form results are directly comparable.
