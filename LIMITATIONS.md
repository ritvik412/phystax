# Limitations

Stated up front, so results are read for what they are.

- **Simulated universes.** The controlled arm renders 2-D scenes with an
  orthographic camera. Measurement errors here are a lower bound for real
  cameras (no perspective, depth, lens distortion or rolling shutter).
- **Real footage is passive.** On IRIS clips the agent cannot run new
  experiments; real-world results test measurement and law identification,
  not experiment design.
- **IRIS ground truth.** Only directly measured geometry (lengths, angles,
  heights) is released; damping and friction values are not. Results that use
  setup metadata (e.g. drop height) depart from the benchmark's no-metadata
  task and are reported separately.
- **Small n for agents.** Agent episodes take minutes each; agent results are
  reported with their n and with bootstrap intervals, and single episodes are
  presented as examples, not findings.
- **SAM results are run-to-run variable.** Every SAM response is cached and
  replayed, so reruns are exact; a separate repeat measures the variability.
- **Synthetic SAM stand-in.** Used only to develop and test plumbing offline.
  Results produced with it are tagged and excluded from every report.
- **Sandbox.** Agent code runs in a local subprocess; a hosted deployment
  should use a platform sandbox.
