# Build plan

Deadline: **Wed Nov 4, 9:00 PM ET = Thu Nov 5, 7:30 AM IST**. Times below are IST.
Each day has one exit criterion. If a day's criterion is not met, fix that
before moving on; later days depend on it.

| Day | Date | Work | Exit criterion |
|---|---|---|---|
| 1 | Tue Oct 27 | Add a payment method and claim Meta credits. Request Baseten and Modal credits (both have limited slots). Create the public repo and push this code. Run `scripts/day1_sam_smoke.py`. | SAM returns tracks for all four body phrases; you know the latency per clip, frames billed, and whether `c` confidence comes back. |
| 2 | Wed Oct 28 | Run `measure_perception` over levels 0–3, 3 seeds. Attend the "verifiable agents" workshop (Agents 102, 9:30 PM IST). | Figure 1: SAM vs classical error by level. A go/no-go on H2: does confidence anti-correlate with error? |
| 3 | Thu Oct 29 | First live agent episodes with Muse Spark on the `oracle` channel, all three families. Bring up Muse Spark Open on Baseten. | One full ledger per family with a passing grade on `oracle`; any tool-calling quirks fixed in `providers.py` / `loop.py`. |
| 4 | Fri Oct 30 | Camera channels with the agent. Realism pass without Muse Image (barred in India): pick 10–20 CC0 photos (tabletop textures, marbles, blocks), cut object sprites out with SAM 3.1 image segmentation, drop them in `assets/`, re-run the smoke test. Wire the Wonder reviewer. | Agent completes `sam` and `sam_conf` episodes; renders look like a lab bench, not test patterns. |
| 5 | Sat Oct 31 | Freeze and hash the protocol (`run_bench protocol`). Full reference grid. Pilot agent grid (1 seed). | `runs/protocol.json` committed; reference table complete. |
| 6 | Sun Nov 1 | Full agent grid: Spark and Spark Open (Glimmer if served). `report.py`. | Tables 1–3 filled with real data; decomposition computed. |
| 7 | Mon Nov 2 | UI polish for the demo: the sealed-prediction moment, the reveal overlay, verify button, replay of the best runs. | A judge can run one episode end to end, or replay one, without help. |
| 8 | Tue Nov 3 | Write-up in the README: results, figures, failure gallery, honest limitations. Last day for troubleshooting support. | README reads as a short paper: question, method, results, what would change our mind. |
| 9 | Wed Nov 4 | Record and edit the 3-minute video (see `demo-script.md`). Post on X for the viral prize. Submit in the evening IST, hours before the cutoff. | Submission confirmed with repo link, video link and track selection. |

## Scope rules for the ten days

- Cut first, in this order, if time runs short: Glimmer arm, Wonder reviewer,
  degradation level 1, the third family. Never cut: the sealed commit, the
  ledger, the reference fitter, the `oracle` vs `sam` vs `sam_conf` comparison.
- Do not start The Well / PDE streams. Fields are not objects, and discovering
  a PDE is a separate project.
- Every number in the README must come from `report.py` output on runs whose
  `sam_backend` is `api`.
