# Submission draft: Cerebral Valley form

Edit anything that does not sound like you. Opinion questions are your call;
the drafts are suggestions based on what happened in this build.

**Team Name:** (your choice; e.g. the project name once finalised)

**Which track:** Indie Developer

**Problem statement:** Push the Frontier

**Most Viral (X post link):** (after posting the 30-second cut)

**Project Description:**
An AI scientist that discovers physical laws through a camera, and a
measurement of where it goes wrong. Muse Spark designs experiments, sees them
only through SAM 3.1 video tracking, and seals a numeric prediction in a
hash-chained ledger before the test is revealed; a deterministic grader scores
it against what really happens. In simulated universes with hidden,
non-textbook force laws, every error splits exactly into measurement,
experiment design and reasoning: through the camera, measurement is about 3% of
the error and reasoning almost all of it. We characterise SAM 3.1 as an
instrument (sub-pixel in the interior, 5–8× worse at image borders, brittle to
object names but not to box prompts, box confidence predictive of error, word
confidence not) and show that correlated tracking errors make textbook
uncertainty badly overconfident. On 7 real 4K pendulum clips from the IRIS
benchmark, exponential damping, the form its ground truth assumes, is rejected
in every clip. A live mode lets anyone record an experiment with a webcam, tap
the object and watch a sealed prediction meet reality.

**Public GitHub Repository:** https://github.com/ritvik412/phystax

**3-Minute Demo Video:** (YouTube link)

**Describe your use of Meta products:**
SAM 3.1 through the Meta Model API is the measuring instrument: video tracking
with word prompts, frame-0 box prompts and tap point prompts, with per-frame
confidence, decoded with the official meta_sam_parser and cached for exact
replay. Muse Spark 1.3 (contributor tier) is the experimentalist: tool calls to
run experiments and sandboxed fitting code, then a schema-constrained sealed
prediction; in live mode it chooses the law from the observed motion. Muse Code
was used to build and test the repository. Muse Glimmer serving is scripted but
was not run (our GPU driver predates the required CUDA version).

**Why are Meta's models important to your solution?**
The question is what a segmentation model's errors do to scientific
conclusions, so the instrument has to be a real foundation model: SAM 3.1 is
the instrument being measured, and its prompt modes and confidence are the
variables. Muse Spark makes the closed loop possible: it plans experiments,
writes and runs fitting code through tools, and commits structured predictions
that can be hashed and graded, cheaply enough (fractions of a cent per episode
on the contributor tier) to run dozens of full episodes.

**Feedback on partner technologies:** (if you used any; Hugging Face Spaces
hosts the replay site)

**How likely to build with Meta Model API:** (your rating)

**Overall opinion of Meta Model API:** (your rating)

**Recommend Meta Model API:** (your rating)

**Recommend Muse Code:** (your rating)

**Biggest obstacle:** Making SAM measurements trustworthy enough for physics:
specific object names were often not detected, word-prompt confidence turned
out to rise with frame number rather than track accuracy, and objects at the
image border bias centroids. Box and point prompts, border-frame filtering and
a calibration of box confidence fixed most of it. Running the open-weight model
locally was blocked by our GPU driver version.

**Meta's models vs others for this work:** (your view) Suggested: SAM 3.1 was
sub-pixel accurate on tracked objects and never lost an object in view, far
beyond a classical tracker under degraded video; Muse Spark followed the
tool-calling protocol reliably and was very cheap on the contributor tier, but
sometimes answered a structured-output request in prose after long tool
sessions, which we handled with retries.

**What matters most when choosing a model or tool:** (your view) Suggested:
reliability of structured outputs and tool calls, measurable accuracy, cost.

**Hardest part of switching:** (your view) Suggested: SAM's prompt and output
format (we built around its object prompts and parser).

**What would stop production use:** (your view) Suggested: run-to-run
variability and confidence semantics in SAM responses, and structured-output
consistency over long agent sessions.
