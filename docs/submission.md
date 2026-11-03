# Submission draft (edit to the form's field limits)

**Title:** PhysTax: measuring what a camera costs an AI scientist

**One line:** An AI scientist discovers hidden physical laws through SAM 3.1,
seals its predictions before the test, and we measure exactly where its errors
come from: the camera, its experiments, or its reasoning.

**Track / theme:** Indie developer · Push the Frontier

**What it does:** PhysTax builds universes with hidden, non-textbook force
laws. A Muse Spark agent designs experiments, sees them only through SAM 3.1
video tracking, fits candidate laws in a sandbox, and commits a sealed
prediction (hashed into a ledger) before new test conditions are revealed. A
deterministic grader scores it against what the hidden law does. Because the
truth is known, each error splits exactly into measurement, design and
reasoning. We also characterise SAM 3.1 itself as an instrument and test the
pipeline on real 4K lab footage from the IRIS benchmark.

**What we found:** Through the camera, measurement accounts for a few percent
of the agent's error; reasoning accounts for the rest. SAM 3.1 is sub-pixel
away from image borders, 5–8× worse at them, brittle to object names but not to
box prompts; box confidence tracks error, word confidence does not; correlated
tracking errors make textbook uncertainty badly overconfident. On 7 real IRIS
pendulum clips, exponential damping, the form the benchmark assumes for its
damping ground truth, is rejected in every clip.

**How Meta models are used:** SAM 3.1 (video, word and box prompts, confidence)
is the measuring instrument. Muse Spark 1.3 is the experimentalist (tool calls,
sandboxed fitting code, schema-constrained sealed predictions). Muse Code was
used to build the repository.

**Why it matters:** Every pipeline that measures physics through a segmenter
(world-model evaluation, real-to-sim, lab automation) inherits its errors. This
gives the error budget and shows that, for current AI scientists, the
bottleneck is reasoning, not perception.

**Links:** repository, replay site (Hugging Face Space), demo video.
