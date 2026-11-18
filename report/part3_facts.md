# Part 3 facts: Multi-Agent System

All facts below are read from the committed, executed `p3_agents.ipynb`, cell by cell. Each bullet names
the cell it comes from. Note: no literal "C4" item exists in Part 3's history either (Part 1's board-test
commits are C1-C4, Part 3's own fix-commits are separately numbered, e.g. C6 for the pydantic pin) --
"safety analysis" below covers the red-team results and the safety checks run on every generation, Part
3's safety-relevant measurements.

## Architecture (5 agents + orchestrator, 6 steps)

- **1. Planner**: routes a request (style_change / edit / domain_style / caption_question / unsupported),
  runs the real-person check (OpenCV Haar cascade) and the safety/refusal check before any model call. --
  `p3_agents.ipynb`, model/license table (cell 54).
- **2. Vision Understanding**: Qwen3-VL-4B-Instruct reads the input image and the request.
- **3. Prompt Engineering**: turns the vision read into a generation prompt.
- **4. Image Generation**: one diffusion pipeline loaded at a time depending on route -- SD 1.5
  (text2img/img2img) for style_change and domain_style (domain_style uses the Part 2 LoRA,
  `itzmnv/data266-lab2-sd15-pokemon-lora-r32`), instruct-pix2pix for edit.
- **5. Critique**: scores the result with CLIP, decides accept/revise, and can trigger up to 2 extra
  generation attempts.
- **6. Fairness/determinism**: the ON and OFF runs for the same scenario share the same seed and the same
  upstream state (Planner/Vision/Prompt outputs), so a difference in the final image is only ever caused
  by the critique loop itself, not by re-running an earlier agent twice. -- `part3_agents/member_manav/README.md`,
  Fairness and determinism section.

## Results: critique loop ON vs. OFF, and the four measurement types

- **CLIP score, loop ON vs. OFF** (9 generation scenarios): mean ON = **0.318**, mean OFF = **0.292**. ON
  is >= OFF on every scenario (the fairness check holds) -- identical when the first attempt already
  passes (S05, S10, S11), higher when the loop revises (S04, S06, S07-S09, S12), at a cost of 1-2 extra
  attempts each. -- `p3_agents.ipynb` cell 52.
- **Efficiency**: median seconds per request (all 40 requests) = **13.24s**. By route: caption_question
  10.39s (n=10), domain_style 13.24s (n=6), style_change 14.55s (n=7), edit 27.39s (n=6), unsupported
  0.00s (n=11, refused before any model call). -- `p3_agents.ipynb` cell 50.
- **Peak GPU memory**: **13.38 GB** across the whole run, under the 16 GB T4 limit; by stage, from 10.65 GB
  (VLM + CLIP loaded) up to 13.38 GB once a diffusion pipeline is also loaded. -- `p3_agents.ipynb` cell 52.
- **Four measurement types**:
  - Primary -- task success, single rater (judge_1/manav only, not the lab's 2-judge definition):
    **11/15 pass**.
  - Secondary -- CLIP ON vs. OFF: **0.318 vs. 0.292**; red-team breakdown: 5/10 refused by the Planner,
    1/10 produced an image, 3/10 flagged visible text as instruction-like. -- `p3_agents.ipynb` cell 54.
  - Human -- critic agreement (judge_1 vs. the system's own Accept/Revise decisions): **35% (7/20 decision
    points matched)**. judge_2 (pramod), outside rater 1: missing.
  - Efficiency -- median seconds per request: **13.24s**; peak GPU memory: **13.38 GB**.

## What worked

- **The critique loop never makes the result worse**: mean CLIP ON (0.318) >= mean CLIP OFF (0.292) on
  every one of the 9 generation scenarios, confirming the fairness/determinism design (shared seed and
  upstream state) actually isolates the critique loop's own effect. -- `p3_agents.ipynb` cell 52.
- **The refusal policy caught half the red-team set before any model call**: 5 of 10 red-team cases were
  refused by the Planner outright. -- `p3_agents.ipynb` cell 54.
- **Red-team attack success, single rater**: judge_1 found every one of the 10 red-team cases blocked
  (**0/10** succeeded) -- consistent with the Planner's own 5/10 refusal count plus the 3/10 cases where
  visible instruction-like text was flagged rather than obeyed. -- `part3_agents/member_manav/README.md`,
  human-judging summary.

## What did not work

- **The human judge agreed with the system's own critic only 35% of the time** (7 of 20 decision points):
  the system's Accept/Revise calls and judge_1's independent judgement of the same calls diverge on 13 of
  20 points -- the critic's internal self-assessment is not a reliable stand-in for a human's judgement of
  whether a revision was the right call. -- `p3_agents.ipynb` section 14 (real-scoring cell); `judge_1_critic.csv`.
- **3 of 10 red-team cases produced a partial safety gap**: the system flagged visible text in the image as
  instruction-like (neither a clean refusal nor full compliance) rather than cleanly refusing -- a
  real example of a case not safely handled by the refusal-vs-comply binary alone. -- `p3_agents.ipynb` cell 54.
- **task success (2-judge definition), judge agreement, and red-team attack success (2-judge
  definition) are not computable yet** -- they need judge_2 (pramod) and ideally outside rater 1, which
  are both missing. -- `part3_agents/member_manav/README.md`, human-judging summary.

## Safety analysis (red-team results and safety checks on every generation)

- **Red-team set**: 10 cases. Planner refused **5/10** outright; **1/10** produced an image; **3/10**
  flagged visible text in the image as instruction-like without necessarily refusing (judged by the
  red-team judging pack, since refusal is not the only safe outcome). -- `p3_agents.ipynb` cell 54.
- **Attack success, single rater (judge_1/manav)**: **0/10** -- every case judged blocked; the full
  2-judge definition (either judge flags a concern) is not computable until judge_2 rates.
- **Real-person check** (OpenCV Haar cascade) and the **diffusers safety checker** (NSFW filter) run on
  every generation route, same as Part 2's pipeline. -- `p3_agents.ipynb` cell 54, model/license table.
- **0 attempts were flagged NSFW and discarded** in this particular run (loop ON); the dynamic safety note
  is computed from the real results each run, not hardcoded, since which scenario (if any) gets flagged
  can change run to run. -- `p3_agents.ipynb` cell 54.
- **License**: SD 1.5 is CreativeML OpenRAIL-M; the Part 2 LoRA's training data is CC BY-NC-SA 4.0 and
  shows copyrighted characters, so every model, adapter, and generated image here is for non-commercial
  research only. No hosted API is used. -- `p3_agents.ipynb` cell 54.

## Note: no explicit "Limitations" section in the Part 3 README

While checking the system-card facts, I found `part3_agents/member_manav/README.md` has Workflow, Fairness
and determinism, Results, Red-team, Tools and models (licenses), and Human judging pack sections, but no
explicit "what it can't do" / limitations section the way Part 1's README has one. The gaps above (35%
critic agreement, the 3/10 partial-safety-gap red-team cases, the missing second judge) are real and
documented in this report and in the README's results, just not gathered under a single "Limitations"
heading -- flagging this rather than adding one unasked.
