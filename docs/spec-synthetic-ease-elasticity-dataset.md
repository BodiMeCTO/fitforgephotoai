# Spec: a synthetic dataset of draped garments with ease and elasticity labels, and what it would take

Status: draft 1, 2026-11-05, written with the prototype in this repo. For Keith to decide on.
This contradicts ADR-0080 decision 4 ("a synthetic training dataset is not a current goal", held
identically in the solver, the API and the hub) and uses the solver in a way ADR-0043 forbids
(see section 7). Nothing in the other repos was changed; this document says what would have to be.

## 1. What it is for

bodi.me gives size advice from a garment's measurements per size, its Ease (the room over the
body it is meant for) and its Elasticity (its stretch), per measurement point. Two things a photo
model could add:

1. **Read ease and elasticity off a photo of a garment worn.** Trained on renders where both are
   known exactly, it would give a first guess for a garment that has no calibration yet (the
   catalogue batch of ADR-0080, step "calibrate"), and a starting point for the auto-calibration
   the virtual try-on needs (the cloth parameters a drape should use for this garment).
2. **Body measurement from a clothed photo.** `fitforgephoto` fits an Anny body to two photos of
   a person in tight clothing. A model that knows how much the garment stands off the body
   (its ease) could do the same through ordinary clothes.

The dataset is the means. It is only worth building at scale if a small one shows the labels are
readable from the pictures (section 8).

## 2. What the prototype builds (this repo)

Per sample: an ANSUR II-sampled Anny body; a GarmentCode shirt drafted for that body at an ease
offset; draped by the solver in one of its library fabrics with the solver's production drape
settings; the solver's QC and fit labels; ease read three ways (designed, pattern, worn); the
fabric's elasticity as class and numbers; the welded mesh; four rendered views with per-sample
lighting, colour and camera jitter; every setting and commit recorded. Packed as WebDataset tar
shards with an index and a card. A small image model (the probe) checks whether the labels can
be read off the renders, with whole bodies held out.

It reaches the solver and the body model only through `photoai/engine.py`. That file is the list
of what the pipeline needs from them.

## 3. What the solver would need (fitforgesolver)

- **S1. A job kind for it.** Today no kind drafts a garment for a sampled body at an ease offset,
  drapes it, labels it and renders it. Proposed: `dataset_sample` (spec: body phenotype or
  sampled target, design, ease offset, fabric, pose, views; result: the record, the welded mesh,
  the renders) or, cheaper, a `sweep` kind that takes a config and a shard and writes a run. The
  boundary law (ADR-0043) then holds: the builder calls the API, the API the solver.
- **S2. The `render` kind has no handler** (issue 2026-11-03-cntr). The deterministic rasterizer
  and the normal-map shader exist in `fitforge/render/`; the kind needs wiring, with the cameras,
  light rig, colours and background as spec fields.
- **S3. A flat-pattern girth reader that uses the body's own levels.** `patterns.drafted_chart`
  reads bust, waist and hip at 2, 20 and 42 cm below the underarm (the PatternFast sidecars'
  rule). On a drafted shirt for a 162 cm body that gave waist 63 cm and hip 63 cm against a
  body of 106 and 112: nonsense. The reader should take the levels from the draft's own
  `waist_line` / `hips_line`, or the label is "pattern ease at the bust" only, as now.
- **S4. A torso-only worn girth.** `fitlabels.compute_girth_ease` measures the garment's outer
  contour over every vertex, so sleeves hanging through the waist plane make a +4 cm shirt read
  -19 cm. The prototype's `photoai/measure.py` restricts the section to the torso panels and
  uses the tailor's blob rule; that belongs in the solver beside the other labels.
- **S5. GPU throughput, measured.** On CPU the production settings take 230 s per drape at a
  2 cm mesh; the production 1 cm mesh is about four times more. The solver's batched pack path
  drapes only the legacy cloth (not the production settings). A number for samples per hour on
  the GPU box, with the production settings and the pack path extended to them, decides
  whether "large" means 10,000 or 1,000,000.
- **S6. The elasticity axis needs calibration before it is a label.** The solver's stretch is
  calibrated for wovens (ADR-0052); bending is not calibrated for its unit-mass load
  (`backend.py` says so); knits were "set by eye". A model trained on these drapes learns the
  solver's cloth, not cloth. The cantilever and strip tests exist (`scripts/cells/`); the knit
  classes need the same against real fabric tests before the class label means a real fabric.
- **S7. Convergence.** Every drape here hit the 400-step ceiling with `converged: false`. Either
  the KE threshold is wrong for the CPU path or the budget is; a dataset of unconverged drapes
  carries that in every label. Record it (done) and decide.
- **S8. Poses.** The prototype drapes on the rest pose only. annyext's pose presets build posed
  bodies; whether a drape on `a_pose` / `arms_down` stays on the body was not tested here.
  A photo dataset needs the photo protocol's poses at least.
- **S9. Garment variety.** Two GarmentCode shirts. Trousers need the waistband hold (the old
  lane had it) and were not run. The graded-pattern archetypes (ADR-0078) are the real garments;
  drafting them at a ladder of sizes on a fixed body is the production equivalent of the ease
  ladder, with ease = chart minus body, which is exactly bodi.me's definition.
- **S10. The old dataset lane.** `fitforge/pipeline`, `surrogate`, `grading.ease_ladder`,
  `retex` are on the deletion list (issue 2026-11-03-dset). The prototype imports none of them.
  Delete them as planned; nothing here holds that up.

## 4. What the API would need (fitforgeapi)

- **A1. A `Dataset` run record**: config hash, seed, grid size, shard progress, per-sample
  status, where the files are. The batch record ADR-0080 adds for calibration is the same shape.
- **A2. A route that submits the sweep as jobs** and serves the records; shard storage on the
  solver's artifact store, paths never served off-box.
- **A3. Nothing in the API reads a photo today.** The model's output (ease and elasticity with
  uncertainty, per point) would land as one more calibration method beside "from the drapes",
  "designed ease", "fit outcomes" (ADR-0080 decision 2), with its method and rung recorded.

## 5. What the body model would need (annyextension)

- **B1. Who owns population sampling.** `annyext.ansur.sample_population` exists and uses the
  real ANSUR II CSVs (the record says `ansur2_copula`). The slider fit reaches many targets but
  flags the rest; the prototype keeps flagged bodies and records it.
- **B2. The plus-size field in a public dataset.** `extreme_weight` bodies are CAESAR-derived
  (WEAR licence). A published dataset must either exclude them or have the licence read for
  derived renders. The prototype's sampler never sets it (the sliders alone).
- **B3. Pose presets that match the photo protocol** (front A-pose, side arms down) with
  jitter: present; see S8 for the drape side.

## 6. Scale and cost

- Measured here: 230 s per sample per CPU core, 4 cores, so about 60 samples an hour. A
  thousand samples is a day of this box. The GPU number (S5) is unknown; the solver's own
  records say a production drape is seconds on the box.
- Renders are cheap (5 s for four views). Storage: about 1.2 MB per sample (four 384x512 views
  in four passes and a welded PLY).
- Grid for a first real dataset: 200 bodies x 4 designs x 6 ease steps x 4 fabrics x 2 poses =
  38,400 samples. On the GPU box at 10 s a drape that is four days; on CPU, a month.

## 7. The decisions this needs

**Keith's answers, 2026-11-05:** (1) spike only, no ADR change yet: ADR-0080 stands and the
solver's deletion of its old dataset lane goes ahead; amend only if the probe shows the labels
are readable. (2) **Add the solver job kind first** (S1, with the API route A2) before any more
samples are built: the in-process call to the solver is not to be extended. (3) The pilot runs
on the solver's GPU box, as the only warp process on it.

1. **Reverse or amend ADR-0080 decision 4** (the dataset is not a goal) if the spike result
   (section 8) warrants it. Until then this repo is a spike, not a track.
2. **ADR-0043 (the API is the only caller).** The prototype imports the solver in-process. The
   fix is S1 (a kind) plus A2 (a route); the alternative is to call this repo a solver tool
   that lives outside the law, which the law does not allow.
3. **Licences for a published dataset**: GarmentCode drafts (MIT), Anny (Apache 2.0), ANSUR II
   (public domain), the CAESAR-derived field (B2). The DXF pattern library is not touched.
   Any released model or dataset needs these read, not guessed.
4. **Which model.** Free backbones that fit a commercial use: DINOv2 (Apache 2.0) or SigLIP 2
   (Apache 2.0) features with small heads; SAM 2 (Apache 2.0) for the garment mask. Not Sapiens
   (non-commercial). Any body estimator must be Anny-based (annyextension forbids SMPL-family
   code and data). Licences to be read before use.

## 8. The spike that decides (recommended before anything above)

1. Build 2,000 to 5,000 samples on the GPU box with S4 in the solver and the rest as the
   prototype (two shirts, the ease ladder, the four fabrics, two poses).
2. Identifiability: does the probe (or DINOv2 features with a linear head) recover worn ease
   and elasticity class on held-out bodies, against the baselines? If elasticity is not
   recoverable from loose garments, say so; it is the expected result.
3. Transfer: a small real set (a few hundred photos with a tape-measured body, the garment's
   chart, and a fit verdict) is the only test that matters. Without it no claim about photos
   can be made. The comfort-window estimator's fit sessions are a possible source.
4. Go or no-go on that, with the GPU throughput number in hand.

## 9. Limits the dataset carries whatever is done

- The pictures are shaded renders: no skin, hair, texture, real light, camera or background.
  Sim-to-real is where this class of project usually fails.
- The body is rigid: a real tight garment compresses flesh, so negative ease looks different.
- Ease and body shape are tangled in a photo (ease = garment minus body). The task must be
  posed with the garment's chart or the body known; the photo supplies the other.
- Elasticity is weakly visible on a loose garment. Expect the fibre content on the label to
  beat the picture; it is the baseline to beat.
