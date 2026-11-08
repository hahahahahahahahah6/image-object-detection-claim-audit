# Documentation claim auditor

Version 0.1.0. An offline, deterministic, standard-library-only CLI for Python 3.9+
that compares documentation claims about finding and boxing things in a picture
against static source patterns. No LLM, network, telemetry, or filesystem writes.
MIT licensed.

## Install and run

```sh
python3 -m pip install .
image-object-detection-claim-audit --version
image-object-detection-claim-audit audit . --json
python3 -m image_object_detection_claim_audit audit . --verbose
image-object-detection-claim-audit audit --ignore docs/archive/ . --fail-on medium
python3 -m unittest discover
python3 -m build
```

The repository defaults to the current directory. Repeat the single-value ignore
option for multiple path prefixes. Verbose decisions and scan scope go to stderr;
JSON goes to stdout and includes verdict and message fields for every finding.
Exit codes are 0 for a pass, 2 for findings at the selected threshold, and 1 for
usage or IO errors. The default threshold is high; medium also fails test-only
claims. Each rule family yields at most one finding.

```text
documented-image-object-detection-claim-false (high; test-only: medium)
documented-no-image-object-detection-claim-false (high)
```

## Scope and matching

Scans root README.md and Markdown below docs/. Triple-backtick fenced blocks are
excluded; inline code remains prose. Matching uses case-insensitive exact
substrings, with no stemming or hyphen normalization. Denial spans are removed
first. Longest-first matching with span blanking makes each occurrence match once
under its longest phrase. No claims is a pass; a denial without contrary code is
a pass. Test fixtures under tests/ can produce a medium finding.

Every repository source file with a .py suffix is checked independently, excluding
Git metadata, bytecode caches and virtual environments. Each paired route must
occur within one module. Imports and aliases are resolved; local function or
class definitions shadow route names. Construction alone is insufficient.

```text
ultralytics.YOLO: construction + instance call or .predict()
torchvision.models.detection: fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn,
  ssd300_vgg16, fcos_resnet50_fpn: construction + instance call
DefaultPredictor: construction + instance call
pipeline: task containing "object-detection" + instance call
readNetFromDarknet: construction + .forward()
CascadeClassifier: construction + .detectMultiScale()
Generic route: e.g. x.detect_objects(...) or imported detect_objects(...)
```

## Differentiation

These neighboring tools check separate capabilities:

```text
image-classification-claim-audit
```

Assigning one label to a whole picture differs from finding and boxing things in it.

```text
image-captioning-claim-audit
```

Describing a picture differs from finding and boxing things in it.

```text
image-generation-claim-audit
```

Creating new images differs from finding things in an existing picture.

```text
image-background-removal-claim-audit
```

Cutting a subject out of a picture differs from finding and boxing it.

```text
image-colorization-claim-audit
```

Adding color to a grey picture differs from finding things in it.

## Limitations

Static pattern matching is not semantic judgement or proof of working behavior.
Exact phrases miss restatements. Strings, comments, unbound references, subprocess
commands, classification, captioning, masks, template matching, tracking, OCR,
contours and circles alone provide no source evidence.

A pipeline call without the task string below deliberately does not count:
the task is what makes it a detector.

```text
object-detection
```

A face-only cascade counts as evidence: faces are things in a picture, so finding
faces qualifies. Generic method names can describe unrelated behavior; the call
below on an unrelated receiver still counts because its method name is shared.

```text
x.detect_objects(image)
```

The exact phrase below also matches a manual annotation tool that lets users draw
boxes by hand. This is a known ambiguity. Live-video prose using the second
phrase also counts; a video detector remains a detector under this counting rule.

```text
bounding box detection tool for labelling training data
detects objects in live video
```

For a README-only self-check, copy this README into an otherwise empty directory
and audit that directory. A full source checkout contains phrase-oracle fixtures
in tests/ and therefore produces the expected medium finding.
