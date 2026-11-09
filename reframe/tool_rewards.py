from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

CROP_TOOL = "crop"
SELECT_TOOL = "select"
CROP_BBOX_SCALE = 1000.0
MAX_CROP_BOXES = 3
NEUTRAL_SUITABILITY = "neutral"
# OG-MOPD repairs a tool call whose gate score is below the threshold of the expected tool.
GATE_IOU = {CROP_TOOL: 0.25, SELECT_TOOL: 0.15}


@dataclass(frozen=True)
class ToolRewardBreakdown:
    total: float
    call: float
    observation: float
    penalty: float


def parse_tool_metadata(value: Any) -> Dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(
            f"tool metadata must be a dict or JSON object, got {type(value)}"
        )
    return dict(value)


def _clip_progress(progress: float) -> float:
    return min(max(float(progress), 0.0), 1.0)


def exponential_call_weight(progress: float) -> float:
    tail = math.exp(-5.0)
    return (math.exp(-5.0 * _clip_progress(progress)) - tail) / (1.0 - tail)


def _valid_bbox(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        return None
    try:
        x1, y1, x2, y2 = (float(x) for x in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in (x1, y1, x2, y2)):
        return None
    if not (
        0.0 <= x1 < x2 <= CROP_BBOX_SCALE
        and 0.0 <= y1 < y2 <= CROP_BBOX_SCALE
    ):
        return None
    return x1, y1, x2, y2


def parse_tool_bbox(value: Any) -> Optional[Tuple[int, int, int, int]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        return None
    if any(
        isinstance(x, bool) or not isinstance(x, (int, float)) for x in value
    ):
        return None
    values = tuple(float(x) for x in value)
    if not all(math.isfinite(x) and x.is_integer() for x in values):
        return None
    if _valid_bbox(values) is None:
        return None
    return tuple(int(x) for x in values)


def parse_crop_boxes(arguments: Dict[str, Any]) -> Optional[list[Tuple[int, int, int, int]]]:
    if ("bbox_2d" in arguments) == ("bboxes_2d" in arguments):
        return None
    values = [arguments["bbox_2d"]] if "bbox_2d" in arguments else arguments["bboxes_2d"]
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_CROP_BOXES:
        return None
    boxes = [parse_tool_bbox(value) for value in values]
    if any(box is None for box in boxes) or len(set(boxes)) != len(boxes):
        return None
    return boxes


def crop_token_budget() -> int:
    budget = int(os.environ.get("REFRAME_CROP_MAX_TOKEN_NUM", "1960"))
    if budget < 4:
        raise ValueError("Crops require a per-bbox visual-token budget of at least 4.")
    return budget


def validate_crop_metadata(metadata: Dict[str, Any]) -> None:
    for key in ("evidence_bboxes", "original_evidence_bboxes"):
        if key in metadata and (not metadata[key] or
                                any(parse_tool_bbox(box) is None for box in metadata[key])):
            raise ValueError(f"invalid integer 0--1000 {key}")
    targets = metadata.get("evidence_target_images")
    if targets is not None and (
        len(targets) != len(metadata.get("evidence_bboxes") or [])
        or any(type(target) is not int or target < 1 for target in targets)
    ):
        raise ValueError("invalid evidence target image IDs")
    for step in metadata.get("canonical_tool_plan") or []:
        if step.get("name") == CROP_TOOL and parse_crop_boxes(step.get("arguments") or {}) is None:
            raise ValueError(f"invalid canonical integer 0--1000 bbox arguments: {step!r}")


def canonical_plan_from_evidence(metadata: Dict[str, Any]) -> list[Dict[str, Any]]:
    if metadata.get("tool_applicable") is not True or metadata.get("evidence_available") is False:
        return []
    tool = metadata.get("preferred_tool")
    if tool == SELECT_TOOL:
        frames = metadata.get("evidence_frames")
        if not frames:
            return []
        arguments = {"target_frames": list(frames)}
    elif tool == CROP_TOOL:
        boxes = metadata.get("evidence_bboxes")
        if not boxes:
            return []
        targets = metadata.get("evidence_target_images") or [1] * len(boxes)
        if len(targets) != len(boxes) or len(set(targets)) != 1:
            raise ValueError("Canonical crop evidence must refer to one source image")
        arguments = {"target_image": targets[0],
                     "bbox_2d" if len(boxes) == 1 else "bboxes_2d":
                     list(boxes[0]) if len(boxes) == 1 else [list(box) for box in boxes]}
    else:
        return []
    return [{"name": tool, "arguments": arguments}]


def compose_bbox(
    parent_bbox: Any,
    child_bbox: Any,
) -> Optional[Tuple[float, float, float, float]]:
    parent = _valid_bbox(parent_bbox)
    child = _valid_bbox(child_bbox)
    if parent is None or child is None:
        return None
    parent_width = parent[2] - parent[0]
    parent_height = parent[3] - parent[1]
    return (
        parent[0] + child[0] / CROP_BBOX_SCALE * parent_width,
        parent[1] + child[1] / CROP_BBOX_SCALE * parent_height,
        parent[0] + child[2] / CROP_BBOX_SCALE * parent_width,
        parent[1] + child[3] / CROP_BBOX_SCALE * parent_height,
    )


def bbox_area_ratio(box: Sequence[float]) -> float:
    return _bbox_area(box) / (CROP_BBOX_SCALE ** 2)


def _bbox_area(box: Sequence[float]) -> float:
    return max(float(box[2]) - float(box[0]), 0.0) * max(
        float(box[3]) - float(box[1]), 0.0
    )


def _bbox_intersection(a: Sequence[float], b: Sequence[float]) -> float:
    width = max(min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])), 0.0)
    height = max(min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])), 0.0)
    return width * height


def bbox_iou(predicted_bbox: Any, gt_bbox: Any) -> Optional[float]:
    predicted = _valid_bbox(predicted_bbox)
    target = _valid_bbox(gt_bbox)
    if predicted is None or target is None:
        return None
    intersection = _bbox_intersection(predicted, target)
    union = _bbox_area(predicted) + _bbox_area(target) - intersection
    return intersection / union if union > 0.0 else 0.0


def _valid_frame_set(value: Any) -> Optional[set[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return None
    try:
        result = {int(x) for x in value}
    except (TypeError, ValueError):
        return None
    if any(x < 1 for x in result):
        return None
    return result


def frame_set_iou(predicted_frames: Any, gt_frames: Any) -> Optional[float]:
    predicted = _valid_frame_set(predicted_frames)
    target = _valid_frame_set(gt_frames)
    if predicted is None or target is None:
        return None
    union = len(predicted | target)
    return len(predicted & target) / union if union else 0.0


def _crop_iou_matrix(
    metadata: Dict[str, Any], call: Dict[str, Any], original: bool
) -> Optional[list[list[float]]]:
    arguments = call.get("arguments") or {}
    local = parse_crop_boxes(arguments)
    if local is None or sum(bbox_area_ratio(box) for box in local) > 0.5:
        return None
    predicted = call.get("absolute_bboxes")
    if predicted is None:
        predicted = [call["absolute_bbox"]] if call.get("absolute_bbox") is not None else local
    if len(predicted) != len(local) or any(_valid_bbox(box) is None for box in predicted):
        return None
    targets = metadata.get("original_evidence_bboxes" if original else "evidence_bboxes")
    if not targets:
        return None
    source = call.get("original_target_image", arguments.get("target_image"))
    target_ids = metadata.get("evidence_target_images") or [None] * len(targets)
    if len(target_ids) != len(targets):
        return None
    return [[(bbox_iou(box, gt) or 0.0) if target is None or source == target else 0.0
             for gt, target in zip(targets, target_ids)] for box in predicted]


def crop_match_scores(
    metadata: Dict[str, Any], call: Dict[str, Any], *, original=False
) -> Optional[Tuple[list[float], int]]:
    from scipy.optimize import linear_sum_assignment

    scores = _crop_iou_matrix(metadata, call, original)
    if scores is None:
        return None
    rows, cols = linear_sum_assignment(scores, maximize=True)
    matched = [0.0] * len(scores)
    for row, col in zip(rows, cols):
        matched[row] = scores[row][col]
    return matched, len(scores[0])


LATER_CALL_WEIGHT = 0.85


def trajectory_crop_score(
    metadata: Dict[str, Any], scored_calls: Iterable[Tuple[float, Dict[str, Any]]]
) -> Tuple[float, float]:
    """Evidence score of every crop the trajectory kept in context, and its plain coverage.

    ``scored_calls`` pairs each successful, valid crop call with its weight (1 for the first
    attempt, LATER_CALL_WEIGHT for later ones). All their boxes enter one one-to-one maximum
    matching against the GT instances with edge weight ``weight * IoU``, so an instance found by
    an early call is not erased by a later one. Boxes beyond the required instance count dilute
    the total: score = matched sum * min(1, required / all boxes). Coverage is the same matching
    without call weights or dilution, divided by the required count (a diagnostic in [0, 1]).
    """
    from scipy.optimize import linear_sum_assignment

    weighted, plain = [], []
    for weight, call in scored_calls:
        for row in _crop_iou_matrix(metadata, call, original=False) or []:
            weighted.append([weight * iou for iou in row])
            plain.append(list(row))
    if not weighted:
        return 0.0, 0.0
    required = len(weighted[0])

    def matched(matrix):
        rows, cols = linear_sum_assignment(matrix, maximize=True)
        return sum(matrix[row][col] for row, col in zip(rows, cols))

    score = matched(weighted) * min(1.0, required / len(weighted))
    coverage = min(1.0, matched(plain) / required)
    return score, coverage


def crop_best_match_score(metadata: Dict[str, Any], call: Dict[str, Any], *, original=False) -> Optional[float]:
    scores = _crop_iou_matrix(metadata, call, original)
    return max((score for row in scores for score in row), default=0.0) if scores is not None else None


def crop_evidence_score(metadata: Dict[str, Any], call: Dict[str, Any]) -> Optional[float]:
    result = crop_match_scores(metadata, call)
    if result is None:
        return None
    matched, target_count = result
    return sum(matched) * min(1.0, target_count / len(matched))


def evidence_graded(metadata: Dict[str, Any]) -> bool:
    if metadata.get("evidence_available") is False:
        return False
    return (metadata.get("tool_applicable") is True
            or metadata.get("tool_suitability") == NEUTRAL_SUITABILITY)


def tool_iou_metrics(metadata: Any, calls: Iterable[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    metadata = parse_tool_metadata(metadata)
    result: Dict[str, Optional[float]] = {"crop": None, "select": None}
    if not evidence_graded(metadata):
        return result

    calls = list(calls)
    expected_tools = {step.get("name") for step in metadata.get("canonical_tool_plan") or []}
    if not expected_tools:
        expected_tools.add(metadata.get("preferred_tool"))

    if CROP_TOOL in expected_tools and metadata.get("evidence_bboxes"):
        crops = [(1.0, call) for call in calls
                 if call.get("name") == CROP_TOOL and call.get("success", True)]
        result["crop"] = trajectory_crop_score(metadata, crops)[1]

    evidence_frames = metadata.get("evidence_frames")
    if SELECT_TOOL in expected_tools and _valid_frame_set(evidence_frames) is not None:
        select_ious = [
            iou for call in calls
            if call.get("name") == SELECT_TOOL and call.get("success", True)
            and (iou := frame_set_iou((call.get("arguments") or {}).get("target_frames"), evidence_frames)) is not None
        ]
        result["select"] = max(select_ious, default=0.0)
    return result


def evidence_score_for_call(
    metadata: Dict[str, Any],
    call: Dict[str, Any],
    expected_tool: Optional[str] = None,
    expected_call: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float]:
    if not call.get("success", True):
        return "invalid", 0.0
    name = call.get("name")
    expected_tool = (
        expected_tool
        or (expected_call or {}).get("name")
        or metadata.get("preferred_tool")
    )
    if name != expected_tool:
        return "invalid", 0.0

    arguments = call.get("arguments") or {}
    if name == CROP_TOOL:
        expected_arguments = (expected_call or {}).get("arguments") or {}
        expected_target = expected_arguments.get("target_image")
        if expected_target is not None:
            try:
                if int(arguments.get("target_image")) != int(expected_target):
                    return "invalid", 0.0
            except (TypeError, ValueError):
                return "invalid", 0.0
        score = crop_evidence_score(metadata, call)
        return ("positive", score) if score is not None else ("invalid", 0.0)
    if name == SELECT_TOOL:
        score = frame_set_iou(arguments.get("target_frames"), metadata.get("evidence_frames"))
        return ("positive", score) if score is not None else ("invalid", 0.0)
    return "invalid", 0.0


def shape_dynamic_tool_reward(
    correctness: float,
    calls: Iterable[Dict[str, Any]],
    metadata: Dict[str, Any],
    progress: float,
    output_valid: bool = True,
) -> ToolRewardBreakdown:
    calls = list(calls)
    applicable = None if metadata.get("evidence_available") is False else metadata.get("tool_applicable")
    graded = evidence_graded(metadata)
    canonical_plan = metadata.get("canonical_tool_plan") or []
    expected_tools = {step.get("name") for step in canonical_plan}
    expected_tools &= {CROP_TOOL, SELECT_TOOL}
    if not expected_tools and metadata.get("preferred_tool") in {CROP_TOOL, SELECT_TOOL}:
        expected_tools.add(metadata["preferred_tool"])

    invalid_count = int(not output_valid)
    failed_call_count = 0
    latest_by_tool: Dict[str, float] = {}
    scored_crops: list[Tuple[float, Dict[str, Any]]] = []
    has_valid_call = False
    seen_calls = set()
    for index, call in enumerate(calls):
        name = call.get("name")
        # A failed retry must not fall back to an earlier successful tool score.
        if graded and name in expected_tools:
            latest_by_tool[name] = 0.0
        if not call.get("success", True) or name not in {CROP_TOOL, SELECT_TOOL}:
            failed_call_count += 1
            continue
        if not graded:
            continue
        if name not in expected_tools:
            invalid_count += 1
            continue
        key = json.dumps(
            {"name": name, "arguments": call.get("arguments") or {},
             "absolute_bbox": call.get("absolute_bbox")},
            sort_keys=True, separators=(",", ":"),
        )
        if key in seen_calls:
            invalid_count += 1
            continue
        seen_calls.add(key)
        expected_call = next((step for step in canonical_plan if step.get("name") == name), None)
        status, score = evidence_score_for_call(
            metadata, call, expected_tool=name,
            expected_call=expected_call if len(canonical_plan) > 1 else None,
        )
        if status == "invalid":
            invalid_count += 1
        else:
            latest_by_tool[name] = score
            has_valid_call = True
            if name == CROP_TOOL:
                scored_crops.append((1.0 if index == 0 else LATER_CALL_WEIGHT, call))
    if graded and CROP_TOOL in expected_tools:
        # Crops already returned stay in context: score them all together, not just the last.
        latest_by_tool[CROP_TOOL] = trajectory_crop_score(metadata, scored_crops)[0]

    weight = exponential_call_weight(progress)
    is_video = metadata.get("preferred_tool") == "select"
    call_coefficient = 0.45 if is_video else 0.3
    penalty_coefficient = -1.2 if is_video else -1.4
    call = call_coefficient * weight if applicable is True and has_valid_call else 0.0
    if applicable is False:
        call = -0.3 * weight * len(calls)
    if SELECT_TOOL in latest_by_tool and len(calls) > 1:
        # Frame selection keeps latest-call scoring, discounted when more than one call ran.
        latest_by_tool[SELECT_TOOL] *= LATER_CALL_WEIGHT
    observation = 0.4 * sum(latest_by_tool.values()) / max(len(expected_tools), 1)
    penalty = (penalty_coefficient * float(invalid_count)
               + penalty_coefficient * float(failed_call_count))
    total = float(correctness) + call + observation + penalty
    return ToolRewardBreakdown(total, call, observation, penalty)


def branch_gate_score(
    metadata: Dict[str, Any],
    call: Dict[str, Any],
    relaxed_iou: float,
) -> float:
    arguments = call.get("arguments") or {}
    original_iou = None
    if call.get("name") == CROP_TOOL:
        # The gate uses the best match: it requires finding evidence, not covering every instance.
        best = [score for score in (crop_best_match_score(metadata, call),
                                    crop_best_match_score(metadata, call, original=True))
                if score is not None]
        return max(best, default=relaxed_iou)
    elif call.get("name") == SELECT_TOOL:
        original_iou = frame_set_iou(
            arguments.get("target_frames"), metadata.get("original_evidence_frames"))
    return max(relaxed_iou, original_iou or 0.0)


_warned_invalid_canonical = False


def should_force_canonical_branch(
    metadata: Dict[str, Any],
    candidate: Optional[Dict[str, Any]],
    canonical: Dict[str, Any],
) -> bool:
    """Repair missing, invalid, or low-IoU tool actions.

    A canonical branch that the current tool limits reject (e.g. a multi-box GT crop whose
    boxes together exceed the 50% area cap) cannot be injected: executing it would fail.
    Such samples keep the student's branch, whatever it is.
    """
    global _warned_invalid_canonical
    expected_tool = canonical.get("name")
    canonical_status, _ = evidence_score_for_call(
        metadata,
        {**canonical, "success": True},
        expected_tool=expected_tool,
        expected_call=canonical,
    )
    if canonical_status == "invalid":
        if not _warned_invalid_canonical:
            _warned_invalid_canonical = True
            print(f"[reframe-gate] canonical branch rejected by tool limits; keeping the student branch "
                  f"(reported once): {canonical}", flush=True)
        return False
    if candidate is None:
        return True
    if (
        metadata.get("composable")
        and canonical.get("name") == SELECT_TOOL
        and (candidate.get("arguments") or {}).get("target_frames")
        != (canonical.get("arguments") or {}).get("target_frames")
    ):
        return True
    candidate_status, candidate_score = evidence_score_for_call(
        metadata,
        {**candidate, "success": True},
        expected_tool=expected_tool,
        expected_call=canonical,
    )
    if candidate_status == "invalid":
        return True
    return branch_gate_score(metadata, candidate, candidate_score) < GATE_IOU[expected_tool]
