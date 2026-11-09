from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import sys
import threading
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

_DECORD_LOCK = threading.Lock()
# Serialize the shared Rust tokenizer backend across rollout worker threads.
_TOKENIZER_LOCK = threading.RLock()


class _LockedTokenizer:
    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __call__(self, *args, **kwargs):
        with _TOKENIZER_LOCK:
            return self._inner(*args, **kwargs)

    def __getattr__(self, name):
        value = getattr(self._inner, name)
        if not callable(value):
            return value

        def locked(*args, **kwargs):
            with _TOKENIZER_LOCK:
                return value(*args, **kwargs)
        return locked

from PIL import Image

from swift.agent_template.qwen3_coder import Qwen3_5AgentTemplate
from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset
from swift.infer_engine.protocol import RolloutOutput
from swift.rewards import ORM, orms
from swift.rollout import MultiTurnScheduler, multi_turns

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from reframe.answer_matching import match_document_answer, unescape_latex_punctuation
from reframe.tool_rewards import (
    CROP_BBOX_SCALE,
    CROP_TOOL,
    MAX_CROP_BOXES,
    parse_crop_boxes,
    crop_token_budget,
    SELECT_TOOL,
    bbox_area_ratio,
    branch_gate_score,
    compose_bbox,
    evidence_score_for_call,
    parse_tool_metadata,
    canonical_plan_from_evidence,
    shape_dynamic_tool_reward,
    should_force_canonical_branch,
    validate_crop_metadata,
)


REFRAME_ROOT = os.environ.get("REFRAME_DATA_ROOT") or os.path.join(REPO_ROOT, "data")
TOOL_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
BOXED_START_PATTERN = re.compile(r"\\boxed\s*\{")
QWEN35_AGENT_TEMPLATE = Qwen3_5AgentTemplate()
MAX_VISUAL_OPERATIONS = 2
MAX_CROP_AREA_RATIO = 0.5
MAX_CROP_ASPECT_RATIO = 150.0
MAX_SELECTED_FRAME_RATIO = 0.5
ORIGINAL_DROP_PROB = 0.5


def _drop_notice(has_video: bool) -> str:
    notice = "\nOriginal visual inputs are dropped and cannot be targeted by tools anymore. Use the stable IDs of returned images"
    return notice + ("; select may only narrow the previous selection." if has_video else ".")


def _tool_instructions(mode: str) -> str:
    if mode == "image":
        return (
            "Answer the question using the images. Cropping is optional: use it only when "
            "a closer view would clarify relevant details. "
            f"Use at most {MAX_VISUAL_OPERATIONS} crop calls. "
            "Use stable 1-based image IDs: original images are numbered in input order; "
            "crops have the IDs reported by the tool. Each bbox is [x_min,y_min,x_max,y_max] "
            "in integer 0–1000 coordinates of the target image: (0,0) is top-left and "
            "(1000,1000) is bottom-right. "
            f"Use bbox_2d for one box or bboxes_2d for up to {MAX_CROP_BOXES} distinct boxes, not both. "
            f"Their total area must be at most {100 * MAX_CROP_AREA_RATIO:g}% of the target image. "
            "A multi-box call counts as one call. Avoid redundant boxes. "
            "Put only the final answer in \\boxed{...}, exactly once, with no text after it: "
            "an option letter for multiple choice, otherwise the shortest sufficient answer."
        )
    common = (
        "Reason about the visual input and answer the question. Visual tools are optional. "
        f"Use at most {MAX_VISUAL_OPERATIONS} visual operations. Call a tool only when a more focused spatial or "
        "temporal view is likely to improve grounding by concentrating relevant evidence and "
        "excluding distractors. If additional focusing is unlikely to improve reliability, "
        "answer directly without a tool call. Preserve enough context to answer the full question; "
        "do not minimize the selected area or frame count at the expense of relevant evidence."
    )
    crop_guidance = (
        " For image or multi-image input, use crop only when the "
        "question-relevant evidence is spatially localized and focusing it is likely "
        "to clarify fine details or reduce distracting context. Include a small spatial margin "
        "and any nearby objects or labels needed to interpret the evidence."
    )
    select_guidance = (
        " For video input, use select only when a smaller subset of frames would "
        "provide more relevant temporal evidence. Keep every relevant event segment with a "
        "neighboring frame before and after it when available; do not fill unrelated gaps."
    )
    protocol = (
        " When a tool is needed, follow the provided Qwen3.5 tool-call protocol, choose "
        "precise arguments, and continue reasoning from the returned visual evidence. "
        "End the response with exactly one final answer in \\boxed{...}, with no text after it. "
        "For multiple-choice questions, put only the option letter inside the box; for other "
        "questions, put only the shortest sufficient answer, without repeating the question."
    )
    crop = (
        ' crop arguments are {"target_image": <1-based integer>, '
        '"bbox_2d": [x_min,y_min,x_max,y_max]}, or replace bbox_2d with '
        f'"bboxes_2d": [[x_min,y_min,x_max,y_max], ...] for up to {MAX_CROP_BOXES} distinct boxes. '
        "A multi-box call returns one image per box and counts as one operation. "
        "Each returned crop has its own visual-token budget. "
        "Do not include unrelated or redundant boxes; these are penalized. "
        "Coordinates use the integer 0-1000 "
        "coordinate grid: (0,0) is top-left and (1000,1000) is bottom-right; "
        f"all boxes in one call must together retain at most {100 * MAX_CROP_AREA_RATIO:g}% "
        "of the source image area. "
        "Use the stable image IDs reported by tools. For a video with no image inputs or "
        "returned images yet, target_image indexes the original candidate frames."
    )
    select = (
        ' select arguments are {"target_frames": [<sorted unique 1-based frame indices>]}; '
        f"each call must remove at least {100 * (1 - MAX_SELECTED_FRAME_RATIO):g}% of the current candidates. "
        "Keep original candidate-frame indices on later calls, and select a strict subset of "
        "the previous selection."
    )
    if mode == "video":
        return common + select_guidance + protocol + select
    return common + crop_guidance + select_guidance + protocol + select + crop


def _tool_schemas(mode: str) -> List[Dict[str, Any]]:
    crop = {
        "type": "function",
        "function": {
            "name": CROP_TOOL,
            "description": (
                "Zoom in on spatially localized evidence using an integer 0-1000 "
                "relative-coordinate box when a focused view is likely to clarify fine "
                "details or reduce distracting context. Keep a small margin and the context "
                "needed to interpret the evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": (
                            "[x_min,y_min,x_max,y_max] on the 0-1000 coordinate grid, where "
                            "(0,0) is top-left and (1000,1000) is bottom-right. The box must "
                            f"retain at most {100 * MAX_CROP_AREA_RATIO:g}% of the source image area."
                        ),
                    },
                    "target_image": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Stable 1-based image ID; use 1 for an original image and the reported "
                            "IDs for tool-returned images. For a video without image inputs or "
                            "returned images, use original candidate-frame indices."
                        ),
                    },
                },
                "required": ["bbox_2d", "target_image"],
            },
        },
    }
    crop["function"]["parameters"]["properties"]["bboxes_2d"] = {
        "type": "array",
        "items": deepcopy(crop["function"]["parameters"]["properties"]["bbox_2d"]),
        "minItems": 1,
        "maxItems": MAX_CROP_BOXES,
        "uniqueItems": True,
        "description": (
            "Distinct boxes on the same target image; their areas must sum to at most "
            f"{100 * MAX_CROP_AREA_RATIO:g}% of that image."
        ),
    }
    crop["function"]["parameters"]["required"] = ["target_image"]
    crop["function"]["parameters"]["oneOf"] = [
        {"required": ["bbox_2d"]}, {"required": ["bboxes_2d"]},
    ]
    if mode == "image":
        crop["function"]["parameters"]["properties"]["target_image"]["description"] = (
            "Stable 1-based image ID: original images are numbered in input order; "
            "use the reported IDs for returned crops."
        )
    select = {
        "type": "function",
        "function": {
            "name": SELECT_TOOL,
            "description": (
                "Select all relevant event segments with neighboring safety frames, "
                "while excluding unrelated gaps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_frames": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "minItems": 1,
                        "uniqueItems": True,
                        "description": (
                            "Sorted unique 1-based frame indices. Each call must remove at least "
                            f"{100 * (1 - MAX_SELECTED_FRAME_RATIO):g}% of the current candidates. "
                            "Later selections must be strict subsets and retain original frame indices."
                        ),
                    }
                },
                "required": ["target_frames"],
            },
        },
    }
    if mode == "image":
        return [crop]
    if mode == "video":
        return [select]
    return [select, crop]


class ReframeDatasetPreprocessor(RowPreprocessor):
    def __init__(self, mode: str, media_root: Optional[str] = None):
        super().__init__()
        self.mode = mode
        self.media_root = media_root
        if mode in {"video", "mix"}:
            from datasets import List as HfList
            from datasets import Value

            self.arrow_feature_overrides = {
                "videos": HfList(HfList(Value(dtype="large_string"))),
            }

    def __call__(self, dataset, **kwargs):
        dataset = super().__call__(dataset, **kwargs)
        if self.mode == "image":
            return dataset
        if not any(dataset["tool_video_path"]):
            return dataset
        return dataset.with_transform(self._decode_raw_video_batch)

    @staticmethod
    def _decode_raw_video_batch(batch):
        import decord

        for row_index, video_path in enumerate(batch["tool_video_path"]):
            if not video_path:
                continue
            frame_numbers = json.loads(batch["tool_video_frame_numbers"][row_index])
            reader = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=2)
            indices = [int(index) for index in frame_numbers]
            if any(index < 0 or index >= len(reader) for index in indices):
                raise ValueError(
                    f"sampled frame number outside 0..{len(reader) - 1}: {video_path}"
                )
            decoded = reader.get_batch(indices).asnumpy()
            max_tokens = int(os.environ.get("VIDEO_MAX_TOKEN_NUM", "128"))
            batch["videos"][row_index] = [[
                _resize_to_token_budget(Image.fromarray(frame), max_tokens)
                for frame in decoded
            ]]
        return batch

    @staticmethod
    def _answer(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return str(value[0]) if value else ""
        return "" if value is None else str(value)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        question = str(row.pop("question"))
        solution = self._answer(row.pop("answer"))
        media = row.pop("image")
        if self.media_root:
            paths = list(media) if isinstance(media, (list, tuple)) else [media]
            media = [
                os.path.join(self.media_root, path)
                if isinstance(path, str) and not os.path.isabs(path)
                and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", path)
                else path for path in paths
            ]
        is_video = bool(row.get("is_video", self.mode == "video"))
        metadata = parse_tool_metadata(row.get("tool_metadata"))
        validate_crop_metadata(metadata)

        row["messages"] = [
            {"role": "system", "content": _tool_instructions(self.mode)},
            {"role": "user", "content": ("<video>" if is_video else "<image>") + question},
        ]
        row["solution"] = solution
        row["tools"] = _tool_schemas(self.mode)
        row["tool_metadata"] = json.dumps(metadata, separators=(",", ":"))
        row["is_video"] = is_video
        row["qid"] = str(row.get("qid", ""))

        row["images"] = None
        row["videos"] = None
        if self.mode == "image":
            row["tool_video_path"] = None
            row["tool_video_frames"] = None
            row["tool_video_frame_numbers"] = None
        else:
            row["tool_video_path"] = ""
            row["tool_video_frames"] = "[]"
            row["tool_video_frame_numbers"] = "[]"
        if is_video:
            frames = list(media) if isinstance(media, (list, tuple)) else [media]
            if metadata.get("video_media_format") == "video":
                if len(frames) != 1:
                    raise ValueError("raw video rows must contain exactly one media path")
                frame_numbers = [
                    int(index) for index in metadata["video_sampled_frame_numbers"]
                ]
                row["tool_video_path"] = str(frames[0])
                row["tool_video_frame_numbers"] = json.dumps(
                    frame_numbers, separators=(",", ":")
                )
                frames = []
                frame_count = len(frame_numbers)
            else:
                row["tool_video_frames"] = json.dumps(frames, separators=(",", ":"))
                frame_count = len(frames)
            row["videos"] = [frames]
            row["messages"][0]["content"] += (
                f" The candidate timeline has {frame_count} frames, numbered 1..{frame_count} "
                "in order. target_frames uses these candidate indices, not seconds or raw-video frame numbers."
                " Time markers such as <1.2 seconds> are synthetic, spacing candidate frames 0.5 seconds "
                "apart; they are not real video time."
            )
        else:
            row["images"] = list(media) if isinstance(media, (list, tuple)) else [media]

        preferred = metadata.get("preferred_tool")
        row["teacher_route"] = "video" if is_video or preferred == SELECT_TOOL else "image"
        return row


def _register_reframe_dataset(name: str, path: str, mode: str) -> None:
    register_dataset(
        DatasetMeta(
            dataset_name=name,
            dataset_path=path,
            preprocess_func=ReframeDatasetPreprocessor(
                mode, media_root=os.environ.get("REFRAME_MEDIA_ROOT") or os.path.dirname(os.path.abspath(path))),
        ),
        exist_ok=True,
    )


for _mode in ("image", "video", "mix"):
    _register_reframe_dataset(f"reframe-{_mode}", os.path.join(REFRAME_ROOT, f"{_mode}.parquet"), _mode)


def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    match = TOOL_PATTERN.search(text or "")
    if match is None:
        return None
    try:
        functions = QWEN35_AGENT_TEMPLATE.get_toolcall(match.group(0))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not functions:
        return None
    function = functions[0]
    try:
        arguments = json.loads(function.arguments)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(arguments, dict):
        return None
    for key, value in arguments.items():
        if not isinstance(value, str):
            continue
        try:
            arguments[key] = json.loads(value)
        except json.JSONDecodeError:
            arguments[key] = value
    return {"name": function.name, "arguments": arguments}


def _truncate_after_first_tool_call(text: str) -> Tuple[str, int]:
    matches = list(TOOL_PATTERN.finditer(text or ""))
    if not matches:
        return text or "", 0
    return (text or "")[:matches[0].end()], max(0, len(matches) - 1)


def _format_tool_call(call: Dict[str, Any]) -> str:
    content = json.dumps(
        {"name": call["name"], "arguments": call["arguments"]},
        ensure_ascii=False,
    )
    return QWEN35_AGENT_TEMPLATE._format_tool_calls([{"content": content}])


def _replace_last_tool_call(text: str, replacement: str) -> Tuple[str, Tuple[int, int]]:
    matches = list(TOOL_PATTERN.finditer(text or ""))
    if matches:
        match = matches[-1]
        result = text[:match.start()] + replacement + text[match.end():]
        return result, (match.start(), match.start() + len(replacement))
    prefix = text.rstrip()
    separator = "\n" if prefix else ""
    result = prefix + separator + replacement
    return result, (len(prefix) + len(separator), len(result))


def _to_pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        value = value.get("path")
    if isinstance(value, str) and value.startswith("data:"):
        value = value.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    raise TypeError(f"unsupported image value: {type(value)}")


def _to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _resize_image_to_token_budget(value: Any, max_tokens: int) -> Image.Image:
    image = _to_pil(value)
    max_pixels = max_tokens * (16 * 2) ** 2
    if image.width * image.height > max_pixels:
        scale = (max_pixels / (image.width * image.height)) ** 0.5
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


def _resize_to_token_budget(value: Any, max_tokens: int) -> str:
    return _to_data_url(_resize_image_to_token_budget(value, max_tokens))


def _visual_grid(width: int, height: int, max_tokens: int) -> Tuple[int, int]:
    from qwen_vl_utils import vision_process

    factor = 16 * 2
    min_tokens = vision_process.IMAGE_MIN_TOKEN_NUM
    try:
        rows, cols = (side // factor for side in vision_process.smart_resize(
            height, width, factor=factor, min_pixels=min_tokens * factor ** 2,
            max_pixels=max_tokens * factor ** 2))
        if rows >= 1 and cols >= 1 and min_tokens <= rows * cols <= max_tokens:
            return rows, cols
    except ValueError:
        pass
    candidates = [
        (rows, max(1, min(max_tokens // rows, round(rows * width / height))))
        for rows in range(1, max_tokens + 1)
    ]
    candidates = [(rows, cols) for rows, cols in candidates
                  if rows * cols >= min_tokens and max(rows, cols) / min(rows, cols) < 150]
    if not candidates:
        raise ValueError(f"no visual grid fits {max_tokens} tokens")
    return min(candidates, key=lambda grid: (abs(math.log(grid[1] / grid[0] * height / width)),
                                             -grid[0] * grid[1]))


def _resize_to_visual_grid(value: Any, max_tokens: int) -> Tuple[str, int]:
    image = _to_pil(value)
    rows, cols = _visual_grid(image.width, image.height, max_tokens)
    size = (cols * 32, rows * 32)
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return _to_data_url(image), rows * cols


class ObservationGatedScheduler(MultiTurnScheduler):
    max_operations = MAX_VISUAL_OPERATIONS
    branch_aware = True

    @property
    def tokenizer(self):
        inner = MultiTurnScheduler.tokenizer.fget(self)
        if inner is None or int(os.environ.get("SWIFT_ROLLOUT_STEP_WORKERS", "1") or 1) <= 1:
            return inner
        return _LockedTokenizer(inner)

    async def on_trajectory_start(self, requests) -> None:
        for request in requests:
            metadata = parse_tool_metadata(request.data_dict.get("tool_metadata"))
            validate_crop_metadata(metadata)
            nested_video = request.videos[0] if request.videos else []
            frames = list(nested_video) if isinstance(nested_video, (list, tuple)) else []
            tool_video_frames = request.data_dict.get("tool_video_frames")
            tool_video_frame_numbers = request.data_dict.get("tool_video_frame_numbers")
            if isinstance(tool_video_frames, str):
                tool_video_frames = json.loads(tool_video_frames)
            if isinstance(tool_video_frame_numbers, str):
                tool_video_frame_numbers = json.loads(tool_video_frame_numbers)
            request.data_dict["_reframe_state"] = {
                "metadata": metadata,
                "tools_enabled": bool(request.data_dict.get("tools_enabled", True)),
                "training": bool(request.data_dict.get("reframe_training", False)),
                "calls": [],
                "cursor": 0,
                "frames": frames,
                "tool_video_path": request.data_dict.get("tool_video_path"),
                "tool_video_frames": tool_video_frames,
                "tool_video_frame_numbers": tool_video_frame_numbers,
                "selected": None,
                "tool_visual_tokens_used": 0,
                "image_sources": [
                    {"value": value, "bbox": [0, 0, 1000, 1000], "original_target_image": index}
                    for index, value in enumerate(request.images, 1)
                ],
                "branch_masks": [],
                "teacher_routes": [],
                "default_teacher_route": request.data_dict.get("teacher_route", "image"),
            }

    @staticmethod
    def _state(request) -> Dict[str, Any]:
        return request.data_dict["_reframe_state"]

    def _canonical(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        metadata = state["metadata"]
        if (not state.get("training", False) or metadata.get("tool_applicable") is False
                or metadata.get("evidence_available") is False):
            return None
        plan = metadata.get("canonical_tool_plan")
        if plan is None:
            plan = canonical_plan_from_evidence(metadata)
        cursor = state["cursor"]
        return deepcopy(plan[cursor]) if cursor < len(plan) else None

    async def on_turn_end(self, infer_request, response_choice, current_turn: int) -> Dict[str, Any]:
        state = self._state(infer_request)
        content = response_choice.message.content or ""
        truncated, overflow = _truncate_after_first_tool_call(content)
        info = {
            "tool_call_boundary_truncated_turns": state.get("boundary_truncated_turns", 0),
            "tool_call_retokenized_turns": state.get("retokenized_turns", 0),
            "output_invalid_reasons": _output_invalid_reasons(content, response_choice.finish_reason),
        }
        if truncated == content:
            return {"rollout_infos": info}

        raw_ids = list(response_choice.token_ids or [])
        prefix_length = self._generation_prefix_length(content, raw_ids)
        token_ids = self.tokenizer.encode(truncated[prefix_length:], add_special_tokens=False)
        sampled_prefix = token_ids == raw_ids[:len(token_ids)]
        logprobs = self._extract_logprobs_from_choice(response_choice)
        if sampled_prefix and len(logprobs) == len(raw_ids) and response_choice.logprobs is not None:
            response_choice.logprobs = deepcopy(response_choice.logprobs)
            response_choice.logprobs["content"] = response_choice.logprobs["content"][:len(token_ids)]
        else:
            response_choice.logprobs = None
        response_choice.message.content = truncated
        response_choice.token_ids = token_ids
        if infer_request.messages and infer_request.messages[-1].get("role") == "assistant":
            infer_request.messages[-1]["content"] = truncated
        state["boundary_truncated_turns"] = state.get("boundary_truncated_turns", 0) + 1
        state["retokenized_turns"] = state.get("retokenized_turns", 0) + int(not sampled_prefix)
        info.update({
            "tool_call_boundary_truncated": True,
            "tool_call_overflow_count": overflow,
            "tool_call_boundary_truncated_turns": state["boundary_truncated_turns"],
            "tool_call_retokenized_turns": state["retokenized_turns"],
            "output_invalid_reasons": _output_invalid_reasons(truncated, response_choice.finish_reason),
        })
        return {"rollout_infos": info}

    def _response_prefix_length(self, content: str, raw_ids: List[int]) -> int:
        decode = getattr(self.tokenizer, "decode", None)
        if decode is None:
            return 0
        for drop in range(min(3, len(raw_ids))):
            generated = decode(raw_ids[:len(raw_ids) - drop], skip_special_tokens=False,
                               spaces_between_special_tokens=False)
            if generated and content.endswith(generated):
                return len(content) - len(generated)
        return 0

    def check_finished(self, infer_request, response_choice, current_turn: int) -> bool:
        state = self._state(infer_request)
        if not state["tools_enabled"]:
            return True
        if len(state["calls"]) >= self.max_operations:
            return True
        if _parse_tool_call(response_choice.message.content) is not None:
            return False
        if response_choice.finish_reason == "length":
            return True
        return not (self.branch_aware and self._canonical(state) is not None)

    def _select(self, infer_request, state: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        target = arguments.get("target_frames")
        if not isinstance(target, list) or not target:
            raise ValueError("Execution Error: target_frames must be a non-empty list.")
        if any(not isinstance(x, int) or isinstance(x, bool) for x in target):
            raise ValueError("Execution Error: target_frames must contain integers.")
        if target != sorted(set(target)):
            raise ValueError("Execution Error: target_frames must be sorted and unique.")
        frames = state["frames"]
        source_frames = state.get("tool_video_frames") or frames
        frame_numbers = state.get("tool_video_frame_numbers") or []
        frame_count = len(frame_numbers) if state.get("tool_video_path") else len(source_frames)
        if frame_count == 0 or target[0] < 1 or target[-1] > frame_count:
            raise ValueError(f"Execution Error: frame indices must be in [1,{frame_count}].")
        previous = state["selected"]
        if (state.get("original_drop") or {}).get("applied") and previous is None:
            raise ValueError("Execution Error: original video is unavailable after drop; only reselect returned frames.")
        if previous is not None and not set(target) < set(previous):
            raise ValueError("Execution Error: a second selection must be a strict subset of the first.")
        source_size = frame_count if previous is None else len(previous)
        if source_size <= 1 or len(target) > max(1, int(MAX_SELECTED_FRAME_RATIO * source_size)):
            raise ValueError(
                f"Execution Error: select must remove at least "
                f"{100 * (1 - MAX_SELECTED_FRAME_RATIO):g}% of candidates."
            )

        max_tool_visual_tokens = int(os.environ.get("REFRAME_MAX_TOOL_VISUAL_TOKENS", "4096"))
        remaining_visual_tokens = max_tool_visual_tokens - state["tool_visual_tokens_used"]
        later_operation = len(state.get("calls", [])) + 1 < self.max_operations and len(target) > 1
        reserved_frames = max(1, int(MAX_SELECTED_FRAME_RATIO * len(target))) if later_operation else 0
        if remaining_visual_tokens < 4 * (len(target) + reserved_frames):
            raise ValueError("Execution Error: selected frames exceed the remaining visual-token budget.")
        per_frame_tokens = min(
            int(os.environ.get("REFRAME_SELECT_MAX_TOKEN_NUM", "400")),
            remaining_visual_tokens // (len(target) + reserved_frames),
        )
        if state.get("tool_video_path"):
            import decord

            # Serialize decord/ffmpeg access; concurrent decoding is unsafe.
            with _DECORD_LOCK:
                reader = decord.VideoReader(
                    state["tool_video_path"], ctx=decord.cpu(0), num_threads=1
                )
                source_indices = [
                    min(max(int(frame_numbers[index - 1]), 0), len(reader) - 1)
                    for index in target
                ]
                sources = [
                    {"video_path": state["tool_video_path"], "frame_number": index,
                     "bbox": [0, 0, 1000, 1000]} for index in source_indices
                ]
                decoded = reader.get_batch(source_indices).asnumpy()
                del reader
            rendered = [_resize_to_visual_grid(Image.fromarray(frame), per_frame_tokens) for frame in decoded]
        else:
            sources = [
                {"value": source_frames[index - 1], "bbox": [0, 0, 1000, 1000]}
                for index in target
            ]
            rendered = [_resize_to_visual_grid(source["value"], per_frame_tokens) for source in sources]
        selected = [image for image, _ in rendered]
        for source, index in zip(sources, target):
            source["original_target_image"] = index
        state["tool_visual_tokens_used"] += sum(tokens for _, tokens in rendered)
        first_index = len(state["image_sources"]) + 2
        infer_request.images.extend(selected)
        state["image_sources"].extend(sources)
        state["selected"] = target
        placeholders = "".join("<image>" for _ in selected)
        return (
            f"<tool_response>Selected original video frames {target} (1-based indices out of "
            f"{frame_count} candidate frames); appended as images "
            f"{first_index} to {first_index + len(selected) - 1}.\n{placeholders}</tool_response>"
        )

    @staticmethod
    def _crop_targets_removed_original(state: Dict[str, Any], target: Any) -> bool:
        drop = state.get("original_drop") or {}
        if not drop.get("applied") or type(target) is not int:
            return False
        if state["frames"] and not state["image_sources"]:
            return 1 <= target <= len(state["frames"])
        if state["frames"] and target == 1:
            return True
        source_index = target - (2 if state["frames"] else 1)
        return 0 <= source_index < drop["original_image_count"]

    def _crop(self, infer_request, state: Dict[str, Any], arguments: Dict[str, Any]) -> Tuple[str, List[Any]]:
        target = arguments.get("target_image")
        if not isinstance(target, int) or isinstance(target, bool):
            raise ValueError("Execution Error: target_image must be a 1-based integer.")
        if self._crop_targets_removed_original(state, target):
            raise ValueError("Execution Error: original image is unavailable after drop; use a returned image ID.")

        if state["frames"] and not state["image_sources"]:
            source_index = target - 1
            frames = state.get("tool_video_frames") or state["frames"]
            numbers = state.get("tool_video_frame_numbers") or []
            frame_count = len(numbers) if state.get("tool_video_path") else len(frames)
            if source_index < 0 or source_index >= frame_count:
                raise ValueError("Execution Error: target_image is outside the video-frame range.")
            source = {"bbox": [0, 0, 1000, 1000], "original_target_image": target}
            if state.get("tool_video_path"):
                source.update(video_path=state["tool_video_path"], frame_number=int(numbers[source_index]))
            else:
                source["value"] = frames[source_index]
        elif state["frames"]:
            source_index = target - 2
            if source_index < 0 or source_index >= len(state["image_sources"]):
                raise ValueError("Execution Error: target_image must reference an appended video image.")
            source = state["image_sources"][source_index]
        else:
            source_index = target - 1
            if source_index < 0 or source_index >= len(state["image_sources"]):
                raise ValueError("Execution Error: target_image is outside the image range.")
            source = state["image_sources"][source_index]

        if source.get("video_path"):
            import decord

            with _DECORD_LOCK:
                reader = decord.VideoReader(source["video_path"], ctx=decord.cpu(0), num_threads=1)
                frame_number = min(max(source["frame_number"], 0), len(reader) - 1)
                image = Image.fromarray(reader[frame_number].asnumpy()).convert("RGB")
                del reader
        else:
            image = _to_pil(source["value"])
        boxes = parse_crop_boxes(arguments)
        if boxes is None:
            raise ValueError(
                f"Execution Error: use exactly one bbox_2d or 1--{MAX_CROP_BOXES} distinct bboxes_2d "
                "with valid integer xyxy coordinates on the 0-1000 grid."
            )
        if sum(bbox_area_ratio(box) for box in boxes) > MAX_CROP_AREA_RATIO:
            raise ValueError(
                f"Execution Error: all boxes in one call must together retain at most "
                f"{100 * MAX_CROP_AREA_RATIO:g}% of the source image area."
            )
        per_crop_tokens = crop_token_budget()
        absolute_boxes, encoded_crops, sources, descriptions = [], [], [], []
        width, height = image.size
        first_id = len(state["image_sources"]) + (2 if state["frames"] else 1)
        target_kind = "original video frame" if state["frames"] and not state["image_sources"] else "image"
        original_kind = "video frame" if state["frames"] else "image"
        for index, box in enumerate(boxes):
            absolute = compose_bbox(source["bbox"], box)
            if absolute is None:
                raise ValueError("Execution Error: crop coordinates cannot be mapped to the original image.")
            crop = image.crop((
                round(absolute[0] / CROP_BBOX_SCALE * width),
                round(absolute[1] / CROP_BBOX_SCALE * height),
                round(absolute[2] / CROP_BBOX_SCALE * width),
                round(absolute[3] / CROP_BBOX_SCALE * height),
            ))
            if crop.width == 0 or crop.height == 0:
                raise ValueError("Execution Error: cropped image is empty after pixel rounding.")
            crop_aspect_ratio = max(crop.width, crop.height) / min(crop.width, crop.height)
            if crop_aspect_ratio >= MAX_CROP_ASPECT_RATIO:
                raise ValueError(
                    f"Execution Error: cropped image aspect ratio {crop_aspect_ratio:.1f} is too extreme."
                )
            encoded_crops.append(_resize_to_visual_grid(crop, per_crop_tokens)[0])
            absolute_boxes.append(list(absolute))
            sources.append({**source, "bbox": list(absolute),
                            "original_target_image": source.get("original_target_image", target)})
            absolute_text = "[" + ", ".join(f"{value:.12g}" for value in absolute) + "]"
            if not state["frames"]:
                description = (
                    f"Returned image {first_id + index}: a crop of image {target}.\n"
                    f"Crop box [x_min,y_min,x_max,y_max] in image {target}'s 0–1000 coordinates: {list(box)}.\n"
                )
                if list(source["bbox"]) != [0, 0, 1000, 1000]:
                    description += (
                        f"Mapped to original image {source.get('original_target_image', target)}'s "
                        f"0–1000 coordinates: {absolute_text}.\n"
                    )
                descriptions.append(description + "<image>")
            else:
                descriptions.append(
                    f"Here is the cropped image {first_id + index} (Image Size: {crop.width}x{crop.height}):\n"
                    f"Target {target_kind} {target}: bbox_xyxy_1000={list(box)}.\n"
                    f"Original {original_kind} {source.get('original_target_image', target)}: "
                    f"original_bbox_xyxy_1000={absolute_text}, source_size_px={width}x{height}.\n<image>"
                )
        infer_request.images.extend(encoded_crops)
        state["image_sources"].extend(sources)
        legend = (
            "Each bbox is [x_min,y_min,x_max,y_max] on a 0-1000 grid "
            "of the named image or original video frame, not pixel coordinates. "
            "(0,0) is the source's top-left and (1000,1000) its bottom-right; "
            "(x_min,y_min) is the crop's top-left and (x_max,y_max) its bottom-right.\n"
        ) if state["frames"] else ""
        response = "<tool_response>" + legend + "\n".join(descriptions) + "</tool_response>"
        return response, absolute_boxes if "bboxes_2d" in arguments else absolute_boxes[0]

    def _execute(self, infer_request, state: Dict[str, Any], call: Dict[str, Any]) -> str:
        name = call.get("name")
        arguments = call.get("arguments") or {}
        trace = {"name": name, "arguments": deepcopy(arguments), "success": False}
        if name == CROP_TOOL:
            trace["crop_on_removed_original"] = self._crop_targets_removed_original(
                state, arguments.get("target_image"))
        try:
            mode = os.environ.get("REFRAME_MODE", "mix")
            if name == SELECT_TOOL:
                if mode == "image":
                    raise ValueError("Execution Error: select is disabled for the image expert.")
                response = self._select(infer_request, state, arguments)
            elif name == CROP_TOOL:
                if mode == "video":
                    raise ValueError("Execution Error: crop is disabled for the video expert.")
                response, absolute_bbox = self._crop(infer_request, state, arguments)
                trace["absolute_bboxes" if "bboxes_2d" in arguments else "absolute_bbox"] = absolute_bbox
                trace["original_target_image"] = state["image_sources"][-1].get("original_target_image")
            else:
                raise ValueError(f"Execution Error: unknown visual tool {name!r}.")
            trace["success"] = True
        except Exception as exc:
            if not (isinstance(exc, ValueError) and str(exc).startswith("Execution Error")):
                trace["infra_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[reframe_rl] tool infra error in {name}: {trace['infra_error']}", file=sys.stderr)
            response = f"<tool_response>{exc}</tool_response>"
        state["calls"].append(trace)
        return response

    def _generation_prefix_length(self, content: str, raw_ids: Optional[List[int]]) -> int:
        if raw_ids:
            return self._response_prefix_length(content, list(raw_ids))
        template = getattr(getattr(self, "infer_engine", None), "template", None)
        prefix = template._get_response_prefix() if template is not None else ""
        return len(prefix) if prefix and content.startswith(prefix) else 0

    def step(self, infer_request, response_choice, current_turn: int) -> Dict[str, Any]:
        state = self._state(infer_request)
        completion = response_choice.message.content or ""
        prefix_length = self._generation_prefix_length(completion, response_choice.token_ids)
        candidate = _parse_tool_call(completion)
        canonical = self._canonical(state)
        sampled_tool_call = candidate is not None
        branch_span = None
        candidate_iou = None
        if self.branch_aware and canonical is not None and candidate is not None:
            status, score = evidence_score_for_call(
                state["metadata"], candidate, expected_call=canonical)
            if status != "invalid":
                candidate_iou = branch_gate_score(state["metadata"], candidate, score)

        if self.branch_aware and canonical is not None:
            if should_force_canonical_branch(state["metadata"], candidate, canonical):
                completion, branch_span = _replace_last_tool_call(completion, _format_tool_call(canonical))
                candidate = canonical
                infer_request.messages[-1]["content"] = completion

        if candidate is None:
            candidate = {"name": None, "arguments": {}}

        if branch_span is not None:
            encoded = self.tokenizer(completion[prefix_length:], add_special_tokens=False,
                                     return_offsets_mapping=True)
            token_ids = encoded["input_ids"]
            span = (branch_span[0] - prefix_length, branch_span[1] - prefix_length)
            branch_mask = [
                int(end > span[0] and start < span[1])
                for start, end in encoded["offset_mapping"]
            ]
            response_choice.message.content = completion
            response_choice.token_ids = token_ids
            response_choice.logprobs = None
        else:
            token_ids = response_choice.token_ids
            if token_ids is None:
                token_ids = self.tokenizer.encode(completion[prefix_length:], add_special_tokens=False)
                response_choice.logprobs = None
            token_ids = list(token_ids)
            branch_mask = [0] * len(token_ids)

        tool_response = self._execute(infer_request, state, candidate)
        state["calls"][-1]["gate_eligible"] = self.branch_aware and canonical is not None
        state["calls"][-1]["gate_candidate_iou"] = candidate_iou
        state["calls"][-1]["injected"] = branch_span is not None
        state["calls"][-1]["sampled"] = sampled_tool_call
        if canonical is not None and state["calls"][-1]["success"] and candidate.get("name") == canonical.get("name"):
            state["cursor"] += 1
        route = {SELECT_TOOL: "video", CROP_TOOL: "image"}.get(
            candidate.get("name"), state["default_teacher_route"]
        )
        state["default_teacher_route"] = route
        state["branch_masks"].append(branch_mask)
        state["teacher_routes"].append([route] * len(token_ids))
        infer_request.messages.append({"role": "user", "content": tool_response})

        return {
            "infer_request": infer_request,
            "response_token_ids": token_ids,
            "response_loss_mask": [1] * len(token_ids),
            "rollout_infos": {
                "tool_calls": deepcopy(state["calls"]),
                "branch_response_mask": deepcopy(state["branch_masks"]),
                "teacher_route_by_turn": deepcopy(state["teacher_routes"]),
                "default_teacher_route": state["default_teacher_route"],
                "images": list(infer_request.images),
            },
        }


class GroundDropScheduler(ObservationGatedScheduler):
    branch_aware = False

    async def on_trajectory_start(self, requests) -> None:
        await super().on_trajectory_start(requests)
        for request in requests:
            state = self._state(request)
            training = state["training"]
            key = f"original-drop:{request.data_dict.get('qid', '')}:{request.uuid or ''}"
            draw = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) / 2**64
            eligible = state["metadata"].get("evidence_available") is not False
            state["original_drop"] = {
                "training": training,
                "selected": training and eligible and draw < ORIGINAL_DROP_PROB,
                "applied": False,
                "original_message_count": len(request.messages),
                "original_image_count": len(request.images),
                "turns": [],
            }

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        result = await super().on_turn_end(infer_request, response_choice, current_turn)
        drop = self._state(infer_request)["original_drop"]
        if drop["training"]:
            drop["turns"].append({
                "messages": deepcopy(infer_request.messages),
                "images": list(infer_request.images),
                "videos": deepcopy(infer_request.videos),
                "dropped": drop["applied"],
                "token_ids": list(response_choice.token_ids or []),
                "logprobs": self._extract_logprobs_from_choice(response_choice),
            })
        return result

    @staticmethod
    def _remove_original_media(infer_request, drop):
        for message in infer_request.messages[:drop["original_message_count"]]:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = content.replace("<image>", "").replace("<video>", "")
            elif isinstance(content, list):
                message["content"] = [
                    item for item in content
                    if item.get("type") not in {"image", "image_url", "video", "video_url"}
                ]
        infer_request.images = infer_request.images[drop["original_image_count"]:]
        infer_request.videos = []
        drop["applied"] = True

    def step(self, infer_request, response_choice, current_turn):
        result = super().step(infer_request, response_choice, current_turn)
        state = self._state(infer_request)
        drop = state["original_drop"]
        if drop["selected"] and not drop["applied"] and state["calls"][-1]["success"]:
            self._remove_original_media(infer_request, drop)
            infer_request.messages[-1]["content"] += _drop_notice(bool(state["frames"]))
        result["rollout_infos"].update({
            "images": list(infer_request.images),
            "videos": deepcopy(infer_request.videos),
            "original_vision_dropped": drop["applied"],
        })
        return result

    async def run(self, infer_request, request_config, **kwargs):
        output = await super().run(infer_request, request_config, **kwargs)
        return self.finalize_rollout(infer_request, request_config, output)

    def finalize_rollout(self, infer_request, request_config, output):
        drop = self._state(infer_request)["original_drop"]
        if not drop["training"]:
            return output
        if request_config.n != 1:
            raise ValueError("Training original-vision drop requires request_config.n=1")
        return self._split_training_turns(output, drop)

    @staticmethod
    def _split_training_turns(output, drop):
        turns = drop["turns"]
        if len(turns) != len(output.response_token_ids):
            raise ValueError("Original-vision snapshots must match assistant token turns")
        final_choice = output.response.choices[0]
        turn_logprobs = [
            list(turn["logprobs"])
            if (turn["token_ids"] == ids and len(turn["logprobs"]) == len(ids)) else []
            for turn, ids in zip(turns, output.response_token_ids)
        ]
        outputs = []
        for index, turn in enumerate(turns):
            current_ids = list(output.response_token_ids[index])
            info = deepcopy(output.rollout_infos)
            info.update({
                "images": turn["images"],
                "videos": turn["videos"],
                "trajectory_final_completion": final_choice.message.content,
                "trajectory_final_finish_reason": final_choice.finish_reason,
                "training_turn_index": index,
                "training_turn_count": len(turns),
                "original_vision_dropped": turn["dropped"],
                "trajectory_original_vision_dropped": drop["applied"],
                "original_vision_drop_selected": drop["selected"],
                "trajectory_rollout_logprob_turns": sum(bool(values) for values in turn_logprobs),
            })
            info["teacher_route_by_turn"] = info.get("teacher_route_by_turn", [])[index:index + 1]
            info["branch_response_mask"] = info.get("branch_response_mask", [])[index:index + 1]
            response = deepcopy(output.response)
            response.choices[0].logprobs = None
            response.choices[0].message.content = turn["messages"][-1]["content"]
            response.choices[0].token_ids = current_ids
            outputs.append(RolloutOutput(
                response=response,
                messages=turn["messages"],
                response_token_ids=[current_ids],
                response_loss_mask=[[1] * len(current_ids)],
                rollout_infos=info,
                rollout_logprobs=[turn_logprobs[index]] if turn_logprobs[index] else [],
            ))
        return outputs


multi_turns["og_mopd"] = ObservationGatedScheduler
multi_turns["gd_grpo"] = GroundDropScheduler


def _boxed_answers(text: str) -> List[Tuple[str, int]]:
    answers = []
    for match in BOXED_START_PATTERN.finditer(text or ""):
        start = index = match.end()
        depth = 1
        while index < len(text):
            if text[index] == "\\":
                index += 2
                continue
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    answers.append((text[start:index], index + 1))
                    break
            index += 1
    return answers


def _extract_answer(text: str) -> str:
    matches = _boxed_answers(text or "")
    if matches:
        return matches[-1][0].strip()
    answer_match = re.findall(r"<answer>(.*?)</answer>", text or "", flags=re.DOTALL | re.IGNORECASE)
    return answer_match[-1].strip() if answer_match else ""


NUMBER_WORDS = {
    word: str(value) for value, word in enumerate(
        ("ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE",
         "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN",
         "SEVENTEEN", "EIGHTEEN", "NINETEEN", "TWENTY"))
}


def _normalize_answer(value: Any) -> str:
    value = ReframeDatasetPreprocessor._answer(value)
    extracted = _extract_answer(value)
    if extracted:
        value = extracted
    value = unescape_latex_punctuation(value)
    text = re.sub(r"\s+", "", value).strip(".$,;").upper()
    return NUMBER_WORDS.get(text, text)


_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
BARE_NUMBER_PATTERN = re.compile(_NUMBER)
UNIT_NUMBER_PATTERN = re.compile(
    rf"(?P<currency>US\$|[$\u00a3\u20ac\u00a5])?\s*(?P<number>{_NUMBER})\s*"
    r"(?:(?P<scale>(?:thousand|million|billion|trillion)s?)"
    r"|(?P<mark>%|percent|\u00b0|\^\s*(?:\\circ|\{\s*\\circ\s*\})|degrees?))?",
    re.IGNORECASE,
)


def _unit_omission_match(first: str, second: str) -> bool:
    first, second = (unescape_latex_punctuation(value).strip() for value in (first, second))
    for with_unit, bare in ((first, second), (second, first)):
        match = UNIT_NUMBER_PATTERN.fullmatch(with_unit)
        if not match or not BARE_NUMBER_PATTERN.fullmatch(bare):
            continue
        if match["currency"] and match["mark"]:
            continue
        if match["number"].replace(",", "") == bare.replace(",", ""):
            return True
    return False


FIELD_LABEL_PREFIX_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9 ()#./_%&-]{0,40}\s*:\s*")


def _strip_field_label(value: str) -> str:
    match = FIELD_LABEL_PREFIX_PATTERN.match(value.strip())
    if not match:
        return value
    remainder = value.strip()[match.end():].strip()
    if not remainder or remainder.startswith("//"):
        return value
    return remainder


def _answer_context(messages: Any) -> str:
    if not isinstance(messages, (list, tuple)):
        return ""
    return next(
        (str(message.get("content", "")) for message in messages
         if isinstance(message, dict) and message.get("role") == "user"),
        "",
    )


CHOICE_REFERENCE_PATTERN = re.compile(r"\\boxed\s*\{\s*([A-Za-z])\s*\}")
LATEX_TEXT_PATTERN = re.compile(r"\\text\s*\{([^{}]*)\}")
CHOICE_PREDICTION_PATTERNS = (
    re.compile(r"([A-Za-z])\.?"),
    re.compile(r"[\(\[]\s*([A-Za-z])\s*[\)\]](?:\s.*)?", re.DOTALL),
    re.compile(r"([A-Za-z])\s*[.):](?:\s.*)?", re.DOTALL),
    re.compile(r"(?:option|choice)\s*:?\s*[\(\[]?\s*([A-Za-z])\s*[\)\]]?\.?", re.IGNORECASE),
)


def _choice_letter(prediction: str) -> Optional[str]:
    text = LATEX_TEXT_PATTERN.sub(r"\1", prediction).strip()
    for pattern in CHOICE_PREDICTION_PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            return match.group(1).upper()
    return None


def _correctness(completion: str, solution: Any, messages: Any = None) -> float:
    prediction = _extract_answer(completion)
    target = ReframeDatasetPreprocessor._answer(solution)
    choice = CHOICE_REFERENCE_PATTERN.fullmatch(target.strip())
    target = _extract_answer(target) or target
    if not prediction.strip() or not target.strip():
        return 0.0
    if choice:
        letter = _choice_letter(prediction)
        if letter is not None:
            return float(letter == choice.group(1).upper())
    field_match = match_document_answer(
        prediction,
        target,
        context=_answer_context(messages),
    )
    if field_match is not None:
        return float(field_match)
    stripped_prediction = _strip_field_label(prediction)
    stripped_target = _strip_field_label(target)
    if stripped_prediction != prediction or stripped_target != target:
        field_match = match_document_answer(
            stripped_prediction,
            stripped_target,
            context=_answer_context(messages),
        )
        if field_match:
            return 1.0
    prediction_forms = {_normalize_answer(prediction), _normalize_answer(stripped_prediction)}
    target_forms = {_normalize_answer(target), _normalize_answer(stripped_target)}
    prediction_forms.discard("")
    target_forms.discard("")
    if prediction_forms & target_forms:
        return 1.0
    return float(any(
        _unit_omission_match(prediction_text, target_text)
        for prediction_text in (prediction, stripped_prediction)
        for target_text in (target, stripped_target)
    ))


def _reference_correctness(completion: str, solution: Any, extra_answer: Any = None, messages: Any = None) -> float:
    references = [solution] + (extra_answer if isinstance(extra_answer, list) else [])
    return max(_correctness(completion, reference, messages) for reference in references)


ILLEGAL_SPECIAL_TOKENS = (
    "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
    "<|object_ref_start|>", "<|object_ref_end|>", "<|vision_start|>",
    "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    "<|im_start|>",
)


def _has_repeated_line(text: str) -> bool:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return any(
        current == following and sum(char.isalpha() for char in current) >= 3
        for current, following in zip(lines, lines[1:])
    )


def _output_invalid_reasons(completion: str, finish_reason: Optional[str] = None) -> List[str]:
    reasons = []
    if "<tool_call>" in completion:
        reasons.append("unexecuted_tool_call")
    if any(token in completion for token in ILLEGAL_SPECIAL_TOKENS):
        reasons.append("special_token")
    boxed_matches = _boxed_answers(completion or "")
    if len(list(BOXED_START_PATTERN.finditer(completion or ""))) != 1:
        reasons.append("boxed_count")
    elif len(boxed_matches) != 1:
        reasons.append("unbalanced_boxed")
    elif completion[boxed_matches[0][1]:].strip():
        reasons.append("trailing_text")
    if completion.count("<tool_response>") != completion.count("</tool_response>"):
        reasons.append("unbalanced_tool_response")
    if _has_repeated_line(completion or ""):
        reasons.append("repeated_text")
    if finish_reason == "length":
        reasons.append("length")
    return reasons


def _valid_output(completion: str) -> bool:
    return not _output_invalid_reasons(completion)


def _progress(kwargs: Dict[str, Any]) -> float:
    state = kwargs.get("trainer_state")
    if state is None:
        return 0.0
    max_steps = max(int(getattr(state, "max_steps", 0)), 1)
    return min(max(float(getattr(state, "global_step", 0)) / max_steps, 0.0), 1.0)


def _reward_parts(completions, solution, tool_metadata, rollout_infos, **kwargs):
    progress = _progress(kwargs)
    results = []
    rollout_infos = rollout_infos or [{} for _ in completions]
    tool_metadata = tool_metadata or [{} for _ in completions]
    finish_reasons = kwargs.get("finish_reason") or [None for _ in completions]
    messages = kwargs.get("messages") or [None for _ in completions]
    extra_answers = kwargs.get("extra_answer") or [[] for _ in completions]
    for completion, target, metadata, info, finish_reason, sample_messages, extras in zip(
        completions,
        solution,
        tool_metadata,
        rollout_infos,
        finish_reasons,
        messages,
        extra_answers,
    ):
        completion = (info or {}).get("trajectory_final_completion", completion)
        finish_reason = (info or {}).get("trajectory_final_finish_reason", finish_reason)
        parsed_metadata = parse_tool_metadata(metadata)
        correctness = _reference_correctness(completion, target, extras, sample_messages)
        results.append(shape_dynamic_tool_reward(
            correctness=correctness,
            calls=[call for call in (info or {}).get("tool_calls", []) if not call.get("infra_error")],
            metadata=parsed_metadata,
            progress=progress,
            output_valid=_valid_output(completion) and finish_reason != "length",
        ))
    return results


class CorrectnessReward(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        infos = kwargs.get("rollout_infos") or [{} for _ in completions]
        messages = kwargs.get("messages") or [None for _ in completions]
        extra_answers = kwargs.get("extra_answer") or [[] for _ in completions]
        return [
            _reference_correctness(
                (info or {}).get("trajectory_final_completion", completion), target, extras, sample_messages)
            for completion, target, info, sample_messages, extras in zip(
                completions, solution, infos, messages, extra_answers)
        ]


class CallReward(ORM):
    def __call__(self, completions, solution, tool_metadata, rollout_infos, **kwargs) -> List[float]:
        return [part.call for part in _reward_parts(
            completions, solution, tool_metadata, rollout_infos, **kwargs
        )]


class ObservationReward(ORM):
    def __call__(self, completions, solution, tool_metadata, rollout_infos, **kwargs) -> List[float]:
        return [part.observation for part in _reward_parts(
            completions, solution, tool_metadata, rollout_infos, **kwargs
        )]


class PenaltyReward(ORM):
    def __call__(self, completions, solution, tool_metadata, rollout_infos, **kwargs) -> List[float]:
        return [part.penalty for part in _reward_parts(
            completions, solution, tool_metadata, rollout_infos, **kwargs
        )]


orms["reframe_answer"] = CorrectnessReward
orms["reframe_call"] = CallReward
orms["reframe_observation"] = ObservationReward
orms["reframe_penalty"] = PenaltyReward
