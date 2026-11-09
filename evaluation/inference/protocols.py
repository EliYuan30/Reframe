from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageChops, ImageStat

from evaluation.benchmarks import IMAGE_DATASETS, VIDEO_DATASETS, DATASETS
from evaluation.inference import tools as native

PROTOCOL_VERSION = 2
MODELS = dict(native.MODELS)
MODES = ("w_tool", "wo_tool")

IMAGE_BENCHMARKS = IMAGE_DATASETS
VIDEO_BENCHMARKS = VIDEO_DATASETS
BENCHMARKS = DATASETS
OFFICIAL_SCORERS = {"videomme_v2": "evaluation.scoring.videomme_v2", "vsi_bench": "evaluation.scoring.vsi_bench",
                    "vstar_video": "evaluation.scoring.vstar_video"}
MAX_TOKENS = {"image": 8192, "video": 16384}
MAX_TURNS = native.MAX_TURNS
MAX_OPERATIONS = native.MAX_OPERATIONS
CROP_TOKENS = 1960
MAX_CROP_BOXES = 3  # Reframe crops at most 3 boxes per call, each at CROP_TOKENS
MIN_PIXELS, MAX_PIXELS = 3136, 4014080
OFFICIAL_MAX_PIXELS = 12845056
MAX_CROP_TOKENS = 16384
PATCH = 28
TIMESTAMP_MODES = ("reframe", "real", "none")
THYME_IMAGE_ALIAS = "/mnt/data/image.png"


def modality(benchmark: str) -> str:
    if benchmark in IMAGE_BENCHMARKS:
        return "image"
    if benchmark in VIDEO_BENCHMARKS:
        return "video"
    raise ValueError(f"Unknown benchmark: {benchmark}")


def sampling(protocol: str, benchmark: str) -> dict[str, Any]:
    """Swift needs </tool_call> in the stop list to retain returned stop strings."""
    if protocol not in MODELS:
        raise ValueError(protocol)
    stop = {"thyme": ["</code>", "</answer>", "</tool_call>"],
            "deepeyesv2": ["</code>", "<|im_end|>", "</tool_call>"]}.get(protocol, ["</tool_call>"])
    return {"max_tokens": MAX_TOKENS[modality(benchmark)], "temperature": 0.0, "top_p": 1.0,
            "top_k": -1, "n": 1, "stop": stop}


def allowed_samplings(protocol: str) -> list[dict[str, Any]]:
    return [sampling(protocol, IMAGE_BENCHMARKS[0]), sampling(protocol, VIDEO_BENCHMARKS[0])]


def smart_resize(height: int, width: int, factor: int = PATCH, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    if max(height, width) / max(1, min(height, width)) > 200:
        raise ValueError("Aspect ratio above 200 is rejected by the Qwen2.5-VL processor")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_tokens(size: tuple[int, int], min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> int:
    height, width = smart_resize(size[1], size[0], min_pixels=min_pixels, max_pixels=max_pixels)
    return (height // PATCH) * (width // PATCH) + 2


def visible_size(size: tuple[int, int], max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    height, width = smart_resize(size[1], size[0], max_pixels=max_pixels)
    return width, height


def header_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


@dataclass(frozen=True)
class Media:
    visible: tuple[str, ...]
    sources: tuple[str, ...]
    is_video: bool


def _local_path(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("path")
    if not isinstance(value, str) or not value or "://" in value or value.startswith("data:"):
        raise ValueError("Only trusted local media paths are accepted")
    return value


def media_for(row: dict[str, Any], benchmark: str) -> Media:
    values = row.get("image", row.get("images"))
    values = list(values) if isinstance(values, (list, tuple)) else [values]
    visible = tuple(_local_path(value) for value in values)
    if not visible:
        raise ValueError("Row has no media")
    if modality(benchmark) == "image":
        if row.get("is_video") and row.get("is_video") not in ("False", "false", 0):
            raise ValueError("Image benchmark row is marked as video")
        return Media(visible, visible, False)
    frames = row.get("tool_video_frames")
    frames = json.loads(frames) if isinstance(frames, str) else frames
    if not frames:
        # Same as Reframe (reframe/plugin.py): without original frames, tools crop the candidate frames.
        frames = list(visible)
    if not isinstance(frames, (list, tuple)) or len(frames) != len(visible):
        raise ValueError("tool_video_frames must list one original frame per candidate frame")
    return Media(visible, tuple(_local_path(value) for value in frames), True)


class ImageBank:
    def __init__(self, paths: tuple[str, ...] | list[str]):
        self.items: list[Any] = list(paths)
        self.sizes: dict[int, tuple[int, int]] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Image.Image:
        if type(index) is not int or not 0 <= index < len(self.items):
            raise IndexError("image index out of range")
        item = self.items[index]
        if isinstance(item, str):
            with Image.open(item) as image:
                item = image.convert("RGB")
            self.items[index] = item
        return item

    def size(self, index: int) -> tuple[int, int]:
        item = self.items[index]
        if isinstance(item, Image.Image):
            return item.size
        if index not in self.sizes:
            self.sizes[index] = header_size(item)
        return self.sizes[index]

    def append(self, image: Image.Image) -> None:
        self.items.append(image)


@dataclass(frozen=True)
class LazyRef:
    index: int


class MultiImageInterpreter(native.ImageInterpreter):
    MAX_USER_VARIABLES = 96

    def __init__(self, bank: ImageBank, protocol: str, aliases: dict[str, int] | None = None,
                 views: list[tuple[int, int]] | None = None):
        if protocol not in native.CODE_PROTOCOLS:
            raise ValueError("Interpreter is only for code protocols")
        self.protocol = protocol
        self.bank = bank
        self.inputs = len(bank)
        self.aliases = dict(aliases or {})
        if views is not None and (protocol != "deepeyesv2" or len(views) != self.inputs):
            raise ValueError("Model-visible views are a DeepEyesV2 extension with one size per input")
        self.views = list(views) if views is not None else None
        self.clipped = 0
        self.frames: dict[int, native.Frame] = {}
        self.original = None
        refs = [LazyRef(index) for index in range(self.inputs)]
        self.env = {"image": refs[0], "images": refs,
                    **{f"image_{index + 1}": ref for index, ref in enumerate(refs)},
                    **{name: native.Symbol(name) for name in ("Image", "plt", "np", "cv2", "math", "display")}}
        self.reserved = len(self.env)
        self.allocated = 0

    def materialize(self, value):
        if not isinstance(value, LazyRef):
            return value
        if value.index not in self.frames:
            image = self.bank[value.index]
            box = (0., 0., float(image.width), float(image.height))
            if self.views is not None and tuple(self.views[value.index]) != image.size:
                image = image.resize(tuple(self.views[value.index]), Image.Resampling.BICUBIC)
            self.frames[value.index] = native.Frame(image, box)
        return self.frames[value.index]

    def crop(self, frame, box):
        if self.views is not None and isinstance(frame, native.Frame) and isinstance(box, (tuple, list)) and len(box) == 4:
            x1, y1, x2, y2 = [round(self.number(v)) for v in box]
            width, height = frame.image.width, frame.image.height
            clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
            if x1 < x2 and y1 < y2 and clipped[0] < clipped[2] and clipped[1] < clipped[3] and clipped != (x1, y1, x2, y2):
                self.clipped += 1
                box = clipped
        return super().crop(frame, box)

    def rerender(self, frames: list[native.Frame]) -> tuple[list[native.Frame], list[int | None]]:
        if self.views is None:
            return frames, [None] * len(frames)
        result, sources = [], []
        for frame in frames:
            match, best = None, None
            for index, view in sorted(self.frames.items()):
                width, height = self.bank.size(index)
                a, b, c, d = frame.box
                if not (-0.5 <= a < c <= width + 0.5 and -0.5 <= b < d <= height + 0.5):
                    continue
                sx, sy = view.image.width / width, view.image.height / height
                region = tuple(int(round(v)) for v in (a * sx, b * sy, c * sx, d * sy))
                if region[2] <= region[0] or region[3] <= region[1]:
                    continue
                reference = view.image.crop(region).convert("RGB")
                output = frame.image.convert("RGB") if frame.image.mode in ("RGB", "RGBA") else None
                if output is None:
                    continue
                if output.size == reference.size:
                    error = 0.0 if ImageChops.difference(output, reference).getbbox() is None else None
                else:
                    reference = reference.resize(output.size, Image.Resampling.BICUBIC)
                    error = max(ImageStat.Stat(ImageChops.difference(output, reference)).mean)
                    error = error if error <= 4.0 else None
                if error is not None and (best is None or error < best):
                    match, best = index, error
            width, height = self.bank.size(match) if match is not None else (0, 0)
            box = (max(0., frame.box[0]), max(0., frame.box[1]), min(float(width), frame.box[2]), min(float(height), frame.box[3]))
            if match is None or round(box[2]) <= round(box[0]) or round(box[3]) <= round(box[1]):
                result.append(frame)
                sources.append(None)
                continue
            crop = self.bank[match].crop(box).convert("RGB")
            result.append(native.Frame(crop, box))
            sources.append(match)
        return result, sources

    def expr(self, node):
        return self.materialize(super().expr(node))

    def assign(self, target, value):
        if isinstance(target, ast.Name) and not target.id.startswith("_"):
            if target.id not in self.work and len(self.work) - self.reserved >= self.MAX_USER_VARIABLES:
                self.fail("Variable limit exceeded")
            self.work[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (tuple, list)) and len(target.elts) == len(value):
            for part, item in zip(target.elts, value):
                self.assign(part, item)
            return
        self.fail("Only bounded variable assignment is supported")

    def call(self, receiver, method, args, kwargs):
        if isinstance(receiver, native.Symbol):
            name = receiver.name + ("." + method if method else "")
            if name in {"Image.open", "cv2.imread"}:
                if (self.protocol != "thyme" or len(args) != 1 or kwargs or type(args[0]) is not str
                        or args[0] not in self.aliases):
                    self.fail("Only the listed in-memory image paths can be opened")
                frame = self.materialize(LazyRef(self.aliases[args[0]]))
                array = name == "cv2.imread"
                return native.Frame(frame.image, frame.box, array, array)
        return super().call(receiver, method, args, kwargs)

    def thyme_inputs(self) -> set[str]:
        return set(self.aliases)


def thyme_aliases(count: int, is_video: bool) -> dict[str, int]:
    if is_video:
        width = max(2, len(str(count)))
        return {f"/mnt/data/frame_{index + 1:0{width}d}.png": index for index in range(count)}
    if count == 1:
        return {THYME_IMAGE_ALIAS: 0}
    return {f"/mnt/data/image_{index + 1}.png": index for index in range(count)}


def frame_labels(row: dict[str, Any], count: int, mode: str) -> tuple[list[str | None], str]:
    """Return per-frame timestamp labels and the effective mode.

    reframe: match the Reframe (Qwen3.5) evaluation of pre-resized frames (metadata fps=2,
    frame indices 0..N-1, Qwen3-VL pairs labelled ``<{k+0.25:.1f} seconds>``
    before frames 2k and 2k+1). real: actual seconds from fps/frame_count with
    uniform rint(linspace(0, frame_count-1, N)) indices (only rows carrying
    fps/frame_count; otherwise falls back to none). none: no labels.
    """
    if mode not in TIMESTAMP_MODES:
        raise ValueError(mode)
    labels: list[str | None] = [None] * count
    if mode == "reframe":
        for pair in range(0, count, 2):
            labels[pair] = f"<{pair // 2 + 0.25:.1f} seconds>"
        return labels, mode
    if mode == "real":
        fps, total = row.get("fps"), row.get("frame_count")
        if isinstance(fps, (int, float)) and fps > 0 and isinstance(total, int) and total > 0:
            for index in range(count):
                position = round(index * (total - 1) / max(1, count - 1))
                labels[index] = f"<{position / fps:.1f} seconds>"
            return labels, mode
    return labels, "none"


def media_block(media: Media, labels: list[str | None]) -> str:
    count = len(media.visible)
    if not media.is_video:
        return "<image>" * count + "\n"
    head = f"The video is given as {count} sampled frames (images 1 to {count}, in temporal order)."
    return head + "\n" + "".join((label or "") + "<image>" for label in labels[:count]) + "\n"


def deepeyes_schema(inputs: int) -> dict[str, Any]:
    schema = native.tool_schema("deepeyes")
    if inputs > 1:
        schema = json.loads(json.dumps(schema))
        schema["function"]["parameters"]["properties"]["image_index"] = {
            "type": "integer",
            "description": f"Which input image to zoom in on, 1-based (1 to {inputs}); default 1. Coordinates refer to that image."}
    return schema


def system_prompt(protocol: str, tools: bool, inputs: int) -> str:
    if not tools or protocol != "deepeyes" or inputs == 1:
        return native.system_prompt(protocol, tools)
    schema = json.dumps(deepeyes_schema(inputs), ensure_ascii=False, separators=(",", ":"))
    return native.system_prompt("pixelreasoner", True).split("<tools>\n", 1)[0] + (
        f"<tools>\n{schema}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>')


def user_prompt(question: str, protocol: str, tools: bool, media: Media, labels: list[str | None],
                source_sizes: list[tuple[int, int]] | None) -> str:
    question = native.literal_text(question)
    block = media_block(media, labels)
    if protocol == "thyme":
        aliases = list(thyme_aliases(len(media.sources), media.is_video))
        sizes = source_sizes or []
        def dims(size):
            return f"{size[0]}x{size[1]}" if size else "unknown"
        if media.is_video:
            leading = len(media.visible)
            frame_sizes = sizes[:leading]
            unique = sorted({dims(size) for size in frame_sizes}) or ["unknown"]
            path = f'"{aliases[0]}" ... "{aliases[leading - 1]}" (frame k is image k, in temporal order)'
            size = ", ".join(f'"{value}"' for value in unique) + " (every frame)" if len(unique) == 1 else ", ".join(
                f'"{dims(value)}"' for value in frame_sizes)
        elif len(aliases) == 1:
            path, size = f'"{aliases[0]}"', f'"{dims(sizes[0]) if sizes else "unknown"}"'
        else:
            path = ", ".join(f'"{alias}"' for alias in aliases)
            size = ", ".join(f'"{dims(value)}"' for value in sizes) if sizes else '"unknown"'
        return (f'{block}{question}\n\n### User Image Path:** {path}\n'
                f'### User Image Size:** {size}\n\n### **Output Format (strict adherence required):**\n\n'
                '<think>Your detailed reasoning process, including any code, should go here.</think>\n'
                "<answer>Your final answer to the user's question goes here.</answer>\n")
    if protocol == "deepeyesv2":
        return block + question + native.V2_USER_SUFFIX
    if protocol == "pixelreasoner":
        return block + question + native.PIXEL_SUFFIX
    suffix = native.DEEPEYES_SUFFIX if tools else "\nThink first, then answer. Format strictly as: <think>...</think> <answer>...</answer> "
    return block + "Question: " + question + suffix


def build_request(row: dict[str, Any], benchmark: str, protocol: str, mode: str, media: Media,
                  timestamp_mode: str = "reframe", source_sizes: list[tuple[int, int]] | None = None
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode not in MODES:
        raise ValueError(mode)
    tools = mode == "w_tool"
    count = len(media.visible)
    labels, effective = frame_labels(row, count, timestamp_mode) if media.is_video else ([None] * count, "none")
    request = {"messages": [
        {"role": "system", "content": system_prompt(protocol, tools, len(media.visible))},
        {"role": "user", "content": user_prompt(str(row["question"]), protocol, tools, media, labels, source_sizes)},
    ], "images": list(media.visible), "tools": []}
    placeholders = sum(message["content"].count("<image>") for message in request["messages"])
    if placeholders != len(request["images"]):
        raise ValueError("Image placeholders do not match images")
    return request, {"timestamp_mode": effective, "inputs": len(media.visible)}


def execute_deepeyes(call: dict[str, Any], bank: ImageBank, visible_sizes: list[tuple[int, int]],
                     inputs: int, max_pixels: int = MAX_PIXELS) -> tuple[Image.Image, list[float], int]:
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    if call.get("name") != "image_zoom_in_tool":
        raise native.UnsupportedOperation(f"Unsupported image operation: {call.get('name')!r}")
    allowed = {"bbox_2d", "label"} | ({"image_index"} if inputs > 1 else set())
    if set(arguments) - allowed:
        raise native.UnsupportedOperation("Unexpected arguments (paths/code are never accepted)")
    values = arguments.get("bbox_2d")
    if not isinstance(values, list) or len(values) != 4 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("bbox_2d must contain four finite numbers")
    index = arguments.get("image_index", 1)
    if type(index) is not int or not 1 <= index <= inputs:
        raise ValueError("image_index must be a valid 1-based input index")
    seen_w, seen_h = visible_size(visible_sizes[index - 1], max_pixels)
    box = [max(0.0, values[0]), max(0.0, values[1]), min(float(seen_w), values[2]), min(float(seen_h), values[3])]
    if min(box[2] - box[0], box[3] - box[1]) <= 0:
        raise ValueError("Native crop must have positive width and height after clipping")
    width, height = bank.size(index - 1)
    sx, sy = width / seen_w, height / seen_h
    source = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
    crop = bank[index - 1].crop(tuple(source)).convert("RGB")
    if min(crop.size) <= 0:
        raise ValueError("Rounded native crop is empty")
    return crop, source, index


def execute_operation(protocol: str, call: dict[str, Any], bank: ImageBank, interpreter: Any,
                      visible_sizes: list[tuple[int, int]], inputs: int, max_pixels: int = MAX_PIXELS
                      ) -> tuple[list[native.Frame], str, dict[str, Any]]:
    if protocol in native.CODE_PROTOCOLS:
        result = native.execute_code(protocol, interpreter, call)
        if protocol == "deepeyesv2" and getattr(interpreter, "views", None) is not None:
            frames, sources = interpreter.rerender(result.frames)
            return frames, result.stdout, {"view_sizes": [list(frame.image.size) for frame in result.frames],
                                           "rerendered_from": [None if index is None else index + 1 for index in sources],
                                           "clipped_crops": interpreter.clipped}
        return result.frames, result.stdout, {}
    if protocol == "pixelreasoner":
        target = (call.get("arguments") or {}).get("target_image", 1) if isinstance(call.get("arguments"), dict) else None
        crop, box = native.execute_native("pixelreasoner", bank, call)
        return [native.Frame(crop, tuple(box))], "", {"target_image": target}
    crop, box, index = execute_deepeyes(call, bank, visible_sizes, inputs, max_pixels)
    return [native.Frame(crop, tuple(box))], "", {"image_index": index}


def balanced_boxed(text: str) -> str | None:
    starts = list(re.finditer(r"\\boxed\s*\{", text or ""))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return None


def unwrap_text(value: str) -> str:
    previous = None
    while previous != value:
        previous = value
        value = re.sub(r"\\(?:text|textbf|mathrm)\s*\{([^{}]*)\}", r"\1", value)
    return value.strip()


def extract_answer(text: str) -> tuple[str | None, str]:
    text = text or ""
    tags = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    inner = tags[-1] if tags else None
    source = "answer_tag"
    if inner is None:
        unclosed = re.search(r"<answer>((?:(?!<answer>).)*)$", text, flags=re.S | re.I)
        if unclosed:
            inner, source = unclosed.group(1), "answer_unclosed"
    if inner is not None:
        boxed = balanced_boxed(inner)
        content = unwrap_text(boxed if boxed is not None else inner)
        if content:
            return content, source + ("_boxed" if boxed is not None else "")
    boxed = balanced_boxed(text)
    if boxed is not None and unwrap_text(boxed):
        return unwrap_text(boxed), "boxed"
    return None, "none"


def answer_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def target_answer(value: Any) -> str:
    value = answer_text(value).strip()
    match = re.fullmatch(r"\\boxed\s*\{([^{}]*)\}", value)
    if match:
        value = match.group(1).strip()
    return f"\\boxed{{{value}}}" if value else ""


def mcq_letter(content: str) -> str | None:
    text = content.strip()
    patterns = (r"\(?([A-Za-z])\)?[.):]?", r"\(([A-Z])\)(?:\s.*)?", r"([A-Z])\s*[.):](?:\s.*)?",
                r"(?:the\s+)?(?:correct\s+|final\s+)?(?:option|choice|answer)\s*(?:is)?\s*:?\s*\(?([A-Za-z])\)?[.):]?(?:\s.*)?")
    for index, pattern in enumerate(patterns):
        match = re.fullmatch(pattern, text, flags=re.S | (re.I if index == 3 else 0))
        if match:
            return match.group(1).upper()
    return None


def final_answer(text: str, gts: str) -> dict[str, Any]:
    content, source = extract_answer(text)
    reference = re.fullmatch(r"\\boxed\{([A-Za-z])\}", gts or "")
    letter = None
    if content is not None and reference:
        letter = mcq_letter(content)
        if letter is not None:
            content = letter
    return {"final_answer": f"\\boxed{{{content}}}" if content is not None else (text or ""),
            "answer_found": content is not None, "answer_source": source,
            "mcq_reference": bool(reference), "mcq_letter_parsed": letter is not None if reference else None}


def deviations() -> dict[str, Any]:
    """Manifest record of every intentional difference from the native protocols."""
    return {
        "all": ["greedy decoding; per-round max_tokens 8192 image / 16384 video",
                "client-owned loop: default max 3 turns and 2 operations (native Thyme uses more rounds and 2048-token rounds); overridable, see manifest budget",
                "each returned image is token-budgeted to 1960 visual tokens by default (28px grid), at most 3 images' worth per call as in Reframe; inputs keep MIN/MAX_PIXELS 3136/4014080 by default; both overridable (official checkpoints: 12845056 pixels), see manifest budget; Swift applies MAX_PIXELS to returned images too, so crop_tokens*784 must not exceed max_pixels",
                "a context overflow after the first turn ends the trajectory (flag context_overflow) and the last assistant text is scored",
                "video: the same 64 candidate frames as the Reframe evaluation at Reframe's 128 visual tokens per frame (evaluation.adapters.external_video), given as ordered images with a one-line frame preamble; tools act on the matching original frames",
                "wo_tool: probe direct system prompt 'Answer without calling tools or writing executable code.'; one turn; emitted calls are never executed"],
        "pixelreasoner": ["video frames are target_image 1..N; crops are appended after them; select_frames is not offered",
                          "bbox_2d is normalized when every value is <1 (scale-free), otherwise divided by the target source size as in the probe"],
        "deepeyes": ["bbox_2d interpreted in model-visible smart_resize pixels and mapped to the original image/frame (official env crops the processor-resized image; official MAX_PIXELS 12.8M vs 4.0M here)",
                     "multi-input image_index (1-based) argument added to the schema only when there are >1 inputs",
                     "legacy clip/nonempty crop validation, not official >30px/aspect<=100"],
        "thyme": ["virtual paths /mnt/data/image.png, /mnt/data/image_K.png or /mnt/data/frame_KK.png with original sizes listed; typed AST interpreter, not a Python sandbox",
                  "wo_tool keeps the native Thyme path/size/output-format prompt",
                  "interpreter thyme-virtual-paths v2 (Thyme only): import os/os.path/random/uuid bind inert symbols; os.path join/basename/dirname/splitext/split are pure posixpath string functions and exists/isfile/isdir query a per-operation in-memory namespace (input aliases, saved paths, no-op makedirs/mkdir directories)",
                  "deterministic randomness: random.randint and np.random.randint return their lower bound, random.choice the first item, uuid4() the fixed UUID 00000000-0000-4000-8000-000000000000 (native names are random; only the virtual file name differs)",
                  "Image.save/cv2.imwrite accept any non-input path under /mnt/data/ (<=8 per operation) and keep the image in memory only; print(path) of a saved path, or printed text containing it as a whitespace/quote-delimited token (native stdout path scan), returns that image; nothing is written to disk",
                  "scalar f-strings and bounded str operations, if/for (<=32 items, <=64 iterations, <=256 executed statements), cv2.resize fx/fy, ImageEnhance Contrast/Sharpness/Brightness/Color, ImageOps autocontrast/equalize/grayscale/invert/mirror/flip, rotate, convert('L'), copy; out-of-bounds crops and other unsupported code stay model-invalid"],
        "deepeyesv2": ["protocol v2: image_i are the model-visible views (source resized to the processor's smart_resize size, as the official sandbox binds the resized input), so code coordinates and image_i.size match what the model sees; outputs that are geometric crops of a view are re-rendered from the closest-matching original image/frame at the tracked source box (a resized whole view becomes the whole source, then the returned-image budget applies), photometric or other edits stay at view resolution; boxes partly outside a view are clipped to it (PIL crop semantics, the native evaluator would reject them), counted as clipped_crops",
                       "typed AST interpreter, not Jupyter; no search tools",
                       "wo_tool keeps the native user suffix, which still mentions Python code"],
    }
