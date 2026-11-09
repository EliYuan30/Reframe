import argparse
import asyncio
import json
import os
from pathlib import Path
from datetime import datetime

INFERENCE_DIR = Path(__file__).resolve().parent

def prepare_request(messages, output):
    import numpy as np
    from PIL import Image
    from swift.infer_engine.protocol import RolloutInferRequest
    from reframe.plugin import _resize_to_token_budget, _tool_instructions, _tool_schemas

    images, videos, source_frames, conversation = [], [], [], []
    for message in messages:
        text = []
        for item in message["content"]:
            if item["type"] == "text":
                text.append(item["text"])
            elif item["type"] == "image":
                images.append(str((INFERENCE_DIR / item["image"]).resolve(strict=True)))
                text.append("<image>")
            elif item["type"] == "video":
                if videos:
                    raise ValueError("This example supports one video per conversation.")
                import decord
                video_path = (INFERENCE_DIR / item["video"]).resolve(strict=True)
                reader = decord.VideoReader(str(video_path), num_threads=1)
                indices = np.linspace(0, len(reader) - 1, min(64, len(reader))).round().astype(int)
                frames_dir = output / f"{Path(item['video']).stem}_input_frames"
                frames_dir.mkdir(exist_ok=True)
                for index, frame in enumerate(reader.get_batch(indices).asnumpy(), 1):
                    path = frames_dir / f"frame_{index:03d}.png"
                    Image.fromarray(frame).save(path)
                    source_frames.append(str(path))
                videos.append([_resize_to_token_budget(path, 128) for path in source_frames])
                (frames_dir / "indices.json").write_text(json.dumps({
                    "source_video": str(video_path),
                    "original_frame_indices_zero_based": indices.tolist(),
                }, indent=2))
                text.append("<video>")
            else:
                raise ValueError(f"Unsupported content type: {item['type']}")
        conversation.append({"role": message["role"], "content": "".join(text)})
    mode = "mix" if videos and images else "video" if videos else "image"
    conversation.insert(0, {"role": "system", "content": _tool_instructions(mode)})
    os.environ["REFRAME_MODE"] = mode
    return RolloutInferRequest(
        messages=conversation, images=images, videos=videos, tools=_tool_schemas(mode),
        data_dict={"tools_enabled": True, "reframe_training": False,
                   "tool_video_frames": source_frames},
    )


def run(args):
    os.environ.update(IMAGE_MAX_TOKEN_NUM="5120", IMAGE_MIN_TOKEN_NUM="4",
                          MAX_PIXELS="5242880", MIN_PIXELS="4096", VIDEO_MAX_TOKEN_NUM="128",
                          REFRAME_CROP_MAX_TOKEN_NUM="1960", REFRAME_SELECT_MAX_TOKEN_NUM="400",
                          REFRAME_MAX_TOOL_VISUAL_TOKENS="4096")
    visual = (INFERENCE_DIR / args.visual.expanduser()).resolve(strict=True)
    mode = "video" if visual.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"} else "image"
    messages = [{"role": "user", "content": [
        {"type": mode, mode: str(visual)},
        {"type": "text", "text": args.question},
    ]}]
    name = visual.stem
    output = INFERENCE_DIR / f"tool_returned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True)
    request = prepare_request(messages, output)

    import torch
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
    from reframe.plugin import ObservationGatedScheduler, _parse_tool_call, _to_pil

    engine = TransformersEngine(args.model, model_type="qwen3_5", torch_dtype=torch.bfloat16,
                                device_map="cuda:0", attn_impl="sdpa")
    scheduler = ObservationGatedScheduler(max_turns=3)
    asyncio.run(scheduler.on_trajectory_start([request]))
    state = scheduler._state(request)
    config = RequestConfig(max_tokens=4096, temperature=1.0, top_p=0.95, stop=["</tool_call>"])
    record = {"model": args.model, "input_messages": messages, "turns": [], "final_answer": None}

    def save():
        (output / f"{name}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for message in request.messages:
        print(f"\n{message['role']}:\n{message['content']}", flush=True)
    save()
    for turn in range(3):
        choice = engine.infer([InferRequest(messages=request.messages, images=request.images,
                                           videos=request.videos, tools=request.tools)],
                              config, use_tqdm=False)[0].choices[0]
        content = choice.message.content or ""
        print(f"\nAssistant (turn {turn + 1}):\n{content}", flush=True)
        request.messages.append({"role": "assistant", "content": content})
        entry = {"assistant": content, "finish_reason": choice.finish_reason}
        record["turns"].append(entry)
        call = _parse_tool_call(content)
        if call is None:
            record["final_answer"] = content
            record["finish_reason"] = choice.finish_reason
            save()
            break
        if turn == 2:
            record["finish_reason"] = "tool_limit"
            save()
            break
        before = len(request.images)
        response = scheduler._execute(request, state, call)
        request.messages.append({"role": "user", "content": response})
        entry.update(tool_call=state["calls"][-1], tool_response=response, returned_images=[])
        for index, value in enumerate(request.images[before:], 1):
            path = output / f"{name}_turn_{turn + 1}_{call['name']}_{index:02d}.png"
            _to_pil(value).save(path)
            entry["returned_images"].append(path.name)
        save()
        print(f"\nTool ({call['name']}):\n{response}", flush=True)
        print(f"Saved {len(entry['returned_images'])} images to {output}", flush=True)
        if entry["tool_call"].get("infra_error"):
            raise RuntimeError(entry["tool_call"]["infra_error"])
    print(f"\nSaved conversation and tool observations to {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--question", required=True)
    run(parser.parse_args())
