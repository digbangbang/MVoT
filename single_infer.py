import os
import argparse
import json
from types import SimpleNamespace

from PIL import Image

import torch
from transformers.generation import StoppingCriteriaList
from transformers import StopStringCriteria

from utils.load_model import load_model


REAL_GOAL_INSTRUCTION = (
    "Task: FrozenLake\n"
    "Determine whether the agent (elf character) can safely reach the gift following the action sequence without falling into the holes. If not, identify the failure reason. The definitions of the actions are as below. \n"
    "* Go up/left/down/right: move one grid space in the absolute up/left/down/right direction. \n"
    "Return A, B or C. \n"
    "Full Action Sequence: <ACTION_SEQ>\n"
    "A. Action Success. \n"
    "B. Action Failed: Fall into the Hole. \n"
    "C. Action Failed: Agent Safe but Fail to Reach Destination. \n"
)

LONG_HORIZON_VISUALIZATION_INSTRUCTION = (
    "<INIT_STATE>\nResponse: <ACTION_HISTORY>"
)

ACTION_DICT = {
    0: "left",
    1: "down",
    2: "right",
    3: "up",
}


def build_prompt_from_json(img_path: str, data_json_path: str) -> str:
    # Find data_id from path like .../<data_id>/<t>.png
    parts = os.path.normpath(img_path).split(os.sep)
    # assume last two parts are <t>.png and <data_id>
    if len(parts) < 2:
        raise ValueError(f"Unexpected image path: {img_path}")
    try:
        data_id = int(parts[-2])
    except Exception:
        raise ValueError(f"Cannot parse data_id from path: {img_path}")

    with open(data_json_path, "r") as f:
        data = json.load(f)

    actions = None
    additional_actions = []
    for env_key, env_dict in data.items():
        ids = env_dict.get("data_id", [])
        if data_id in ids:
            idx = ids.index(data_id)
            actions = env_dict.get("actions", [])[idx]
            additional_actions = env_dict.get("additional_actions", [[]])[idx] if env_dict.get("additional_actions") else []
            break
    if actions is None:
        raise ValueError(f"data_id {data_id} not found in {data_json_path}")

    action_seq = "".join([f"Go {ACTION_DICT[a]}. " for a in (actions + additional_actions)])
    init_state_text = "Initial State: <image>"
    history_text = ""  # first-step prompt
    input_text = LONG_HORIZON_VISUALIZATION_INSTRUCTION.replace("<INIT_STATE>", init_state_text).replace("<ACTION_HISTORY>", history_text)
    input_text = REAL_GOAL_INSTRUCTION + input_text
    input_text = input_text.replace("<ACTION_SEQ>", action_seq)
    return input_text


def main():
    parser = argparse.ArgumentParser(description="Single-image inference with FrozenLake prompt")
    parser.add_argument("--image_path", required=True, help="Path to the input PNG")
    parser.add_argument("--data_json", default="data_samples/frozenlake/level3/data.json", help="data.json for action sequence lookup")
    parser.add_argument("--max_new_tokens", type=int, default=1400)
    parser.add_argument("--image_seq_length", type=int, default=1024)
    parser.add_argument('--do_eval', default=True)
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save generated images")
    
    args = parser.parse_args()

    # Build args for load_model
    lm_args = SimpleNamespace(
        model="anole",
        image_seq_length=args.image_seq_length,
        model_ckpt=None,
        do_eval=True,
        do_train=False,
    )
    mp = load_model(lm_args)
    model, processor = mp["model"], mp["processor"]
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Prompt
    input_text = build_prompt_from_json(args.image_path, args.data_json)

    print("===== INPUT PROMPT =====")
    print(input_text)

    # Tokenize text
    tokenized = processor(
        text=input_text,
        padding="max_length",
        return_tensors="pt",
        max_length=2600,
    )
    tokenized = {k: v.to(model.device) for k, v in tokenized.items()}

    # Prepare pixel_values for recursive_generate (it will handle image tokens replacement)
    img = Image.open(args.image_path).convert("RGB")
    pixel_values = processor.image_processor(img, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.bfloat16)

    stopping = StoppingCriteriaList([StopStringCriteria(stop_strings=["<reserved08706>", "</s>"], tokenizer=processor.tokenizer)])

    # Decide save directory for generated images
    save_dir = args.save_dir or os.path.join(os.path.dirname(args.image_path), "_single_out")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        pred_text, _, _, _ = model.recursive_generate(
            processor=processor,
            input_text=input_text,
            save_dir=save_dir,
            inputs=None,
            max_new_tokens=args.max_new_tokens,
            stopping_criteria=stopping,
            multimodal_generation_mode="interleaved-text-image",
            pixel_values=pixel_values,
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
        )

    print("===== MODEL OUTPUT =====")
    print(pred_text)
    print(f"Generated images (if any) are saved step-wise under: {save_dir}")


if __name__ == "__main__":
    main()


'''
bug TODO infer停止不了？

cd /hpc2hdd/home/zli404/workspace/MVoT && source ~/miniconda3/bin/activate mvot && python single_infer.py --image_path data_samples/frozenlake/ocr_level/0/0.png --data_json data_samples/frozenlake/level3/data.json --max_new_tokens 400 --save_dir outputs/single_image_demo
'''