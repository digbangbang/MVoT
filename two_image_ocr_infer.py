import os
import re
import argparse
from types import SimpleNamespace

from PIL import Image

import torch
from transformers.generation import StoppingCriteriaList
from transformers import StopStringCriteria

from utils.postprocess_logits_utils import split_token_sequence
from utils.load_model import load_model


PROMPT_NO_ACTIONS = (
    "Task: FrozenLake\n"
    "Use the images to read the action instructions and determine the outcome.\n"
    "Return A, B or C.\n"
    "A. Action Success.\n"
    "B. Action Failed: Fall into the Hole.\n"
    "C. Action Failed: Agent Safe but Fail to Reach Destination.\n"
    "Initial State and Instructions:\n<image>\n<image>\n"
)


def main():
    parser = argparse.ArgumentParser(description="Two-image OCR-level inference. Pass two OCR image paths.")
    parser.add_argument("--image_path1", type=str, required=True, help="Direct path to OCR image 1")
    parser.add_argument("--image_path2", type=str, required=True, help="Direct path to OCR image 2")
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--image_seq_length", type=int, default=1024)
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save decoded images (if any)")

    args = parser.parse_args()

    img_path1, img_path2 = args.image_path1, args.image_path2
    if not os.path.exists(img_path1):
        raise FileNotFoundError(f"image_path1 not found: {img_path1}")
    if not os.path.exists(img_path2):
        raise FileNotFoundError(f"image_path2 not found: {img_path2}")

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

    # Prompt with two <image> placeholders
    input_text = PROMPT_NO_ACTIONS

    print("===== INPUT PROMPT =====")
    print(input_text)
    print("Image 1:", img_path1)
    print("Image 2:", img_path2)

    tokenized = processor(
        text=input_text,
        padding="max_length",
        return_tensors="pt",
        max_length=2600,
    )
    tokenized = {k: v.to(model.device) for k, v in tokenized.items()}

    # Prepare pixel values for both images
    img1 = Image.open(img_path1).convert("RGB")
    img2 = Image.open(img_path2).convert("RGB")
    pv1 = processor.image_processor(img1, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.bfloat16)
    pv2 = processor.image_processor(img2, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.bfloat16)

    stopping = StoppingCriteriaList([StopStringCriteria(stop_strings=["<reserved08706>", "</s>"], tokenizer=processor.tokenizer)])

    # Output dir
    base_dir = args.save_dir or os.path.join(os.getcwd(), "_two_image_ocr_out")
    os.makedirs(base_dir, exist_ok=True)

    with torch.no_grad():
        # Get tokens for both images and concatenate
        t1 = model.model.model.get_image_tokens(pv1).to(torch.int64).reshape(-1)
        t2 = model.model.model.get_image_tokens(pv2).to(torch.int64).reshape(-1)
        all_img_tokens = torch.cat([t1, t2], dim=0)

        input_ids = tokenized["input_ids"].clone()
        mask = (input_ids == model.config.image_token_id)
        num_placeholders = int(mask.sum().item())
        if num_placeholders != all_img_tokens.numel():
            # Try to handle when tokenizer expanded to blocks per image
            # It should equal 2 * model.image_token_num
            expected = getattr(model, "image_token_num", None)
            if expected is not None and num_placeholders == 2 * expected:
                # OK, lengths match expected; ensure our concatenated tokens length equals num_placeholders
                if all_img_tokens.numel() != num_placeholders:
                    # Resize by trunc/pad as last resort
                    if all_img_tokens.numel() > num_placeholders:
                        all_img_tokens = all_img_tokens[:num_placeholders]
                    else:
                        pad = torch.full((num_placeholders - all_img_tokens.numel(),), processor.tokenizer.pad_token_id, dtype=all_img_tokens.dtype, device=all_img_tokens.device)
                        all_img_tokens = torch.cat([all_img_tokens, pad], dim=0)
            else:
                raise RuntimeError(
                    f"Mismatch between number of <image> placeholders ({num_placeholders}) and provided image tokens ({all_img_tokens.numel()})."
                )

        input_ids[mask] = all_img_tokens

        generated_tokens, _ = model.generate(
            input_ids=input_ids,
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            stopping_criteria=stopping,
            multimodal_generation_mode="interleaved-text-image",
        )

    parts = split_token_sequence(
        tokens=generated_tokens,
        image_seq_length=model.image_token_num,
        boi=model.config.boi_token_id,
        eoi=model.config.eoi_token_id,
        max_length=generated_tokens.shape[-1],
        pad_token_id=model.config.eos_token_id,
    )
    pred_text = processor.batch_decode(parts['texts'], skip_special_tokens=True)[0]
    if parts["images"] is not None:
        for i, img_tokens in enumerate(parts["images"]):
            out = model.decode_image_tokens(img_tokens.to(model.device))
            out = processor.postprocess_pixel_values(out).squeeze()
            out = Image.fromarray(out.permute(1, 2, 0).detach().cpu().numpy())
            out.save(os.path.join(base_dir, f"{i}.jpg"))

    print("===== MODEL OUTPUT =====")
    print(pred_text)
    print(f"Generated images (if any) are saved under: {base_dir}")


if __name__ == "__main__":
    main()
