import os
import json
import argparse
import shutil
from PIL import Image, ImageDraw, ImageFont

ACTION_DICT = {
    0: "left",
    1: "down",
    2: "right",
    3: "up",
}


def load_font(preferred_size: int = 22) -> ImageFont.FreeTypeFont:
    # Try common fonts, then fallback to PIL default.
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, preferred_size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_text_on_bottom(img: Image.Image, text: str, pad_px: int = 64, margin: int = 8) -> Image.Image:
    w, h = img.size
    new_h = h + pad_px
    canvas = Image.new("RGB", (w, new_h), color=(255, 255, 255))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    # scale font roughly with image width
    base_font_size = max(14, min(28, w // 22))
    font = load_font(base_font_size)
    # center vertically within the pad area
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max(margin, (w - text_w) // 2)
    y = h + max(0, (pad_px - text_h) // 2)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return canvas


def process_sequence(src_dir: str, dst_dir: str, data_id: str, actions: list, pad_px: int) -> None:
    # For each t in actions, put next instruction on image t.png
    for t, act in enumerate(actions):
        src_img = os.path.join(src_dir, str(data_id), f"{t}.png")
        dst_img = os.path.join(dst_dir, str(data_id), f"{t}.png")
        if not os.path.exists(src_img):
            continue
        try:
            with Image.open(src_img) as im:
                im = im.convert("RGB")
                direction = ACTION_DICT.get(int(act), None)
                if direction is None:
                    text = ""
                else:
                    text = f"Go {direction}."
                out = draw_text_on_bottom(im, text, pad_px=pad_px)
                out.save(dst_img)
        except Exception:
            # If anything fails, leave the original already-copied image
            pass


def main():
    parser = argparse.ArgumentParser(description="Augment FrozenLake images by adding bottom text with next action.")
    parser.add_argument("--src_dir", default="data_samples/frozenlake/level3", help="Source folder (original)")
    parser.add_argument("--dst_dir", default="data_samples/frozenlake/ocr_level", help="Destination folder (augmented copy)")
    parser.add_argument("--pad_px", type=int, default=64, help="Padding pixels to add at image bottom")
    parser.add_argument("--data_json", default=None, help="Path to data.json; defaults to <src_dir>/data.json")
    args = parser.parse_args()

    src_dir = args.src_dir
    dst_dir = args.dst_dir
    data_json = args.data_json or os.path.join(src_dir, "data.json")

    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source dir not found: {src_dir}")
    if not os.path.exists(data_json):
        raise FileNotFoundError(f"data.json not found: {data_json}")

    # Copy the entire directory tree first (keeps original untouched)
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    if not os.path.exists(dst_dir):
        shutil.copytree(src_dir, dst_dir)

    with open(data_json, "r") as f:
        data = json.load(f)

    # Iterate environments and samples
    for env_key, env_dict in data.items():
        actions_list = env_dict.get("actions", [])
        data_ids = env_dict.get("data_id", [])
        for idx, data_id in enumerate(data_ids):
            actions = actions_list[idx]
            # Ensure destination subdir exists
            os.makedirs(os.path.join(dst_dir, str(data_id)), exist_ok=True)
            process_sequence(src_dir, dst_dir, data_id, actions, pad_px=args.pad_px)

    print(f"Augmented images written under: {dst_dir}")


if __name__ == "__main__":
    main()
