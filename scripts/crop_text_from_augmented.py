import os
import argparse
from PIL import Image


def crop_bottom_text(img: Image.Image, pad_px: int) -> Image.Image:
    """Given an augmented image (original + bottom pad with text),
    return only the bottom pad area containing the text.

    If the image height is less than or equal to pad_px, return the whole image.
    """
    w, h = img.size
    if pad_px >= h:
        return img.copy()
    # Crop the bottom band: [h - pad_px, h)
    return img.crop((0, h - pad_px, w, h))


def process_dir(src_dir: str, dst_dir: str, pad_px: int) -> None:
    for root, _, files in os.walk(src_dir):
        for fn in files:
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            src_path = os.path.join(root, fn)
            rel = os.path.relpath(src_path, start=src_dir)
            out_path = os.path.join(dst_dir, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                with Image.open(src_path) as im:
                    im = im.convert("RGB")
                    cropped = crop_bottom_text(im, pad_px)
                    cropped.save(out_path)
            except Exception as e:
                # Best-effort; skip problematic files
                print(f"[WARN] Failed {src_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Keep only the bottom text area from augmented images.")
    parser.add_argument("--src_dir", default="data_samples/frozenlake/ocr_level", help="Source augmented images directory")
    parser.add_argument("--dst_dir", default="data_samples_frozenlake/text", help="Destination directory for text-only images")
    parser.add_argument("--pad_px", type=int, default=64, help="Padding pixels used when augmenting (height of text area)")
    args = parser.parse_args()

    if not os.path.exists(args.src_dir):
        raise FileNotFoundError(f"Source dir not found: {args.src_dir}")

    os.makedirs(args.dst_dir, exist_ok=True)

    process_dir(args.src_dir, args.dst_dir, args.pad_px)
    print(f"Text-only images written under: {args.dst_dir}")


if __name__ == "__main__":
    main()
