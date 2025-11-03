import torch
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os

def doctr_trocr_ocr(image_path: str, save_to_file: bool = True):
    # -------------------------------------------------------------
    # 1️⃣ Load Models
    # -------------------------------------------------------------
    print("Loading models...")
    doctr_model = ocr_predictor(pretrained=True)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trocr_model.to(device)

    # -------------------------------------------------------------
    # 2️⃣ Load Image
    # -------------------------------------------------------------
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    doc = DocumentFile.from_images([image_path])

    # -------------------------------------------------------------
    # 3️⃣ Detect text regions with docTR
    # -------------------------------------------------------------
    print("Running text detection (docTR)...")
    doctr_result = doctr_model(doc)
    exported = doctr_result.export()
    page = exported['pages'][0]

    word_boxes = []
    for block in page['blocks']:
        for line in block['lines']:
            for word in line['words']:
                ((x1, y1), (x2, y2)) = word['geometry']
                x1, y1, x2, y2 = (
                    int(x1 * width), int(y1 * height),
                    int(x2 * width), int(y2 * height)
                )
                word_boxes.append((x1, y1, x2, y2))

    if not word_boxes:
        print("⚠️ No text detected in the image.")
        return ""

    print(f"✅ Detected {len(word_boxes)} text regions")

    # -------------------------------------------------------------
    # 4️⃣ Recognize each region with TrOCR
    # -------------------------------------------------------------
    results = []
    print("Running text recognition (TrOCR)...")
    for (x1, y1, x2, y2) in word_boxes:
        crop = image.crop((x1, y1, x2, y2))
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue
        pixel_values = processor(crop, return_tensors="pt").pixel_values.to(device)
        generated_ids = trocr_model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if text:
            results.append({"bbox": (x1, y1, x2, y2), "text": text})

    if not results:
        print("⚠️ No readable text recognized.")
        return ""

    # -------------------------------------------------------------
    # 5️⃣ Sort & Group into readable text lines
    # -------------------------------------------------------------
    results.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    lines = []
    current_line = []
    line_threshold = 20  # tweak depending on image resolution

    for r in results:
        if not current_line:
            current_line.append(r)
            continue
        prev_y = current_line[-1]["bbox"][1]
        curr_y = r["bbox"][1]
        if abs(curr_y - prev_y) < line_threshold:
            current_line.append(r)
        else:
            current_line.sort(key=lambda w: w["bbox"][0])
            lines.append(" ".join(w["text"] for w in current_line))
            current_line = [r]

    if current_line:
        current_line.sort(key=lambda w: w["bbox"][0])
        lines.append(" ".join(w["text"] for w in current_line))

    # -------------------------------------------------------------
    # 6️⃣ Print & optionally save results
    # -------------------------------------------------------------
    final_text = "\n".join(lines)
    print("\n=== OCR TEXT OUTPUT ===\n")
    print(final_text)

    if save_to_file:
        out_path = os.path.splitext(image_path)[0] + "_ocr.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"\n💾 Text saved to: {out_path}")

    return final_text


# -------------------------------------------------------------
# 🔰 Run Example
# -------------------------------------------------------------
if __name__ == "__main__":
    image_path = "/home/max/Pictures/Screenshots/Screenshot from 2025-10-01 21-09-12.png"  # change this to your image path
    doctr_trocr_ocr(image_path)
