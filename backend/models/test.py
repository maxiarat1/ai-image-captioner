import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

# --- CONFIG ---
model_path = "nanonets/Nanonets-OCR-s"
image_path = "/home/max/Pictures/Screenshots/Screenshot from 2025-09-21 22-05-40.png"

# --- LOAD MODEL ---
print("Loading model...")

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)  # force fast processor

device = model.device
print(f"✅ Model loaded on: {device}")

# --- OCR FUNCTION ---
def ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=1024):
    """
    Extract text, tables, equations, and images from a document image using Nanonets OCR-S.
    """
    prompt = """Extract the text from the above document as if you were reading it naturally.
    Return the tables in HTML format. Return the equations in LaTeX representation.
    If there is an image in the document and an image caption is not present, add a small description
    of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>.
    Watermarks should be wrapped in <watermark></watermark>.
    Page numbers should be wrapped in <page_number></page_number>.
    Prefer using ☐ and ☑ for check boxes.
    """

    # --- Load and resize image ---
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((1024, 1024))

    # --- Use chat-style input (required for Qwen2VL) ---
    messages = [
        {"role": "system", "content": "You are a helpful OCR assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]

    print("Applying chat template...")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("Encoding inputs...")
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

    print(f"Generating output (max_new_tokens={max_new_tokens})...")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    print("Decoding output...")
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    print("✅ OCR complete.")
    return output_text


# --- RUN OCR ---
print("Starting OCR...")
result = ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=1024)

# --- DISPLAY RESULT ---
print("\n--- OCR RESULT ---\n")
print(result[:3000])
