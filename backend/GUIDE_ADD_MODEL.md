# Adding New Models

**99% of models require only a single line of configuration - no coding needed.**

## Quick Start

1. Open `backend/models_config.jsonl`
2. Copy a similar model's configuration line
3. Edit the model ID and name
4. Save and restart the backend

---

## Table of Contents

1. [What You Need](#what-you-need)
2. [Choose Your Model Type](#choose-your-model-type)
3. [Add the Configuration](#add-the-configuration)
4. [Common Examples](#common-examples)
5. [Configuration Reference](#configuration-reference)
6. [Advanced: Custom Handlers](#advanced-custom-handlers)
7. [Troubleshooting](#troubleshooting)

---

## What You Need

Before adding a model, get this info from the HuggingFace model page:

1. **Model ID** - Example: `Salesforce/blip2-opt-2.7b`
2. **What it does** - Captioning? Tagging? OCR?
3. **Model type** - Usually in the model card or README

That's all you need for most models!

---

## Choose Your Model Type

Select the type that matches your model's functionality:

### Image Captioning (VLM)
- **Type:** `hf_vlm`
- **Examples:** BLIP, BLIP2, LLaVA, Florence
- **Use when:** Model generates text descriptions from images

### Image Tagging
- **Type:** `hf_tagger`
- **Examples:** WD-ViT, WD-EVA02
- **Use when:** Model outputs multiple tags/labels per image
- **Note:** Requires a `selected_tags.csv` file

### Text Recognition (OCR)
- **Type:** `hf_ocr`
- **Examples:** Nanonets-OCR, Chandra
- **Use when:** Model extracts text from images

### Image Classification
- **Type:** `hf_classifier`
- **Examples:** ViT, ResNet
- **Use when:** Model classifies images into categories

### ONNX Models
- **Type:** `onnx_tagger`
- **Examples:** WD14-ConvNext
- **Use when:** You have an ONNX format model

### Custom Models
- **Type:** `hf_vlm_custom`
- **Use when:** Model requires unique handling (rarely needed)

---

## Add the Configuration

### Basic Template

Open `backend/models_config.jsonl` and add ONE line:

```json
{"model_key": "unique-name", "model_id": "huggingface/model-id", "type": "MODEL_TYPE", "processor_class": "AutoProcessor", "model_class": "AutoModel", "category": "general", "description": "Short description", "vlm_capable": true, "supports_prompts": true, "supports_batch": true, "default_precision": "bfloat16"}
```

**Just change:**
- `unique-name` → Your choice (lowercase, use hyphens)
- `huggingface/model-id` → From HuggingFace
- `MODEL_TYPE` → From table above
- `description` → What users see in UI
- Set `vlm_capable`, `supports_prompts` to `true` or `false`

**Then restart:** `python backend/app.py`

---

## Common Examples

### Example 1: Captioning Model (BLIP2)

```json
{"model_key": "blip2", "model_id": "Salesforce/blip2-opt-2.7b", "type": "hf_vlm", "processor_class": "Blip2Processor", "model_class": "Blip2ForConditionalGeneration", "category": "general", "description": "BLIP2 - Image captioning", "vlm_capable": true, "supports_prompts": true, "supports_batch": true, "default_precision": "bfloat16"}
```

### Example 2: Anime Tagging Model

```json
{"model_key": "wd-tagger", "model_id": "SmilingWolf/wd-vit-large-tagger-v3", "type": "hf_tagger", "processor_class": "AutoImageProcessor", "model_class": "AutoModelForImageClassification", "category": "anime", "description": "WD Tagger - Anime tags", "vlm_capable": false, "supports_prompts": false, "supports_batch": true, "default_precision": "float32", "requires_tags_csv": true}
```

**Don't forget:** Taggers need `selected_tags.csv` in `backend/database/`

### Example 3: OCR Model

```json
{"model_key": "nanonets-ocr", "model_id": "nanonets/Nanonets-OCR-s", "type": "hf_ocr", "processor_class": "AutoProcessor", "model_class": "AutoModelForImageTextToText", "category": "ocr", "description": "Nanonets OCR", "vlm_capable": false, "supports_prompts": false, "supports_batch": true, "default_precision": "bfloat16", "processor_config": {"trust_remote_code": true}, "model_config": {"trust_remote_code": true}}
```

### Example 4: ONNX Model (Fast!)

```json
{"model_key": "wd-onnx", "model_id": "SmilingWolf/wd-v1-4-convnext-tagger-v2", "type": "onnx_tagger", "category": "anime", "description": "Fast WD Tagger (ONNX)", "vlm_capable": false, "supports_prompts": false, "supports_batch": true, "image_size": 448, "requires_tags_csv": true, "input_tensor_name": "input_1:0"}
```

---

## Configuration Reference

### Required Fields (Always Needed)

```json
{
  "model_key": "unique-name",           // Your choice, lowercase with hyphens
  "model_id": "org/model-name",         // From HuggingFace
  "type": "hf_vlm",                     // See types above
  "category": "general",                // UI category: general, anime, ocr
  "description": "What users see"       // Short, clear description
}
```

### Common Optional Fields

```json
{
  "processor_class": "AutoProcessor",           // Usually AutoProcessor
  "model_class": "AutoModel",                   // Check HF docs
  "vlm_capable": true,                          // Can do VLM tasks?
  "supports_prompts": true,                     // Takes text prompts?
  "supports_batch": true,                       // Batch processing?
  "default_precision": "bfloat16"               // float32, float16, bfloat16, 4bit, 8bit
}
```

### Advanced Options (Use When Needed)

```json
{
  "supported_precisions": ["float32", "4bit"],  // Limit precision choices
  "processor_config": {"trust_remote_code": true},  // For special models
  "model_config": {"trust_remote_code": true},      // For special models
  "requires_tags_csv": true,                        // For taggers only
  "special_params": ["thinking_mode"]               // Custom parameters
}
```

### When to Use `trust_remote_code`

Add this if the model card says "requires trust_remote_code":

```json
"processor_config": {"trust_remote_code": true},
"model_config": {"trust_remote_code": true}
```

### Precision Tips

Most models work with: `"default_precision": "bfloat16"`

If you get errors, try:
1. `"float32"` - Safest, uses more VRAM
2. `"4bit"` - For huge models, saves VRAM
3. Limit with: `"supported_precisions": ["float32", "bfloat16"]`

---

## Advanced: Custom Handlers

**99% of models don't need this!** Only use if your model has truly unique requirements.

### When You Actually Need This

Your model needs custom code if it:
- Uses a unique conversation format (like Janus)
- Has special preprocessing not in HuggingFace
- Needs custom post-processing logic
- Has model-specific generation methods

**Examples:** Janus, R4B, TrOCR (already included!)

### Creating a Custom Handler

#### Step 1: Create Handler File

Create `backend/models/handlers/my_model_handler.py`:

```python
"""My Custom Model Handler"""
import logging
from PIL import Image
from typing import Dict, Optional
from .hf_vlm_handler import HuggingFaceVLMHandler

logger = logging.getLogger(__name__)


class MyModelHandler(HuggingFaceVLMHandler):
    """Handler for MyModel with special requirements."""

    def infer_single(self, image: Image.Image, prompt: Optional[str] = None,
                    parameters: Optional[Dict] = None) -> str:
        """Custom inference logic."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")

        try:
            # Your custom logic here
            # Example: special prompt formatting, unique generation, etc.
            result = "Your custom inference"
            return result
        except Exception as e:
            logger.exception("Error in custom inference: %s", e)
            return f"Error: {str(e)}"
```

#### Step 2: Register Handler

Add to `backend/models/handlers/__init__.py`:

```python
from .my_model_handler import MyModelHandler

__all__ = [
    # ... existing handlers ...
    'MyModelHandler',
]
```

#### Step 3: Register in Factory

Add to `CUSTOM_HANDLER_MAP` in `backend/models/adapter_factory.py`:

```python
CUSTOM_HANDLER_MAP = {
    'janus': JanusHandler,
    'r4b': R4BHandler,
    # ... other handlers ...
    'my-model': MyModelHandler,  # Add your handler
}
```

#### Step 4: Use in Config

```json
{"model_key": "my-special-model", "model_id": "org/model", "type": "hf_vlm_custom", "custom_handler": "my-model", "processor_class": "AutoProcessor", "model_class": "AutoModel", "category": "general", "description": "My special model", "vlm_capable": true, "supports_prompts": true, "supports_batch": true, "default_precision": "bfloat16"}
```

**Key points:**
- Handler file name: `my_model_handler.py`
- Class name: `MyModelHandler` (CamelCase)
- Registry key: `'my-model'` (lowercase with hyphens)
- Use `type: "hf_vlm_custom"` in config

---

## Complete Walkthrough: Adding a New Model

Let's add **CogVLM-Chat** step by step!

### What We Have

Looking at HuggingFace page for `THUDM/cogvlm-chat-hf`:
- It's a captioning model (VLM)
- Uses standard HuggingFace transformers
- Supports prompts
- Works with bfloat16

### Step 1: Open Config File

Open `backend/models_config.jsonl` in your editor.

### Step 2: Copy Similar Model

Find a similar model (like BLIP2) and copy its line.

### Step 3: Edit the Line

Change to:
```json
{"model_key": "cogvlm-chat", "model_id": "THUDM/cogvlm-chat-hf", "type": "hf_vlm", "processor_class": "AutoProcessor", "model_class": "AutoModelForCausalLM", "category": "general", "description": "CogVLM - Conversational image understanding", "vlm_capable": true, "supports_prompts": true, "supports_batch": true, "default_precision": "bfloat16"}
```

### Step 4: Add to End of File

Paste it as a new line at the bottom of `models_config.jsonl`.

### Step 5: Save and Restart

```bash
# Save the file
# Then restart backend:
python backend/app.py
```

### Step 6: Test!

The model appears in your UI automatically. Select it and try captioning an image!

### What Happened Behind the Scenes

The system automatically:
- Loaded your configuration
- Created an adapter with the VLM handler
- Added UI controls (precision, max tokens, temperature, etc.)
- Configured batch processing
- Enabled prompt support

**Result:** Full model integration with a single configuration line.

---

## Troubleshooting

### Model Not Showing in UI

**Checklist:**
1. Verify each model is on a single line (JSONL format)
2. Ensure `model_key` is unique
3. Restart the backend completely
4. Check backend logs for errors

**Quick fix:** Copy a working model's configuration and modify it.

### Model Won't Load

**Try these steps in order:**
1. Verify model exists on HuggingFace
2. Set `"default_precision": "float32"` (safest option)
3. Add `trust_remote_code` if the model card requires it:
   ```json
   "processor_config": {"trust_remote_code": true},
   "model_config": {"trust_remote_code": true}
   ```
4. Check VRAM availability - consider 4bit/8bit quantization for large models

### Precision Errors

Some models only support specific precisions. Restrict them explicitly:

```json
"supported_precisions": ["float32", "bfloat16"]
```

**Common configurations:**
- BLIP: `["float32"]` only
- Most modern models: `["float32", "bfloat16"]`

### Tagger Not Working

**Required steps for tagger models:**
1. Download `selected_tags.csv` from the model repository
2. Place it in `backend/database/`
3. Restart the backend

### Best Practices

1. **Start minimal** - Begin with basic configuration, add options incrementally
2. **Reference existing models** - Use `models_config.jsonl` as a template source
3. **Monitor logs** - Backend provides detailed error messages
4. **Default to float32** - Use this precision when uncertain

---

## Quick Reference

### Minimal VLM Config
```json
{"model_key": "name", "model_id": "org/model", "type": "hf_vlm", "processor_class": "AutoProcessor", "model_class": "AutoModel", "category": "general", "description": "Description", "vlm_capable": true, "supports_prompts": true, "supports_batch": true, "default_precision": "bfloat16"}
```

### Minimal Tagger Config
```json
{"model_key": "name", "model_id": "org/model", "type": "hf_tagger", "processor_class": "AutoImageProcessor", "model_class": "AutoModelForImageClassification", "category": "anime", "description": "Description", "vlm_capable": false, "supports_prompts": false, "supports_batch": true, "default_precision": "float32", "requires_tags_csv": true}
```

### Minimal OCR Config
```json
{"model_key": "name", "model_id": "org/model", "type": "hf_ocr", "processor_class": "AutoProcessor", "model_class": "AutoModelForImageTextToText", "category": "ocr", "description": "Description", "vlm_capable": false, "supports_prompts": false, "supports_batch": true, "default_precision": "bfloat16"}
```

---

## Summary

**Standard workflow for most models:**

1. Locate model on HuggingFace
2. Copy similar configuration from `models_config.jsonl`
3. Modify `model_key`, `model_id`, and `description`
4. Save and restart backend

**No coding required** - configuration-driven model integration.

**For reference:** Existing models in `models_config.jsonl` provide working examples.
