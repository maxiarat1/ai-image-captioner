# Guide: Adding a New Model to AI Image Captioner

This guide explains how to add a new AI model to the system. The architecture uses a config-driven approach that makes adding models straightforward - in most cases, you only need to add a single line to a configuration file!

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Step-by-Step Guide](#step-by-step-guide)
4. [Model Types](#model-types)
5. [Configuration Options](#configuration-options)
6. [Advanced: Custom Handlers](#advanced-custom-handlers)
7. [Complete Example](#complete-example)

---

## Overview

The system uses three layers:

```
┌─────────────────────────────────────┐
│  models_config.jsonl                │  ← Add your model here (1 line!)
│  Single-line model definitions      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Model Adapter Factory              │  ← Auto-generates adapters
│  Creates adapters from config       │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Handlers (by model type)           │  ← Reusable implementations
│  hf_vlm, hf_tagger, hf_ocr, etc.   │
└─────────────────────────────────────┘
```

**Most models** can be added by just editing `models_config.jsonl` - no code changes needed!

---

## Quick Start

### For Standard HuggingFace Models

1. Open `backend/models_config.jsonl`
2. Add a new line with your model configuration
3. Restart the backend
4. Your model is ready to use!

**That's it!** The system will automatically:
- Create the adapter
- Load the model
- Handle inference
- Add it to the UI

---

## Step-by-Step Guide

### Step 1: Identify Your Model Type

Choose the appropriate type based on what your model does:

| Type | Description | Examples |
|------|-------------|----------|
| `hf_vlm` | Vision-Language Models (captioning) | BLIP, BLIP2, LLaVA |
| `hf_tagger` | Image tagging/classification | WD-ViT, WD-EVA02 |
| `hf_ocr` | Optical Character Recognition | Nanonets-OCR, Chandra |
| `hf_classifier` | Standard ImageNet classifiers | ViT, ResNet |
| `onnx_tagger` | ONNX-based taggers | WD14-ConvNext |
| `hf_ocr_trocr` | TrOCR-specific models | TrOCR variants |
| `hf_vlm_custom` | VLMs needing custom logic | Janus, R4B |

### Step 2: Find Required Information

You need these details from HuggingFace:

- **Model ID**: `organization/model-name` (from HuggingFace model card)
- **Processor Class**: Usually `AutoProcessor` (check model docs)
- **Model Class**: E.g., `BlipForConditionalGeneration`, `AutoModel`

### Step 3: Add Configuration Line

Open `backend/models_config.jsonl` and add a new line:

```json
{"model_key": "my-model", "model_id": "org/model-name", "type": "hf_vlm", "processor_class": "AutoProcessor", "model_class": "AutoModelForCausalLM", "category": "general", "description": "My custom model", "vlm_capable": true, "supports_prompts": true, "supports_batch": true, "default_precision": "float32"}
```

**Important**: Each model must be on a single line (JSONL format).

### Step 4: Restart Backend

```bash
python backend/app.py
```

### Step 5: Test Your Model

The model will appear in the UI automatically. Try generating a caption!

---

## Model Types

### Vision-Language Models (hf_vlm)

**Use for**: Image captioning, visual question answering

**Example**:
```json
{"model_key": "my-vlm", "model_id": "Salesforce/blip2-opt-2.7b", "type": "hf_vlm", "processor_class": "Blip2Processor", "model_class": "Blip2ForConditionalGeneration", "category": "general", "description": "BLIP2 for captioning", "vlm_capable": true, "supports_prompts": true, "supports_batch": true, "default_precision": "bfloat16"}
```

**Supports**:
- Text prompts
- Batch processing
- Multiple precision modes (float32, float16, bfloat16, 4bit, 8bit)
- Flash Attention 2

---

### Image Taggers (hf_tagger)

**Use for**: Multi-label classification, anime tagging

**Example**:
```json
{"model_key": "my-tagger", "model_id": "SmilingWolf/wd-vit-large-tagger-v3", "type": "hf_tagger", "processor_class": "AutoImageProcessor", "model_class": "AutoModelForImageClassification", "category": "anime", "description": "Anime tagging model", "vlm_capable": false, "supports_prompts": false, "supports_batch": true, "default_precision": "float32", "requires_tags_csv": true}
```

**Note**: Tagger models need a `selected_tags.csv` file in `backend/database/`.

---

### OCR Models (hf_ocr)

**Use for**: Text extraction, document understanding

**Example**:
```json
{"model_key": "my-ocr", "model_id": "nanonets/Nanonets-OCR-s", "type": "hf_ocr", "processor_class": "AutoProcessor", "model_class": "AutoModelForImageTextToText", "category": "ocr", "description": "OCR with table support", "vlm_capable": false, "supports_prompts": false, "supports_batch": true, "default_precision": "bfloat16", "processor_config": {"trust_remote_code": true}, "model_config": {"trust_remote_code": true}}
```

---

### ONNX Models (onnx_tagger)

**Use for**: Optimized inference with ONNX Runtime

**Example**:
```json
{"model_key": "my-onnx", "model_id": "SmilingWolf/wd-v1-4-convnext-tagger-v2", "type": "onnx_tagger", "category": "anime", "description": "Fast ONNX tagging", "vlm_capable": false, "supports_prompts": false, "supports_batch": true, "image_size": 448, "requires_tags_csv": true, "input_tensor_name": "input_1:0"}
```

---

## Configuration Options

### Required Fields

| Field | Description | Example |
|-------|-------------|---------|
| `model_key` | Unique identifier (lowercase, use hyphens) | `"blip2"` |
| `model_id` | HuggingFace model ID | `"Salesforce/blip2-opt-2.7b"` |
| `type` | Model type (see above) | `"hf_vlm"` |
| `category` | UI category | `"general"`, `"anime"`, `"ocr"` |
| `description` | User-facing description | `"Fast image captioning"` |

### HuggingFace Models

| Field | Description | Example |
|-------|-------------|---------|
| `processor_class` | Processor class name | `"AutoProcessor"` |
| `model_class` | Model class name | `"AutoModel"` |

### Capability Flags

| Field | Description | Default |
|-------|-------------|---------|
| `vlm_capable` | Can do VLM tasks | `false` |
| `supports_prompts` | Accepts text prompts | `false` |
| `supports_batch` | Batch processing | `false` |

### Optional Configuration

| Field | Description | Example |
|-------|-------------|---------|
| `default_precision` | Default dtype | `"float32"`, `"bfloat16"` |
| `supported_precisions` | Limit precision options | `["float32", "4bit", "8bit"]` |
| `special_params` | Custom parameters | `["thinking_mode"]` |
| `processor_config` | Processor kwargs | `{"trust_remote_code": true}` |
| `model_config` | Model kwargs | `{"torch_dtype": "auto"}` |
| `requires_tags_csv` | Needs tags CSV file | `true` |

### Precision Restrictions

Some models don't support all precision modes. Restrict them using `supported_precisions`:

```json
{"model_key": "r4b", ..., "supported_precisions": ["float32", "4bit", "8bit"]}
```

**Common restrictions**:
- **BLIP**: `["float32"]` only
- **BLIP2**: `["float32", "bfloat16"]` only
- **R4B**: `["float32", "4bit", "8bit"]` (no float16/bfloat16)

---

## Advanced: Custom Handlers

If your model needs special logic (unique inference, custom preprocessing), create a custom handler.

### When You Need a Custom Handler

- Custom message format (like Janus, R4B)
- Special preprocessing steps
- Unique output post-processing
- Model-specific parameters

### Creating a Custom Handler

1. Add handler to `backend/models/handlers/custom_handlers.py`:

```python
class MyCustomHandler(HuggingFaceVLMHandler):
    """Custom handler for my model."""
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Custom inference logic."""
        # Your custom logic here
        pass
```

2. Register in `backend/models/adapter_factory.py`:

```python
CUSTOM_HANDLER_MAP = {
    'janus': JanusHandler,
    'r4b': R4BHandler,
    'my_custom': MyCustomHandler,  # Add your handler
}
```

3. Use in config with `type: "hf_vlm_custom"`:

```json
{"model_key": "my-model", "type": "hf_vlm_custom", "custom_handler": "my_custom", ...}
```

---

## Complete Example

Let's add a fictional model called **"SuperVision-7B"** - a new vision-language model.

### Step 1: Gather Information

From HuggingFace model card:
- Model ID: `awesome-ai/supervision-7b`
- Type: Vision-language model (captioning)
- Processor: `AutoProcessor`
- Model Class: `AutoModelForVision2Seq`
- Supports prompts: Yes
- Works well with: bfloat16

### Step 2: Create Configuration

```json
{"model_key": "supervision-7b", "model_id": "awesome-ai/supervision-7b", "type": "hf_vlm", "processor_class": "AutoProcessor", "model_class": "AutoModelForVision2Seq", "category": "general", "description": "SuperVision-7B - Advanced image understanding with 7B parameters", "vlm_capable": true, "supports_prompts": true, "supports_batch": true, "default_precision": "bfloat16", "processor_config": {"trust_remote_code": true}, "model_config": {"trust_remote_code": true}}
```

### Step 3: Add to Config File

Open `backend/models_config.jsonl` and add the line at the end:

```jsonl
{"model_key": "blip", ...}
{"model_key": "blip2", ...}
...
{"model_key": "supervision-7b", "model_id": "awesome-ai/supervision-7b", "type": "hf_vlm", "processor_class": "AutoProcessor", "model_class": "AutoModelForVision2Seq", "category": "general", "description": "SuperVision-7B - Advanced image understanding with 7B parameters", "vlm_capable": true, "supports_prompts": true, "supports_batch": true, "default_precision": "bfloat16", "processor_config": {"trust_remote_code": true}, "model_config": {"trust_remote_code": true}}
```

### Step 4: What Happens Automatically

When you restart the backend, the system will:

1. ✅ Load the configuration from JSONL
2. ✅ Register the model as `supervision-7b`
3. ✅ Create a UnifiedModelAdapter using HuggingFaceVLMHandler
4. ✅ Add these parameters to the UI:
   - Precision (float32, float16, bfloat16, 4bit, 8bit)
   - Flash Attention toggle
   - Max Tokens (10-500)
   - Do Sample checkbox
   - Temperature (0.1-2.0, depends on do_sample)
   - Top P (0.0-1.0, depends on do_sample)
   - Top K (0-100, depends on do_sample)
   - Num Beams (1-10)
5. ✅ Add to "General" category in model selector
6. ✅ Support batch processing
7. ✅ Handle model loading with bfloat16 precision
8. ✅ Support text prompts

### Step 5: Using the Model

```python
# In UI or via API
response = requests.post('/caption', json={
    'model': 'supervision-7b',
    'prompt': 'Describe this image in detail.',
    'parameters': {
        'max_new_tokens': 200,
        'do_sample': True,
        'temperature': 0.7
    }
})
```

---

## Troubleshooting

### Model not appearing in UI

1. Check JSONL syntax (use a JSON validator)
2. Ensure `model_key` is unique
3. Restart backend completely
4. Check backend logs for errors

### Model fails to load

1. Verify `model_id` exists on HuggingFace
2. Check `processor_class` and `model_class` names
3. Try adding `"processor_config": {"trust_remote_code": true}`
4. Check VRAM/RAM requirements

### Precision errors

Add `supported_precisions` to restrict to working precisions:
```json
"supported_precisions": ["float32", "float16"]
```

### Model needs special parameters

Add to `special_params` and handle in handler:
```json
"special_params": ["my_custom_param"]
```

---

## Summary

**For 90% of models**: Just add one line to `models_config.jsonl`!

**Key points**:
- Use correct `type` for your model
- Set capability flags accurately (`vlm_capable`, `supports_prompts`, etc.)
- Restrict `supported_precisions` if needed
- Add `trust_remote_code` if required by model
- Each config is a single JSON line (JSONL format)

**Need help?**
- Check existing models in `models_config.jsonl` for examples
- See handler implementations in `backend/models/handlers/`

