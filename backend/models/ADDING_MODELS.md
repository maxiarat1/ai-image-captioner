# Adding New AI Models Guide

This guide explains how to add new AI vision-language models to the AI Image Captioner application.

## Architecture Overview

The application uses an **adapter pattern** for models:
- Each model has its own adapter class that extends `BaseModelAdapter`
- Models are registered in `app.py` and load on-demand
- Frontend UI is auto-generated from model parameters

## Quick Checklist

To add a new model, you need to:

- [ ] Create model adapter in `backend/models/your_model_adapter.py`
- [ ] Import and register model in `backend/app.py`
- [ ] (Optional) Add precision defaults in `backend/config.py`
- [ ] (Optional) Add dependencies to `requirements.txt`
- [ ] Test the model works

## Step-by-Step Guide

### Step 1: Create Model Adapter

Create a new file in `backend/models/` (e.g., `my_model_adapter.py`):

```python
import torch
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq  # Adjust imports
from .base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)

class MyModelAdapter(BaseModelAdapter):
    # Parameters that should NOT be passed to model.generate()
    # Add any special parameters your model uses (precision, batch_size, etc.)
    SPECIAL_PARAMS = {'precision', 'use_flash_attention', 'batch_size'}

    def __init__(self, model_id: str = "organization/model-name"):
        """
        Initialize your model adapter.

        Args:
            model_id: HuggingFace model ID or path
        """
        super().__init__(model_id)
        self.model_id = model_id
        self.device = self._init_device(torch)
        self.quantization_config = None

    def load_model(self, precision="float16", use_flash_attention=False) -> None:
        """
        Load the model and processor.

        Args:
            precision: Model precision (float32/float16/bfloat16/4bit/8bit)
            use_flash_attention: Whether to use Flash Attention 2
        """
        try:
            logger.info("Loading MyModel %s on %s with %s precision…",
                       self.model_id, self.device, precision)

            # Load processor/tokenizer
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )

            # Setup pad token for batch processing
            self._setup_pad_token()

            # Create quantization config if needed (for 4bit/8bit)
            self.quantization_config = self._create_quantization_config(precision)

            model_kwargs = {
                "trust_remote_code": True,
                "quantization_config": self.quantization_config
            }

            # Set dtype based on precision
            if precision not in ["4bit", "8bit"]:
                dtype = self._get_dtype(precision)
                if dtype != "auto":
                    model_kwargs["torch_dtype"] = dtype

            # Setup flash attention if requested
            if use_flash_attention:
                self._setup_flash_attention(
                    model_kwargs, precision, force_bfloat16=False
                )

            # Load model
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                **model_kwargs
            )

            # Move to device if not using quantization
            if precision not in ["4bit", "8bit"]:
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("MyModel loaded successfully (Precision: %s)", precision)

        except Exception as e:
            logger.exception("Error loading MyModel %s: %s", self.model_id, e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        """
        Generate caption for a single image.

        Args:
            image: PIL Image
            prompt: Optional text prompt
            parameters: Generation parameters

        Returns:
            Generated caption string
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Ensure image is RGB
            image = self._ensure_rgb(image)

            # Set default prompt if needed
            if not prompt or not prompt.strip():
                prompt = "Describe this image in detail."

            # Process inputs (adjust based on your model's API)
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors='pt'
            )

            # Move inputs to device with proper dtype
            model_dtype = next(self.model.parameters()).dtype
            inputs = self._move_inputs_to_device(inputs, self.device, model_dtype)

            # Build generation parameters (filter out special params)
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)

            # Set defaults if needed
            if 'max_new_tokens' not in gen_params:
                gen_params['max_new_tokens'] = 200

            logger.debug("MyModel params: %s", gen_params)

            # Generate output
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)

            # Decode output (skip input tokens)
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][prompt_length:]
            caption = self.processor.decode(new_tokens, skip_special_tokens=True).strip()

            return caption if caption else "Unable to generate description."

        except Exception as e:
            logger.exception("Error generating caption with MyModel: %s", e)
            return f"Error: {str(e)}"

    def generate_captions_batch(self, images: list, prompts: list = None, parameters: dict = None) -> list:
        """
        Generate captions for multiple images using batch processing.

        Args:
            images: List of PIL Images
            prompts: Optional list of prompts (one per image)
            parameters: Generation parameters

        Returns:
            List of generated captions
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Ensure we have prompts for all images
            if prompts is None:
                prompts = ["Describe this image in detail."] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")

            # Convert all images to RGB
            processed_images = self._ensure_rgb(images)

            # Process batch with padding
            inputs = self.processor(
                text=prompts,
                images=processed_images,
                return_tensors='pt',
                padding=True
            )

            # Move inputs to device with proper dtype
            model_dtype = next(self.model.parameters()).dtype
            inputs = self._move_inputs_to_device(inputs, self.device, model_dtype)

            # Build generation parameters
            gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)
            if 'max_new_tokens' not in gen_params:
                gen_params['max_new_tokens'] = 200

            # Generate for batch
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_params)

            # Trim input tokens from outputs
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            # Batch decode
            captions = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True
            )

            # Clean up results
            results = [caption.strip() if caption.strip() else "Unable to generate description."
                      for caption in captions]

            return results

        except Exception as e:
            logger.exception("Error generating captions in batch with MyModel: %s", e)
            return self._format_batch_error(e, len(images))

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.processor is not None

    def get_available_parameters(self) -> list:
        """
        Define generation parameters for frontend UI.

        Returns:
            List of parameter definitions (creates UI controls automatically)
        """
        return [
            {
                "name": "Max New Tokens",
                "param_key": "max_new_tokens",
                "type": "number",
                "min": 1,
                "max": 500,
                "step": 1,
                "description": "Maximum number of tokens to generate"
            },
            {
                "name": "Temperature",
                "param_key": "temperature",
                "type": "number",
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Sampling temperature for randomness"
            },
            {
                "name": "Top P",
                "param_key": "top_p",
                "type": "number",
                "min": 0,
                "max": 1,
                "step": 0.01,
                "description": "Nucleus sampling probability threshold"
            },
            {
                "name": "Do Sample",
                "param_key": "do_sample",
                "type": "checkbox",
                "description": "Enable sampling instead of greedy decoding"
            },
            {
                "name": "Precision",
                "param_key": "precision",
                "type": "select",
                "options": [
                    {"value": "float16", "label": "Float16 (Recommended)"},
                    {"value": "bfloat16", "label": "BFloat16"},
                    {"value": "float32", "label": "Float32 (Full)"},
                    {"value": "4bit", "label": "4-bit Quantized"},
                    {"value": "8bit", "label": "8-bit Quantized"}
                ],
                "description": "Model precision mode (requires model reload)"
            },
            {
                "name": "Use Flash Attention 2",
                "param_key": "use_flash_attention",
                "type": "checkbox",
                "description": "Enable Flash Attention for better performance"
            },
            {
                "name": "Batch Size",
                "param_key": "batch_size",
                "type": "number",
                "min": 1,
                "max": 16,
                "step": 1,
                "description": "Number of images to process simultaneously"
            }
        ]

    def unload(self) -> None:
        """Clean up model from memory."""
        if hasattr(self, 'current_precision_params'):
            delattr(self, 'current_precision_params')
        if hasattr(self, 'quantization_config'):
            self.quantization_config = None
        super().unload()
```

### Step 2: Register Model in Backend

Edit `backend/app.py`:

```python
# 1. Add import at the top
from models.my_model_adapter import MyModelAdapter

# 2. Add to MODEL_METADATA dictionary
MODEL_METADATA = {
    # ... existing models ...
    'my-model': {
        'description': "MyModel - Brief description of what it does",
        'adapter': MyModelAdapter,
        'adapter_args': {'model_id': "organization/model-name"}
    }
}
```

### Step 3: Add Precision Defaults (Optional)

If your model supports precision modes, edit `backend/config.py`:

```python
PRECISION_DEFAULTS = {
    # ... existing models ...
    'my-model': {'precision': 'float16', 'use_flash_attention': False}
}
```

**Common precision choices:**
- `float16`: Good default for most models (~half VRAM of float32)
- `bfloat16`: Better numerical stability, used by models trained in bf16
- `float32`: Full precision, highest VRAM usage

### Step 4: Add Dependencies (If Needed)

If your model requires special packages, add to `requirements.txt`:

```
# Your Model Dependencies
special-package>=1.0.0
```

### Step 5: Test Your Model

```bash
# Install dependencies
pip install -r requirements.txt

# Start backend
cd backend
python app.py

# Your model should now appear in the frontend dropdown!
```

## Base Adapter Helper Methods

The `BaseModelAdapter` class provides useful helper methods:

### Device & Precision
- `_init_device(torch)` - Auto-detect best device (CUDA/MPS/CPU)
- `_get_dtype(precision)` - Convert precision string to torch dtype
- `_create_quantization_config(precision)` - Setup 4bit/8bit quantization
- `_setup_flash_attention(kwargs, precision, force_bfloat16)` - Enable Flash Attention 2

### Image Processing
- `_ensure_rgb(images)` - Convert images to RGB mode
- `_move_inputs_to_device(inputs, device, dtype)` - Move tensors to device with proper dtype

### Parameters
- `_filter_generation_params(params, exclude_keys)` - Filter out special params
- `_setup_pad_token()` - Setup tokenizer pad token for batch processing

### Error Handling
- `_format_batch_error(error, batch_size)` - Format error for batch processing

## Parameter Types Reference

Frontend UI is auto-generated from `get_available_parameters()`:

### Number (Slider)
```python
{
    "name": "Temperature",
    "param_key": "temperature",
    "type": "number",
    "min": 0,
    "max": 2,
    "step": 0.1,
    "description": "Sampling temperature"
}
```

### Checkbox (Boolean)
```python
{
    "name": "Do Sample",
    "param_key": "do_sample",
    "type": "checkbox",
    "description": "Enable sampling"
}
```

### Select (Dropdown)
```python
{
    "name": "Precision",
    "param_key": "precision",
    "type": "select",
    "options": [
        {"value": "float16", "label": "Float16"},
        {"value": "float32", "label": "Float32"}
    ],
    "description": "Model precision mode"
}
```

## Special Parameters

Some parameters are handled specially and should be in `SPECIAL_PARAMS`:

- **`precision`**: Controls model loading precision (requires reload)
- **`use_flash_attention`**: Enables Flash Attention 2 (requires reload)
- **`batch_size`**: Frontend batching control (not passed to model)

These are filtered out before being passed to `model.generate()`.

## Examples in Codebase

Look at these existing adapters for reference:

1. **`blip_adapter.py`** - Simple, fast model (good starting point)
2. **`llava_phi3_adapter.py`** - Chat-based model with prompt formatting
3. **`r4b_adapter.py`** - Advanced model with precision modes
4. **`olmocr_adapter.py`** - Specialized model with custom output processing

## Troubleshooting

### Model not appearing in frontend
- Check `MODEL_METADATA` registration in `app.py`
- Restart backend server
- Check browser console for errors

### Import errors
- Verify adapter class name matches import in `app.py`
- Check file is named `*_adapter.py`
- Ensure class extends `BaseModelAdapter`

### Precision errors
- Not all models support all precision modes
- Some models require specific dtypes (check model docs)
- Flash Attention requires compatible GPU and package installation

### Processor errors
- Some models need special processor handling
- Check model's HuggingFace page for correct usage
- LLaVA models may need `patch_size` workaround (see `llava_phi3_adapter.py`)

## Best Practices

1. **Always convert images to RGB**: Use `self._ensure_rgb(image)`
2. **Handle missing prompts**: Provide sensible defaults
3. **Log important steps**: Use `logger.info()` and `logger.debug()`
4. **Use torch.no_grad()**: Disable gradients during inference
5. **Proper error handling**: Return error messages instead of crashing
6. **Batch support**: Implement `generate_captions_batch()` for better performance
7. **Memory cleanup**: Call `super().unload()` in your `unload()` method

## Contributing

When adding models to the main codebase:
1. Test with multiple images and prompts
2. Test batch processing
3. Test different precision modes
4. Document any special requirements
5. Add model to this guide's examples section

## Questions?

Check existing adapter implementations or open an issue on GitHub!
