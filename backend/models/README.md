# Adding New AI Models

This guide explains how to add a new AI model adapter to the project.

## Quick Steps

1. **Create a new adapter file** in `backend/models/` (e.g., `my_model_adapter.py`)
2. **Inherit from `BaseModelAdapter`**
3. **Implement required methods**
4. **Register the model** in `backend/app.py` → `MODEL_METADATA`
5. **Add precision defaults** in `backend/config.py` → `PRECISION_DEFAULTS` (if using precision options)

## Required Implementation

### 1. Class Structure

```python
from .base_adapter import BaseModelAdapter
import logging

logger = logging.getLogger(__name__)

class MyModelAdapter(BaseModelAdapter):
    # Define special parameters that shouldn't be passed to model.generate()
    SPECIAL_PARAMS = {'batch_size', 'precision', 'custom_param'}
    
    def __init__(self):
        super().__init__("my-model-name")
        self.device = self._init_device(torch)
```

### 2. Required Methods

#### `load_model(self) -> None`
Load the model and processor. Called when model is first used.

```python
def load_model(self) -> None:
    logger.info("Loading model...")
    self.processor = AutoProcessor.from_pretrained("model-id")
    self.model = AutoModel.from_pretrained("model-id")
    self.model.to(self.device)
    self.model.eval()
    self._setup_pad_token()  # For batch processing
```

#### `generate_caption(self, image, prompt, parameters) -> str`
Generate caption for a single image.

```python
def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
    if not self.is_loaded():
        raise RuntimeError("Model not loaded")
    
    try:
        image = self._ensure_rgb(image)
        
        # Filter and sanitize parameters
        gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)
        gen_params = self._sanitize_generation_params(gen_params)
        
        # Process inputs
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)
        
        return self.processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logger.exception("Error: %s", e)
        return f"Error: {str(e)}"
```

**Note for library-specific parameters:** If your model uses a custom generation function (not standard `model.generate()`), you may need to map parameter names. For example, Chandra's `generate_hf()` uses `max_output_tokens` instead of `max_new_tokens`:

```python
# Extract and map library-specific parameters
max_output_tokens = parameters.get("max_new_tokens") if parameters else None
gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)
gen_params.pop("max_new_tokens", None)  # Remove since passed separately

# Call library-specific function
result = generate_hf(batch, model, max_output_tokens=max_output_tokens, **gen_params)
```

#### `is_loaded(self) -> bool`
Check if model is loaded in memory.

```python
def is_loaded(self) -> bool:
    return self.model is not None and self.processor is not None
```

#### `get_available_parameters(self) -> list`
Define UI parameters with proper grouping for the node editor.

```python
def get_available_parameters(self) -> list:
    return [
        {
            "name": "Do Sample",
            "param_key": "do_sample",
            "type": "checkbox",
            "description": "Enable sampling mode (conflicts with num_beams>1)",
            "group": "sampling"  # Groups: sampling, beam_search, general
        },
        {
            "name": "Temperature",
            "param_key": "temperature",
            "type": "number",
            "min": 0.1,
            "max": 2,
            "step": 0.1,
            "description": "Sampling temperature (requires do_sample=true)",
            "depends_on": "do_sample",  # Show dependency
            "group": "sampling"
        },
        {
            "name": "Num Beams",
            "param_key": "num_beams",
            "type": "number",
            "min": 1,
            "max": 20,
            "step": 1,
            "description": "Beam search (conflicts with do_sample=true)",
            "group": "beam_search"
        },
        # Add more parameters...
    ]
```

### 3. Optional Methods

#### `generate_captions_batch(self, images, prompts, parameters) -> list`
Batch processing for better performance (optional but recommended).

```python
def generate_captions_batch(self, images: list, prompts: list = None, parameters: dict = None) -> list:
    # Implement batch processing
    # See blip_adapter.py or blip2_adapter.py for examples
    pass
```

## Helper Methods Available

From `BaseModelAdapter`:

- `_ensure_rgb(images)` - Convert images to RGB
- `_filter_generation_params(parameters, exclude_keys)` - Filter valid params
- `_sanitize_generation_params(parameters)` - Apply generation rules (handles do_sample/num_beams conflicts)
- `_setup_pad_token()` - Setup padding for batch processing
- `_init_device(torch_module)` - Get best device (CUDA/MPS/CPU)
- `_move_inputs_to_device(inputs, device, dtype)` - Move tensors to device
- `_create_quantization_config(precision)` - Create quantization config
- `_get_dtype(precision)` - Map precision string to torch dtype
- `_setup_flash_attention(model_kwargs, precision)` - Setup Flash Attention 2

## Register the Model

### Step 1: Add to MODEL_METADATA in `backend/app.py`

```python
from models.my_model_adapter import MyModelAdapter

MODEL_METADATA = {
    # ... existing models ...
    'my-model': {
        'category': 'general',  # Categories: 'general', 'anime', 'multimodal', 'ocr'
        'description': 'Brief description of the model',
        'adapter': MyModelAdapter,
        'adapter_args': {'model_id': "huggingface/model-id"}
    }
}
```

### Step 2: Add Precision Defaults in `backend/config.py` (REQUIRED for precision support)

**IMPORTANT**: If your model supports precision options (float16, bfloat16, float32, 4bit, 8bit), you MUST add it to `PRECISION_DEFAULTS` in `backend/config.py`. Without this, precision parameters won't be extracted and passed to `load_model()`.

```python
PRECISION_DEFAULTS = {
    # ... existing models ...
    'my-model': {'precision': 'float16', 'use_flash_attention': False}
}
```

This ensures:
- The precision parameter is extracted from user selections
- It's passed to `load_model(precision='...', use_flash_attention=...)`
- The model loads with the correct precision instead of always using the default

**Common mistake**: Forgetting this step means precision selection in the UI won't work - the model will always load with the hardcoded default in `load_model()`.

## Parameter Groups & Dependencies

Use these guidelines for parameter configuration:

- **Group: "sampling"** - Parameters requiring `do_sample=true` (temperature, top_p, top_k)
- **Group: "beam_search"** - Parameters requiring `num_beams>1` (length_penalty, early_stopping)
- **Group: "general"** - Parameters that work in both modes (repetition_penalty, max_length)
- **depends_on** - Shows which parameter enables this one (auto-disables in UI when dependency not met)

The `_sanitize_generation_params()` method automatically handles conflicts:
- Disables sampling params when `do_sample=False`
- Disables beam search params when `num_beams<=1`
- Prevents `do_sample=True` with `num_beams>1`

## Examples

See existing adapters for reference:
- **Simple model**: `blip_adapter.py`
- **With precision options**: `blip2_adapter.py`
- **Chat-based model**: `llava_phi3_adapter.py`
- **OCR model**: `trocr_adapter.py`
- **Library-specific parameters**: `chandra_adapter.py` (custom generate function with parameter mapping)

## Testing

After implementation:
1. Start the backend: `python backend/app.py`
2. Check model appears in frontend dropdown
3. Test single image generation
4. Test batch processing (if implemented)
5. Verify parameter UI shows correct groups and dependencies

---

## Quick Reference (Pseudocode)

```python
# 1. Create adapter class
class MyModelAdapter(BaseModelAdapter):
    SPECIAL_PARAMS = {'batch_size', 'precision'}  # Non-generation params
    
    def __init__(self):
        super().__init__("model-name")
        self.device = self._init_device(torch)
    
    # 2. Load model
    def load_model(self):
        self.processor = load_processor()
        self.model = load_model().to(self.device).eval()
        self._setup_pad_token()
    
    # 3. Generate caption
    def generate_caption(self, image, prompt, parameters):
        image = self._ensure_rgb(image)
        gen_params = self._filter_generation_params(parameters, self.SPECIAL_PARAMS)
        gen_params = self._sanitize_generation_params(gen_params)  # Auto-fix conflicts
        inputs = self.processor(image, prompt).to(self.device)
        outputs = self.model.generate(**inputs, **gen_params)
        return self.processor.decode(outputs)
    
    # 4. Check if loaded
    def is_loaded(self):
        return self.model is not None
    
    # 5. Define parameters with groups
    def get_available_parameters(self):
        return [
            {"name": "Do Sample", "param_key": "do_sample", "type": "checkbox", "group": "sampling"},
            {"name": "Temperature", "param_key": "temperature", "type": "number", 
             "depends_on": "do_sample", "group": "sampling"},
            {"name": "Num Beams", "param_key": "num_beams", "type": "number", "group": "beam_search"},
            # ... more params
        ]

# 6. Register in backend/app.py
MODEL_METADATA = {
    'my-model': {
        'category': 'general',
        'description': 'Model description',
        'adapter': MyModelAdapter,
        'adapter_args': {'model_id': "org/model"}
    }
}

# 7. Add precision defaults in backend/config.py (if using precision options)
PRECISION_DEFAULTS = {
    'my-model': {'precision': 'float16', 'use_flash_attention': False}
}
```

**That's it!** The base class handles device management, parameter validation, conflict resolution, and UI integration.
