import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig
from .base_adapter import BaseModelAdapter

class R4BAdapter(BaseModelAdapter):
    """YannQi/R-4B model adapter for advanced image reasoning"""

    def __init__(self):
        super().__init__("YannQi/R-4B")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantization_config = None

    def _create_quantization_config(self, precision):
        """Create quantization configuration based on precision parameter"""
        if precision == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif precision == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        return None

    def _get_torch_dtype(self, precision):
        """Get torch dtype from precision parameter"""
        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        # Default to float32 for consistency with test.py control
        return precision_map.get(precision, torch.float32)

    def _extract_final_result(self, caption: str) -> str:
        """Extract only the final result from R4B thinking output"""
        if not caption:
            return caption

        # For auto/long mode: extract content after </think> tag
        if "</think>" in caption:
            parts = caption.split("</think>")
            if len(parts) > 1:
                final_part = parts[-1].strip()
                # Remove any remaining newlines and extra whitespace
                return final_part.replace('\n', ' ').strip()

        # For short mode: remove "Auto-Thinking Output: " prefix
        if caption.startswith("Auto-Thinking Output: "):
            return caption[len("Auto-Thinking Output: "):].strip()

        # Return as-is if no known pattern found
        return caption.strip()

    def load_model(self, precision="float32", use_flash_attention=False) -> None:
        """Load R-4B model and processor with configurable precision and optimizations"""
        try:
            print(f"Loading R-4B model on {self.device} with {precision} precision...")

            # Create quantization config for 4bit/8bit precision
            self.quantization_config = self._create_quantization_config(precision)

            # Load processor
            self.processor = AutoProcessor.from_pretrained("YannQi/R-4B", trust_remote_code=True)

            # Prepare model loading arguments
            model_kwargs = {
                "trust_remote_code": True,
                "quantization_config": self.quantization_config,
            }

            # Set dtype for non-quantized precision modes
            if precision not in ["4bit", "8bit"]:
                model_kwargs["torch_dtype"] = self._get_torch_dtype(precision)

            # Add Flash Attention support if available and requested
            if use_flash_attention and torch.cuda.is_available():
                try:
                    import flash_attn
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    print("Using Flash Attention 2 for improved performance")
                except ImportError:
                    print("Flash Attention not available, using default attention")

            # Load model
            self.model = AutoModel.from_pretrained("YannQi/R-4B", **model_kwargs)

            # Move to device if not quantized (quantized models handle device placement automatically)
            if precision not in ["4bit", "8bit"]:
                self.model.to(self.device)

            self.model.eval()

            print(f"R-4B model loaded successfully! (Precision: {precision})")

        except Exception as e:
            print(f"Error loading R-4B model: {e}")
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        """Generate caption for the given image using R-4B with configurable parameters"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Debug logging
            print(f"R4B Adapter: generate_caption called")
            print(f"   Prompt: '{prompt}'")
            print(f"   Parameters received: {parameters}")

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Build generation parameters - only include what user explicitly provided
            gen_params = {}

            # Thinking mode for chat template (separate from generation params)
            thinking_mode = 'auto'  # Default for chat template

            # Process parameters - only include what was explicitly set
            if parameters:
                print(f"   Processing parameters...")

                # Handle thinking mode separately (for chat template)
                if 'thinking_mode' in parameters:
                    thinking_mode_value = parameters.get('thinking_mode')
                    if thinking_mode_value in ['auto', 'short', 'long']:
                        thinking_mode = thinking_mode_value
                    else:
                        print(f"   Invalid thinking_mode '{thinking_mode_value}', using default 'auto'")

                # Get all generation param keys (excluding special params like precision, use_flash_attention, thinking_mode)
                generation_param_keys = [p['param_key'] for p in self.get_available_parameters()
                                        if p['type'] in ['number'] and p['param_key'] not in ['precision', 'use_flash_attention', 'thinking_mode']]

                # Only pass parameters that were explicitly set by the user
                for param_key in generation_param_keys:
                    if param_key in parameters:
                        gen_params[param_key] = parameters[param_key]
            else:
                print(f"   No parameters provided, using model defaults...")

            print(f"   Thinking mode: {thinking_mode}")
            print(f"   Generation parameters: {gen_params if gen_params else 'Using library defaults'}")
            print("=" * 60)

            # Prepare prompt
            if not prompt:
                prompt = "Describe this image."

            # Create conversation messages in the format expected by R-4B
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Apply chat template with thinking mode
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                thinking_mode=thinking_mode
            )

            # Process inputs
            inputs = self.processor(
                images=image,
                text=text,
                return_tensors="pt"
            )

            # Match dtype to model’s parameters
            model_dtype = next(self.model.parameters()).dtype
            inputs = {
                k: (v.to(self.device, dtype=model_dtype) if torch.is_floating_point(v) else v.to(self.device))
                for k, v in inputs.items()
            }

            # Generate response - library will use its own defaults for unspecified params
            generated_ids = self.model.generate(**inputs, **gen_params)

            # Extract only the new tokens (response)
            output_ids = generated_ids[0][len(inputs["input_ids"][0]):]

            # Decode response
            caption = self.processor.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            # Extract final result from thinking output
            final_result = self._extract_final_result(caption)
            return final_result if final_result else "Unable to generate description."

        except Exception as e:
            print(f"Error generating caption with R-4B: {e}")
            return f"Error: {str(e)}"

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.processor is not None

    def get_available_parameters(self) -> list:
        """Get list of available parameters for R-4B model"""
        return [
            {
                "name": "Max New Tokens",
                "param_key": "max_new_tokens",
                "type": "number",
                "min": 1,
                "max": 32768,
                "step": 1,
                "description": "Maximum number of new tokens to generate"
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
                "name": "Top K",
                "param_key": "top_k",
                "type": "number",
                "min": 0,
                "max": 200,
                "step": 1,
                "description": "Top-k sampling: limit to k highest probability tokens"
            },
            {
                "name": "Thinking Mode",
                "param_key": "thinking_mode",
                "type": "select",
                "options": [
                    {"value": "auto", "label": "Auto"},
                    {"value": "short", "label": "Short"},
                    {"value": "long", "label": "Long"}
                ],
                "description": "Verbosity of reasoning process"
            },
            {
                "name": "Precision",
                "param_key": "precision",
                "type": "select",
                "options": [
                    {"value": "float32", "label": "Float32 (Full)"},
                    {"value": "float16", "label": "Float16 (Half)"},
                    {"value": "bfloat16", "label": "BFloat16"},
                    {"value": "4bit", "label": "4-bit Quantized"},
                    {"value": "8bit", "label": "8-bit Quantized"}
                ],
                "description": "Model precision mode (requires model reload)"
            },
            {
                "name": "Use Flash Attention 2",
                "param_key": "use_flash_attention",
                "type": "checkbox",
                "description": "Enable Flash Attention for better performance (requires flash-attn package)"
            }
        ]

    def unload(self) -> None:
        """
        Unload the R-4B model and clear precision parameters.
        This ensures complete cleanup when changing precision settings.
        """
        # Clear precision tracking
        if hasattr(self, 'current_precision_params'):
            delattr(self, 'current_precision_params')
        
        if hasattr(self, 'quantization_config'):
            self.quantization_config = None
            
        # Call parent unload method
        super().unload()
