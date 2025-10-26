import torch
import logging
from utils.torch_utils import pick_device, force_cpu_mode
from PIL import Image
from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig
from .base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)

class R4BAdapter(BaseModelAdapter):
    def __init__(self):
        super().__init__("YannQi/R-4B")
        self.device = pick_device(torch)
        self.quantization_config = None

    def _create_quantization_config(self, precision):
        if precision == "4bit":
            return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                     bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        elif precision == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        return None

    def _get_dtype(self, precision):
        precision_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        return precision_map.get(precision, torch.float32)

    def _extract_final_result(self, caption: str) -> str:
        if not caption:
            return caption
        if "</think>" in caption:
            parts = caption.split("</think>")
            if len(parts) > 1:
                return parts[-1].replace('\n', ' ').strip()
        if caption.startswith("Auto-Thinking Output: "):
            return caption[len("Auto-Thinking Output: "):].strip()
        return caption.strip()

    def load_model(self, precision="float32", use_flash_attention=False) -> None:
        try:
            logger.info("Loading R-4B model on %s with %s precision…", self.device, precision)

            self.quantization_config = self._create_quantization_config(precision)

            self.processor = AutoProcessor.from_pretrained("YannQi/R-4B", trust_remote_code=True, use_fast=True)
            if not self.processor:
                raise RuntimeError("Processor loading failed")

            model_kwargs = {"trust_remote_code": True, "quantization_config": self.quantization_config}

            if precision not in ["4bit", "8bit"]:
                model_kwargs["dtype"] = self._get_dtype(precision)

            if use_flash_attention and not force_cpu_mode() and torch.cuda.is_available():
                try:
                    import flash_attn
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using Flash Attention 2")
                except ImportError:
                    logger.info("Flash Attention not available, using default. You can install it via 'pip install flash-attn --no-build-isolation' if GPU is compatible.")

            self.model = AutoModel.from_pretrained("YannQi/R-4B", **model_kwargs)
            if not self.model:
                raise RuntimeError("Model loading failed")

            if precision not in ["4bit", "8bit"]:
                self.model.to(self.device)

            self.model.eval()
            logger.info("R-4B model loaded successfully (Precision: %s)", precision)

        except Exception as e:
            logger.exception("Error loading R-4B model: %s", e)
            raise

    def generate_caption(self, image: Image.Image, prompt: str = None, parameters: dict = None) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            gen_params = {}
            thinking_mode = 'auto'

            if parameters:
                logger.debug("R4B parameters: %s", parameters)
                if 'thinking_mode' in parameters:
                    thinking_mode_value = parameters.get('thinking_mode')
                    if thinking_mode_value in ['auto', 'short', 'long']:
                        thinking_mode = thinking_mode_value

                generation_param_keys = [p['param_key'] for p in self.get_available_parameters()
                                        if p['type'] == 'number' and p['param_key'] not in
                                        ['precision', 'use_flash_attention', 'thinking_mode']]

                for param_key in generation_param_keys:
                    if param_key in parameters:
                        gen_params[param_key] = parameters[param_key]

            prompt = prompt or "Describe this image."

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]

            text = self.processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True, thinking_mode=thinking_mode)

            inputs = self.processor(images=image, text=text, return_tensors="pt")

            model_dtype = next(self.model.parameters()).dtype
            inputs = {k: (v.to(self.device, dtype=model_dtype) if torch.is_floating_point(v) else v.to(self.device))
                     for k, v in inputs.items()}

            generated_ids = self.model.generate(**inputs, **gen_params)
            output_ids = generated_ids[0][len(inputs["input_ids"][0]):]
            caption = self.processor.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            final_result = self._extract_final_result(caption)
            return final_result if final_result else "Unable to generate description."

        except Exception as e:
            logger.exception("Error generating caption with R-4B: %s", e)
            return f"Error: {str(e)}"

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def get_available_parameters(self) -> list:
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
        if hasattr(self, 'current_precision_params'):
            delattr(self, 'current_precision_params')
        if hasattr(self, 'quantization_config'):
            self.quantization_config = None
        super().unload()
