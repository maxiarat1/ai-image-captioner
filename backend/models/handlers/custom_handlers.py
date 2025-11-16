"""
Custom Model Handlers
Specialized handlers for models that need unique logic (Janus, R4B, TrOCR)
"""
import torch
from PIL import Image
from typing import Any, Dict, List, Optional
import logging
from transformers import AutoProcessor, VisionEncoderDecoderModel
from .hf_vlm_handler import HuggingFaceVLMHandler
from .hf_ocr_handler import HuggingFaceOCRHandler

logger = logging.getLogger(__name__)


class JanusHandler(HuggingFaceVLMHandler):
    """
    Custom handler for Janus models (requires janus package).
    Inherits from HF VLM handler with specialized processor handling.
    """
    
    def _get_processor_class(self):
        """Import VLChatProcessor from janus package for Janus models."""
        try:
            from janus.models import VLChatProcessor
            return VLChatProcessor
        except ImportError:
            logger.error(
                "Janus models require the 'janus' package. "
                "Install it via: pip install git+https://github.com/deepseek-ai/Janus.git"
            )
            raise
    
    def _get_model_class(self):
        """Use AutoModelForCausalLM for Janus."""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM
    
    def load(self, precision: str = None, use_flash_attention: bool = False, **kwargs) -> None:
        """Load Janus model with special processor handling."""
        try:
            from utils.torch_utils import pick_device
            from transformers import AutoImageProcessor
            
            self.device = pick_device(torch)
            
            precision = precision or self.config.get('default_precision', 'bfloat16')
            
            # Validate precision against supported_precisions if specified
            supported_precisions = self.config.get('supported_precisions')
            if supported_precisions and precision not in supported_precisions:
                logger.warning(
                    "%s does not support %s precision. Falling back to %s. "
                    "Supported precisions: %s",
                    self.model_key, precision, self.config.get('default_precision', 'bfloat16'),
                    ', '.join(supported_precisions)
                )
                precision = self.config.get('default_precision', 'bfloat16')
            
            logger.info("Loading %s on %s with %s precision…", self.model_key, self.device, precision)
            
            # Load Janus-specific processor with explicit fast image processor class
            processor_class = self._get_processor_class()
            self.processor = processor_class.from_pretrained(
                self.model_id,
                fast_image_processor_class=AutoImageProcessor
            )
            
            # Setup pad token for batch processing
            self._setup_pad_token()
            
            # Prepare model loading kwargs
            model_kwargs = self.config.get('model_config', {}).copy()
            
            # Handle quantization
            quantization_config = self._create_quantization_config(precision)
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            else:
                model_kwargs['torch_dtype'] = self._get_dtype(precision)
            
            # Setup flash attention if requested
            if use_flash_attention:
                self._setup_flash_attention(model_kwargs, precision)
            
            # Load model
            model_class = self._get_model_class()
            self.model = model_class.from_pretrained(self.model_id, **model_kwargs)
            
            # Move to device if not quantized
            if precision not in ["4bit", "8bit"]:
                self.model.to(self.device)
            
            # Setup generation config pad token
            self._setup_generation_config_pad_token()
            
            self.model.eval()
            logger.info("%s loaded successfully", self.model_key)
            
        except Exception as e:
            logger.exception("Error loading %s: %s", self.model_key, e)
            raise
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Generate caption using Janus-specific conversation format."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            image = self._ensure_rgb(image)
            
            # Use default prompt if none provided
            prompt_text = prompt.strip() if prompt and prompt.strip() else "Describe this image in detail."
            
            # Janus uses conversation format with specific role markers
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{prompt_text}",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            # Prepare inputs using VLChatProcessor
            prepare_inputs = self.processor(
                conversations=conversation,
                images=[image],
                force_batchify=True
            )
            
            # Move inputs to device and match model dtype
            prepare_inputs = self._move_inputs_to_device(prepare_inputs, match_model_dtype=True)
            
            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params = self._sanitize_generation_params(gen_params)
            
            # Set default generation parameters
            default_params = {
                'pad_token_id': self.processor.tokenizer.eos_token_id,
                'bos_token_id': self.processor.tokenizer.bos_token_id,
                'eos_token_id': self.processor.tokenizer.eos_token_id,
                'max_new_tokens': 512,
                'do_sample': False,
                'use_cache': True
            }
            
            # Merge with user parameters (user params take precedence)
            final_params = {**default_params, **gen_params}
            
            # Generate
            with torch.no_grad():
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    **final_params
                )
            
            # Decode
            result = self.processor.tokenizer.decode(
                outputs[0].cpu().tolist(),
                skip_special_tokens=True
            )
            
            return result.strip() if result else "Unable to generate description."
            
        except Exception as e:
            logger.exception("Error with Janus inference: %s", e)
            return f"Error: {str(e)}"
    
    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """Generate captions for multiple images using Janus conversation format."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            # Ensure all images are RGB
            images = [self._ensure_rgb(img) for img in images]
            
            # Prepare prompts
            if prompts is None:
                prompts = [None] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")
            
            # Process each image separately as Janus processor doesn't support true batching
            results = []
            for image, prompt in zip(images, prompts):
                prompt_text = prompt.strip() if prompt and prompt.strip() else "Describe this image in detail."
                
                # Build conversation for this image
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{prompt_text}",
                        "images": [image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                
                # Prepare inputs
                prepare_inputs = self.processor(
                    conversations=conversation,
                    images=[image],
                    force_batchify=True
                )
                
                # Move inputs to device and match model dtype
                prepare_inputs = self._move_inputs_to_device(prepare_inputs, match_model_dtype=True)
                
                # Prepare generation parameters
                gen_params = self._filter_generation_params(parameters)
                gen_params = self._sanitize_generation_params(gen_params)
                
                # Set default generation parameters
                default_params = {
                    'pad_token_id': self.processor.tokenizer.eos_token_id,
                    'bos_token_id': self.processor.tokenizer.bos_token_id,
                    'eos_token_id': self.processor.tokenizer.eos_token_id,
                    'max_new_tokens': 512,
                    'do_sample': False,
                    'use_cache': True
                }
                
                # Merge with user parameters
                final_params = {**default_params, **gen_params}
                
                # Generate
                with torch.no_grad():
                    inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                    outputs = self.model.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        **final_params
                    )
                
                # Decode
                result = self.processor.tokenizer.decode(
                    outputs[0].cpu().tolist(),
                    skip_special_tokens=True
                )
                
                results.append(result.strip() if result else "Unable to generate description.")
            
            return results
            
        except Exception as e:
            logger.exception("Error in batch Janus generation: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)


class R4BHandler(HuggingFaceVLMHandler):
    """
    Custom handler for R4B model with thinking mode support.
    R4B only supports float32, 4bit, and 8bit precision modes.
    """
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Generate caption with R4B thinking mode support."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            image = self._ensure_rgb(image)
            
            # Extract thinking mode
            thinking_mode = self._extract_thinking_mode(parameters)
            
            # Build prompt
            user_prompt = prompt.strip() if prompt and prompt.strip() else "Describe this image."
            
            # R4B requires structured message format with image object
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt}
                ]
            }]
            
            # Apply chat template with thinking_mode parameter
            text_prompt = self.processor.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True, 
                thinking_mode=thinking_mode
            )
            
            # Process inputs with the formatted text
            inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")
            
            # Move inputs to device and convert floating point tensors to model dtype
            inputs = self._move_inputs_to_device(inputs, match_model_dtype=True)
            
            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params = self._sanitize_generation_params(gen_params)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_params)
            
            # Decode only the generated portion (skip input tokens)
            output_ids = generated_ids[0][len(inputs["input_ids"][0]):]
            result = self.processor.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Extract final result (remove thinking tags)
            return self._extract_final_result(result)
            
        except Exception as e:
            logger.exception("Error with R4B inference: %s", e)
            return f"Error: {str(e)}"
    
    def _extract_thinking_mode(self, parameters: Optional[Dict]) -> str:
        """Extract and validate thinking_mode parameter."""
        if not parameters:
            return 'auto'
        thinking_mode = parameters.get('thinking_mode', 'auto')
        return thinking_mode if thinking_mode in ['auto', 'short', 'long'] else 'auto'
    
    def _extract_final_result(self, caption: str) -> str:
        """Extract final caption from R4B output (removes thinking tags)."""
        if not caption:
            return caption
        if "</think>" in caption:
            parts = caption.split("</think>")
            if len(parts) > 1:
                return parts[-1].replace('\n', ' ').strip()
        if caption.startswith("Auto-Thinking Output: "):
            return caption[len("Auto-Thinking Output: "):].strip()
        return caption.strip()
    
    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """Generate captions for multiple images using R4B's chat template format."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            # Ensure all images are RGB
            images = [self._ensure_rgb(img) for img in images]
            
            # Prepare prompts
            if prompts is None:
                prompts = [None] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")
            
            # Extract thinking mode
            thinking_mode = self._extract_thinking_mode(parameters)
            
            # Build text prompts for each image using chat template
            text_prompts = []
            for prompt in prompts:
                user_prompt = prompt.strip() if prompt and prompt.strip() else "Describe this image in detail."
                
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt}
                    ]
                }]
                
                # Apply chat template
                text_prompt = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True, 
                    thinking_mode=thinking_mode
                )
                text_prompts.append(text_prompt)
            
            # Process batch with images and text prompts
            inputs = self.processor(images=images, text=text_prompts, return_tensors="pt", padding=True)
            
            # Move inputs to device and match model dtype
            inputs = self._move_inputs_to_device(inputs, match_model_dtype=True)
            
            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params = self._sanitize_generation_params(gen_params)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_params)
            
            # Decode each result, skipping input tokens
            results = []
            input_lengths = inputs["input_ids"].shape[1]
            for i, generated in enumerate(generated_ids):
                output_ids = generated[input_lengths:]
                result = self.processor.decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                results.append(self._extract_final_result(result))
            
            return results
            
        except Exception as e:
            logger.exception("Error in batch R4B generation: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)


class TrOCRHandler(HuggingFaceOCRHandler):
    """
    Custom handler for TrOCR with doctr text detection support.
    
    Two-stage OCR approach:
    1. Word-level text detection using doctr's ocr_predictor
    2. Text recognition using TrOCR on detected word regions
    3. Intelligent line reconstruction by grouping words based on vertical proximity
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.detector = None
    
    def load(self, precision: str = None, use_flash_attention: bool = False, **kwargs) -> None:
        """Load TrOCR model with doctr text detection."""
        try:
            from utils.torch_utils import pick_device
            self.device = pick_device(torch)
            
            precision = precision or self.config.get('default_precision', 'float32')
            use_fast = kwargs.get('use_fast', True)
            
            logger.info("Loading %s on %s with %s precision…", self.model_key, self.device, precision)
            
            # Load doctr text detector
            if self.config.get('requires_text_detection'):
                logger.info("Loading doctr text detector...")
                from doctr.models import ocr_predictor
                self.detector = ocr_predictor(pretrained=True)
            
            # Load TrOCR processor and model
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            processor_config = self.config.get('processor_config', {})
            processor_config['use_fast'] = use_fast
            self.processor = TrOCRProcessor.from_pretrained(self.model_id, **processor_config)
            
            # Prepare model loading
            model_kwargs = {}
            quantization_config = self._create_quantization_config(precision)
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            else:
                model_kwargs['torch_dtype'] = self._get_dtype(precision)
            
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id, **model_kwargs)
            
            if precision not in ["4bit", "8bit"]:
                self.model.to(self.device)
            
            self.model.eval()
            
            logger.info("%s loaded successfully", self.model_key)
            
        except Exception as e:
            logger.exception("Error loading %s: %s", self.model_key, e)
            raise
    
    def _detect_text_regions(self, image: Image.Image) -> List[tuple]:
        """
        Detect text regions at word-level using doctr.
        Returns list of bounding boxes in absolute pixel coordinates (x1, y1, x2, y2).
        """
        import tempfile
        import os
        from doctr.io import DocumentFile
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path, format='PNG')
        
        try:
            doc = DocumentFile.from_images(tmp_path)
            result = self.detector(doc)
            exported = result.export()
            
            if not exported.get('pages'):
                return []
            
            page = exported['pages'][0]
            width, height = image.size
            boxes = []
            
            # Extract word-level bounding boxes
            for block in page.get('blocks', []):
                for line in block.get('lines', []):
                    for word in line.get('words', []):
                        ((x1, y1), (x2, y2)) = word['geometry']
                        # Convert normalized coordinates to absolute pixels
                        x1, y1, x2, y2 = (
                            int(x1 * width), int(y1 * height),
                            int(x2 * width), int(y2 * height)
                        )
                        boxes.append((x1, y1, x2, y2))
            
            return boxes
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _group_words_into_lines(self, word_results: List[Dict], line_threshold: int = 20) -> List[str]:
        """
        Group recognized words into lines based on vertical proximity.
        
        Args:
            word_results: List of dicts with 'bbox' (x1, y1, x2, y2) and 'text'
            line_threshold: Maximum vertical distance (pixels) to group words into same line
            
        Returns:
            List of strings, each representing a line of text
        """
        if not word_results:
            return []
        
        # Sort words by top-to-bottom first, then left-to-right
        word_results.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
        
        lines = []
        current_line = []
        
        for word_result in word_results:
            if not current_line:
                current_line.append(word_result)
                continue
            
            prev_y = current_line[-1]["bbox"][1]  # y1 coordinate of previous word
            curr_y = word_result["bbox"][1]  # y1 coordinate of current word
            
            if abs(curr_y - prev_y) < line_threshold:
                # Same line
                current_line.append(word_result)
            else:
                # New line - sort current line left-to-right and save
                current_line.sort(key=lambda w: w["bbox"][0])
                line_text = " ".join(w["text"] for w in current_line if w["text"])
                if line_text:
                    lines.append(line_text)
                current_line = [word_result]
        
        # Handle last line
        if current_line:
            current_line.sort(key=lambda w: w["bbox"][0])
            line_text = " ".join(w["text"] for w in current_line if w["text"])
            if line_text:
                lines.append(line_text)
        
        return lines
    
    def _recognize_text_from_crop(self, crop: Image.Image, parameters: Optional[Dict] = None) -> str:
        """
        Recognize text from a cropped image region using TrOCR.
        """
        pixel_values = self.processor(images=crop, return_tensors="pt").pixel_values
        model_dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(device=self.device, dtype=model_dtype)

        gen_params = self._filter_generation_params(parameters)

        # Ensure dtype alignment and safe inference
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=model_dtype):
            generated_ids = self.model.generate(pixel_values, **gen_params)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Extract text using TrOCR with doctr text detection."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            image = self._ensure_rgb(image)
            
            # Extract OCR-specific parameters
            params = parameters or {}
            line_separator = params.get("line_separator", "\\n")
            # Handle escaped newline from frontend
            if line_separator == "\\n":
                line_separator = "\n"
            line_threshold = params.get("line_threshold", 20)
            
            # If detector is available, use two-stage OCR
            if self.detector:
                # Detect word-level bounding boxes
                boxes = self._detect_text_regions(image)
                num_boxes = len(boxes)
                logger.info("TrOCR detected %d word box(es) in image", num_boxes)

                if num_boxes == 0:
                    logger.info("No text regions detected in image")
                    return ""

                # Recognize text from each word box
                word_results = []
                for (x1, y1, x2, y2) in boxes:
                    # Skip very small crops
                    if (x2 - x1) < 5 or (y2 - y1) < 5:
                        continue
                    
                    crop = image.crop((x1, y1, x2, y2))
                    text = self._recognize_text_from_crop(crop, parameters)
                    if text:
                        word_results.append({"bbox": (x1, y1, x2, y2), "text": text})

                logger.info("Successfully recognized %d words", len(word_results))

                # If no readable text was recognized, return empty
                if not word_results:
                    logger.info("No readable text recognized in detected regions")
                    return ""

                # Group words into lines based on vertical proximity
                lines = self._group_words_into_lines(word_results, line_threshold)
                
                # Apply final line separator
                final_text = line_separator.join(lines)
                logger.info("Total lines reconstructed: %d", len(lines))
                return final_text if final_text else ""
            
            else:
                # Fallback: process whole image without detection
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate
                gen_params = self._filter_generation_params(parameters)
                
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, **gen_params)
                
                # Decode
                result = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                return result.strip()
            
        except Exception as e:
            logger.exception("Error with TrOCR: %s", e)
            return f"Error: {str(e)}"
    
    def is_loaded(self) -> bool:
        """Check if model and processor are loaded (detector is optional)."""
        return self.model is not None and self.processor is not None


class LFM2Handler(HuggingFaceVLMHandler):
    """
    Custom handler for LFM2 model which requires chat template format.
    """
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Generate caption using LFM2's chat template format."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            image = self._ensure_rgb(image)
            
            # Use default prompt if none provided
            prompt_text = prompt.strip() if prompt and prompt.strip() else "What is in this image?"
            
            # LFM2 requires conversation format with chat template
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]
            
            # Apply chat template to get properly formatted inputs
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            )
            
            # Move inputs to device and match model dtype
            inputs = self._move_inputs_to_device(inputs, match_model_dtype=True)
            
            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params = self._sanitize_generation_params(gen_params)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_params)
            
            # Decode the full response
            full_response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract just the assistant's response
            return self._extract_response(full_response)
            
        except Exception as e:
            logger.exception("Error with LFM2 inference: %s", e)
            return f"Error: {str(e)}"
    
    def _extract_response(self, full_response: str) -> str:
        """Extract the assistant's response from the full decoded output."""
        # The full response includes the conversation, extract just the answer
        if "assistant" in full_response.lower():
            parts = full_response.split("assistant", 1)
            if len(parts) > 1:
                # Clean up the response
                response = parts[1].strip()
                # Remove any leading colons or whitespace
                response = response.lstrip(":").strip()
                return response if response else "Unable to generate description."
        
        # Fallback: just return the full response cleaned up
        return full_response.strip() if full_response.strip() else "Unable to generate description."
    
    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """Generate captions for multiple images using LFM2's chat template format."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            # Ensure all images are RGB
            images = [self._ensure_rgb(img) for img in images]
            
            # Prepare prompts
            if prompts is None:
                prompts = [None] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")
            
            # Process each image separately as LFM2 requires chat template per image
            results = []
            for image, prompt in zip(images, prompts):
                prompt_text = prompt.strip() if prompt and prompt.strip() else "What is in this image?"
                
                # Build conversation for this image
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                ]
                
                # Apply chat template
                inputs = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                )
                
                # Move inputs to device and match model dtype
                inputs = self._move_inputs_to_device(inputs, match_model_dtype=True)
                
                # Prepare generation parameters
                gen_params = self._filter_generation_params(parameters)
                gen_params = self._sanitize_generation_params(gen_params)
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_params)
                
                # Decode and extract response
                full_response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                results.append(self._extract_response(full_response))
            
            return results
            
        except Exception as e:
            logger.exception("Error in batch LFM2 generation: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)


class LlavaPhiHandler(HuggingFaceVLMHandler):
    """
    Custom handler for LLaVA-Phi-3 model which requires specific prompt formatting
    and keyword arguments for processor.
    """
    
    def load(self, precision: str = "float16", use_flash_attention: bool = False) -> None:
        """Load model with patch_size workaround for LLaVA processor bug."""
        # Call parent load first
        super().load(precision, use_flash_attention)
        
        # Fix LLaVA processor bug: patch_size not set
        if hasattr(self.processor, 'patch_size') and self.processor.patch_size is None:
            if hasattr(self.processor, 'image_processor') and hasattr(self.processor.image_processor, 'patch_size'):
                self.processor.patch_size = self.processor.image_processor.patch_size
                logger.info("Set patch_size from image_processor: %s", self.processor.patch_size)
            else:
                # Default patch size for LLaVA models
                self.processor.patch_size = 14
                logger.warning("patch_size not found in config, using default: 14")
    
    def _format_prompt(self, prompt: Optional[str]) -> str:
        """
        Format prompt with LLaVA-Phi-3 special tokens.
        Template: <|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n
        """
        if not prompt or not prompt.strip():
            prompt = "Describe this image in detail."
        else:
            prompt = prompt.strip()
        
        # Ensure <image> token is present
        if "<image>" not in prompt:
            formatted_prompt = f"<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"
        else:
            # User already included <image> token, just wrap with chat template
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        
        return formatted_prompt
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Generate caption using LLaVA-Phi-3's specific format."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            image = self._ensure_rgb(image)
            
            # Format prompt with special tokens
            formatted_prompt = self._format_prompt(prompt)
            
            # LLaVA processor requires keyword arguments
            inputs = self.processor(
                text=formatted_prompt,
                images=image,
                return_tensors='pt'
            )
            
            # Move inputs to device and match model dtype
            inputs = self._move_inputs_to_device(inputs, match_model_dtype=True)
            
            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            
            # Set defaults before sanitization
            if 'max_new_tokens' not in gen_params:
                gen_params['max_new_tokens'] = 200
            if 'do_sample' not in gen_params:
                gen_params['do_sample'] = False  # Greedy decoding by default
            
            gen_params = self._sanitize_generation_params(gen_params)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)
            
            # Decode output, skipping input tokens to get only the generated response
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][prompt_length:]
            caption = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
            
            return caption if caption else "Unable to generate description."
            
        except Exception as e:
            logger.exception("Error with LLaVA-Phi-3 inference: %s", e)
            return f"Error: {str(e)}"
    
    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """Generate captions for multiple images using LLaVA-Phi-3's format."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            # Ensure all images are RGB
            images = [self._ensure_rgb(img) for img in images]
            
            # Prepare prompts
            if prompts is None:
                prompts = [None] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")
            
            # Format all prompts with special tokens
            formatted_prompts = [self._format_prompt(p) for p in prompts]
            
            # LLaVA processor requires keyword arguments with text list
            inputs = self.processor(
                text=formatted_prompts,
                images=images,
                return_tensors='pt',
                padding=True
            )
            
            # Move inputs to device and match model dtype
            inputs = self._move_inputs_to_device(inputs, match_model_dtype=True)
            
            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            
            # Set defaults before sanitization
            if 'max_new_tokens' not in gen_params:
                gen_params['max_new_tokens'] = 200
            if 'do_sample' not in gen_params:
                gen_params['do_sample'] = False  # Greedy decoding by default
            
            gen_params = self._sanitize_generation_params(gen_params)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)
            
            # Decode outputs, skipping input tokens for each
            results = []
            prompt_length = inputs["input_ids"].shape[1]
            for output in output_ids:
                new_tokens = output[prompt_length:]
                caption = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
                results.append(caption if caption else "Unable to generate description.")
            
            return results
            
        except Exception as e:
            logger.exception("Error in batch LLaVA-Phi-3 generation: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)


class NanonetsOCRHandler(HuggingFaceOCRHandler):
    """
    Custom handler for Nanonets OCR model which requires chat template format
    with system and user messages.
    """
    
    def _default_prompt(self) -> str:
        """Default OCR prompt for comprehensive text extraction."""
        return (
            "Extract the text from the above document as if you were reading it naturally.\n"
            "Return the tables in HTML format. Return the equations in LaTeX representation.\n"
            "If there is an image in the document and an image caption is not present, add a small description\n"
            "of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>.\n"
            "Watermarks should be wrapped in <watermark></watermark>.\n"
            "Page numbers should be wrapped in <page_number></page_number>.\n"
            "Prefer using ☐ and ☑ for check boxes."
        )
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Extract text using Nanonets OCR with chat template format."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            image = self._ensure_rgb(image)
            
            # Use default OCR prompt if none provided
            prompt_text = (prompt or self._default_prompt()).strip()
            
            # Build chat-style messages (required by Nanonets processor)
            messages = [
                {"role": "system", "content": "You are a helpful OCR assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]
            
            # Apply chat template to get formatted text
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process with both text and images
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
            
            # Move to device and match model dtype
            inputs = self._move_inputs_to_device(inputs, match_model_dtype=True)
            
            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params.setdefault("do_sample", False)  # Deterministic by default for OCR
            gen_params = self._sanitize_generation_params(gen_params)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)
            
            # Skip prompt tokens and decode only the generated response
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[:, prompt_len:]
            text_output = self.processor.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            
            return text_output.strip() if text_output else ""
            
        except Exception as e:
            logger.exception("Error with Nanonets OCR: %s", e)
            return f"Error: {str(e)}"
    
    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """Extract text from multiple images using batch processing."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            # Ensure all images are RGB
            images = [self._ensure_rgb(img) for img in images]
            
            # Prepare prompts (use default if not provided)
            if prompts is None:
                default_prompt = self._default_prompt()
                prompts = [default_prompt] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")
            
            # Build chat messages for each image
            texts = []
            for i, (image, prompt_text) in enumerate(zip(images, prompts)):
                prompt_text = (prompt_text or self._default_prompt()).strip()
                
                messages = [
                    {"role": "system", "content": "You are a helpful OCR assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                ]
                
                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                texts.append(text)
            
            # Process batch with both text and images
            inputs = self.processor(
                text=texts, images=images, padding=True, return_tensors="pt"
            )
            
            # Move to device and match model dtype
            inputs = self._move_inputs_to_device(inputs, match_model_dtype=True)
            
            # Prepare generation parameters
            gen_params = self._filter_generation_params(parameters)
            gen_params.setdefault("do_sample", False)  # Deterministic by default for OCR
            gen_params = self._sanitize_generation_params(gen_params)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_params)
            
            # Skip prompt tokens and decode each result
            prompt_len = inputs["input_ids"].shape[1]
            results = []
            for output in output_ids:
                new_tokens = output[prompt_len:]
                text_output = self.processor.tokenizer.decode(
                    new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                results.append(text_output.strip() if text_output else "")
            
            return results
            
        except Exception as e:
            logger.exception("Error in batch Nanonets OCR: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)


class ChandraOCRHandler(HuggingFaceOCRHandler):
    """
    Custom handler for Chandra OCR model which uses the chandra package's generate_hf function.
    """
    
    def load(self, precision: str = None, use_flash_attention: bool = False, **kwargs) -> None:
        """Load Chandra OCR model (only supports float precisions)."""
        try:
            from utils.torch_utils import pick_device
            self.device = pick_device(torch)
            
            # Chandra only supports float precisions (no quantization)
            precision = precision or self.config.get('default_precision', 'bfloat16')
            supported_precisions = ["float32", "float16", "bfloat16"]
            if precision not in supported_precisions:
                logger.warning("Chandra OCR only supports float precisions. Falling back to bfloat16.")
                precision = "bfloat16"
            
            logger.info("Loading %s on %s with %s precision…", self.model_key, self.device, precision)
            
            # Load processor
            processor_config = self.config.get('processor_config', {})
            self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_config)
            
            # Setup pad token
            self._setup_pad_token()
            
            # Prepare model loading kwargs
            model_kwargs = self.config.get('model_config', {}).copy()
            model_kwargs['torch_dtype'] = self._get_dtype(precision)
            
            # Setup flash attention if requested
            if use_flash_attention:
                self._setup_flash_attention(model_kwargs, precision)
            
            # Load model
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **model_kwargs)
            self.model.to(self.device)
            
            # Attach processor to model (required by Chandra's generate_hf)
            self.model.processor = self.processor
            
            # Configure generation settings for deterministic OCR output
            if self.processor.tokenizer.pad_token_id is not None:
                self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
            
            # Use greedy decoding for deterministic OCR
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
            
            self.model.eval()
            logger.info("%s loaded successfully", self.model_key)
            
        except Exception as e:
            logger.exception("Error loading %s: %s", self.model_key, e)
            raise
    
    def infer_single(self, image: Image.Image, prompt: Optional[str] = None, 
                    parameters: Optional[Dict] = None) -> str:
        """Extract text using Chandra OCR with generate_hf."""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            from chandra.model.hf import generate_hf
            from chandra.model.schema import BatchInputItem
            
            logger.debug("Chandra single image processing")
            
            image = self._ensure_rgb(image)
            
            # Extract parameters
            params = parameters or {}
            max_output_tokens = params.get('max_new_tokens', 512)
            chandra_preset = params.get('chandra_preset', 'user_prompt')
            
            # Create BatchInputItem based on preset selection
            if chandra_preset == "ocr_layout":
                batch_item = BatchInputItem(image=image, prompt_type="ocr_layout")
            elif chandra_preset == "ocr":
                batch_item = BatchInputItem(image=image, prompt_type="ocr")
            else:  # user_prompt (default)
                if prompt:
                    batch_item = BatchInputItem(image=image, prompt=prompt)
                else:
                    batch_item = BatchInputItem(image=image, prompt_type="ocr_layout")
            
            # Generate using Chandra's generate_hf
            with torch.inference_mode():
                results = generate_hf([batch_item], self.model, max_output_tokens=max_output_tokens)
            
            # Extract raw text from GenerationResult
            caption = results[0].raw if results and hasattr(results[0], 'raw') else ""
            return caption.strip() if caption else "Unable to generate description."
            
        except Exception as e:
            logger.exception("Error with Chandra OCR: %s", e)
            return f"Error: {str(e)}"
    
    def infer_batch(self, images: List[Image.Image], prompts: Optional[List[str]] = None,
                   parameters: Optional[Dict] = None) -> List[str]:
        """
        Extract text from multiple images.
        Note: Chandra processes images sequentially internally, so we process one at a time.
        """
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.model_key} not loaded")
        
        try:
            logger.info(f"Chandra processing {len(images)} images sequentially")
            
            # Prepare prompts
            if prompts is None:
                prompts = [None] * len(images)
            elif len(prompts) != len(images):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of images ({len(images)})")
            
            # Process each image individually (Chandra doesn't benefit from batching)
            results = []
            for image, prompt in zip(images, prompts):
                result = self.infer_single(image, prompt, parameters)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.exception("Error in Chandra OCR processing: %s", e)
            error_msg = f"Error: {str(e)}"
            return [error_msg] * len(images)
