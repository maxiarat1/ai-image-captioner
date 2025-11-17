"""
Inference Strategies
Provides reusable strategies for prompt formatting and response extraction.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Prompt Formatting Strategies
# ============================================================================

class PromptStrategy(ABC):
    """Base class for prompt formatting strategies."""

    @abstractmethod
    def format_single(self, processor, image: Image.Image, prompt: Optional[str],
                     device, supports_prompts: bool) -> Dict:
        """
        Format inputs for a single image.

        Returns:
            Dictionary of inputs ready for model.generate()
        """
        pass

    @abstractmethod
    def format_batch(self, processor, images: List[Image.Image], prompts: Optional[List[str]],
                    device, supports_prompts: bool) -> Dict:
        """
        Format inputs for a batch of images.

        Returns:
            Dictionary of inputs ready for model.generate()
        """
        pass


class DefaultPromptStrategy(PromptStrategy):
    """Standard HF processor formatting."""

    def format_single(self, processor, image: Image.Image, prompt: Optional[str],
                     device, supports_prompts: bool) -> Dict:
        if prompt and prompt.strip() and supports_prompts:
            inputs = processor(image, prompt.strip(), return_tensors="pt")
        else:
            inputs = processor(image, return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}

    def format_batch(self, processor, images: List[Image.Image], prompts: Optional[List[str]],
                    device, supports_prompts: bool) -> Dict:
        if prompts and prompts[0] and supports_prompts:
            text_prompts = [p.strip() if p else "" for p in prompts]
            inputs = processor(images=images, text=text_prompts,
                             return_tensors="pt", padding=True)
        else:
            inputs = processor(images=images, return_tensors="pt", padding=True)
        return {k: v.to(device) for k, v in inputs.items()}


class ChatTemplateStrategy(PromptStrategy):
    """Chat template formatting (for Nanonets, LFM2)."""

    def __init__(self, system_message: Optional[str] = None):
        self.system_message = system_message

    def format_single(self, processor, image: Image.Image, prompt: Optional[str],
                     device, supports_prompts: bool) -> Dict:
        messages = self._build_messages(image, prompt or "Describe this image.")

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process with both text and images
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}

    def format_batch(self, processor, images: List[Image.Image], prompts: Optional[List[str]],
                    device, supports_prompts: bool) -> Dict:
        if prompts is None:
            prompts = ["Describe this image."] * len(images)

        # Build chat messages for each image
        texts = []
        for image, prompt_text in zip(images, prompts):
            messages = self._build_messages(image, prompt_text or "Describe this image.")
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)

        # Process batch
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}

    def _build_messages(self, image: Image.Image, prompt: str) -> List[Dict]:
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        })
        return messages


class ConversationStrategy(PromptStrategy):
    """Janus-style conversation formatting."""

    def format_single(self, processor, image: Image.Image, prompt: Optional[str],
                     device, supports_prompts: bool) -> Dict:
        prompt_text = prompt.strip() if prompt and prompt.strip() else "Describe this image in detail."

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt_text}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        prepare_inputs = processor(
            conversations=conversation,
            images=[image],
            force_batchify=True
        )

        return prepare_inputs

    def format_batch(self, processor, images: List[Image.Image], prompts: Optional[List[str]],
                    device, supports_prompts: bool) -> Dict:
        # Janus doesn't support true batching, so we return None
        # This signals to use sequential processing
        return None


class R4BThinkingStrategy(PromptStrategy):
    """R4B model with thinking mode support."""

    def __init__(self):
        self.thinking_mode = 'auto'

    def set_thinking_mode(self, mode: str):
        """Set thinking mode from parameters."""
        if mode in ['auto', 'short', 'long']:
            self.thinking_mode = mode

    def format_single(self, processor, image: Image.Image, prompt: Optional[str],
                     device, supports_prompts: bool) -> Dict:
        user_prompt = prompt.strip() if prompt and prompt.strip() else "Describe this image."

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt}
            ]
        }]

        # Apply chat template with thinking_mode
        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking_mode=self.thinking_mode
        )

        # Process inputs
        inputs = processor(images=image, text=text_prompt, return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}

    def format_batch(self, processor, images: List[Image.Image], prompts: Optional[List[str]],
                    device, supports_prompts: bool) -> Dict:
        if prompts is None:
            prompts = [None] * len(images)

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

            text_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                thinking_mode=self.thinking_mode
            )
            text_prompts.append(text_prompt)

        inputs = processor(images=images, text=text_prompts, return_tensors="pt", padding=True)
        return {k: v.to(device) for k, v in inputs.items()}


class LlavaPhiStrategy(PromptStrategy):
    """LLaVA-Phi-3 specific prompt formatting."""

    def _format_prompt(self, prompt: Optional[str]) -> str:
        """Format prompt with LLaVA-Phi-3 special tokens."""
        if not prompt or not prompt.strip():
            prompt = "Describe this image in detail."
        else:
            prompt = prompt.strip()

        # Ensure <image> token is present
        if "<image>" not in prompt:
            formatted_prompt = f"<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"
        else:
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

        return formatted_prompt

    def format_single(self, processor, image: Image.Image, prompt: Optional[str],
                     device, supports_prompts: bool) -> Dict:
        formatted_prompt = self._format_prompt(prompt)
        inputs = processor(text=formatted_prompt, images=image, return_tensors='pt')
        return {k: v.to(device) for k, v in inputs.items()}

    def format_batch(self, processor, images: List[Image.Image], prompts: Optional[List[str]],
                    device, supports_prompts: bool) -> Dict:
        if prompts is None:
            prompts = [None] * len(images)

        formatted_prompts = [self._format_prompt(p) for p in prompts]
        inputs = processor(text=formatted_prompts, images=images, return_tensors='pt', padding=True)
        return {k: v.to(device) for k, v in inputs.items()}


class LFM2Strategy(PromptStrategy):
    """LFM2 model chat template formatting."""

    def format_single(self, processor, image: Image.Image, prompt: Optional[str],
                     device, supports_prompts: bool) -> Dict:
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
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

        return {k: v.to(device) for k, v in inputs.items()}

    def format_batch(self, processor, images: List[Image.Image], prompts: Optional[List[str]],
                    device, supports_prompts: bool) -> Dict:
        # LFM2 processes images sequentially, return None to trigger fallback
        return None


# ============================================================================
# Response Extraction Strategies
# ============================================================================

class ResponseStrategy(ABC):
    """Base class for response extraction strategies."""

    @abstractmethod
    def extract_single(self, processor, output_ids, input_ids=None) -> str:
        """Extract response from single generation output."""
        pass

    @abstractmethod
    def extract_batch(self, processor, output_ids, input_ids=None) -> List[str]:
        """Extract responses from batch generation output."""
        pass


class DefaultResponseStrategy(ResponseStrategy):
    """Standard decode and strip."""

    def extract_single(self, processor, output_ids, input_ids=None) -> str:
        result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return result.strip() if result else ""

    def extract_batch(self, processor, output_ids, input_ids=None) -> List[str]:
        results = processor.batch_decode(output_ids, skip_special_tokens=True)
        return [r.strip() for r in results]


class SkipInputTokensStrategy(ResponseStrategy):
    """Decode only generated tokens, skipping input."""

    def extract_single(self, processor, output_ids, input_ids=None) -> str:
        if input_ids is not None and len(input_ids) > 0:
            # Skip input tokens
            input_length = input_ids.shape[1] if hasattr(input_ids, 'shape') else len(input_ids[0])
            new_tokens = output_ids[0][input_length:]
            result = processor.decode(new_tokens, skip_special_tokens=True)
        else:
            result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return result.strip() if result else ""

    def extract_batch(self, processor, output_ids, input_ids=None) -> List[str]:
        results = []
        if input_ids is not None:
            input_length = input_ids.shape[1]
            for output in output_ids:
                new_tokens = output[input_length:]
                result = processor.decode(new_tokens, skip_special_tokens=True)
                results.append(result.strip() if result else "")
        else:
            results = processor.batch_decode(output_ids, skip_special_tokens=True)
            results = [r.strip() for r in results]
        return results


class ThinkingTagsStrategy(ResponseStrategy):
    """Extract final result by removing thinking tags (R4B)."""

    def extract_single(self, processor, output_ids, input_ids=None) -> str:
        # First skip input tokens
        if input_ids is not None and len(input_ids) > 0:
            input_length = input_ids.shape[1] if hasattr(input_ids, 'shape') else len(input_ids[0])
            new_tokens = output_ids[0][input_length:]
            result = processor.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Extract content after thinking tags
        return self._extract_final_result(result)

    def extract_batch(self, processor, output_ids, input_ids=None) -> List[str]:
        results = []
        if input_ids is not None:
            input_length = input_ids.shape[1]
            for output in output_ids:
                new_tokens = output[input_length:]
                result = processor.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                results.append(self._extract_final_result(result))
        else:
            decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
            results = [self._extract_final_result(r) for r in decoded]
        return results

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


class AssistantResponseStrategy(ResponseStrategy):
    """Extract assistant's response from conversation (LFM2)."""

    def extract_single(self, processor, output_ids, input_ids=None) -> str:
        full_response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return self._extract_response(full_response)

    def extract_batch(self, processor, output_ids, input_ids=None) -> List[str]:
        full_responses = processor.batch_decode(output_ids, skip_special_tokens=True)
        return [self._extract_response(r) for r in full_responses]

    def _extract_response(self, full_response: str) -> str:
        """Extract the assistant's response from the full decoded output."""
        if "assistant" in full_response.lower():
            parts = full_response.split("assistant", 1)
            if len(parts) > 1:
                response = parts[1].strip()
                response = response.lstrip(":").strip()
                return response if response else "Unable to generate description."

        return full_response.strip() if full_response.strip() else "Unable to generate description."


class JanusResponseStrategy(ResponseStrategy):
    """Janus-specific response extraction using tokenizer."""

    def extract_single(self, processor, output_ids, input_ids=None) -> str:
        result = processor.tokenizer.decode(
            output_ids[0].cpu().tolist(),
            skip_special_tokens=True
        )
        return result.strip() if result else "Unable to generate description."

    def extract_batch(self, processor, output_ids, input_ids=None) -> List[str]:
        results = []
        for output in output_ids:
            result = processor.tokenizer.decode(
                output[0].cpu().tolist(),
                skip_special_tokens=True
            )
            results.append(result.strip() if result else "Unable to generate description.")
        return results
