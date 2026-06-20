import copy
import cv2
import math
import os
import time
import ast

from PIL import Image
import numpy as np
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

from findingdory.policies.llm.vlm_agent import VLMAgent
from findingdory.policies.llm.utils import save_response
from habitat.core.logging import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QwenAgent(VLMAgent):
    """LLM based agent in Habitat environments."""

    def __init__(self, config) -> None:
        """
        :param config: QWEN config
        """
        super().__init__(config)

        local_files_only = getattr(config, "local_files_only", False)
        cache_dir = getattr(config, "cache_dir", None)
        self.model_name = config.model
        self.model_loader = getattr(config, "model_loader", "qwen2_5_vl")
        self.enable_thinking = getattr(config, "enable_thinking", None)
        self.max_new_tokens = getattr(config, "max_new_tokens", 256)
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir

        self.processor = AutoProcessor.from_pretrained(
            config.model,
            min_pixels=128 * 28 * 28,
            max_pixels=256 * 28 * 28,
            use_fast=False,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )

        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # bnb_4bit_use_double_quant=True,
        )

        model_cls = (
            AutoModelForImageTextToText
            if self.model_loader == "auto_image_text"
            else Qwen2_5_VLForConditionalGeneration
        )
        self.model = model_cls.from_pretrained(
            config.model,
            device_map="auto",
            # torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2",
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        ).eval()

    def get_vlm_response(self, images, prompt, path=None):
        torch.cuda.empty_cache()

        mm_prompt = None
        if path:
            mm_prompt = path
            mm_type = "video" if path.endswith(".mp4") else "image"
        elif len(images) == 1:
        # else:
            mm_prompt = self.save_image(images[0])
            mm_type = "image"
        else:
            mm_prompt = self.save_video(images, fps=1)
            mm_type = "video"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": mm_type, mm_type: mm_prompt},
                    {"type": "text", "text": prompt},
                ]
            }
        ]

        # Preparation for inference
        chat_template_kwargs = {}
        if self.enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = self.enable_thinking
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        except TypeError:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        for key, value in list(video_kwargs.items()):
            if isinstance(value, list) and len(value) == 1:
                video_kwargs[key] = value[0]
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.decode(
            generated_ids_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # Clear CUDA cache to free up GPU memory
        torch.cuda.empty_cache()

        return output_text

    def run_vlm(self, frames, lang_goal):
        assert len(frames) > 0, "No frames to process"

        prompt = copy.deepcopy(self.prompt)
        prompt = prompt.replace("{goal}", lang_goal)

        # Clear CUDA cache to free up GPU memory
        torch.cuda.empty_cache()
        llm_response_raw = self.get_vlm_response(frames, prompt)
        # Print the full raw LLM response (including COT/thinking)
        logger.info("=" * 80)
        logger.info(f">>> RAW LLM RESPONSE (full output including COT/thinking):")
        logger.info(llm_response_raw)
        logger.info("<<< END RAW LLM RESPONSE")
        logger.info("=" * 80)
        llm_response = self.extract_info_from_response(llm_response_raw)

        save_response(
            llm_response_raw,
            self.output_folder_with_episode_index,
            model_name="qwen_agent",
            goal=lang_goal
        )

        return self.extract_frame_indices_from_response(llm_response, frames)

    def load_frames_and_run(self, video_path, lang_goal):
        prompt = copy.deepcopy(self.prompt)
        prompt = prompt.replace("{goal}", lang_goal)
        llm_response = self.get_vlm_response(images=None, prompt=prompt, path=video_path)
        print("LLM Response: ", llm_response.encode("utf-8"))
