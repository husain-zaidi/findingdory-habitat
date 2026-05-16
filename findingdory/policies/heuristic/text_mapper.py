#!/usr/bin/env python3
import copy
import gzip
import json
import math
import re
import warnings
from collections import Counter

import numpy as np
import torch
from qwen_vl_utils import process_vision_info

from findingdory.policies.heuristic.vlm_mapper import VLMMapperAgent
from findingdory.policies.llm.utils import load_text, save_response


class TextMapperAgent(VLMMapperAgent):
    """
    Two-stage text agent from the appendix:
    1. Summarize frame chunks into structured text with the VLM.
    2. Reason over the concatenated text history to predict frame indices.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.summary_prompt = load_text(config.summary_prompt_file)
        self.reasoning_prompt = load_text(config.reasoning_prompt_file)
        self.summary_chunk_size = min(config.summary_chunk_size, 250)
        self.summary_images_per_chunk = min(
            getattr(config, "summary_images_per_chunk", 8), self.summary_chunk_size
        )
        self.summary_mode = getattr(config, "summary_mode", "symbolic")
        self._category_maps_ready = False
        self._obj_category_by_id = {}
        self._recep_category_by_id = {}
        self._other_obj_category_by_id = {}

    def process_frames_for_vlm(self, save_images=True):
        """
        Override the base implementation so the text agent keeps the full frame
        history and handles chunking explicitly during summarization.
        """
        if self._frame_num_to_original_frame_num is None:
            self._frame_num_to_original_frame_num = list(range(len(self._observations)))
        return super(VLMMapperAgent, self).process_frames_for_vlm(save_images=save_images)

    def get_text_response(self, prompt):
        torch.cuda.empty_cache()
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=96)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.decode(
            generated_ids_trimmed[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        torch.cuda.empty_cache()
        return output_text

    def get_chunk_summary_response(self, images, prompt, chunk_idx):
        torch.cuda.empty_cache()
        sampled_images = images
        if len(images) > self.summary_images_per_chunk:
            sample_indices = np.linspace(
                0, len(images) - 1, self.summary_images_per_chunk, dtype=int
            ).tolist()
            sampled_images = [images[idx] for idx in sample_indices]

        image_paths = self.save_images(sampled_images, chunk_idx)
        content = [{"type": "image", "image": image_path} for image_path in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.decode(
            generated_ids_trimmed[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        torch.cuda.empty_cache()
        return output_text

    def _build_history_entry(self, summary, start_idx, end_idx):
        start_obs = self._observations[start_idx]
        end_obs = self._observations[end_idx]
        history_entry = {
            "frame_start": start_idx,
            "frame_end": end_idx,
            "time_of_day_start": start_obs["time_of_day"],
            "time_of_day_end": end_obs["time_of_day"],
            "room_name": summary.get("room_name", ""),
            "picking_placing_or_navigating": summary.get(
                "picking_placing_or_navigating", "nav"
            ),
            "object_being_manipulated": summary.get("object_being_manipulated", ""),
            "receptacle_being_manipulated": summary.get(
                "receptacle_being_manipulated", ""
            ),
            "other_objects_in_scene": summary.get("other_objects_in_scene", []),
        }
        return history_entry

    def _ensure_category_maps(self):
        if self._category_maps_ready or self.env_config is None:
            return

        dataset_path = self.env_config.habitat.dataset.data_path
        with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
            dataset_payload = json.load(f)

        self._obj_category_by_id = {
            int(idx) + 1: name
            for name, idx in dataset_payload["obj_category_to_obj_category_id"].items()
        }
        self._recep_category_by_id = {
            int(idx) + 1: name
            for name, idx in dataset_payload["recep_category_to_recep_category_id"].items()
        }
        self._other_obj_category_by_id = {
            int(idx) + 1: name
            for name, idx in dataset_payload["other_obj_category_to_other_obj_category_id"].items()
            if name
        }
        self._category_maps_ready = True

    def _categories_from_segmentation(self, obs, key, category_by_id):
        raw = obs.get(key)
        if raw is None:
            return []
        values = np.unique(raw)
        categories = []
        for value in values:
            value = int(value)
            if value == 0:
                continue
            if value in category_by_id:
                categories.append(category_by_id[value])
        return categories

    def _find_best_frame_for_category(
        self,
        category_name,
        segmentation_key,
        category_by_id,
        nav_only=False,
    ):
        category_id = None
        for cur_id, cur_name in category_by_id.items():
            if cur_name == category_name:
                category_id = cur_id
                break
        if category_id is None:
            return None

        best_idx = None
        best_score = -1.0
        for idx, obs in enumerate(self._observations):
            if nav_only and obs.get("manipulation_mode"):
                continue
            segmentation = obs.get(segmentation_key)
            if segmentation is None:
                continue
            segmentation = np.asarray(segmentation)
            if segmentation.ndim > 2:
                segmentation = np.squeeze(segmentation)
            if segmentation.ndim != 2:
                continue
            mask = segmentation == category_id
            area = int(np.sum(mask))
            if area <= 0:
                continue

            coords = np.argwhere(mask)
            centroid_y, centroid_x = coords.mean(axis=0)
            height, width = segmentation.shape[:2]
            center_y = (height - 1) / 2.0
            center_x = (width - 1) / 2.0
            max_dist = math.sqrt(center_y**2 + center_x**2) + 1e-6
            dist = math.sqrt((centroid_y - center_y) ** 2 + (centroid_x - center_x) ** 2)
            center_weight = max(0.05, 1.0 - (dist / max_dist))
            score = float(area) * center_weight

            if score > best_score:
                best_score = score
                best_idx = idx
        if best_score <= 0:
            return None
        return best_idx

    def _direct_retrieval_from_goal(self, lang_goal):
        self._ensure_category_maps()

        goal = lang_goal.lower().strip()
        match = re.match(r"navigate to (?:a|an|the) ([a-z0-9_ \-]+)\.?$", goal)
        if not match:
            return None

        category_name = match.group(1).strip().replace(" ", "_").replace("-", "_")
        candidates = [
            self._find_best_frame_for_category(
                category_name,
                "all_object_segmentation",
                self._obj_category_by_id,
                nav_only=True,
            ),
            self._find_best_frame_for_category(
                category_name,
                "receptacle_segmentation",
                self._recep_category_by_id,
                nav_only=True,
            ),
            self._find_best_frame_for_category(
                category_name,
                "other_object_segmentation",
                self._other_obj_category_by_id,
                nav_only=True,
            ),
        ]
        if not any(candidate is not None for candidate in candidates):
            candidates = [
                self._find_best_frame_for_category(
                    category_name, "all_object_segmentation", self._obj_category_by_id
                ),
                self._find_best_frame_for_category(
                    category_name, "receptacle_segmentation", self._recep_category_by_id
                ),
                self._find_best_frame_for_category(
                    category_name, "other_object_segmentation", self._other_obj_category_by_id
                ),
            ]
        candidates = [candidate for candidate in candidates if candidate is not None]
        if not candidates:
            return None
        return [candidates[0]]

    def _get_segmentation_area(self, obs, segmentation_key, category_name, category_by_id):
        segmentation = obs.get(segmentation_key)
        if segmentation is None:
            return 0
        segmentation = np.asarray(segmentation)
        if segmentation.ndim > 2:
            segmentation = np.squeeze(segmentation)
        if segmentation.ndim != 2:
            return 0

        category_id = None
        for cur_id, cur_name in category_by_id.items():
            if cur_name == category_name:
                category_id = cur_id
                break
        if category_id is None:
            return 0
        return int(np.sum(segmentation == category_id))

    def _find_best_interaction_frame(
        self,
        object_category=None,
        receptacle_category=None,
        require_manipulation=True,
        prefer_latest=True,
    ):
        self._ensure_category_maps()

        candidates = []
        for idx, obs in enumerate(self._observations):
            if require_manipulation and not obs.get("manipulation_mode"):
                continue

            obj_area = 0
            recep_area = 0
            if object_category is not None:
                obj_area = max(
                    self._get_segmentation_area(
                        obs,
                        "all_object_segmentation",
                        object_category,
                        self._obj_category_by_id,
                    ),
                    self._get_segmentation_area(
                        obs,
                        "other_object_segmentation",
                        object_category,
                        self._other_obj_category_by_id,
                    ),
                )
                if obj_area <= 0:
                    continue
            if receptacle_category is not None:
                recep_area = self._get_segmentation_area(
                    obs,
                    "receptacle_segmentation",
                    receptacle_category,
                    self._recep_category_by_id,
                )
                if recep_area <= 0:
                    continue

            score = float(obj_area + recep_area)
            if prefer_latest:
                score += idx * 1e-3
            candidates.append((score, idx))

        if not candidates:
            return None
        candidates.sort()
        return candidates[-1][1]

    def _direct_history_retrieval_from_goal(self, lang_goal):
        goal = lang_goal.lower().strip().rstrip(".")

        match = re.match(
            r"navigate to (?:the|a|an) ([a-z0-9_ \-]+) that you interacted with yesterday",
            goal,
        )
        if match:
            object_category = match.group(1).strip().replace(" ", "_").replace("-", "_")
            frame_idx = self._find_best_interaction_frame(object_category=object_category)
            return [frame_idx] if frame_idx is not None else None

        match = re.match(
            r"navigate to (?:a|an|the) ([a-z0-9_ \-]+) you picked an object from",
            goal,
        )
        if match:
            receptacle_category = match.group(1).strip().replace(" ", "_").replace("-", "_")
            frame_idx = self._find_best_interaction_frame(receptacle_category=receptacle_category)
            return [frame_idx] if frame_idx is not None else None

        match = re.match(
            r"navigate to (?:a|an|the) ([a-z0-9_ \-]+) you placed an object on",
            goal,
        )
        if match:
            receptacle_category = match.group(1).strip().replace(" ", "_").replace("-", "_")
            frame_idx = self._find_best_interaction_frame(receptacle_category=receptacle_category)
            return [frame_idx] if frame_idx is not None else None

        return None

    def _infer_room_name(self, visible_receptacles, visible_other_objects):
        names = set(visible_receptacles + visible_other_objects)
        if {"toilet", "bathtub", "sink"} & names:
            return "bathroom"
        if {"bed", "hamper", "chest_of_drawers"} & names:
            return "bedroom"
        if {"counter", "cabinet", "oven", "microwave", "toaster"} & names:
            return "kitchen"
        if {"couch", "tv", "bench"} & names:
            return "living_room"
        if {"desk", "filing_cabinet"} & names:
            return "office"
        return ""

    def _summarize_chunk_symbolic(self, start_idx, end_idx):
        self._ensure_category_maps()

        obj_counts = Counter()
        recep_counts = Counter()
        other_counts = Counter()
        manip_obj_counts = Counter()
        manip_recep_counts = Counter()
        manip_steps = 0

        for obs in self._observations[start_idx : end_idx + 1]:
            obj_names = self._categories_from_segmentation(
                obs, "all_object_segmentation", self._obj_category_by_id
            )
            recep_names = self._categories_from_segmentation(
                obs, "receptacle_segmentation", self._recep_category_by_id
            )
            other_names = self._categories_from_segmentation(
                obs, "other_object_segmentation", self._other_obj_category_by_id
            )

            obj_counts.update(obj_names)
            recep_counts.update(recep_names)
            other_counts.update(other_names)

            if obs.get("manipulation_mode"):
                manip_steps += 1
                manip_obj_counts.update(obj_names)
                manip_recep_counts.update(recep_names)

        object_being_manipulated = manip_obj_counts.most_common(1)[0][0] if manip_obj_counts else ""
        receptacle_being_manipulated = (
            manip_recep_counts.most_common(1)[0][0] if manip_recep_counts else ""
        )

        if manip_steps == 0:
            nav_mode = "nav"
        elif sum(manip_obj_counts.values()) >= sum(manip_recep_counts.values()):
            nav_mode = "picking"
        else:
            nav_mode = "placing"

        visible_receptacles = [name for name, _ in recep_counts.most_common(5)]
        visible_other_objects = [name for name, _ in other_counts.most_common(8)]

        return {
            "room_name": self._infer_room_name(visible_receptacles, visible_other_objects),
            "picking_placing_or_navigating": nav_mode,
            "object_being_manipulated": object_being_manipulated,
            "receptacle_being_manipulated": receptacle_being_manipulated,
            "other_objects_in_scene": visible_other_objects + visible_receptacles,
        }

    def _extract_summary_from_response(self, response):
        parsed = self.extract_info_from_response(response)
        if parsed is not None:
            return parsed

        def extract_string(key):
            patterns = [
                rf'"{key}"\s*:\s*"([^"]+)"',
                rf"{key}\s*:\s*\"([^\"]+)\"",
                rf"{key}\s*:\s*([A-Za-z_ ][A-Za-z_ \-]+)",
                rf"{key.replace('_', ' ')}\s*:\s*([A-Za-z0-9_ ][A-Za-z0-9_ \-]+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, response, flags=re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            return ""

        def extract_list(key):
            match = re.search(rf'"{key}"\s*:\s*\[(.*?)\]', response, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                label_match = re.search(
                    rf"{key.replace('_', ' ')}\s*:\s*(.+)",
                    response,
                    flags=re.IGNORECASE,
                )
                if not label_match:
                    return []
                raw_value = label_match.group(1).split("```")[0].strip()
                return [item.strip(" -\"'") for item in raw_value.split(",") if item.strip()][:20]
            values = re.findall(r'"([^"]+)"', match.group(1))
            return values[:20]

        room_name = extract_string("room_name") or extract_string("Room name")
        object_name = extract_string("object_being_manipulated") or extract_string("Object being manipulated")
        receptacle_name = extract_string("receptacle_being_manipulated") or extract_string("Receptacle being manipulated")
        nav_value = (
            extract_string("picking_placing_or_navigating")
            or extract_string("Picking, placing, or navigating")
        ).lower()
        if "pick" in nav_value:
            nav_value = "picking"
        elif "plac" in nav_value:
            nav_value = "placing"
        else:
            nav_value = "nav"

        summary = {
            "room_name": room_name,
            "picking_placing_or_navigating": nav_value,
            "object_being_manipulated": object_name,
            "receptacle_being_manipulated": receptacle_name,
            "other_objects_in_scene": extract_list("other_objects_in_scene") or extract_list("Other objects in scene"),
        }

        if any(
            [
                summary["room_name"],
                summary["object_being_manipulated"],
                summary["receptacle_being_manipulated"],
                summary["other_objects_in_scene"],
            ]
        ):
            return summary
        return None

    def _summarize_chunk(self, images, start_idx, end_idx, chunk_idx):
        if self.summary_mode == "symbolic":
            return self._build_history_entry(
                self._summarize_chunk_symbolic(start_idx, end_idx),
                start_idx,
                end_idx,
            )

        prompt = copy.deepcopy(self.summary_prompt)
        prompt = prompt.replace("{chunk size}", str(len(images)))
        response = self.get_chunk_summary_response(images, prompt, chunk_idx)
        print("Chunk Summary Response: ", response.encode("utf-8"))
        parsed = self._extract_summary_from_response(response)
        if parsed is None:
            warnings.warn(
                f"Failed to parse text-agent chunk summary for frames {start_idx}-{end_idx}",
                RuntimeWarning,
            )
            parsed = {
                "room_name": "",
                "picking_placing_or_navigating": "nav",
                "object_being_manipulated": "",
                "receptacle_being_manipulated": "",
                "other_objects_in_scene": [],
            }
        return self._build_history_entry(parsed, start_idx, end_idx)

    def _build_text_history(self, frames):
        history_entries = []
        num_chunks = math.ceil(len(frames) / self.summary_chunk_size)

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.summary_chunk_size
            end_idx = min(len(frames), start_idx + self.summary_chunk_size) - 1
            chunk_frames = frames[start_idx : end_idx + 1]
            history_entries.append(
                self._summarize_chunk(chunk_frames, start_idx, end_idx, chunk_idx)
            )

        return history_entries

    def _goal_expects_multiple_indices(self, lang_goal):
        goal = lang_goal.lower()
        multi_keywords = [
            "all ",
            " in order",
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "revisit all",
            "multiple",
            "sequence",
        ]
        return any(keyword in goal for keyword in multi_keywords)

    def _extract_frame_indices_from_raw_text(self, raw_response, frames, lang_goal):
        patterns = [
            r"frame_indices\s*[:=]\s*\[([^\]]+)\]",
            r"frame indices\s*[:=]\s*\[([^\]]+)\]",
            r"frame_index(?:es)?\s*[:=]\s*\[([^\]]+)\]",
            r'"frame_indices"\s*:\s*\[([^\]]+)\]',
            r'"frame_index"\s*:\s*(\d+)',
            r"frame index\s*[:=]\s*(\d+)",
            r'"output"\s*:\s*\[([^\]]+)\]',
        ]
        for pattern in patterns:
            match = re.search(pattern, raw_response, flags=re.IGNORECASE)
            if not match:
                continue
            values = [int(value) for value in re.findall(r"-?\d+", match.group(1))]
            if not values:
                continue
            if not self._goal_expects_multiple_indices(lang_goal):
                values = values[:1]
            return [int(v) if 0 <= int(v) < len(frames) else -1 for v in values]

        # Fall back to the first integer list that appears after any output-like cue.
        cue_patterns = [
            r"(?:output|answer|result)\s*[:=]\s*(.+)",
            r"(?:navigate to|therefore)[^\n]*?(?:frame|index)[^\n]*",
        ]
        for cue_pattern in cue_patterns:
            cue_match = re.search(cue_pattern, raw_response, flags=re.IGNORECASE | re.DOTALL)
            if not cue_match:
                continue
            search_region = cue_match.group(1) if cue_match.lastindex else raw_response
            list_match = re.search(r"\[([^\]]+)\]", search_region)
            if list_match:
                values = [int(value) for value in re.findall(r"-?\d+", list_match.group(1))]
                if values:
                    if not self._goal_expects_multiple_indices(lang_goal):
                        values = values[:1]
                    return [int(v) if 0 <= int(v) < len(frames) else -1 for v in values]
        return [-1]

    def run_vlm(self, frames, lang_goal):
        assert len(frames) > 0, "No frames to process"

        direct_frame_indices = self._direct_retrieval_from_goal(lang_goal)
        if direct_frame_indices is not None:
            save_response(
                json.dumps(
                    {
                        "mode": "direct_symbolic_retrieval",
                        "goal": lang_goal,
                        "frame_indices": direct_frame_indices,
                    },
                    indent=2,
                ),
                self.output_folder_with_episode_index,
                model_name="qwen_text_agent",
                goal=lang_goal,
            )
            return direct_frame_indices

        direct_history_frame_indices = self._direct_history_retrieval_from_goal(lang_goal)
        if direct_history_frame_indices is not None:
            save_response(
                json.dumps(
                    {
                        "mode": "direct_symbolic_history_retrieval",
                        "goal": lang_goal,
                        "frame_indices": direct_history_frame_indices,
                    },
                    indent=2,
                ),
                self.output_folder_with_episode_index,
                model_name="qwen_text_agent",
                goal=lang_goal,
            )
            return direct_history_frame_indices

        history_entries = self._build_text_history(frames)
        history_json = json.dumps(history_entries, indent=2)

        prompt = copy.deepcopy(self.reasoning_prompt)
        prompt = prompt.replace("{history}", history_json)
        prompt = prompt.replace("{goal}", lang_goal)

        raw_response = self.get_text_response(prompt)
        print("Text Agent Response: ", raw_response.encode("utf-8"))
        llm_response = self.extract_info_from_response(raw_response)

        save_response(
            json.dumps(
                {
                    "history": history_entries,
                    "raw_reasoning_response": raw_response,
                    "reasoning_response": llm_response,
                },
                indent=2,
            ),
            self.output_folder_with_episode_index,
            model_name="qwen_text_agent",
            goal=lang_goal,
        )

        if llm_response is not None:
            frame_indices = self.extract_frame_indices_from_response(llm_response, frames)
            if frame_indices != [-1]:
                if not self._goal_expects_multiple_indices(lang_goal):
                    return frame_indices[:1]
                return frame_indices

        return self._extract_frame_indices_from_raw_text(raw_response, frames, lang_goal)
