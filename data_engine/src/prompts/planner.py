"""Planner prompt builder — TGP-Planner CoT reasoning (CoT-SFT).

Format based on TGP Planner QA.txt.
"""

import json

from ir.schema import SampleIR
from utils.coord import format_past_timeline, format_traj
from .base import PromptBuilder, register_prompt

_QUESTION = """\
You are an expert driver. Suppose you are driving. Let's complete the following tasks step by step.
Input:
- 1 frame of front-view image collected from the ego-vehicle at the present timestep
Picture 1: <image> the front view of the ego-vehicle
- Current high-level intent (string): {command}
- 1.5-second past trajectory(3 steps at 2 Hz), each trajectory point format (x:float, y:float, heading:float): {timeline}
- To ensure a safe trajectory, you should pay attention to objects close to the key points of trajectory in the image: {tgp_kps}

Task 1: Critical Objects and Conditions Detection
Decide whether at least one critical instance of each class could influence the ego-vehicle's future path (no omissions). A vehicle can be a car, bus, truck, motorcyclist, scooter, etc. traffic_element includes traffic signs and traffic lights. road_hazard may include hazardous road conditions, road debris, obstacles, etc. A conflicting_vehicle is a vehicle that may potentially conflict with the ego's future path. Output "yes" or "no" for every class (no omissions).
   Object classes to audit:
     - nearby_vehicle
     - conflicting_pedestrian
     - cyclist
     - construction
     - traffic_element
     - weather_condition
     - road_hazard
     - emergency_vehicle
     - animal
     - special_vehicle
     - conflicting_vehicle
     - door_opening_vehicle

Task 2: Natural Language Explanation
(thinking output should be wrapped in <thinking>...</thinking>)
Compose a 100-word concise natural-language description of the optimal future 4-second trajectory for the ego vehicle that the expert driver (you) plans and explain why the expert driver plans to execute this trajectory.
   - Mention only the classes you marked "yes" in the previous task.
   - Describe how each of those critical objects or conditions influences the optimal trajectory.
   - Do not invent objects or conditions not present in the input.

Task 3: Meta-Behaviour Selection
Assign exactly one category from each list. Choose the label that best summarises the overall behaviour of the optimal future trajectory:
   - speed ∈ {{ keep, accelerate, decelerate }}
   - command ∈ {{ straight, yield, left_turn, right_turn, lane_follow, lane_change_left, lane_change_right, reverse }}
   Choose the label that best summarises the overall behaviour of the optimal future trajectory.
   - If none fits, use 'other', but do this sparingly.

Task 4: Future Trajectory Prediction
(answer output should be wrapped in <answer>...</answer>)
Given the input, critical objects/conditions, natural language explanation, and meta-behaviour, predict the optimal 4-second normalized future trajectory (8 steps at 2 Hz) of the ego vehicle. Predict 8 normalized future trajectory points in [PT, ...] format. Each point is (x, y, heading).
Output format (strict JSON, no extra keys, no markdown codeblock chars(```), no commentary):
{{
  "critical_objects": {{ ... }},
  "explanation": "100-word description...",
  "meta_behaviour": {{ "speed": "...", "command": "..." }},
  "future_trajectory": "<answer>[PT, ...]</answer>"
}}"""


@register_prompt("planner")
class PlannerBuilder(PromptBuilder):

    def build(self, sample: SampleIR) -> dict:
        return {
            "id": sample.sample_id,
            "image": [sample.images[0]] if sample.images else [],
            "system": "",
            "conversations": [
                {"from": "human", "value": self._question(sample)},
                {"from": "gpt", "value": self._answer(sample)},
            ],
        }

    def _question(self, sample: SampleIR) -> str:
        tgp_kps = []
        if sample.trajectory_2d_tgp:
            tgp_kps = [{"point_2d": p["point_2d"]} for p in sample.trajectory_2d_tgp]

        return _QUESTION.format(
            command=sample.metadata.command.upper(),
            timeline=format_past_timeline(sample.trajectory or []),
            tgp_kps=json.dumps(tgp_kps),
        )

    def _answer(self, sample: SampleIR) -> str:
        import collections
        cobjs = {}
        mbeh = {}
        expl = ""
        traj = ""

        if sample.pseudo_label:
            cobjs = sample.pseudo_label.critical_objects or {}
            mbeh = sample.pseudo_label.meta_behaviour or {}
            raw = sample.pseudo_label.explanation or ""
            expl = f"<thinking>{raw}</thinking>"

        if sample.normalized_trajectory:
            traj = f"<answer>{format_traj(sample.normalized_trajectory)}</answer>"

        # OrderedDict to match QA.txt key order: CO → explanation → MB → FT
        answer = collections.OrderedDict()
        answer["critical_objects"] = cobjs
        answer["explanation"] = expl
        answer["meta_behaviour"] = mbeh
        answer["future_trajectory"] = traj
        return json.dumps(answer, indent=2, ensure_ascii=False)
