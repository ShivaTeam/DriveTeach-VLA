"""Poutine prompt builder — VLM pseudo-labeling input."""

import json
import math

from ir.schema import SampleIR
from utils.coord import format_point, describe_past, describe_future
from .base import PromptBuilder, register_prompt

_OUTPUT_SCHEMA = """\
{
  "critical_objects": {
    "nearby_vehicle": "yes | no",
    "conflicting_pedestrian": "yes | no",
    "cyclist": "yes | no",
    "construction": "yes | no",
    "traffic_element": "yes | no",
    "weather_condition": "yes | no",
    "road_hazard": "yes | no",
    "emergency_vehicle": "yes | no",
    "animal": "yes | no",
    "special_vehicle": "yes | no",
    "conflicting_vehicle": "yes | no",
    "door_opening_vehicle": "yes | no"
  },
  "meta_behaviour": {
    "speed": "keep | accelerate | decelerate | other",
    "command": "straight | yield | left_turn | right_turn | lane_follow | lane_change_left | lane_change_right | reverse | other"
  },
  "explanation": "100-word description that references only the classes marked 'yes'"
}"""

_SYSTEM = """\
You are an expert labeller of driving scenarios.
Input:
-  1 frame of front-view image collected from the ego-vehicle at the present timestep with future trajectory visualization
- Current high-level intent (string)
- {past_desc}
- Expert {future_desc}
Task:
1. Inspect the input and decide, for each object class below, whether at least one critical instance of that class is present (i.e., it materially affects the ego-vehicle's future trajectory ). A vehicle can be a car, bus, truck, motorcyclist, scooter, etc. traffic_element includes traffic signs and traffic lights. road_hazard may include hazardous road conditions, road debris, obstacles, etc. A conflicting_vehicle is a vehicle that may potentially conflict with the ego's future path.
2. Output "yes" or "no" for every class (no omissions).
3. From the expert's future trajectory, assign exactly one category from each list:
   - speed ∈ {{ keep, accelerate, decelerate }}
   - command ∈ {{ straight, yield, left_turn, right_turn, lane_follow, lane_change_left, lane_change_right, reverse }}
   Choose the label that best summarises the overall behaviour of the expert future trajectory.
   - If none fits, use 'other', but do this sparingly.
   - Your output must be corresponded to the given expert trajectory.
4. Compose a concise 100-word natural-language description explaining why the expert trajectory was executed.
   - Mention only the classes you marked "yes"
   - Describe how each of those critical objects or conditions influences the trajectory.
   - Do not invent objects or conditions not present in the input.

Output format (strict JSON, no extra keys, no markdown codeblock chars(```), no commentary):
{schema}"""

_QUESTION = """\
- 1 frame of front-view image collected from the ego-vehicle at the present timestep with future trajectory visualization:
Picture 1: <image> the front view of the ego-vehicle with trajectory visualization. The red line indecates the future trajectory.
- Current high-level intent: {command}
Each trajectory point format: (x:float, y:float, heading:float)
- {past_desc}: {past_str}
- Expert {future_desc}: {future_str}
- Expert {future_desc} velocity length({n_steps} steps at 2 Hz): {vel_str}
Your output must be corresponded to this expert trajectory. Please output strict JSON, no extra keys, no markdown codeblock chars(```), no commentary"""


@register_prompt("poutine_label")
class PoutineLabelBuilder(PromptBuilder):

    def build(self, sample: SampleIR) -> dict:
        return {
            "id": sample.sample_id,
            "image": [sample.images[0]] if sample.images else [],
            "system": self._system(sample),
            "conversations": [
                {"from": "human", "value": self._question(sample)},
                {"from": "gpt", "value": ""},
            ],
        }

    def _system(self, sample: SampleIR) -> str:
        past_desc = describe_past(sample.trajectory or [])
        future_desc = describe_future(sample.trajectory or [])
        return _SYSTEM.format(
            past_desc=past_desc,
            future_desc=future_desc,
            schema=_OUTPUT_SCHEMA,
        )

    def _question(self, sample: SampleIR) -> str:
        tr = sample.trajectory or []
        past_desc = describe_past(tr)
        future_desc = describe_future(tr)
        n_steps = len([p for p in tr if p.t >= -1e-9])
        return _QUESTION.format(
            command=sample.metadata.command,
            past_desc=past_desc,
            future_desc=future_desc,
            past_str=self._fmt_timeline(sample, future=False),
            future_str=self._fmt_timeline(sample, future=True),
            vel_str=self._fmt_vel(sample),
            n_steps=n_steps,
        )

    def _fmt_timeline(self, sample: SampleIR, future: bool) -> str:
        if sample.trajectory is None:
            return "none"
        if future:
            pts = sorted([p for p in sample.trajectory if p.t > 1e-9], key=lambda p: p.t)
            items = [f"    - t+{i+1}: {format_point(p)}" for i, p in enumerate(pts)]
        else:
            pts = sorted([p for p in sample.trajectory if p.t <= 1e-9], key=lambda p: p.t)
            n = len(pts)
            items = []
            for i, p in enumerate(pts):
                label = "t-0" if p.t >= -1e-9 else f"t-{n - 1 - i}"
                items.append(f"    - {label}: {format_point(p)}")
        return "".join(items)

    def _fmt_vel(self, sample: SampleIR) -> str:
        """Format velocity magnitudes from actual ego velocity data.

        Computes sqrt(vx² + vy²) for current + future frames (9 values).
        """
        # Find current frame index (the one with images)
        current_idx = None
        for i, f in enumerate(sample.frames):
            if f.images:
                current_idx = i
                break
        if current_idx is None:
            return "[]"

        n_future = 8  # 8 future steps
        vels = []
        for i in range(current_idx, min(current_idx + n_future + 1, len(sample.frames))):
            vx, vy = sample.frames[i].velocity[0], sample.frames[i].velocity[1]
            vels.append(f"{math.hypot(vx, vy):.2f}")
        return f"[{', '.join(vels)}]"
