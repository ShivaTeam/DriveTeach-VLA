"""Prompter prompt builder — TGP-Prompter 2D-TGP regression (DVD pretraining).

Format based on TGP Prompter QA.txt.
"""

from ir.schema import SampleIR
from utils.coord import format_past_timeline
from .base import PromptBuilder, register_prompt

SYSTEM = (
    "You are a vehicle trajectory prediction model for autonomous driving. "
    "Your task is to predict the ego vehicle's 4-second trajectory based on "
    "the following inputs: single-view image from front cameras, ego vehicle "
    "states (position), and discrete navigation commands. The input provides "
    "a 2-second history, and your output should ensure a safe trajectory for "
    "the next 4 seconds.\nYour predictions will be evaluated through a "
    "non-reactive 4-second simulation with an LQR controller and background "
    "actors following their recorded trajectories. The better your predictions, "
    "the higher your score."
)

_QUESTION = """\
<image>
As an autonomous driving system, predict the vehicle's trajectory based on:
1. Visual perception from front camera view.
2. Historical motion context (last 4 timesteps): {timeline}
3. Active navigation command: [{command}]
Output requirements:
- Predict pixel coordinates of 8 future trajectory points(Trajectory points may be invisible so coordinates could be negative / outside of the boundary)
- Each point format: {{"point_2d": [x:int, y:int], "heading": float}}
- Ensure that every [x, y] position in the image will be safe, empty road.
- Use [PT, ...] to encapsulate the trajectory
- Maintain numerical precision to 2 decimal places"""


@register_prompt("prompter")
class PrompterBuilder(PromptBuilder):

    def build(self, sample: SampleIR) -> dict:
        return {
            "id": sample.sample_id,
            "image": [sample.images[0]] if sample.images else [],
            "bbox_image": sample.bbox_image or "",
            "system": SYSTEM,
            "conversations": [
                {"from": "human", "value": self._question(sample)},
                {"from": "gpt", "value": self._answer(sample)},
            ],
        }

    def _question(self, sample: SampleIR) -> str:
        return _QUESTION.format(
            timeline=format_past_timeline(sample.trajectory or []),
            command=sample.metadata.command.upper(),
        )

    def _answer(self, sample: SampleIR) -> str:
        if sample.trajectory_2d_tgp is None:
            return ""
        pts = []
        for p in sample.trajectory_2d_tgp:
            pts.append(
                '{"point_2d": [%d, %d], "heading": %.2f}'
                % (p["point_2d"][0], p["point_2d"][1], p["heading"])
            )
        return f"Here is the planning trajectory [PT, {', '.join(pts)}]."
