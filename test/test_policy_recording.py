import importlib.util
from pathlib import Path

import pytest
import torch

_SUCCESS_PATH = Path(__file__).parents[1] / "kinova_lfd/data_collection/success.py"
_SUCCESS_SPEC = importlib.util.spec_from_file_location("recording_success", _SUCCESS_PATH)
assert _SUCCESS_SPEC is not None and _SUCCESS_SPEC.loader is not None
_SUCCESS_MODULE = importlib.util.module_from_spec(_SUCCESS_SPEC)
_SUCCESS_SPEC.loader.exec_module(_SUCCESS_MODULE)
cube_goal_success = _SUCCESS_MODULE.cube_goal_success


def test_cube_goal_success_uses_position_delta() -> None:
    cube = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]])
    goal = torch.tensor([[1.03, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]])

    assert cube_goal_success(cube, goal, delta=0.05)
    assert not cube_goal_success(cube, goal, delta=0.02)


def test_cube_goal_success_rejects_empty_pose_batches() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        cube_goal_success(torch.empty((0, 7)), torch.empty((0, 7)), delta=0.05)
