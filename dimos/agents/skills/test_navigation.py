# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain_core.messages import HumanMessage
import pytest

from dimos.agents.skills.navigation import NavigationSkillContainer
from dimos.core.module import Module
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs import PoseStamped, Quaternion, Vector3
from dimos.msgs.sensor_msgs import Image
from dimos.navigation.base import NavigationState


class FakeCamera(Module):
    color_image: Out[Image]


class FakeOdom(Module):
    odom: Out[PoseStamped]


class MockedStopNavSkill(NavigationSkillContainer):
    rpc_calls: list[str] = []

    def __init__(self):
        Module.__init__(self)
        self._skill_started = True

    def _cancel_goal_and_stop(self):
        pass


class MockedExploreNavSkill(NavigationSkillContainer):
    rpc_calls: list[str] = []

    def __init__(self):
        Module.__init__(self)
        self._skill_started = True

    def _start_exploration(self, timeout):
        return "Exploration completed successfuly"

    def _cancel_goal_and_stop(self):
        pass


class MockedSemanticNavSkill(NavigationSkillContainer):
    rpc_calls: list[str] = []

    def __init__(self):
        Module.__init__(self)
        self._skill_started = True

    def _navigate_by_tagged_location(self, query):
        return None

    def _navigate_to_object(self, query):
        return None

    def _navigate_using_semantic_map(self, query):
        return f"Successfuly arrived at '{query}'"


class MockedStairNavSkill(NavigationSkillContainer):
    rpc_calls: list[str] = []

    def __init__(self) -> None:
        Module.__init__(self)
        self._skill_started = True
        self._latest_odom = PoseStamped(position=Vector3(), orientation=Quaternion())
        self.cancel_called = False
        self.set_goal_calls: list[PoseStamped] = []

    def _cancel_goal_and_stop(self) -> None:
        self.cancel_called = True


def test_climb_stairs_3d_requires_odom() -> None:
    skill = MockedStairNavSkill()
    skill._latest_odom = None

    result = skill.climb_stairs_3d(goal_x=1.0, goal_y=1.0, total_height=0.3, steps=3)

    assert result == "No odometry data received yet, cannot start 3D stair navigation."


def test_climb_stairs_3d_rejects_non_positive_timeout() -> None:
    skill = MockedStairNavSkill()

    result = skill.climb_stairs_3d(
        goal_x=1.0,
        goal_y=1.0,
        total_height=0.3,
        steps=3,
        timeout_per_step=0.0,
    )

    assert result == "timeout_per_step must be greater than 0."


def test_climb_stairs_3d_succeeds_with_all_waypoints_reached() -> None:
    skill = MockedStairNavSkill()

    def set_goal_rpc(goal: PoseStamped) -> None:
        skill.set_goal_calls.append(goal)

    skill.get_rpc_calls = lambda *args: (
        set_goal_rpc,
        lambda: NavigationState.IDLE,
        lambda: True,
    )

    result = skill.climb_stairs_3d(goal_x=2.0, goal_y=1.0, total_height=0.6, steps=3)

    assert "Completed stair climbing plan to (2.00, 1.00)" in result
    assert len(skill.set_goal_calls) == 3
    assert skill.set_goal_calls[-1].position.z == pytest.approx(0.6)


def test_climb_stairs_3d_times_out_and_cancels(monkeypatch: pytest.MonkeyPatch) -> None:
    skill = MockedStairNavSkill()
    skill.get_rpc_calls = lambda *args: (
        lambda _goal: None,
        lambda: NavigationState.ACTIVE,
        lambda: False,
    )

    timestamps = iter([0.0, 0.5, 1.1])
    monkeypatch.setattr("dimos.agents.skills.navigation.time.time", lambda: next(timestamps))
    monkeypatch.setattr("dimos.agents.skills.navigation.time.sleep", lambda _s: None)

    result = skill.climb_stairs_3d(
        goal_x=1.0,
        goal_y=0.0,
        total_height=0.4,
        steps=1,
        timeout_per_step=1.0,
    )

    assert "timed out at waypoint 1/1 after 1.0s" in result
    assert skill.cancel_called


@pytest.mark.slow
def test_stop_movement(agent_setup) -> None:
    history = agent_setup(
        blueprints=[
            FakeCamera.blueprint(),
            FakeOdom.blueprint(),
            MockedStopNavSkill.blueprint(),
        ],
        messages=[HumanMessage("Stop moving. Use the stop_movement tool.")],
    )

    assert "stopped" in history[-1].content.lower()


@pytest.mark.slow
def test_start_exploration(agent_setup) -> None:
    history = agent_setup(
        blueprints=[
            FakeCamera.blueprint(),
            FakeOdom.blueprint(),
            MockedExploreNavSkill.blueprint(),
        ],
        messages=[
            HumanMessage("Take a look around for 10 seconds. Use the start_exploration tool.")
        ],
    )

    assert "explor" in history[-1].content.lower()


@pytest.mark.slow
def test_go_to_semantic_location(agent_setup) -> None:
    history = agent_setup(
        blueprints=[
            FakeCamera.blueprint(),
            FakeOdom.blueprint(),
            MockedSemanticNavSkill.blueprint(),
        ],
        messages=[HumanMessage("Go to the bookshelf. Use the navigate_with_text tool.")],
    )

    assert "success" in history[-1].content.lower()
