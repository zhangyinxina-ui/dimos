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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from dimos.msgs.geometry_msgs import PoseStamped


@dataclass
class PoseGraphNode:
    """A node in the SLAM pose graph, representing a keyframe."""

    index: int
    pose: PoseStamped  # SE3 pose in map frame
    scan: Any  # open3d.geometry.PointCloud (not serializable, stored in-process)

    def pose_matrix(self) -> np.ndarray:
        """Convert the node's pose to a 4x4 SE3 matrix."""
        return pose_stamped_to_matrix(self.pose)


@dataclass
class ScanMatchResult:
    """Result of an ICP scan matching operation."""

    fitness: float  # ICP fitness score (0-1, higher is better)
    rmse: float  # ICP RMSE (lower is better)
    transformation: np.ndarray  # 4x4 SE3 transformation matrix


def pose_stamped_to_matrix(pose: PoseStamped) -> np.ndarray:
    """Convert a PoseStamped to a 4x4 homogeneous transformation matrix."""
    q = pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w

    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def matrix_to_pose_stamped(matrix: np.ndarray, frame_id: str = "map", ts: float = 0.0) -> PoseStamped:
    """Convert a 4x4 homogeneous transformation matrix to a PoseStamped."""
    from dimos.msgs.geometry_msgs import PoseStamped, Quaternion, Vector3

    rotation = matrix[:3, :3]
    tx, ty, tz = matrix[0, 3], matrix[1, 3], matrix[2, 3]

    # Convert rotation matrix to quaternion using scipy
    from scipy.spatial.transform import Rotation

    quat = Rotation.from_matrix(rotation).as_quat()  # [x, y, z, w]

    return PoseStamped(
        position=Vector3(tx, ty, tz),
        orientation=Quaternion(quat[0], quat[1], quat[2], quat[3]),
        frame_id=frame_id,
        ts=ts,
    )
