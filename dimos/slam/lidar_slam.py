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
import threading
import time

import numpy as np
from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs import PoseStamped, Quaternion, Vector3
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.slam.types import (
    PoseGraphNode,
    ScanMatchResult,
    matrix_to_pose_stamped,
    pose_stamped_to_matrix,
)
from dimos.utils.logging_config import setup_logger

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

logger = setup_logger()


@dataclass
class LidarSlamConfig(ModuleConfig):
    """Configuration for LidarSlamModule."""

    voxel_size: float = 0.1
    keyframe_distance: float = 0.5  # meters between keyframes
    icp_max_distance: float = 1.0  # max correspondence distance for ICP
    loop_closure_enabled: bool = True
    loop_closure_interval: int = 30  # scans between loop closure attempts
    loop_closure_search_radius: float = 5.0  # meters to search for loop closures
    loop_closure_fitness_threshold: float = 0.3  # min fitness for a valid loop closure


_IDENTITY_POSE = lambda ts=0.0: PoseStamped(
    position=Vector3(0, 0, 0),
    orientation=Quaternion(0, 0, 0, 1),
    frame_id="map",
    ts=ts,
)


class LidarSlamModule(Module):
    """Incremental LiDAR SLAM with ICP scan matching and optional loop closure.

    Subscribes to raw lidar pointclouds and odometry, produces corrected
    SLAM poses and transformed pointclouds.
    """

    default_config = LidarSlamConfig
    config: LidarSlamConfig

    # Inputs
    lidar: In[PointCloud2]
    raw_odom: In[PoseStamped]

    # Outputs
    slam_pose: Out[PoseStamped]
    lidar_corrected: Out[PointCloud2]

    def __init__(self, cfg=None, **kwargs) -> None:
        if not HAS_OPEN3D:
            raise ImportError(
                "open3d is required for LidarSlamModule. "
                "Install with: pip install open3d"
            )
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._current_pose: PoseStamped = _IDENTITY_POSE()
        self._submap: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        self._pose_graph_nodes: list[PoseGraphNode] = []
        self._scan_count: int = 0
        self._last_match_fitness: float = 1.0
        self._last_match_rmse: float = 0.0
        self._cumulative_drift_estimate: float = 0.0
        self._last_odom: PoseStamped | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(Disposable(self.lidar.subscribe(self._on_lidar_frame)))
        self._disposables.add(Disposable(self.raw_odom.subscribe(self._on_odom)))
        logger.info("LidarSlamModule started")

    @rpc
    def stop(self) -> None:
        self._submap = o3d.geometry.PointCloud()
        self._pose_graph_nodes.clear()
        self._last_odom = None
        logger.info("LidarSlamModule stopped")
        super().stop()

    def _on_odom(self, odom: PoseStamped) -> None:
        with self._lock:
            self._last_odom = odom

    def _on_lidar_frame(self, frame: PointCloud2) -> None:
        with self._lock:
            self._process_frame(frame)

    def _process_frame(self, frame: PointCloud2) -> None:
        self._scan_count += 1

        pcd = frame.pointcloud
        if pcd.is_empty():
            return

        downsampled = pcd.voxel_down_sample(self.config.voxel_size)

        if self._submap.is_empty():
            # First frame: initialize at origin
            self._submap += downsampled
            self._current_pose = _IDENTITY_POSE(ts=frame.ts or time.time())
            self._add_keyframe(downsampled)
            self._publish_outputs(frame, downsampled, np.eye(4))
            return

        # ICP scan matching against the accumulated submap
        initial_guess = pose_stamped_to_matrix(self._current_pose)
        result = self._run_icp(downsampled, self._submap, initial_guess)

        if result is not None:
            self._last_match_fitness = result.fitness
            self._last_match_rmse = result.rmse
            self._current_pose = matrix_to_pose_stamped(
                result.transformation,
                frame_id="map",
                ts=frame.ts or time.time(),
            )
            # Track drift estimate (difference between ICP and initial guess)
            diff = np.linalg.norm(result.transformation[:3, 3] - initial_guess[:3, 3])
            self._cumulative_drift_estimate += diff * 0.1
        elif self._last_odom is not None:
            # ICP failed -- fall back to odometry
            self._current_pose = PoseStamped(
                position=self._last_odom.position,
                orientation=self._last_odom.orientation,
                frame_id="map",
                ts=frame.ts or time.time(),
            )

        # Keyframe insertion
        if self._should_add_keyframe():
            self._add_keyframe(downsampled)
            self._submap += downsampled.transform(
                pose_stamped_to_matrix(self._current_pose)
            )

        # Loop closure
        if (
            self.config.loop_closure_enabled
            and self._scan_count % self.config.loop_closure_interval == 0
        ):
            self._attempt_loop_closure(downsampled)

        # Publish outputs
        self._publish_outputs(frame, downsampled, pose_stamped_to_matrix(self._current_pose))

    def _run_icp(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        initial_guess: np.ndarray,
    ) -> ScanMatchResult | None:
        """Run ICP registration between source and target."""
        try:
            result = o3d.pipelines.registration.registration_icp(
                source,
                target,
                self.config.icp_max_distance,
                initial_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
            return ScanMatchResult(
                fitness=result.fitness,
                rmse=result.inlier_rmse,
                transformation=np.asarray(result.transformation),
            )
        except Exception as e:
            logger.warning(f"ICP failed: {e}")
            return None

    def _should_add_keyframe(self) -> bool:
        """Check if current pose is far enough from last keyframe."""
        if not self._pose_graph_nodes:
            return True

        last = self._pose_graph_nodes[-1].pose
        last_pos = np.array([last.x, last.y, last.z])
        curr_pos = np.array([self._current_pose.x, self._current_pose.y, self._current_pose.z])
        return float(np.linalg.norm(curr_pos - last_pos)) >= self.config.keyframe_distance

    def _add_keyframe(self, scan: o3d.geometry.PointCloud) -> None:
        """Add a keyframe to the pose graph."""
        self._pose_graph_nodes.append(
            PoseGraphNode(
                index=len(self._pose_graph_nodes),
                pose=PoseStamped(
                    position=self._current_pose.position,
                    orientation=self._current_pose.orientation,
                    frame_id="map",
                    ts=self._current_pose.ts,
                ),
                scan=scan,
            )
        )

    def _attempt_loop_closure(self, current_scan: o3d.geometry.PointCloud) -> None:
        """Try to detect and close loops by matching against older keyframes."""
        if len(self._pose_graph_nodes) < 3:
            return

        min_idx = max(0, len(self._pose_graph_nodes) - 50)
        current_pos = np.array([self._current_pose.x, self._current_pose.y, self._current_pose.z])

        for i in range(min_idx):
            node = self._pose_graph_nodes[i]
            node_pos = np.array([node.pose.x, node.pose.y, node.pose.z])
            distance = np.linalg.norm(current_pos - node_pos)

            if distance > self.config.loop_closure_search_radius:
                continue

            result = self._run_icp(current_scan, node.scan, np.eye(4))
            if result is None:
                continue

            if result.fitness > self.config.loop_closure_fitness_threshold:
                logger.info(
                    f"Loop closure detected with keyframe {i}: "
                    f"fitness={result.fitness:.3f}, rmse={result.rmse:.3f}"
                )
                self._apply_loop_correction(i, result.transformation)
                break

    def _apply_loop_correction(self, anchor_idx: int, correction: np.ndarray) -> None:
        """Apply a loop closure correction to the pose graph."""
        total = len(self._pose_graph_nodes) - anchor_idx
        if total <= 0:
            return

        for step, node_idx in enumerate(range(anchor_idx, len(self._pose_graph_nodes))):
            alpha = (step + 1) / total
            node = self._pose_graph_nodes[node_idx]
            old_matrix = pose_stamped_to_matrix(node.pose)

            # Interpolated correction
            interpolated = np.eye(4)
            interpolated[:3, :3] = correction[:3, :3] ** alpha
            interpolated[:3, 3] = correction[:3, 3] * alpha

            node.pose = matrix_to_pose_stamped(
                interpolated @ old_matrix, frame_id="map", ts=node.pose.ts
            )

        # Update current pose from the last corrected node
        if self._pose_graph_nodes:
            last = self._pose_graph_nodes[-1]
            self._current_pose = PoseStamped(
                position=last.pose.position,
                orientation=last.pose.orientation,
                frame_id="map",
                ts=self._current_pose.ts,
            )

    def _publish_outputs(
        self,
        original_frame: PointCloud2,
        corrected_scan: o3d.geometry.PointCloud,
        transform: np.ndarray,
    ) -> None:
        """Publish corrected pose and transformed pointcloud."""
        self.slam_pose.publish(self._current_pose)

        if not corrected_scan.is_empty():
            corrected_scan.transform(transform)
            self.lidar_corrected.publish(
                PointCloud2(
                    pointcloud=corrected_scan,
                    frame_id="map",
                    ts=original_frame.ts or time.time(),
                )
            )

    @rpc
    def get_pose(self) -> PoseStamped:
        """Return the current SLAM pose."""
        with self._lock:
            return self._current_pose

    @rpc
    def get_quality_metrics(self) -> dict:
        """Return SLAM quality metrics.

        Returns a dict with:
        - match_fitness: last ICP fitness score (0-1)
        - match_rmse: last ICP RMSE
        - submap_size: number of points in the submap
        - keyframe_count: number of keyframes
        - drift_estimate: estimated cumulative drift in meters
        - scan_count: total scans processed
        """
        with self._lock:
            return {
                "match_fitness": self._last_match_fitness,
                "match_rmse": self._last_match_rmse,
                "submap_size": len(self._submap.points),
                "keyframe_count": len(self._pose_graph_nodes),
                "drift_estimate": self._cumulative_drift_estimate,
                "scan_count": self._scan_count,
            }

    @rpc
    def reset(self) -> bool:
        """Clear all SLAM state and reset to origin."""
        with self._lock:
            self._current_pose = _IDENTITY_POSE()
            self._submap = o3d.geometry.PointCloud()
            self._pose_graph_nodes.clear()
            self._scan_count = 0
            self._last_match_fitness = 1.0
            self._last_match_rmse = 0.0
            self._cumulative_drift_estimate = 0.0
            self._last_odom = None
            logger.info("LidarSlamModule reset")
            return True

    @skill
    def check_localization_quality(self) -> str:
        """Assess the current SLAM localization quality.

        Returns a human-readable assessment of SLAM quality including
        match scores, submap size, keyframe count, and drift estimate.
        """
        metrics = self.get_quality_metrics()
        fitness = metrics["match_fitness"]
        rmse = metrics["match_rmse"]

        if fitness > 0.8 and rmse < 0.1:
            quality = "EXCELLENT"
        elif fitness > 0.6 and rmse < 0.3:
            quality = "GOOD"
        elif fitness > 0.3:
            quality = "FAIR"
        else:
            quality = "POOR"

        return (
            f"SLAM Quality: {quality}\n"
            f"  ICP Fitness: {fitness:.3f} | RMSE: {rmse:.4f}\n"
            f"  Keyframes: {metrics['keyframe_count']} | "
            f"Submap points: {metrics['submap_size']}\n"
            f"  Estimated drift: {metrics['drift_estimate']:.3f}m"
        )


lidar_slam = LidarSlamModule.blueprint
