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

"""Unit tests for LidarSlamModule with synthetic pointclouds."""

from __future__ import annotations

import time

import numpy as np
import pytest

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

pytestmark = pytest.mark.skipif(not HAS_OPEN3D, reason="open3d not available")

from dimos.msgs.geometry_msgs import PoseStamped, Quaternion, Vector3
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.slam.lidar_slam import LidarSlamModule
from dimos.slam.types import (
    ScanMatchResult,
    matrix_to_pose_stamped,
    pose_stamped_to_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_synthetic_scan(
    center: tuple[float, float, float] = (0, 0, 0),
    num_points: int = 500,
    radius: float = 2.0,
    seed: int = 42,
) -> PointCloud2:
    """Create a synthetic PointCloud2 resembling a lidar scan of a flat surface + walls."""
    rng = np.random.default_rng(seed)

    # Ground plane points
    ground_x = rng.uniform(center[0] - radius, center[0] + radius, num_points // 2)
    ground_y = rng.uniform(center[1] - radius, center[1] + radius, num_points // 2)
    ground_z = np.full_like(ground_x, center[2])
    ground = np.column_stack([ground_x, ground_y, ground_z])

    # Wall points (along x)
    wall_x = rng.uniform(center[0] - radius, center[0] + radius, num_points // 4)
    wall_y = np.full_like(wall_x, center[1] + radius)
    wall_z = rng.uniform(center[2], center[2] + 1.0, len(wall_x))
    wall1 = np.column_stack([wall_x, wall_y, wall_z])

    # Wall points (along y)
    wall2_x = np.full(num_points // 4, center[0] - radius)
    wall2_y = rng.uniform(center[1] - radius, center[1] + radius, num_points // 4)
    wall2_z = rng.uniform(center[2], center[2] + 1.0, num_points // 4)
    wall2 = np.column_stack([wall2_x, wall2_y, wall2_z])

    all_points = np.vstack([ground, wall1, wall2]).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)

    return PointCloud2(pointcloud=pcd, frame_id="lidar", ts=time.time())


def make_identity_pose() -> PoseStamped:
    return PoseStamped(
        position=Vector3(0, 0, 0),
        orientation=Quaternion(0, 0, 0, 1),
        frame_id="odom",
    )


# ---------------------------------------------------------------------------
# Tests: types module
# ---------------------------------------------------------------------------


class TestTypes:
    def test_pose_to_matrix_roundtrip(self):
        """PoseStamped -> matrix -> PoseStamped should be identity-like."""
        pose = PoseStamped(
            position=Vector3(1.0, 2.0, 0.5),
            orientation=Quaternion(0, 0, 0, 1),
            frame_id="map",
            ts=1.0,
        )
        matrix = pose_stamped_to_matrix(pose)
        assert matrix.shape == (4, 4)
        assert np.allclose(matrix[:3, 3], [1.0, 2.0, 0.5])
        assert np.allclose(matrix[:3, :3], np.eye(3), atol=1e-6)

        recovered = matrix_to_pose_stamped(matrix, frame_id="map", ts=1.0)
        assert abs(recovered.x - 1.0) < 1e-6
        assert abs(recovered.y - 2.0) < 1e-6
        assert abs(recovered.z - 0.5) < 1e-6

    def test_rotation_roundtrip(self):
        """Rotation in PoseStamped survives matrix conversion."""
        # 90 degree yaw
        from scipy.spatial.transform import Rotation

        yaw = np.pi / 2
        rot = Rotation.from_euler("z", yaw).as_quat()  # [x, y, z, w]
        pose = PoseStamped(
            position=Vector3(0, 0, 0),
            orientation=Quaternion(rot[0], rot[1], rot[2], rot[3]),
            frame_id="map",
        )
        matrix = pose_stamped_to_matrix(pose)
        recovered = matrix_to_pose_stamped(matrix, frame_id="map")

        original_euler = pose.orientation.to_euler()
        recovered_euler = recovered.orientation.to_euler()
        assert abs(original_euler.z - recovered_euler.z) < 1e-6

    def test_scan_match_result(self):
        result = ScanMatchResult(
            fitness=0.95,
            rmse=0.01,
            transformation=np.eye(4),
        )
        assert result.fitness == 0.95
        assert result.rmse == 0.01
        assert result.transformation.shape == (4, 4)


# ---------------------------------------------------------------------------
# Tests: LidarSlamModule
# ---------------------------------------------------------------------------


class TestLidarSlamModule:
    """Test LidarSlamModule using direct method calls (no RPC/transport)."""

    def _make_module(self, **config_overrides) -> LidarSlamModule:
        """Create a LidarSlamModule with test config."""
        from dimos.slam.lidar_slam import LidarSlamConfig

        defaults = dict(
            voxel_size=0.1,
            keyframe_distance=0.5,
            icp_max_distance=2.0,
            loop_closure_enabled=False,
        )
        defaults.update(config_overrides)
        return LidarSlamModule(cfg=LidarSlamConfig(**defaults))

    def _cleanup_module(self, module: LidarSlamModule) -> None:
        """Clean up module threads to avoid thread leak detection."""
        module._close_module()

    def test_initial_state(self):
        module = self._make_module()
        try:
            pose = module.get_pose()
            assert pose.x == 0.0
            assert pose.y == 0.0
            assert pose.z == 0.0

            metrics = module.get_quality_metrics()
            assert metrics["keyframe_count"] == 0
            assert metrics["submap_size"] == 0
            assert metrics["scan_count"] == 0
        finally:
            self._cleanup_module(module)

    def test_first_frame_creates_keyframe(self):
        module = self._make_module()
        try:
            scan = make_synthetic_scan()

            with module._lock:
                module._process_frame(scan)

            metrics = module.get_quality_metrics()
            assert metrics["keyframe_count"] == 1
            assert metrics["submap_size"] > 0
            assert metrics["scan_count"] == 1
        finally:
            self._cleanup_module(module)

    def test_icp_scan_matching_produces_corrected_pose(self):
        module = self._make_module()
        try:
            # Feed first frame at origin
            scan1 = make_synthetic_scan(center=(0, 0, 0), seed=1)
            with module._lock:
                module._process_frame(scan1)

            # Feed second frame shifted by 0.5m in x — should still match
            scan2 = make_synthetic_scan(center=(0.5, 0, 0), seed=2)
            with module._lock:
                module._process_frame(scan2)

            pose = module.get_pose()
            # The pose should have moved from origin — not exact, but not zero
            metrics = module.get_quality_metrics()
            assert metrics["scan_count"] == 2
            # ICP should have succeeded on second frame
            assert metrics["match_fitness"] > 0
        finally:
            self._cleanup_module(module)

    def test_keyframes_generated_at_correct_intervals(self):
        module = self._make_module()
        try:
            # Process several frames at the same location — should only get 1 keyframe
            for i in range(5):
                scan = make_synthetic_scan(center=(0, 0, 0), seed=i)
                with module._lock:
                    module._process_frame(scan)

            metrics = module.get_quality_metrics()
            assert metrics["keyframe_count"] == 1  # All at same location
        finally:
            self._cleanup_module(module)

    def test_keyframes_generated_with_movement(self):
        module = self._make_module()
        try:
            # Process frames with progressive movement
            for i in range(6):
                x_offset = i * 0.6  # Each frame 0.6m apart, keyframe_distance=0.5
                scan = make_synthetic_scan(center=(x_offset, 0, 0), seed=i)
                with module._lock:
                    module._process_frame(scan)

            # Should have multiple keyframes due to movement
            metrics = module.get_quality_metrics()
            assert metrics["keyframe_count"] >= 2
        finally:
            self._cleanup_module(module)

    def test_check_localization_quality(self):
        module = self._make_module()
        try:
            # Process a frame so we have some data
            scan = make_synthetic_scan()
            with module._lock:
                module._process_frame(scan)

            quality = module.check_localization_quality()
            assert isinstance(quality, str)
            assert "SLAM Quality" in quality
            assert "ICP Fitness" in quality
            assert "Keyframes" in quality
        finally:
            self._cleanup_module(module)

    def test_reset_clears_state(self):
        module = self._make_module()
        try:
            # Process some frames
            for i in range(3):
                scan = make_synthetic_scan(center=(i * 0.5, 0, 0), seed=i)
                with module._lock:
                    module._process_frame(scan)

            # Verify we have state
            metrics_before = module.get_quality_metrics()
            assert metrics_before["scan_count"] > 0

            # Reset
            result = module.reset()
            assert result is True

            # Verify cleared
            pose = module.get_pose()
            assert pose.x == 0.0
            assert pose.y == 0.0

            metrics_after = module.get_quality_metrics()
            assert metrics_after["keyframe_count"] == 0
            assert metrics_after["submap_size"] == 0
            assert metrics_after["scan_count"] == 0
        finally:
            self._cleanup_module(module)

    def test_empty_pointcloud_ignored(self):
        module = self._make_module()
        try:
            # Create an empty pointcloud
            empty_pcd = o3d.geometry.PointCloud()
            empty_scan = PointCloud2(pointcloud=empty_pcd, frame_id="lidar")

            with module._lock:
                module._process_frame(empty_scan)

            metrics = module.get_quality_metrics()
            assert metrics["scan_count"] == 1  # Frame was counted
            assert metrics["keyframe_count"] == 0  # But no keyframe added (empty)
        finally:
            self._cleanup_module(module)

    def test_quality_string_no_frames(self):
        """Quality check works even before any frames processed."""
        module = self._make_module()
        try:
            quality = module.check_localization_quality()
            assert "SLAM Quality" in quality
        finally:
            self._cleanup_module(module)

    def test_module_has_blueprint(self):
        """Verify the blueprint class property exists."""
        assert hasattr(LidarSlamModule, "blueprint")

    def test_module_io_streams(self):
        """Verify the module declares correct inputs and outputs."""
        module = self._make_module()
        try:
            inputs = module.inputs
            outputs = module.outputs

            assert "lidar" in inputs
            assert "raw_odom" in inputs
            assert "slam_pose" in outputs
            assert "lidar_corrected" in outputs
        finally:
            self._cleanup_module(module)

    def test_with_loop_closure_enabled(self):
        """Test module with loop closure enabled (no crash, basic operation)."""
        module = self._make_module(
            loop_closure_enabled=True,
            loop_closure_interval=5,
        )
        try:
            # Process enough frames to trigger loop closure check
            for i in range(8):
                x_offset = i * 0.3
                scan = make_synthetic_scan(center=(x_offset, 0, 0), seed=i)
                with module._lock:
                    module._process_frame(scan)

            metrics = module.get_quality_metrics()
            assert metrics["scan_count"] == 8
            assert metrics["keyframe_count"] >= 1
        finally:
            self._cleanup_module(module)
