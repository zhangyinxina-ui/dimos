# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dimos.msgs.geometry_msgs import Pose, Vector3
from dimos.msgs.nav_msgs import OccupancyGrid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_costmap(
    width: int = 20,
    height: int = 20,
    resolution: float = 0.05,
) -> OccupancyGrid:
    """Create a synthetic OccupancyGrid with known coverage."""
    rng = np.random.default_rng(42)
    grid = rng.choice(
        np.array([-1, 0, 100], dtype=np.int8),
        size=(height, width),
        p=[0.10, 0.80, 0.10],
    )
    return OccupancyGrid(
        grid=grid,
        resolution=resolution,
        origin=Pose(position=Vector3(1.0, 2.0, 0.0)),
        frame_id="map",
    )


def _make_manager(tmp_path: Path, **config_overrides):
    """Instantiate PatrolMapManager without starting its RPC / timer machinery."""
    from dimos.mapping.patrol_map_manager import PatrolMapConfig, PatrolMapManager

    cfg = PatrolMapConfig(
        map_save_dir=str(tmp_path),
        map_name="test_map",
        auto_save_interval=300.0,  # long interval so auto-save doesn't fire in tests
        min_coverage_percent=80.0,
    )
    for k, v in config_overrides.items():
        setattr(cfg, k, v)

    mgr = PatrolMapManager.__new__(PatrolMapManager)
    # Manually initialise attributes we need without calling full __init__
    mgr._current_costmap = None
    mgr._current_pose = None
    mgr._coverage_stats = {}
    mgr._map_changed = False
    mgr._auto_save_timer = None
    mgr._lock = __import__("threading").Lock()
    mgr._started = False
    mgr.config = cfg
    # Create mock publish methods
    mgr.patrol_costmap = MagicMock()
    mgr.map_metadata = MagicMock()
    mgr.slam_pose = MagicMock()
    mgr.global_costmap = MagicMock()
    mgr.global_map = MagicMock()
    return mgr


# ---------------------------------------------------------------------------
# Tests — Coverage computation
# ---------------------------------------------------------------------------

class TestCoverageComputation:
    def test_compute_coverage_basic(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        # 3x3 grid: 6 free (value 0), 2 occupied (>=50: 100 and 50), 1 unknown (-1)
        grid = np.array([
            [0, 0, 100],
            [-1, 0, 50],
            [0, 0, 0],
        ], dtype=np.int8)
        stats = mgr._compute_coverage(grid)
        assert abs(stats["free_pct"] - (6 / 9 * 100)) < 0.1  # 66.67%
        assert abs(stats["occupied_pct"] - (2 / 9 * 100)) < 0.1  # 22.22%
        assert abs(stats["unknown_pct"] - (1 / 9 * 100)) < 0.1  # 11.11%

    def test_compute_coverage_empty_grid(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        grid = np.array([], dtype=np.int8)
        stats = mgr._compute_coverage(grid)
        assert stats["free_pct"] == 0.0
        assert stats["occupied_pct"] == 0.0
        assert stats["unknown_pct"] == 100.0

    def test_compute_coverage_all_free(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        grid = np.zeros((10, 10), dtype=np.int8)
        stats = mgr._compute_coverage(grid)
        assert stats["free_pct"] == 100.0
        assert stats["occupied_pct"] == 0.0
        assert stats["unknown_pct"] == 0.0

    def test_compute_coverage_all_unknown(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        grid = np.full((5, 5), -1, dtype=np.int8)
        stats = mgr._compute_coverage(grid)
        assert stats["free_pct"] == 0.0
        assert stats["unknown_pct"] == 100.0


# ---------------------------------------------------------------------------
# Tests — Save / Load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        costmap = _make_costmap()

        # Inject costmap
        with mgr._lock:
            mgr._current_costmap = costmap
            mgr._coverage_stats = mgr._compute_coverage(costmap.grid)

        # Save
        assert mgr.save_map("warehouse_floor_1") is True

        # Verify files exist
        map_dir = tmp_path / "warehouse_floor_1"
        assert (map_dir / "costmap.npy").exists()
        assert (map_dir / "metadata.json").exists()

        # Verify metadata content
        with open(map_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["width"] == costmap.width
        assert meta["height"] == costmap.height
        assert abs(meta["resolution"] - costmap.resolution) < 1e-9
        assert "coverage" in meta
        assert "timestamp" in meta

        # Reset internal state
        with mgr._lock:
            mgr._current_costmap = None
            mgr._coverage_stats = {}

        # Load
        assert mgr.load_map("warehouse_floor_1") is True

        # Verify costmap restored
        with mgr._lock:
            loaded = mgr._current_costmap
        assert loaded is not None
        np.testing.assert_array_equal(loaded.grid, costmap.grid)
        assert loaded.width == costmap.width
        assert loaded.height == costmap.height

    def test_save_no_costmap_returns_false(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        assert mgr.save_map("empty_map") is False

    def test_load_nonexistent_returns_false(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        assert mgr.load_map("no_such_map") is False

    def test_load_missing_files_returns_false(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        # Create directory but without files
        (tmp_path / "incomplete_map").mkdir()
        assert mgr.load_map("incomplete_map") is False


# ---------------------------------------------------------------------------
# Tests — List maps
# ---------------------------------------------------------------------------

class TestListMaps:
    def test_list_maps_empty(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        assert mgr.list_maps() == []

    def test_list_maps_after_save(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        costmap = _make_costmap()
        with mgr._lock:
            mgr._current_costmap = costmap
            mgr._coverage_stats = mgr._compute_coverage(costmap.grid)

        mgr.save_map("map_a")
        mgr.save_map("map_b")

        maps = mgr.list_maps()
        assert sorted(maps) == ["map_a", "map_b"]

    def test_list_maps_ignores_incomplete_dirs(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        (tmp_path / "partial").mkdir()
        assert mgr.list_maps() == []


# ---------------------------------------------------------------------------
# Tests — Map stats
# ---------------------------------------------------------------------------

class TestMapStats:
    def test_get_map_stats_with_data(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        costmap = _make_costmap()
        with mgr._lock:
            mgr._current_costmap = costmap
            mgr._coverage_stats = mgr._compute_coverage(costmap.grid)

        stats = mgr.get_map_stats()
        assert "coverage" in stats
        assert stats["width"] == 20
        assert stats["height"] == 20
        assert "map_complete" in stats

    def test_get_map_stats_no_data(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        stats = mgr.get_map_stats()
        assert stats["coverage"] == {}
        assert stats["map_complete"] is False


# ---------------------------------------------------------------------------
# Tests — is_map_complete
# ---------------------------------------------------------------------------

class TestMapComplete:
    def test_map_complete_above_threshold(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        # 90% free, 5% occupied = 95% explored > 80% threshold
        with mgr._lock:
            mgr._coverage_stats = {"free_pct": 90.0, "occupied_pct": 5.0, "unknown_pct": 5.0}
        assert mgr.is_map_complete() is True

    def test_map_not_complete_below_threshold(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        # 50% free, 10% occupied = 60% explored < 80% threshold
        with mgr._lock:
            mgr._coverage_stats = {"free_pct": 50.0, "occupied_pct": 10.0, "unknown_pct": 40.0}
        assert mgr.is_map_complete() is False

    def test_map_complete_empty_stats(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        assert mgr.is_map_complete() is False


# ---------------------------------------------------------------------------
# Tests — Auto-save timer
# ---------------------------------------------------------------------------

class TestAutoSave:
    def test_auto_save_fires_when_map_changed(self, tmp_path: Path):
        mgr = _make_manager(tmp_path, auto_save_interval=0.1)
        mgr._started = True

        costmap = _make_costmap()
        with mgr._lock:
            mgr._current_costmap = costmap
            mgr._coverage_stats = mgr._compute_coverage(costmap.grid)
            mgr._map_changed = True

        # Patch _schedule_auto_save to prevent timer thread creation
        with patch.object(mgr, "_schedule_auto_save"):
            mgr._auto_save_tick()

        # Should have saved
        map_dir = tmp_path / "test_map"
        assert (map_dir / "costmap.npy").exists()

        mgr._started = False

    def test_auto_save_skips_when_no_change(self, tmp_path: Path):
        mgr = _make_manager(tmp_path, auto_save_interval=0.1)
        mgr._started = True

        costmap = _make_costmap()
        with mgr._lock:
            mgr._current_costmap = costmap
            mgr._coverage_stats = mgr._compute_coverage(costmap.grid)
            mgr._map_changed = False

        # Patch _schedule_auto_save to prevent timer thread creation
        with patch.object(mgr, "_schedule_auto_save"):
            mgr._auto_save_tick()

        # Should NOT have saved
        map_dir = tmp_path / "test_map"
        assert not map_dir.exists()

        mgr._started = False


# ---------------------------------------------------------------------------
# Tests — Skills return valid strings
# ---------------------------------------------------------------------------

class TestSkills:
    def _setup_mgr_with_costmap(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        costmap = _make_costmap()
        with mgr._lock:
            mgr._current_costmap = costmap
            mgr._coverage_stats = mgr._compute_coverage(costmap.grid)
        return mgr

    def test_save_current_map_skill(self, tmp_path: Path):
        mgr = self._setup_mgr_with_costmap(tmp_path)
        result = mgr.save_current_map("warehouse_floor_1")
        assert isinstance(result, str)
        assert "warehouse_floor_1" in result
        assert "Saved" in result

    def test_save_current_map_skill_no_data(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        result = mgr.save_current_map("nowhere")
        assert isinstance(result, str)
        assert "No map data" in result

    def test_load_saved_map_skill(self, tmp_path: Path):
        mgr = self._setup_mgr_with_costmap(tmp_path)
        mgr.save_map("office_2f")

        result = mgr.load_saved_map("office_2f")
        assert isinstance(result, str)
        assert "office_2f" in result
        assert "Loaded" in result

    def test_load_saved_map_skill_not_found(self, tmp_path: Path):
        mgr = self._setup_mgr_with_costmap(tmp_path)
        result = mgr.load_saved_map("nonexistent")
        assert isinstance(result, str)
        assert "not found" in result

    def test_check_map_coverage_skill(self, tmp_path: Path):
        mgr = self._setup_mgr_with_costmap(tmp_path)
        result = mgr.check_map_coverage()
        assert isinstance(result, str)
        assert "Map coverage:" in result
        assert "free" in result.lower()
        assert "obstacles" in result.lower()
        assert "unknown" in result.lower()
        assert "Map quality:" in result

    def test_check_map_coverage_skill_no_data(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        result = mgr.check_map_coverage()
        assert isinstance(result, str)
        assert "No map data" in result

    def test_list_available_maps_skill_empty(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        result = mgr.list_available_maps()
        assert isinstance(result, str)
        assert "No saved maps" in result

    def test_list_available_maps_skill_with_maps(self, tmp_path: Path):
        mgr = self._setup_mgr_with_costmap(tmp_path)
        mgr.save_map("warehouse_floor_1")
        mgr.save_map("office_2f")

        result = mgr.list_available_maps()
        assert isinstance(result, str)
        assert "Available maps:" in result
        assert "warehouse_floor_1" in result
        assert "office_2f" in result


# ---------------------------------------------------------------------------
# Tests — On costmap handler
# ---------------------------------------------------------------------------

class TestOnCostmapHandler:
    def test_on_costmap_updates_state(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        costmap = _make_costmap()

        mgr._on_costmap(costmap)

        assert mgr._current_costmap is not None
        assert mgr._map_changed is True
        assert len(mgr._coverage_stats) == 3
        mgr.patrol_costmap.publish.assert_called_once()
        mgr.map_metadata.publish.assert_called_once()

    def test_on_costmap_publishes_valid_metadata(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        costmap = _make_costmap()

        mgr._on_costmap(costmap)

        call_args = mgr.map_metadata.publish.call_args[0][0]
        meta = json.loads(call_args)
        assert "timestamp" in meta
        assert "coverage" in meta
        assert "map_complete" in meta


# ---------------------------------------------------------------------------
# Tests — Module starts and stops cleanly (without full RPC)
# ---------------------------------------------------------------------------

class TestModuleLifecycle:
    def test_manager_can_be_instantiated(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)
        assert mgr is not None
        assert mgr._current_costmap is None

    def test_coverage_quality_levels(self, tmp_path: Path):
        mgr = _make_manager(tmp_path)

        # Excellent: >= 90% explored
        with mgr._lock:
            mgr._coverage_stats = {"free_pct": 85.0, "occupied_pct": 10.0, "unknown_pct": 5.0}
        result = mgr.check_map_coverage()
        assert "excellent" in result

        # Good: 70-90% explored
        with mgr._lock:
            mgr._coverage_stats = {"free_pct": 65.0, "occupied_pct": 10.0, "unknown_pct": 25.0}
        result = mgr.check_map_coverage()
        assert "good" in result

        # Moderate: 50-70% explored
        with mgr._lock:
            mgr._coverage_stats = {"free_pct": 45.0, "occupied_pct": 10.0, "unknown_pct": 45.0}
        result = mgr.check_map_coverage()
        assert "moderate" in result

        # Poor: < 50% explored
        with mgr._lock:
            mgr._coverage_stats = {"free_pct": 20.0, "occupied_pct": 10.0, "unknown_pct": 70.0}
        result = mgr.check_map_coverage()
        assert "poor" in result
