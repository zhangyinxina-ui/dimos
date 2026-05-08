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

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.global_config import GlobalConfig, global_config
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs import PoseStamped
from dimos.msgs.nav_msgs import OccupancyGrid
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

try:
    from dimos_lcm.std_msgs import String as LCMString
except ImportError:
    LCMString = str  # type: ignore[assignment, misc]


@dataclass
class PatrolMapConfig(ModuleConfig):
    map_save_dir: str = "~/.dimos/maps"
    map_name: str = "default"
    auto_save_interval: float = 60.0
    min_coverage_percent: float = 80.0
    patrol_inflation_radius: float = 0.3


class PatrolMapManager(Module):
    """Manages patrol map lifecycle: save, load, coverage tracking, and auto-save.

    Subscribes to global_costmap, applies safety inflation for patrol navigation,
    tracks coverage statistics, and periodically auto-saves maps to disk.
    """

    default_config = PatrolMapConfig
    config: PatrolMapConfig

    # Inputs
    slam_pose: In[PoseStamped]
    global_costmap: In[OccupancyGrid]
    global_map: In[PointCloud2]

    # Outputs
    patrol_costmap: Out[OccupancyGrid]
    map_metadata: Out[LCMString]

    def __init__(self, cfg: GlobalConfig = global_config, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._global_config = cfg

        self._current_costmap: OccupancyGrid | None = None
        self._current_pose: PoseStamped | None = None
        self._coverage_stats: dict[str, float] = {}
        self._map_changed: bool = False
        self._auto_save_timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._started: bool = False

    # -- Lifecycle --

    @rpc
    def start(self) -> None:
        super().start()
        self._started = True

        # Subscribe to costmap input
        self._disposables.add(
            Disposable(self.global_costmap.subscribe(self._on_costmap))
        )

        # Subscribe to pose input
        self._disposables.add(
            Disposable(self.slam_pose.subscribe(self._on_pose))
        )

        # Optionally load previously saved map on startup
        save_dir = self._resolve_map_dir(self.config.map_name)
        if save_dir.exists():
            self.load_map(self.config.map_name)
            logger.info(f"Loaded existing map '{self.config.map_name}' on startup")

        # Start auto-save timer
        self._schedule_auto_save()

    @rpc
    def stop(self) -> None:
        self._started = False
        if self._auto_save_timer is not None:
            self._auto_save_timer.cancel()
            self._auto_save_timer = None
        super().stop()

    # -- Input handlers --

    def _on_costmap(self, costmap: OccupancyGrid) -> None:
        with self._lock:
            self._current_costmap = costmap
            self._map_changed = True
            self._coverage_stats = self._compute_coverage(costmap.grid)

        # Apply safety inflation and publish patrol costmap
        inflated = self._inflate_for_patrol(costmap)
        self.patrol_costmap.publish(inflated)

        # Publish metadata
        metadata_str = self._build_metadata_json()
        self.map_metadata.publish(metadata_str)

    def _on_pose(self, pose: PoseStamped) -> None:
        self._current_pose = pose

    # -- Inflation --

    def _inflate_for_patrol(self, costmap: OccupancyGrid) -> OccupancyGrid:
        """Apply additional safety inflation for patrol navigation."""
        from dimos.mapping.occupancy.inflation import simple_inflate

        try:
            return simple_inflate(costmap, self.config.patrol_inflation_radius)
        except Exception:
            logger.warning("Patrol inflation failed, using original costmap")
            return costmap

    # -- Coverage computation --

    def _compute_coverage(self, grid: np.ndarray) -> dict[str, float]:
        total = grid.size
        if total == 0:
            return {"free_pct": 0.0, "occupied_pct": 0.0, "unknown_pct": 100.0}
        free = np.sum(grid == 0)
        occupied = np.sum(grid >= 50)
        unknown = np.sum(grid == -1)
        return {
            "free_pct": float(free / total * 100),
            "occupied_pct": float(occupied / total * 100),
            "unknown_pct": float(unknown / total * 100),
        }

    # -- Metadata --

    def _build_metadata_json(self) -> str:
        meta: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coverage": self._coverage_stats,
            "map_complete": self.is_map_complete(),
        }
        if self._current_costmap is not None:
            meta["width"] = self._current_costmap.width
            meta["height"] = self._current_costmap.height
            meta["resolution"] = self._current_costmap.resolution
        return json.dumps(meta)

    # -- Auto-save --

    def _schedule_auto_save(self) -> None:
        if not self._started:
            return
        self._auto_save_timer = threading.Timer(
            self.config.auto_save_interval, self._auto_save_tick
        )
        self._auto_save_timer.daemon = True
        self._auto_save_timer.start()

    def _auto_save_tick(self) -> None:
        if not self._started:
            return
        with self._lock:
            changed = self._map_changed
            self._map_changed = False
        if changed:
            try:
                self.save_map(self.config.map_name)
                logger.info(f"Auto-saved map '{self.config.map_name}'")
            except Exception:
                logger.error("Auto-save failed", exc_info=True)
        self._schedule_auto_save()

    # -- Map directory helpers --

    def _resolve_base_dir(self) -> Path:
        return Path(os.path.expanduser(self.config.map_save_dir))

    def _resolve_map_dir(self, name: str) -> Path:
        return self._resolve_base_dir() / name

    # -- RPC methods --

    @rpc
    def save_map(self, name: str) -> bool:
        """Save current costmap to disk under the given name."""
        with self._lock:
            costmap = self._current_costmap
            stats = dict(self._coverage_stats)

        if costmap is None:
            logger.warning("No costmap data to save")
            return False

        map_dir = self._resolve_map_dir(name)
        map_dir.mkdir(parents=True, exist_ok=True)

        # Save costmap numpy array
        np.save(str(map_dir / "costmap.npy"), costmap.grid)

        # Save metadata
        metadata: dict[str, Any] = {
            "width": costmap.width,
            "height": costmap.height,
            "resolution": costmap.resolution,
            "frame_id": costmap.frame_id,
            "origin": {
                "x": costmap.origin.position.x,
                "y": costmap.origin.position.y,
                "z": costmap.origin.position.z,
            },
            "coverage": stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "map_complete": self.is_map_complete(),
        }
        with open(map_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Optionally save pointcloud if open3d is available
        self._save_pointcloud_if_available(map_dir)

        logger.info(f"Saved map '{name}' to {map_dir}")
        return True

    def _save_pointcloud_if_available(self, map_dir: Path) -> None:
        """Attempt to save pointcloud.ply if open3d is available."""
        try:
            import open3d as o3d  # type: ignore[import-untyped]

            # If we have global_map input data, try to save it
            # For now, skip if no pointcloud data available
            pass
        except ImportError:
            pass

    @rpc
    def load_map(self, name: str) -> bool:
        """Load a previously saved map from disk."""
        map_dir = self._resolve_map_dir(name)
        if not map_dir.exists():
            logger.warning(f"Map directory not found: {map_dir}")
            return False

        costmap_path = map_dir / "costmap.npy"
        metadata_path = map_dir / "metadata.json"

        if not costmap_path.exists() or not metadata_path.exists():
            logger.warning(f"Map files missing in {map_dir}")
            return False

        try:
            grid = np.load(str(costmap_path))
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            from dimos.msgs.geometry_msgs import Pose, Vector3

            origin = Pose(
                position=Vector3(
                    metadata["origin"]["x"],
                    metadata["origin"]["y"],
                    metadata["origin"]["z"],
                )
            )

            costmap = OccupancyGrid(
                grid=grid,
                resolution=metadata["resolution"],
                origin=origin,
                frame_id=metadata.get("frame_id", "world"),
            )

            with self._lock:
                self._current_costmap = costmap
                self._coverage_stats = self._compute_coverage(grid)
                self._map_changed = True

            # Publish the loaded costmap
            self.patrol_costmap.publish(costmap)
            self.map_metadata.publish(self._build_metadata_json())

            logger.info(f"Loaded map '{name}' from {map_dir}")
            return True
        except Exception:
            logger.error(f"Failed to load map '{name}'", exc_info=True)
            return False

    @rpc
    def list_maps(self) -> list[str]:
        """List all saved maps."""
        base_dir = self._resolve_base_dir()
        if not base_dir.exists():
            return []
        return sorted(
            d.name
            for d in base_dir.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        )

    @rpc
    def get_map_stats(self) -> dict:
        """Return coverage stats, dimensions, and age."""
        with self._lock:
            stats = dict(self._coverage_stats)
            costmap = self._current_costmap

        result: dict[str, Any] = {"coverage": stats}

        if costmap is not None:
            result["width"] = costmap.width
            result["height"] = costmap.height
            result["resolution"] = costmap.resolution
            result["total_cells"] = costmap.total_cells

        # Check saved map age
        map_dir = self._resolve_map_dir(self.config.map_name)
        meta_path = map_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                result["saved_timestamp"] = meta.get("timestamp", "unknown")
            except Exception:
                pass

        result["map_complete"] = self.is_map_complete()
        return result

    @rpc
    def is_map_complete(self) -> bool:
        """Check if map coverage meets the minimum threshold."""
        with self._lock:
            stats = self._coverage_stats
        # Map is "complete" when unknown is low enough (i.e., enough is explored)
        explored_pct = stats.get("free_pct", 0.0) + stats.get("occupied_pct", 0.0)
        return explored_pct >= self.config.min_coverage_percent

    # -- Skills --

    @skill
    def save_current_map(self, name: str) -> str:
        """Save the current map with a given name for later use.

        Args:
            name (str): Name to save the map under

        Returns:
            str: Outcome of the save operation
        """
        with self._lock:
            costmap = self._current_costmap
            stats = dict(self._coverage_stats)

        if costmap is None:
            return "No map data available to save."

        success = self.save_map(name)
        if not success:
            return f"Failed to save map '{name}'."

        free_area_m2 = 0.0
        if costmap.total_cells > 0:
            cell_area = costmap.resolution ** 2
            free_cells = int(stats.get("free_pct", 0) / 100 * costmap.total_cells)
            free_area_m2 = free_cells * cell_area

        coverage_pct = stats.get("free_pct", 0.0) + stats.get("occupied_pct", 0.0)
        return f"Saved map '{name}' ({free_area_m2:.1f} m² explored, {coverage_pct:.0f}% coverage)"

    @skill
    def load_saved_map(self, name: str) -> str:
        """Load a previously saved map for patrol navigation.

        Args:
            name (str): Name of the map to load

        Returns:
            str: Outcome of the load operation
        """
        map_dir = self._resolve_map_dir(name)
        meta_path = map_dir / "metadata.json"

        if not map_dir.exists() or not meta_path.exists():
            return f"Map '{name}' not found."

        saved_date = "unknown"
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            ts = meta.get("timestamp", "")
            if ts:
                saved_date = ts[:10]
        except Exception:
            pass

        success = self.load_map(name)
        if not success:
            return f"Failed to load map '{name}'."

        return f"Loaded map '{name}' from {saved_date}"

    @skill
    def check_map_coverage(self) -> str:
        """Check how much of the environment has been mapped.

        Returns:
            str: Map coverage statistics
        """
        with self._lock:
            stats = dict(self._coverage_stats)

        if not stats:
            return "No map data available yet."

        free = stats.get("free_pct", 0.0)
        occ = stats.get("occupied_pct", 0.0)
        unk = stats.get("unknown_pct", 0.0)

        explored = free + occ
        if explored >= 90:
            quality = "excellent"
        elif explored >= 70:
            quality = "good"
        elif explored >= 50:
            quality = "moderate"
        else:
            quality = "poor"

        return (
            f"Map coverage: {free:.1f}% free, {occ:.1f}% obstacles, "
            f"{unk:.1f}% unknown. Map quality: {quality}."
        )

    @skill
    def list_available_maps(self) -> str:
        """List all saved maps available for loading.

        Returns:
            str: List of available maps with metadata
        """
        maps = self.list_maps()
        if not maps:
            return "No saved maps available."

        entries: list[str] = []
        for name in maps:
            meta_path = self._resolve_map_dir(name) / "metadata.json"
            saved_date = "unknown date"
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                ts = meta.get("timestamp", "")
                if ts:
                    saved_date = ts[:10]
            except Exception:
                pass
            entries.append(f"{name} (saved {saved_date})")

        return f"Available maps: {', '.join(entries)}"


patrol_map_manager = PatrolMapManager.blueprint

__all__ = ["PatrolMapManager", "PatrolMapConfig", "patrol_map_manager"]
