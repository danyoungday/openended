from __future__ import annotations

from collections import deque
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


class DoorKeyRewardWrapper(gym.Wrapper):
    """
    Reward shaping wrapper for MiniGrid DoorKey.

    Implements:
      1) Keep env terminal reward for reaching goal
      2) Milestones:
         - first time picking up the key: +0.2
         - first time opening/unlocking the door: +0.3
      3) Potential-based progress shaping via shortest-path distance deltas:
         - no key:     alpha * (d_key(s)  - d_key(s'))
         - has key & door closed: alpha * (d_door(s) - d_door(s'))
         - door open:  alpha * (d_goal(s) - d_goal(s'))
      4) Per-step cost: -0.01

    Notes:
      - Distances are computed on the *full* grid (not partial obs).
      - Locked/closed door is treated as an obstacle unless allow_closed_door_in_key_distance=True
        (sometimes useful if you want d_key to ignore the locked door; for standard DoorKey it
         usually doesn't matter since key is on the start side).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        alpha: float = 0.02,
        step_cost: float = -0.01,
        key_milestone: float = 0.2,
        door_milestone: float = 0.3,
        allow_closed_door_in_key_distance: bool = False,
    ):
        super().__init__(env)
        self.alpha = float(alpha)
        self.step_cost = float(step_cost)
        self.key_milestone = float(key_milestone)
        self.door_milestone = float(door_milestone)
        self.allow_closed_door_in_key_distance = bool(
            allow_closed_door_in_key_distance
        )

        # Episode state
        self._got_key_once = False
        self._opened_door_once = False
        self._prev_has_key = False
        self._prev_door_open = False

        self._prev_d_key: int | None = None
        self._prev_d_door: int | None = None
        self._prev_d_goal: int | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)

        self._got_key_once = False
        self._opened_door_once = False

        self._prev_has_key = self._has_key()
        self._prev_door_open = self._door_open()

        # Initialize previous distances for delta shaping
        self._prev_d_key = self._dist_to("key", treat_door_as_open=self.allow_closed_door_in_key_distance)
        self._prev_d_door = self._dist_to("door", treat_door_as_open=False)
        self._prev_d_goal = self._dist_to("goal", treat_door_as_open=self._prev_door_open)

        return obs, info

    def step(
        self, action
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        # Snapshot pre-step state for milestone detection & progress deltas
        prev_has_key = self._has_key()
        prev_door_open = self._door_open()

        # Distances from s (pre-step)
        d_key_s = self._dist_to(
            "key",
            treat_door_as_open=self.allow_closed_door_in_key_distance,
        )
        d_door_s = self._dist_to("door", treat_door_as_open=False)
        d_goal_s = self._dist_to("goal", treat_door_as_open=prev_door_open)

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # If environment terminated by reaching goal, keep its terminal reward
        # (we can still add step_cost/progress; but typically you'd keep shaping small anyway)
        shaped_reward = float(base_reward)

        # Always apply a small step cost (including terminal step)
        shaped_reward += self.step_cost

        # Post-step state
        has_key = self._has_key()
        door_open = self._door_open()

        # Milestone: first time picking up key
        if (not self._got_key_once) and (not prev_has_key) and has_key:
            shaped_reward += self.key_milestone
            self._got_key_once = True

        # Milestone: first time door becomes open (unlocked/opened)
        if (not self._opened_door_once) and (not prev_door_open) and door_open:
            shaped_reward += self.door_milestone
            self._opened_door_once = True

        # Distances from s' (post-step)
        d_key_sp = self._dist_to(
            "key",
            treat_door_as_open=self.allow_closed_door_in_key_distance,
        )
        d_door_sp = self._dist_to("door", treat_door_as_open=False)
        d_goal_sp = self._dist_to("goal", treat_door_as_open=door_open)

        # Progress shaping based on current phase
        # (use pre-step phase; it’s a bit more stable around key/door transitions)
        prog = 0.0
        if not prev_has_key:
            prog = self._delta_reward(d_key_s, d_key_sp)
        elif not prev_door_open:
            prog = self._delta_reward(d_door_s, d_door_sp)
        else:
            prog = self._delta_reward(d_goal_s, d_goal_sp)

        shaped_reward += prog

        # Update cached state (not strictly needed, but useful if you extend wrapper)
        self._prev_has_key = has_key
        self._prev_door_open = door_open
        self._prev_d_key = d_key_sp
        self._prev_d_door = d_door_sp
        self._prev_d_goal = d_goal_sp

        return obs, shaped_reward, terminated, truncated, info

    # -----------------------
    # Helpers
    # -----------------------

    def _delta_reward(self, d_s: int | None, d_sp: int | None) -> float:
        """
        alpha * (d(s) - d(s'))
        If a distance is undefined (target not found / unreachable), return 0.
        """
        if d_s is None or d_sp is None:
            return 0.0
        # If unreachable on one side, keep it safe and neutral
        if d_s >= 10**8 or d_sp >= 10**8:
            return 0.0
        return self.alpha * float(d_s - d_sp)

    def _has_key(self) -> bool:
        c = getattr(self.env, "carrying", None)
        return c is not None and getattr(c, "type", None) == "key"

    def _door_open(self) -> bool:
        door = self._find_obj("door")
        if door is None:
            return False
        return bool(getattr(door, "is_open", False))

    def _find_obj(self, obj_type: str):
        """Return the first object with .type == obj_type, else None."""
        grid = getattr(self.env, "grid", None)
        if grid is None:
            return None
        for j in range(grid.height):
            for i in range(grid.width):
                o = grid.get(i, j)
                if o is not None and getattr(o, "type", None) == obj_type:
                    return o
        return None

    def _find_obj_pos(self, obj_type: str) -> tuple[int, int] | None:
        """Return (x,y) of first object of given type, else None."""
        grid = getattr(self.env, "grid", None)
        if grid is None:
            return None
        for j in range(grid.height):
            for i in range(grid.width):
                o = grid.get(i, j)
                if o is not None and getattr(o, "type", None) == obj_type:
                    return (i, j)
        return None

    def _dist_to(self, obj_type: str, *, treat_door_as_open: bool) -> int | None:
        """
        Shortest-path distance from agent to the target object type on the full grid.

        - Uses 4-neighborhood BFS.
        - Treats walls as blocked.
        - Treats lava as blocked (conservative).
        - Treats the door as passable only if open OR treat_door_as_open=True.
        - Allows stepping onto the target tile itself even if it would normally be non-overlappable,
          to avoid 'inf' distances for key/goal tiles.
        """
        grid = getattr(self.env, "grid", None)
        agent_pos = getattr(self.env, "agent_pos", None)
        if grid is None or agent_pos is None:
            return None

        target = self._find_obj_pos(obj_type)
        if target is None:
            return None

        start = (int(agent_pos[0]), int(agent_pos[1]))
        W, H = grid.width, grid.height

        # BFS
        q = deque()
        q.append(start)
        dist = {start: 0}

        def neighbors(p):
            x, y = p
            if x > 0:
                yield (x - 1, y)
            if x < W - 1:
                yield (x + 1, y)
            if y > 0:
                yield (x, y - 1)
            if y < H - 1:
                yield (x, y + 1)

        def passable(cell, pos) -> bool:
            # Always allow stepping onto target cell to measure distance-to-target.
            if pos == target:
                return True

            if cell is None:
                return True

            t = getattr(cell, "type", None)

            if t == "wall":
                return False
            if t == "lava":
                return False

            if t == "door":
                # Passable if open, else only if we are treating it as open for this distance.
                if getattr(cell, "is_open", False):
                    return True
                return treat_door_as_open

            # Otherwise use minigrid's overlap logic if available
            can_overlap = getattr(cell, "can_overlap", None)
            if callable(can_overlap):
                return bool(cell.can_overlap())
            return False

        while q:
            cur = q.popleft()
            if cur == target:
                return dist[cur]

            for nb in neighbors(cur):
                if nb in dist:
                    continue
                cell = grid.get(nb[0], nb[1])
                if not passable(cell, nb):
                    continue
                dist[nb] = dist[cur] + 1
                q.append(nb)

        # Unreachable
        return 10**9