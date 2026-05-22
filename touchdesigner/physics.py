"""
physics.py — Force-directed graph layout engine (Fruchterman-Reingold style).

Forces applied each tick:
  1. Coulomb repulsion between every node pair.
  2. Hooke spring attraction along each edge.
  3. Weak gravity towards the canvas centre (prevents drift).

The engine runs in a background daemon thread and mutates Node.x / .y directly
so the renderer reads live positions every frame with zero latency.

Thread safety: a single Lock guards all node-position reads/writes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from core.graph_state import GraphState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameter set (tuned for 1280×720 canvas)
# ---------------------------------------------------------------------------
DEFAULT_REPULSION = 4500.0    # Coulomb coefficient (px² per tick)
DEFAULT_SPRING_K = 0.045      # Spring stiffness
DEFAULT_SPRING_L = 160.0      # Rest length (px)
DEFAULT_DAMPING = 0.83        # Velocity multiplier per tick [0, 1]
DEFAULT_GRAVITY = 0.009       # Centre-pull strength
DEFAULT_DT = 1.0              # Time-step
DEFAULT_MAX_SPEED = 45.0      # Velocity clamp (px/tick)
DEFAULT_TPS = 60.0            # Ticks per second


class PhysicsEngine:
    """Force-directed layout simulation operating on a :class:`~core.graph_state.GraphState`.

    The simulation runs in a daemon thread started by :meth:`start` and stopped
    by :meth:`stop`.  Node positions are updated in-place each tick and are safe
    to read from the render thread.

    Example::

        engine = PhysicsEngine(graph, canvas_width=1280, canvas_height=720)
        engine.start()
        # render loop reads graph.nodes[i].x / .y each frame
        engine.stop()
    """

    def __init__(
        self,
        graph: GraphState,
        canvas_width: float = 1280.0,
        canvas_height: float = 720.0,
        repulsion: float = DEFAULT_REPULSION,
        spring_k: float = DEFAULT_SPRING_K,
        spring_l: float = DEFAULT_SPRING_L,
        damping: float = DEFAULT_DAMPING,
        gravity: float = DEFAULT_GRAVITY,
        dt: float = DEFAULT_DT,
        max_speed: float = DEFAULT_MAX_SPEED,
        ticks_per_second: float = DEFAULT_TPS,
    ) -> None:
        """Initialise the physics engine.

        Args:
            graph: GraphState whose nodes will be simulated.
            canvas_width: Rendering canvas width in pixels.
            canvas_height: Rendering canvas height in pixels.
            repulsion: Coulomb repulsion coefficient.
            spring_k: Hooke spring stiffness along edges.
            spring_l: Spring natural/rest length (pixels).
            damping: Per-tick velocity multiplier (0 = instant stop, 1 = no damping).
            gravity: Centralising gravity strength.
            dt: Integration time-step.
            max_speed: Maximum node speed in px/tick.
            ticks_per_second: Target simulation frequency.
        """
        self.graph = graph
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.repulsion = repulsion
        self.spring_k = spring_k
        self.spring_l = spring_l
        self.damping = damping
        self.gravity = gravity
        self.dt = dt
        self.max_speed = max_speed
        self._interval = 1.0 / ticks_per_second
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background simulation thread (idempotent)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="physics-engine"
        )
        self._thread.start()
        logger.debug("PhysicsEngine started.")

    def stop(self) -> None:
        """Stop the simulation thread and wait for it to exit."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.debug("PhysicsEngine stopped.")

    def scatter(self, margin: float = 80.0) -> None:
        """Randomise node positions within the canvas (thread-safe).

        Args:
            margin: Pixel margin from each canvas edge.
        """
        rng = np.random.default_rng()
        with self._lock:
            for node in self.graph.nodes:
                if not node.pinned:
                    node.x = float(rng.uniform(margin, self.canvas_width - margin))
                    node.y = float(rng.uniform(margin, self.canvas_height - margin))
                    node.vx = 0.0
                    node.vy = 0.0

    def tick(self) -> None:
        """Advance the simulation by one time-step (thread-safe).

        Computes all forces, integrates velocities and positions, and clamps
        nodes to the canvas boundary.
        """
        with self._lock:
            nodes = self.graph.nodes
            n = len(nodes)
            if n == 0:
                return

            pos = np.array([[nd.x, nd.y] for nd in nodes], dtype=np.float64)
            vel = np.array([[nd.vx, nd.vy] for nd in nodes], dtype=np.float64)
            forces = np.zeros((n, 2), dtype=np.float64)
            cx = self.canvas_width / 2.0
            cy = self.canvas_height / 2.0

            # --- Coulomb repulsion (all pairs) ---
            for i in range(n):
                diff = pos[i] - pos            # (n, 2)
                dist = np.linalg.norm(diff, axis=1) + 1e-6  # (n,)
                dist[i] = 1e6                  # self-force → zero
                rep = self.repulsion / (dist ** 2)
                direction = diff / dist[:, None]
                forces[i] += (rep[:, None] * direction).sum(axis=0)

            # --- Spring attraction along edges ---
            id_to_idx = {nd.id: idx for idx, nd in enumerate(nodes)}
            for edge in self.graph.edges:
                i = id_to_idx.get(edge.source_id)
                j = id_to_idx.get(edge.target_id)
                if i is None or j is None:
                    continue
                delta = pos[j] - pos[i]
                dist = np.linalg.norm(delta) + 1e-6
                stretch = dist - self.spring_l
                f = self.spring_k * stretch * (delta / dist)
                forces[i] += f
                forces[j] -= f

            # --- Gravity towards centre ---
            forces[:, 0] += self.gravity * (cx - pos[:, 0])
            forces[:, 1] += self.gravity * (cy - pos[:, 1])

            # --- Integrate ---
            for i, nd in enumerate(nodes):
                if nd.pinned:
                    vel[i] = 0.0
                    continue
                vel[i] = (vel[i] + forces[i] * self.dt) * self.damping
                speed = np.linalg.norm(vel[i])
                if speed > self.max_speed:
                    vel[i] = vel[i] / speed * self.max_speed
                pos[i] += vel[i] * self.dt
                pos[i, 0] = float(np.clip(pos[i, 0], 0.0, self.canvas_width))
                pos[i, 1] = float(np.clip(pos[i, 1], 0.0, self.canvas_height))

            # --- Write back ---
            for i, nd in enumerate(nodes):
                nd.x = float(pos[i, 0])
                nd.y = float(pos[i, 1])
                nd.vx = float(vel[i, 0])
                nd.vy = float(vel[i, 1])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Background thread: tick at the configured rate."""
        while self._running:
            t0 = time.perf_counter()
            try:
                self.tick()
            except Exception as exc:
                logger.warning("Physics tick error: %s", exc)
            sleep = self._interval - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
