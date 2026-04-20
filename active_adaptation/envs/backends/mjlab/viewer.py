import torch
import viser
from mjlab.sim import Simulation
from mjlab.viewer.viser import ViserMujocoScene

from active_adaptation.envs.env_base import _EnvBase


class MjLabViewer:
    """
    Different from `mjlab.viewer.viser.viewer.ViserPlayViewer`, this
    viewer is not responsible for stepping the environment and is updated
    synchronously from the environment step loop.
    """

    def __init__(self, env: _EnvBase, sim: Simulation):
        self.env = env
        self.sim = sim

        self._server = viser.ViserServer(label="mjlab")
        self._is_setup = False

    def setup(self):
        if self._is_setup:
            return

        self._scene = ViserMujocoScene(
            self._server,
            self.sim.mj_model,
            self.env.num_envs,
        )
        self._scene.debug_visualization_enabled = True
        self._scene.env_idx = 0

        tabs = self._server.gui.add_tab_group()
        with tabs.add_tab("Scene", icon=viser.Icon.SETTINGS):
            self._scene.create_scene_gui()
        with tabs.add_tab("Visualization", icon=viser.Icon.EYE):
            self._scene.create_overlay_gui()
        with tabs.add_tab("Groups", icon=viser.Icon.LAYERS_INTERSECT):
            self._scene.create_groups_gui()
        self._is_setup = True

    @property
    def scene(self) -> ViserMujocoScene | None:
        return getattr(self, "_scene", None)

    def add_batched_axes(self, name: str):
        axes_handle = self._server.scene.add_batched_axes(
            name=name,
            batched_wxyzs=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(
                self.env.num_envs, 4
            ),
            batched_positions=torch.tensor([[0.0, 0.0, 0.0]]).expand(
                self.env.num_envs, 3
            ),
            batched_scales=torch.tensor([[1.0, 1.0, 1.0]]).expand(
                self.env.num_envs, 3
            ),
        )
        return axes_handle

    def add_line_segments(
        self, name: str, colors: tuple[float, float, float] | torch.Tensor
    ):
        lines_handle = self._server.scene.add_line_segments(
            name=name,
            points=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).expand(
                self.env.num_envs, 2, 3
            ),
            colors=colors,
        )
        return lines_handle

    def clear(self):
        if self._scene is None:
            return
        self._scene.clear()

    def update(self):
        if self._scene is None:
            raise RuntimeError("MjLab viewer is not set up.")
        with self._server.atomic():
            self._scene.update(self.sim.data)
            if self._scene.needs_update:
                self._scene.refresh_visualization()
            self._server.flush()

    def close(self):
        self._server.stop()
