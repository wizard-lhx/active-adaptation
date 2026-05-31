from typing import cast

from active_adaptation.assets import AssetCfg
from active_adaptation.envs.backends.motrixsim.adapter import (
    MotrixSceneAdapter,
    MotrixSimAdapter,
)
from active_adaptation.envs.env_base import _EnvBase
from active_adaptation.registry import Registry


class MotrixsimBackendEnv(_EnvBase):
    """MotrixSim (Motphys) backend env: scene/sim construction. CPU, MJCF-native."""

    def __init__(self, cfg, device: str, headless: bool = True):
        super().__init__(cfg, device, headless)
        self.robot = self.scene.articulations["robot"]

    def setup_scene(self):
        from active_adaptation.envs.backends.motrixsim.motrixsim_sim import (
            MotrixScene,
            MotrixSim,
        )

        registry = Registry.instance()
        asset_cfg = cast(AssetCfg, registry.get("asset", self.cfg.robot.name))

        physics_dt = float(
            self.cfg.sim.get(
                "motrixsim_physics_dt", self.cfg.sim.get("mujoco_physics_dt", 0.005)
            )
        )

        class SceneCfg:
            robot = asset_cfg.motrixsim()

        scene = MotrixScene(
            SceneCfg(), num_envs=self.cfg.num_envs, device=str(self.device), physics_dt=physics_dt,
            step_dt=float(self.cfg.sim.step_dt),
        )
        sim = MotrixSim(scene)
        self.scene = MotrixSceneAdapter(scene)
        self.sim = MotrixSimAdapter(sim)
        self.terrain_type = self.cfg.get("terrain", "plane")


__all__ = ["MotrixsimBackendEnv"]
