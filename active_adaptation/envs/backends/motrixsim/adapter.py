from typing_extensions import override

from active_adaptation.envs.adapters import SimAdapter, SceneAdapter


class MotrixSimAdapter(SimAdapter):
    def __init__(self, sim):
        self._sim = sim

    def get_physics_dt(self) -> float:
        return self._sim.get_physics_dt()

    def has_gui(self) -> bool:
        return self._sim.has_gui()

    def step(self, render: bool = False) -> None:
        self._sim.step(render=render)

    def render(self) -> None:
        pass

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None:
        pass

    def __getattr__(self, name):
        return getattr(self._sim, name)


class MotrixSceneAdapter(SceneAdapter):
    def __init__(self, scene):
        self._scene = scene

    @property
    def num_envs(self) -> int:
        return self._scene.num_envs

    def reset(self, env_ids) -> None:
        self._scene.reset(env_ids)

    def update(self, dt: float) -> None:
        self._scene.update(dt)

    def write_data_to_sim(self) -> None:
        self._scene.write_data_to_sim()

    @override
    def zero_external_wrenches(self) -> None:
        for asset in self._scene.articulations.values():
            if getattr(asset, "has_external_wrench", False):
                asset._external_force_b.zero_()
                asset._external_torque_b.zero_()
                asset.has_external_wrench = False

    @property
    def articulations(self) -> dict:
        return self._scene.articulations

    @property
    def sensors(self) -> dict:
        return self._scene.sensors

    @property
    def env_origins(self):
        return self._scene.env_origins

    @property
    def ground_mesh(self):
        # plane terrain: GroundQuery short-circuits height_at("plane") and ignores the
        # mesh, so None is fine. (Rough-terrain height queries would need a real mesh.)
        return None

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def __getitem__(self, key):
        return self._scene[key]


__all__ = ["MotrixSimAdapter", "MotrixSceneAdapter"]
