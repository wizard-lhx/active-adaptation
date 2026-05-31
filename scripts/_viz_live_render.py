"""LIVE Go2 renderer (process B) — opens an NVIDIA-GPU Bevy window and renders
whatever pose the env-writer process (A) publishes to shared memory.

Imports ONLY motrixsim/mujoco/numpy — NOT torch — because importing torch in the
same process segfaults the NVIDIA Vulkan driver. State arrives via /dev/shm.
Run with: VK_ICD_FILENAMES=<nvidia_icd> DISPLAY=:1 python _viz_live_render.py
"""
import os, time, tempfile, numpy as np, mujoco
import motrixsim as mx

GO2 = "/home/ran/active-adaptation/active_adaptation/assets/Go2/mjcf/go2.xml"
SCENE = "/home/ran/active-adaptation/active_adaptation/assets/Go2/mjcf/scene_viz.xml"
DOF_FILE = "/dev/shm/go2_live_dof.npy"


def lookat_quat_wxyz(eye, target, up=(0, 0, 1)):
    eye = np.array(eye, float); target = np.array(target, float); up = np.array(up, float)
    f = target - eye; f /= np.linalg.norm(f); r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    m = np.column_stack([r, u, -f]); tr = m.trace()
    if tr > 0:
        s = np.sqrt(tr + 1) * 2; w = 0.25 * s; x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    return [w, x, y, z]


def build_render_model():
    spec = mujoco.MjSpec.from_file(GO2)
    # go2.xml paints every geom rgba="1 1 1 1" (white) -> blows out. Tint to gray.
    for geo in spec.geoms:
        rr = list(geo.rgba)
        if rr[0] > 0.95 and rr[1] > 0.95 and rr[2] > 0.95:
            geo.rgba = [0.55, 0.56, 0.6, 1.0]
    # MotrixSim's default headlight is very bright -> controlled values
    hl = spec.visual.headlight
    hl.ambient = [float(os.environ.get("HL_A", 0.3))] * 3
    hl.diffuse = [float(os.environ.get("HL_D", 0.4))] * 3
    hl.specular = [0.0, 0.0, 0.0]
    g = spec.worldbody.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_PLANE; g.name = "floor"; g.size = [0, 0, 0.05]; g.friction = [1, 0.1, 0.1]
    g.rgba = [0.13, 0.14, 0.16, 1.0]  # dark floor
    light = spec.worldbody.add_light()
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.pos = [0, 0, 4]; light.dir = [0, 0, -1]
    light.diffuse = [float(os.environ.get("LIGHT_D", 0.45))] * 3; light.specular = [0.1, 0.1, 0.1]
    cam = spec.worldbody.add_camera()
    cam.name = "track"; cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
    eye = (-1.3, -1.3, 0.8); cam.pos = list(eye); cam.quat = lookat_quat_wxyz(eye, (0, 0, 0.2))
    spec.compile()
    fd, p = tempfile.mkstemp(suffix=".xml", dir=os.path.dirname(GO2), prefix="_rendmodel_")
    os.close(fd); open(p, "w").write(spec.to_xml())
    try:
        return mx.load_model(p)
    finally:
        os.remove(p)


def main():
    model = mx.load_model(SCENE)  # go2 + checker floor + skybox + visual headlight (umi-on-legs look)
    data = mx.SceneData(model, batch=(1,))
    dof = np.asarray(data.dof_pos).copy(); dof[0, :7] = [0, 0, 0.35, 0, 0, 0, 1]
    data.set_dof_pos(dof.astype(np.float32), model); model.forward_kinematic(data)

    print("[render] opening NVIDIA window on DISPLAY=%s" % os.environ.get("DISPLAY"), flush=True)
    app = mx.render.RenderApp(headless=False); app.__enter__()
    app.launch(model, batch=1)
    app.system_camera.active = False
    app.set_main_camera(model.cameras.tolist()[0])
    app.sync(data)
    print("[render] RUNNING — close the window to stop.", flush=True)

    while True:
        if getattr(app, "is_closed", False):
            print("[render] window closed."); break
        try:
            d = np.load(DOF_FILE)            # (1, nq) float32 from the env writer
            data.set_dof_pos(d.astype(np.float32), model)
            model.forward_kinematic(data)
        except Exception:
            pass                              # writer not ready / mid-write — keep last pose
        app.sync(data)
        time.sleep(0.02)
    app.__exit__(None, None, None)


if __name__ == "__main__":
    main()
