# Effective Impedance Diagnostics (`dev/algoeef`)

本文说明 `dev/algoeef` 分支增加的等效阻抗诊断功能，包括计算假设、代码组织、训练日志、play 数据记录、交互可视化和已知限制。

该功能是**只读诊断**：它在线性化当前策略，但不参与动作采样、PPO loss、反向传播或 optimizer step。关闭诊断后，原 `ppo_symaug` 的训练行为不应改变。

## Quick Start

以下命令均从 active-adaptation 仓库根目录执行。

### 训练并记录 WandB 指标

```bash
uv run --project venv/isaac51 scripts/train_ppo.py \
  task=B2/B2Z1Loco \
  backend=isaac \
  algo=ppo_symaug_eff \
  algo.eff_impedance.enabled=true
```

`ppo_symaug_eff` 的训练诊断默认每 10 次 `train_op` 计算一次：

```bash
algo.eff_impedance_interval=10
```

### Play 时记录矩阵和视频

```bash
uv run --project venv/isaac51 scripts/play.py \
  task=B2/B2Z1Loco \
  backend=isaac \
  algo=ppo_symaug_eff \
  checkpoint_path='run:<entity>/<project>/<run_id>' \
  eff_impedance_play=true \
  task.num_envs=1 \
  +task.command.teleop=true \
  record_video=true
```

关键开关是：

```text
algo=ppo_symaug_eff
eff_impedance_play=true
```

`eff_impedance_play=true` 会由 play reporter 启用 probe，不要求额外设置 `algo.eff_impedance.enabled=true`。

输出位于同一个 Hydra play run 目录：

```text
outputs_play/<date>/<run>/videos/<task>-<time>.mp4
outputs_play/<date>/<run>/eff_impedance/eff_impedance_timeseries.npz
```

### 交互查看

```bash
uv run --project venv/isaac51 scripts/visualize_eff_impedance.py \
  --video /absolute/path/to/video.mp4 \
  --npz /absolute/path/to/eff_impedance_timeseries.npz
```

交互控制：

| 操作 | 作用 |
|---|---|
| `Space` | 自动播放 / 暂停 |
| 长按 `Left` / `Right` | 连续后退 / 前进一个 play step |
| 拖动 Slider | 定位 play step，并暂停自动播放 |
| `V` | 显示 / 隐藏矩阵单元和当前特征值数值 |

## Branch Summary

当前分支基于 `dev/b2z1wt`，保留原 `ppo_symaug`，通过新的 `algo=ppo_symaug_eff` 显式启用诊断能力。

| 文件 | 作用 |
|---|---|
| `active_adaptation/learning/diagnostics/__init__.py` | diagnostics package 入口 |
| `active_adaptation/learning/diagnostics/eff_impedance.py` | Jacobian、等效矩阵、诊断、采样、play recorder |
| `active_adaptation/learning/ppo/ppo_symaug_eff.py` | `ppo_symaug` 适配、Hydra 注册、参数自动推导、训练日志接入 |
| `scripts/play.py` | play 采样、矩阵计算、NPZ/video 输出目录接入 |
| `scripts/visualize_eff_impedance.py` | 视频、矩阵和诊断的离线交互回放 |

分支还包含以下任务配置差异：

```yaml
# cfg/task/B2/B2Z1Loco.yaml
base_height_range: [0.45, 0.45]
```

它固定了 B2Z1Loco 的高度命令，但**不属于等效阻抗公式或诊断所必需的修改**。

## What Is Computed

### Policy linearization

设策略均值为：

```text
mu(o) in R^n
```

其中 `n` 是受控关节数。诊断使用分布的均值 `loc`，不是 rollout 中随机采样的 action。

在采样工作点处计算：

```text
Jq  = d mu / d q       shape: (B, n, n)
Jqd = d mu / d qdot    shape: (B, n, n)
```

实现使用：

```python
jac_fn = torch.func.jacrev(mean_single)
full_jac = torch.func.vmap(jac_fn)(obs)
```

`jacrev` 对单个 observation 求 Jacobian，`vmap` 将同一计算批量应用到 `B` 个工作点。

### Action Space I

当前 B2Z1Loco 使用位置目标动作，默认配置为 `action_space="I"`。记：

```text
Kp = diag(kp)
Kd = diag(kd)
A  = diag(alpha)
I  = identity
```

其中 `alpha` 是每个动作到位置目标的 `action_scaling`。代码计算：

```text
Keff = Kp (I - A Jq)
Deff = Kd - Kp A Jqd
```

矩阵元素含义：

- 对角项 `Keff[i, i]` / `Deff[i, i]`：关节 `i` 对自身状态变化的局部等效刚度 / 阻尼。
- 非对角项 `M[i, j]`：关节 `j` 的状态变化通过策略引起关节 `i` 动作变化，表示策略产生的关节耦合。

因此等效参数通常是完整矩阵，不再只是每个关节一个标量。

### Action Space II

代码还保留位置目标加速度前馈形式：

```text
Deff       = Kd (I - beta dt A Jq) - Kp A Jqd
Meff_delta = -Kd beta dt A Jqd
```

使用 Action Space II 时必须配置 `beta` 和 `dt`。当前默认 Action Space I 不产生 `Meff_delta`，当前实现也不构造绝对 `Meff`，只在 Action Space II 返回相对惯性增量。

## Automatic Configuration

`ppo_symaug_eff.PPOPolicy.from_env()` 在策略创建时从环境自动填充诊断配置。

### Controlled joints

策略递归展开普通或 `ConcatenatedAction` action manager，并要求每个 manager 满足：

```text
action_dim == number of controlled joint_ids
```

所有 manager 的关节按 action 拼接顺序合并，所以 Jacobian 行、Kp/Kd 和 action scaling 使用同一顺序。

### Kp and Kd

Kp/Kd 从运行环境中的实际：

```python
asset.data.joint_stiffness
asset.data.joint_damping
```

读取。当前实现先对 env 维求平均，再用于该批工作点：

```text
kp = joint_stiffness[:, joint_ids].mean(dim=0)
kd = joint_damping[:, joint_ids].mean(dim=0)
```

因此 `task.num_envs=1` 时是该环境的实际值；多环境且 Kp/Kd 被 domain randomization 后，诊断使用跨环境平均增益，而不是逐环境增益。

### Alpha

`alpha` 直接来自 action manager 的 `action_scaling`。标量 scaling 会扩展到全部动作，向量 scaling 按 action 顺序拼接。

### q and qd slices

代码遍历：

```python
env.observation_groups["policy"].funcs
```

按 observation 实际拼接顺序累积 offset，并识别：

```text
joint_pos / joint_pos_multistep
joint_vel / joint_vel_multistep
```

如果 actor input 还包含独立 `command` key，会先加上 command 宽度，因为 `ppo_symaug` actor input 顺序为：

```text
[command, policy]
```

对于 `joint_*_multistep`，历史张量按 `(steps, joints)` 展平，index 0 是最新更新的历史帧。当前 slice 只选择第一个连续 `n` 关节块，也就是最新帧，不会同时对全部历史帧求导并合并。

## Mean Network Adapter

`ppo_symaug` 的 actor 不是一个可以直接以 observation tensor 调用的普通网络。`_PpoSymaugMean` 完成：

```text
raw actor input
  -> VecNorm normalize
  -> actor.get_dist_params()
  -> loc
```

这避免直接调用内部 `ModuleList`，并确保 Jacobian 对应训练时真实使用的归一化策略均值。

求 Jacobian 时：

- 网络切换到 eval。
- 临时关闭参数 `requires_grad`。
- 仍允许对 observation 求导。
- 计算结束后恢复网络 train/eval 状态和参数梯度标记。

这不同于 `torch.no_grad()` 或 `torch.inference_mode()`：后两者会阻止 observation Jacobian 所需的 autograd。

## Sampling and Compute Flow

### Operating point sampling

`EffImpedanceProbe.sample_operating_points()`：

1. 将 TensorDict 组装成 `[command, policy]` actor input。
2. 展平 time/env 等前导 batch 维。
3. 按 `sample_stride` 选择工作点。
4. 截断到 `max_points`。
5. 将 observation、Kp、Kd `detach().cpu()` 后缓存。

CPU 缓存用于减少 rollout GPU 显存占用。真正计算 Jacobian 时，observation 和增益会移动回 mean network 所在 device。

### Compute pipeline

```text
cached operating points
  -> policy mean adapter
  -> Jq / Jqd via jacrev + vmap
  -> Keff / Deff / optional Meff_delta
  -> NumPy matrix diagnostics on CPU
  -> training scalar logs or play NPZ
  -> reset cached operating points
```

### Training timing

`ppo_symaug_eff.train_op()` 在执行原 PPO update 之前计算诊断。默认：

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `eff_impedance.enabled` | `false` | 是否启用训练诊断 |
| `eff_impedance_interval` | `10` | 第 0 次及以后每 10 次 `train_op` 计算 |
| `eff_impedance.sample_stride` | `1` | 展平 batch 中每隔多少点保留一个 |
| `eff_impedance.max_points` | `4096` | 单次诊断最多工作点数 |
| `eff_impedance.log_scalars` | `true` | 是否输出 scalar/per-joint 日志 |

interval 控制**计算频率**，不是先积累 10 个 rollout window 再统一平均；到达 interval 时会采样当前 `train_op` 的 tensordict、立即计算并 reset。

### Play timing

`EffImpedancePlayConfig` 当前默认：

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `update_interval` | `1` | 每个 play step 计算一次 |
| `sample_mode` | `latest` | 每次 sample 前 reset，只保留当前 step |
| `max_points` | `256` | 当前 report 最多工作点数 |
| `show_viewer` | `false` | 不在仿真进程中画 live heatmap |
| `record_npz` | `true` | 记录离线数据 |
| `autosave_interval` | `1` | 每增加一条记录就原子保存 |

play 中 observation/video frame 在 `torch.inference_mode()` 内采集；`maybe_report()` 在该上下文外执行 Jacobian 计算。

使用 `sample_mode="latest"` 和 `task.num_envs=1` 时，每个记录 step 通常只有一个工作点。多环境时同一 step 会采集多个环境，recorder 保存它们的平均 `Keff/Deff`，再对该平均矩阵计算 NPZ diagnostics。

`EffImpedancePlayConfig` 目前没有作为 Hydra 子配置暴露；CLI 只暴露 `eff_impedance_play` 总开关。修改其他 play 默认值需要在代码中传入或调整该 dataclass。

## Diagnostics

### Matrix diagnostics

完整 `Keff/Deff` 可以是非对称矩阵。稳定性相关特征使用对称部分：

```text
sym(M) = 0.5 (M + M^T)
```

随后计算：

| 指标 | 含义 |
|---|---|
| `*_diag` | 完整矩阵的对角元素 |
| `*_sym_eigvals` | `sym(M)` 的全部特征值，升序排列 |
| `*_sym_min_eig` | 最小特征值，即 spectrum index 0 |
| `*_sym_cond` | `np.linalg.cond(sym(M))`，默认 2-norm condition number |
| `*_sym_neg_count` | 负特征值数量 |
| `*_sym_neg_frac` | 负特征值数量 / 关节数 |

特征值 index 是从小到大的**排序位置**，不是关节编号。每个特征值对应一个由多个关节构成的 eigenvector；当前 NPZ 不记录 eigenvector，因此不能仅凭 index 定位具体关节。

### Training log semantics

训练日志通过 `eff_impedance/` 前缀输出。主要 scalar 包括：

```text
eff_impedance/Keff_diag_mean|min|max
eff_impedance/Deff_diag_mean|min|max
eff_impedance/Keff_sym_min_eig_mean|min
eff_impedance/Deff_sym_min_eig_mean|min
eff_impedance/Keff_sym_cond_mean
eff_impedance/Deff_sym_cond_mean
eff_impedance/Keff_sym_neg_frac
eff_impedance/Deff_sym_neg_frac
```

逐关节对角项：

```text
eff_impedance/Keff_diag/joint_00
eff_impedance/Deff_diag/joint_00
eff_impedance/Keff_diag_offset/joint_00
eff_impedance/Deff_diag_offset/joint_00
...
```

`diag_offset` 分别是：

```text
Keff_diag - nominal Kp
Deff_diag - nominal Kd
```

注意同名 `*_sym_neg_frac` 在两个输出路径的统计对象不同：

- **Training scalar log**：采样工作点中 `min_eig < 0` 的工作点比例。
- **Play NPZ**：当前保存的平均矩阵中，负特征值占全部特征值的比例。

## NPZ Schema

设：

```text
T = 记录的 report 数
n = 受控关节 / action 数
```

| 字段 | Shape | 含义 |
|---|---|---|
| `steps` | `(T,)` | play step |
| `num_points` | `(T,)` | 该 report 使用的工作点数 |
| `Keff` | `(T, n, n)` | 工作点平均后的等效刚度矩阵 |
| `Deff` | `(T, n, n)` | 工作点平均后的等效阻尼矩阵 |
| `Meff_delta` | `(T, n, n)` | Action Space II 时可选 |
| `Keff_diag`, `Deff_diag` | `(T, n)` | 对角元素 |
| `Keff_sym_eigvals`, `Deff_sym_eigvals` | `(T, n)` | 对称部分全部特征值 |
| `Keff_sym_min_eig`, `Deff_sym_min_eig` | `(T,)` | 最小特征值 |
| `Keff_sym_cond`, `Deff_sym_cond` | `(T,)` | 条件数 |
| `Keff_sym_neg_count`, `Deff_sym_neg_count` | `(T,)` | 负特征值数量 |
| `Keff_sym_neg_frac`, `Deff_sym_neg_frac` | `(T,)` | 负特征值比例 |

recorder 在内存中保留时间序列，并采用临时文件加原子替换：

```text
.eff_impedance_timeseries.npz.tmp
  -> np.savez_compressed
  -> replace eff_impedance_timeseries.npz
```

如果写临时文件时进程中断，旧的正式 NPZ 不会被半写文件覆盖。`autosave_interval=1` 提高了中断恢复能力，但会在每个记录 step 重新压缩完整已累计序列，长时间 play 时可能带来 I/O 开销。

旧版本 NPZ 只有 `steps/num_points/Keff/Deff`，不包含交互 viewer 需要的特征字段，需要重新运行 play 生成。

## Interactive Viewer

窗口包含：

- 视频画面。
- 全局零中心色标的 `Keff/Deff` heatmap；蓝色为负、白色为零、红色为正。
- 当前 step 的 K/D symmetric eigenvalue spectrum。
- 全程 K/D minimum eigenvalue 曲线。
- 全程 K/D condition number 曲线（log y-axis）。
- 当前 video step、实际使用的 matrix step、工作点数及 scalar 摘要。

对齐规则：

```text
video_step = frame_index + 1
matrix = latest recorded NPZ step <= video_step
```

如果 `update_interval > 1`，两个 matrix record 之间的视频帧会保持显示最近一次矩阵。

自动播放按视频 metadata 中的 FPS 设置 timer。Matplotlib/imageio 实际刷新速度可能低于录制 FPS，但每次回调仍前进一个 play step，不会改变视频和矩阵的 step 对应关系。

## Interpretation

### Diagonal and coupling

- 对角项适合观察每个关节自身的局部等效 K/D。
- 非对角项表示策略导致的跨关节耦合，不应当直接解释为某关节自己的标量增益。

### Eigenvalue spectrum

对于 `n=12`，每个矩阵有 12 个按升序排列的特征值：

```text
index 0 = minimum eigenvalue
...
index 11 = maximum eigenvalue
```

- `min_eig > 0`：对称部分正定。
- `min_eig = 0`：存在弱方向或奇异方向。
- `min_eig < 0`：至少存在一个负方向。

负 `Deff` eigenvalue 表示该局部组合速度方向不再耗散；负 `Keff` eigenvalue 表示该局部组合位移方向不表现为恢复刚度。它们是局部诊断信号，不等价于机器人一定会发生全局不稳定。

### Condition number

condition number 很大表示矩阵在不同组合方向上的响应尺度差异很大，或接近奇异。它不说明具体是哪一个关节；要定位方向需要 eigenvector 或 singular vector，而当前 recorder 未保存这些向量。

## Known Limitations

1. 结果是策略在采样 observation 附近的局部一阶线性化，不是对整个状态空间成立的全局机械阻抗。
2. 当前公式描述 policy + joint PD action mapping，不包含完整刚体动力学、接触、时延、饱和、摩擦和环境闭环的全部影响。
3. 训练使用策略均值 Jacobian，不包含 action sampling noise 对闭环响应的影响。
4. 历史 q/qd observation 只对最新展平块求导，其他历史帧的策略敏感度没有纳入当前矩阵。
5. 多环境 Kp/Kd 当前先跨 env 平均；有 per-env gain randomization 时不能表示每个 env 的独立阻抗。
6. play 多工作点保存平均矩阵；`eig(mean(M))` 一般不等于 `mean(eig(M))`。NPZ 特征与显示的平均矩阵一致，但不代表逐工作点特征平均。
7. eigenvalue rank 在不同 step 之间可能发生 mode swap；index 相同不保证对应同一物理 eigenvector。
8. 当前默认仅验证 Action Space I；Action Space II 需要明确提供 `beta/dt` 并单独验证控制定义。
9. interactive viewer 假定 NPZ 使用当前 schema，不计算缺失特征。
10. Jacobian 与频繁压缩 NPZ 都有额外开销；实时控制部署不应直接启用该诊断路径。

## Review Checklist

合并或扩展该功能时，至少检查：

- action 顺序、joint_ids、Kp/Kd、alpha 和 q/qd slice 顺序一致。
- `Jq/Jqd` 最后两维均为 `(n, n)`。
- `maybe_report()` 位于 `torch.inference_mode()` 外。
- training diagnostics 不改变 PPO optimizer graph。
- video frame 和 NPZ step 使用相同的 play step 定义。
- NPZ 新字段同步更新 viewer。
- 使用 `uv run --project venv/isaac51 ...` 完成 Isaac 相关验证。

