# Benchmark 环境启动与复位优化

MineRL / MineStudio 的「慢」主要来自：**首次启动 JVM + Minecraft 进程**、**每次 reset 时的世界重载**。下面是可以做的优化与边界。

---

## 1. 已实现的优化：复用 env，只 reset

**默认开启**：同一 task 的多个 episode 共用一个 `MinecraftSim`，只在 episode 之间调用 `env.reset()`，不再每个 episode 都新建进程。

- **效果**：首个 episode 仍有一次完整启动（warmup 几十秒），后续 episode 只有 reset 成本，无再次启动 JVM 的开销。
- **关闭方式**：加 `--no-reuse-env` 会恢复「每个 episode 新建 env」的旧行为（调试或排查问题时可用）。

---

## 2. MineStudio 提供的可调参数

| 参数 | 默认值 | 说明 | 对启动/复位的影响 |
|------|--------|------|-------------------|
| `num_empty_frames` | 20 | reset 后跳过的空帧数（避免黑屏） | 减小可略缩短每次 reset 后的等待（少跑几步 no-op），例如改为 10 或 5。用 `--num-empty-frames` 传入。 |
| `render_size` | (640, 360) | 游戏渲染分辨率 | 在创建 `MinecraftSim` 时若传入更小分辨率，可能略微减轻 Java 端负载，对启动/复位有轻微帮助。当前 benchmark 使用默认值。 |
| `obs_size` | (224, 224) | 给策略的观测分辨率 | 只影响 Python 端 resize，对 Minecraft 启动几乎无影响。 |

MineStudio 文档里**没有**专门的「启动加速」章节，只有：

- **FastResetCallback**：通过传送玩家实现「快速复位」，**不重置方块/实体**，挖矿、击杀等任务在第二次及以后会没有目标，**不能用于本 benchmark**。
- **SpeedTestCallback**：用于测 step 耗时，不改变启动行为。

因此：在「必须完整重置世界」的前提下，**框架层能做的启动优化就是复用 env + 上述可调参数**。

---

## 3. MineRL 官方说明（渲染与 step，非启动）

[MineRL Performance tips](https://minerl.readthedocs.io/en/latest/notes/performance-tips.html) 主要讲的是：

- **xvfb 渲染**：用 xvfb 时渲染在 CPU，整体会慢 2–3 倍，主要影响 **step()** 的 FPS，对「首次启动」帮助有限。
- **GPU 渲染**：用 VirtualGL / vglrun 或带 GPU 的 Docker，让 Minecraft 用 GPU 渲染，可明显提升 step 速度，**不会显著减少 JVM/世界首次加载时间**。

结论：**启动/复位速度**主要受 JVM + 世界加载制约，MineRL/MineStudio 没有提供更多「一键加速启动」的接口；**step 速度**可以通过 GPU 渲染、减小 `render_size` 等进一步优化。

---

## 4. 建议总结

| 目标 | 做法 |
|------|------|
| 减少「每 task 总时间」 | 保持默认 **复用 env**（不加 `--no-reuse-env`）。 |
| 略减每次 reset 后等待 | 使用 `--num-empty-frames 10` 或 `5`（若画面可接受）。 |
| 提高 step 帧率 | 使用 GPU 渲染（VirtualGL/vglrun 或 GPU Docker），并视情况减小 `render_size`（需改代码或后续支持配置）。 |
| 不破坏挖矿/击杀等任务 | **不要**使用 FastResetCallback，必须用完整 `env.reset()`。 |

如需进一步压榨启动时间，只能从 Java/Minecraft 侧入手（如 JVM 参数、世界预生成等），已超出当前 MineStudio/MineRL 的封装范围。

---

## 5. 实例池与任务评估分离（可选）

若希望**并行跑多个任务**以缩短总墙钟时间，可采用「固定大小实例池 + 任务与评估分离」：预先收集任务顺序，创建 N 个 worker，每个 worker 接到任务后创建该任务对应的 env、跑完全部 episode 后关闭 env，再接下一任务。这样前 N 个任务会并行初始化 N 个实例，资源占用约 N×(1–2) GB 内存。详见 [benchmark_instance_pool_design.md](benchmark_instance_pool_design.md)。
