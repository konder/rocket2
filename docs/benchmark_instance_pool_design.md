# Benchmark 实例池与任务评估分离方案

## 1. 单实例资源占用（MineRL/MineStudio）

| 资源 | 量级 | 说明 |
|------|------|------|
| **内存** | ~1–2 GB / 实例 | 每个实例 = 1 个 Java 进程（Minecraft 客户端），JVM 堆 + 原生内存。与 `-Xmx`、渲染分辨率相关。 |
| **端口** | 1 / 实例 | InstanceManager 按端口区分实例（`_malmo_base_port` 起递增）。 |
| **CPU** | 1 核可共享 | 游戏循环与渲染主要占 1 核，多实例可分布到多核。 |
| **GPU** | 可选 | 若用 GPU 渲染（VirtualGL），多实例可能共享或各占少量显存。 |

结论：**每个实例资源占用不小**，机器内存是主要瓶颈（例如 32GB 上同时跑约 10–15 个实例需谨慎）。

---

## 2. 方案概述：实例池 + 任务评估分离

思路：

1. **预先收集任务顺序**：从 `eval_tasks_paper.yaml` 等得到任务列表与每任务 episode 数。
2. **固定大小的实例池**：池大小 `pool_size = N`（如 4 或 8），由内存与需求折中。
3. **并行初始化**：启动时创建 N 个 **worker**（进程或线程），每个 worker 在“接到任务时”再创建该任务对应的 env，从而 N 个任务可同时各自创建 env，实现 **并行初始化**。
4. **执行流程**：从池中取一个空闲 worker → 分配任务 T → 该 worker 为 T 创建 env（用 T 的 `env_conf`），跑完 T 的全部 episode（同一 env 内 reset），关闭 env → 结果回传，worker 归还池，可接下一任务。
5. **全部结束后销毁**：所有任务跑完后，关闭所有 worker（若 worker 内还持有 env 则关闭 env），释放资源。

要点：

- **不能**用“一个通用 env 跑所有任务”：不同任务的 `env_conf` 不同（世界、初始物品、成功条件等），MinecraftSim 在构造时绑定了 callbacks/env_conf，因此 **每个任务必须用自己的 env**。
- **可以**做的是：**池 = N 个 worker**，每个 worker 在接到任务时 **按该任务创建 env**，跑完该任务全部 episode 后 **关闭 env**，再接下一个任务。这样：
  - 同一时刻最多 N 个 env（N 个 Java 进程），资源可控；
  - 前 N 个任务会触发 N 次 env 创建，相当于 **并行初始化 N 个实例**；
  - 任务与评估在逻辑上分离（调度器只发“任务 + 参数”，worker 只负责“拿任务 → 建 env → 跑 episode → 关 env → 报结果”）。

---

## 3. 可行性结论

**可行。** 推荐实现方式：**进程池 + 每任务建/关 env**。

- **进程池**：主进程维护任务队列，N 个子进程各为一个 worker；主进程向 worker 发送 `(task, num_episodes, ...)`，worker 内部 `create_env_from_paper_task(task, ...)` → `run_single_episode` 循环 → `env.close()` → 返回该任务的结果。
- **并行初始化**：第一个 N 个任务被分配到 N 个 worker 时，N 个 env 会在 N 个进程中同时创建，即并行初始化。
- **内存**：同一时刻最多 N 个 Java 进程，约 N×(1–2) GB，需根据本机内存设定 `pool_size`。
- **与现有脚本兼容**：`create_env_from_paper_task`、`run_single_episode`、成功判定等均可复用；仅增加“任务队列 + worker 进程 + 结果汇总”的调度层。

可选：若希望进一步减少“重复创建/销毁同一任务配置的 env”，可做 **按任务名/配置的 env 复用**（同一 task 的多个 episode 仍复用同一 env，只 reset；不同 task 之间不复用），当前“复用 env 只 reset”的逻辑已经支持这一点，在 worker 内按任务建一个 env、跑完该任务所有 episode 再关即可。

---

## 4. 实现要点（建议）

1. **任务顺序**：从 YAML 解析出 `List[(task, num_episodes)]`，顺序固定。
2. **Worker 协议**：主进程通过 `multiprocessing.Queue` 或 `multiprocessing.Pool.apply_async` 下发 `(task_dict, num_episodes, max_steps, output_dir, ...)`；worker 返回 `{task_name, episode_results, ...}`。
3. **Worker 内逻辑**：  
   `env = create_env_from_paper_task(task, env_conf_dir, num_empty_frames)` → 循环 `run_single_episode(agent, env, goal_gen, task, ...)` 共 `num_episodes` 次 → `env.close()` → 返回该 task 的汇总结果。  
   Agent / goal_gen 可在 worker 内按需创建，或由主进程传入（若可序列化）。
4. **Agent / GoalGenerator**：若使用 GPU，多进程共享 GPU 需注意 CUDA 与 fork 的兼容性（通常每个进程各建一份 model，或用 spawn 启动子进程）；若仅 CPU，则无此问题。
5. **池大小**：建议通过 CLI `--pool-size N` 暴露，默认 1（退化为当前串行），N>1 时启用上述调度。

这样即实现“固定大小实例池 + 并行初始化 + 任务与评估分离”；“全部 episode 完毕后销毁”体现在每个 worker 在完成当前任务的**所有** episode 后关闭 env，并在全部任务结束后退出 worker 进程。
