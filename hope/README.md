# Hope 单机 8 卡训练

[mirroropd/mirroropd.hope](mirroropd/mirroropd.hope) 参考 MD-OPD 的 `hope/opdonly/opdonly.hope`，沿用用户组、80GB 队列和平台镜像。申请 1 个 worker、8 张 80GB GPU、64 vCPU、256 GiB 内存和 128 GiB 共享内存。

将最新源码包上传并解包到开发机后，在仓库根目录提交：

```bash
cd /home/hadoop-hmart-poistar/dolphinfs_hdd_hadoop-hmart-poistar/fanghaotian/MirrorOPD-verl
make assets-check
make opd-submit
```

提交入口等价于 `cd hope/mirroropd && hope run mirroropd.hope`。实际训练由集群 worker 执行，开发机负责提交任务。配置中的脚本路径使用共享挂载在任务内的路径 `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/MirrorOPD-verl`，与开发机的上述项目目录对应。

Hope worker 直接执行 `examples/mirroropd/run_${DOMAIN:-math}_deepseek_r1_1_5b_fsdp2.sh`，不再经本地包装脚本。入口和训练默认值随源码包更新；默认 shared_sync 共用八卡，standalone 保留原 4+4。

开发机本地 `.hope` 的 `worker.script` 需要一次性替换，具体命令、环境归属、预检和迁移步骤见[Hope 直连入口](../examples/mirroropd/docs/infra/2026-10-22-共享八卡运行与架构.md#hope-直连入口)。`hope/` 仍不随源码包更新，接收端不需要再维护或同步独立 `run.sh`。

Hope 配置中的 `[docker]` 由平台选择作业运行镜像；任务使用已准备好的 `.venv-native`。更换镜像或环境后，在目标 GPU 容器中单独执行依赖和 GPU 运行验收，入口本身不再自动执行原包装脚本的预检。

提交前可以直接预览领域配方，不启动 Ray、访问 GPU 或创建实验目录：

```bash
DRY_RUN=1 bash examples/mirroropd/run_math_deepseek_r1_1_5b_fsdp2.sh
DRY_RUN=1 bash examples/mirroropd/run_code_deepseek_r1_1_5b_fsdp2.sh
DRY_RUN=1 bash examples/mirroropd/run_if_deepseek_r1_1_5b_fsdp2.sh
CONFIG_ONLY=1 bash examples/mirroropd/run_math_deepseek_r1_1_5b_fsdp2.sh
```

正式参数统一维护在[共用训练逻辑](../examples/mirroropd/_run_single_teacher_fsdp2.sh)，领域入口分别是 [math](../examples/mirroropd/run_math_deepseek_r1_1_5b_fsdp2.sh)、[code](../examples/mirroropd/run_code_deepseek_r1_1_5b_fsdp2.sh) 和 [IF](../examples/mirroropd/run_if_deepseek_r1_1_5b_fsdp2.sh)。步数、batch、验证数量和 TP 解释见[配方说明](../examples/mirroropd/docs/训练配方.md)。数学训练默认 1 epoch；用全量 17917 行 DAPO、全局 batch 64 时是 279 步，每 20 步验证，并在最后一步验证。日志写入仓库 `wandb/`，checkpoint 和验证输出写入 `outputs/<实验名>/`。
