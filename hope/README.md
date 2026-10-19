# Hope 单机 8 卡训练

[mirroropd/mirroropd.hope](mirroropd/mirroropd.hope) 参考 MD-OPD 的 `hope/opdonly/opdonly.hope`，沿用用户组、80GB 队列和平台镜像。申请 1 个 worker、8 张 80GB GPU、64 vCPU、256 GiB 内存和 128 GiB 共享内存。

将最新源码包上传并解包到开发机后，在仓库根目录提交：

```bash
cd /home/hadoop-hmart-poistar/dolphinfs_hdd_hadoop-hmart-poistar/fanghaotian/MirrorOPD-verl
make assets-check
make opd-submit
```

提交入口等价于 `cd hope/mirroropd && hope run mirroropd.hope`。实际训练由集群 worker 执行，开发机负责提交任务。配置中的工作目录使用共享挂载在任务内的路径 `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/MirrorOPD-verl`，与开发机的上述项目目录对应。

[mirroropd/run.sh](mirroropd/run.sh) 是 Hope worker 的统一入口，按 `DOMAIN=math|code|if` 选择 `examples/mirroropd/run_<领域>_deepseek_r1_1_5b_fsdp2.sh`，默认选择 `math`。它使用 `<仓库绝对路径>/.venv-native/bin/python`，student 和 teacher 各 4 卡，推理 TP 均为 1。模型和数据默认读取仓库相邻的 `model/` 与 `data/`。入口保留调度器提供的 `CUDA_VISIBLE_DEVICES`；未提供时使用任务内逻辑卡 `0,1,2,3,4,5,6,7`。启动前运行原生依赖检查，并在每张可见 GPU 上执行一个小矩阵乘法，随后进入正式训练。

在开发机本地 Hope 配置的 worker 环境设置 `afo.role.worker.env.DOMAIN = math`、`code` 或 `if`。IF 还需要设置共享路径的 `OPENCOMPASS_ROOT` 和 `NLTK_DATA`。旧的无领域前缀配方入口已删除；接收新版源码时，应单独同步本机维护的 `hope/mirroropd/run.sh`。

Hope 配置中的 `[docker]` 由平台选择作业运行镜像；本项目不执行 Docker 命令或使用 uv 安装。任务复用已准备好的 `.venv-native`，不自动重建或安装环境。镜像沿用参考文件，其中预装依赖是否与开发机一致，要以任务里的 `native-check` 输出为准；不能用开发机上的导入成功替代集群 GPU 验收。

提交前可以预览命令，不启动 Ray、访问 GPU 或创建实验目录：

```bash
DRY_RUN=1 bash hope/mirroropd/run.sh
DOMAIN=code DRY_RUN=1 bash hope/mirroropd/run.sh
DOMAIN=if DRY_RUN=1 bash hope/mirroropd/run.sh
CONFIG_ONLY=1 bash hope/mirroropd/run.sh
```

正式参数统一维护在[共用训练逻辑](../examples/mirroropd/_run_single_teacher_fsdp2.sh)，领域入口分别是 [math](../examples/mirroropd/run_math_deepseek_r1_1_5b_fsdp2.sh)、[code](../examples/mirroropd/run_code_deepseek_r1_1_5b_fsdp2.sh) 和 [IF](../examples/mirroropd/run_if_deepseek_r1_1_5b_fsdp2.sh)。步数、batch、验证数量和 TP 解释见[配方说明](../examples/mirroropd/README.md)。数学训练默认 1 epoch；用全量 17917 行 DAPO、全局 batch 64 时是 279 步，每 20 步验证，并在最后一步验证。日志写入仓库 `wandb/`，checkpoint 和验证输出写入 `outputs/<实验名>/`。
