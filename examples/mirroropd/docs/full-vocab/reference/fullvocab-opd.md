一、背景
Full-Vocab OPD 在完整词表上对齐 Teacher 与 Student 的概率分布，相比 K1、Top-K 词表能够提供更丰富的监督信号。然而，完整词表会带来了显著的系统开销。

以 Qwen3.5-27B 为例，词表大小 V 为 248320，假设 batch-size B 为 128、序列长度 T 为 128K，采用 BF16 精度存储，则单个 Teacher 完整 logits ([B, T, V]) 的数据存储高达 7.58 TiB。

如此规模的数据量，带来了巨大的传输、计算、显存压力，在 Multi-Teacher 场景下，更进一步加剧了这些问题。

针对上述挑战，我们调研和借鉴了现有方案，从系统多个层面进行了优化，最终基于 VeRL 框架在生产环境中实现了高效、稳定地运行。

二、优化方案
2.1 传输优化
传输 Hidden States，替代全量 Logits
Teacher 侧不再输出全词表 logits，而是输出 LM head 之前的 hidden states，再由训练端通过冻结的 Teacher LM head 重建 logits，并计算 Full-vocab KL Loss。

整个传输数据量从 [B, T, V] 降至 [B, T, H]。以 Qwen3.5-27B 为例，由于 H (5120) << V（248320），传输量减少约 98%，从 7.58 TiB 降至 160 GiB（B=128，T=128K，BF16）。


按 Loss Mask 筛选 Hidden States
在实际训练中，Prompt、Tool Result 和 Padding 等位置的 token 通常不参与 Loss 计算，因此无需传输这些位置的 hidden states。仅保留参与 Loss 计算位置的 hidden states，可以进一步降低传输与存储开销。

2.2 读写调度优化
我们使用了 TransferQueue（TQ）在 Teacher 计算和训练端之间传递 hidden states。数据规模较大场景下，串行读写容易造成计算等待，我们从以下 3 个方面优化数据的读写：

异步写入：与 Teacher 计算重叠
Teacher 侧采用分批异步写入机制，将已完成的计算结果及时写入 TQ，使数据写入与后续 Teacher 计算重叠执行，减少同步等待。


提前预取：与 Student 计算重叠
训练端按 micro-batch 顺序提前预取 hidden states，在计算当前 micro-batch 时，预取后续 micro-batch 所需的数据，减少计算前的数据等待。


SP 组内统一读取，减少重复传输
TQ 以样本为单位存储和读取数据，而在序列并行（SP）场景下，同一样本的 token 可能会被切分至多个 rank，多个 rank 同时读取完整样本会造成重复浪费。

因此，我们让 SP 组内的 leader rank 统一读取样本，再按各 rank 负责的 token 范围切分并分发，避免多个 rank 重复拉取相同数据。

当然，理想方案是由各 SP rank 直接读取其负责计算的 token 数据，从源头消除冗余读取与跨 rank 数据分发开销。但该方案需要对 TQ 的数据读取接口进行适配，暂留作后续优化方向。

2.3 显存与计算优化
由于我们只传输了 hidden states，训练端计算 KL Loss 时，需要先计算完整词表的 logits 和概率分布，全部 token 一起计算会产生庞大的中间张量，带来较高的显存占用。为此，我们将 Teacher / Student 的 LM head projection 与 KL Loss 计算 封装成独立算子，通过二维分块和算子融合，降低显存开销并提升计算效率。

二维分块与重计算，降低显存峰值
整体计算沿 token 和 vocab 两个维度分块，默认块大小为 1024 × 8192。

前向：遍历词表计算 Teacher / Student 的 logits 及归一化常数，并缓存当前 Token 块的完整 logits；随后从缓存中逐块读取并累加 KL，避免重复投影。不同 Token 块复用缓存空间，前向结束后释放。

反向：按块重新计算 logits 并求取梯度，无需保留前向缓存，通过重计算降低显存占用。

算子融合：减少显存读写
分块计算虽然降低了显存占用，但 LM head projection 与 KL 计算若独立执行，仍会产生额外的中间数据读写开销。

为此，我们将 LM head projection 与后续 KL 计算进行融合，分别完成归一化统计、KL 累加，使中间结果尽可能在 Kernel 内直接使用，减少显存读写与 Kernel 启动开销，提升计算效率。


2.4 Packing 优化
Multi-Teacher 场景下，训练端存在多个 Teacher LM head，Loss 计算时需要根据 token 所属的 Teacher，选择对应的 LM head 进行计算。当显存不足以同时容纳所有 LM head 时，需要按需从内存加载和切换不同的 head。此时，样本的 Packing 顺序会影响 LM head 的计算效率。

下面分析对比按长度 Packing 与按 Teacher Packing 两种策略对 LM head 计算效率的影响。

按长度 Packing：同一 Head 的计算容易分散
传统的方案是按长度 Packing，优先考虑填充率与负载均衡，但同一 micro-batch 可能混入多个 Teacher 的样本。训练端需要按 Teacher 拆分 token，分组执行 projection 和 KL。

当同一 Teacher 的 token 分散在多个 micro-batch 中时，会产生多轮小块计算。如果 GPU 上可缓存的 head 数量有限，会频繁切换与重复加载。

按 Teacher Packing：聚集 Token，连续复用 Head
在当前训练 batch 内，先按 teacher ID 聚集样本并顺序 Packing，使同一 Teacher 的 token 尽可能集中，从而增加每次 head 计算处理的有效 Loss token 数量，减少切换与重复加载。

但这种策略可能降低 token 填充率，并影响 micro-batch 间的负载均衡，增加 micro-batch 的数量。因此，我们在 不同 Teacher 交界处允许少量混装，以兼顾 token 填充率、负载均衡与 head 复用效率。


Teacher-aware packing 在 teacher 的交界处，允许同一 micro-batch（MB2、MB3）混合不同 teacher 的数据
三、实验
3.1 整体对比
与普通的 K1 OPD（青色线）对比，Full-Vocab OPD（棕色线）的 Loss 下降趋势更好，同时 单步耗时（step time）相当。



3.2 分析实验
1）传输优化
实验配置：qwen3.5-27B、8 节点、FSDP2、SP=8、batch size=1024、seqlen=128 k

优化配置	Teacher 总耗时 (s)	Actor 总耗时 (s)	hidden-states 大小 (GiB)
传输全部 hidden states	85.29	135.56	58.57
+ 按 loss 位置传输	55.17	123.34	33.34
+ 异步写入	51.06	123.24	33.34
+ (提前预取 + SP 优化)	51.78	110.21	33.34
指标说明：

Teacher 总耗时：Teacher 前向计算及 Hidden States 写入 TQ 的总耗时。
Actor 总耗时：数据读取与分发、前反向计算及优化器更新的总耗时。
实验结论：

按 Loss 位置传输使数据量减少 43.1%；异步写入与提前预取分别降低 Teacher 侧和 Actor 侧的等待。三项优化叠加后，Teacher 与 Actor 完整耗时较基线分别下降 39.3% 和 18.7%。
2）显存与计算优化
基于前面介绍的二维分块、算子融合方案，我们实现了基于 cuBLAS 的 Full-vocab KL 计算算子（以下简称 Chunked cuBLAS）。

我们首先测试了算子在不同分块配置下的前反向耗时与峰值显存，最终选择了兼顾计算效率与显存占用的 1024 × 8192 分块配置。


我们还对比三种实现的计算效率和显存：Full Logits（不分块、不融合）、Chunked cuBLAS（我们的实现）和 Liger-kernel 0.8.4（前后向融合计算）

Token 数	实现	前向耗时（ms）	反向耗时（ms）	总耗时（ms）	峰值显存（GiB）
8K	Full Logits	113.95	110.96	224.54	42.85
Chunked cuBLAS	173.53	214.99	388.47	7.87
Liger-kernel 0.8.4	344.95	0.72	345.67	13.41
16k	Full Logits	—	—	—	OOM
Chunked cuBLAS	346.55	423.98	770.70	8.19
Liger-kernel 0.8.4	506.18	0.75	507.08	14.95
8K token 下，Chunked cuBLAS 相比 Full Logits 将峰值显存降低 81.6%，代价是更长的计算耗时；

16K token 下，Full Logits 出现 OOM，Chunked cuBLAS 与 Liger-kernel 均可运行：前者显存更低，后者速度更快。

3）Packing 优化
我们对比了 按长度 Packing 与按 Teacher Packing 两种策略的效果。

实验配置：8 个 Teacher，Batch Size=1024，Packing Length=128K，单卡最多缓存 1 个 Teacher LM head。

指标	按长度 Packing	按 Teacher Packing
microbatch 数量	29	29
平均 teacher / microbatch	4.52	1.21
单卡 head H2D 次数	205	17
head H2D 累计耗时	1.25 s	0.11 s
整体 KL 计算耗时	32.49 s	31.55 s
实验结果：两种策略均生成 29 个 micro-batch，但按 Teacher Packing 将每个 micro-batch 的平均 Teacher 数从 4.52 降至 1.21，LM head H2D 次数从 205 次降至 17 次，传输耗时减少约 91%。由于这部分加载时间占整体时间不大，对整体计算耗时的改善相对有限。

四、设计与实现参考
VeRL：本文实现所基于的训练框架，其 OPD 文档介绍了蒸馏训练流程、Teacher 资源管理与多 Teacher 路由。
DeepSeek-V4 技术报告，§5.2.2：介绍了 teacher hidden-state 缓存、训练时重建 logits、多 teacher head 调度，以及异步加载与卸载。
NeMo RL Full-vocabulary MOPD：提供 hidden-state payload、分块 KL 和 teacher head 生命周期管理。
Liger-Kernel：提供从 student/teacher hidden states 与 head 出发，分块完成 projection、蒸馏 loss 和 student 梯度计算的实现思路。
