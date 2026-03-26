1. 怎么做那个专家moe的负载均衡?
2. 位置编码细讲一下
3. mha,gqa,mla细讲
4. qwen3 论文主要更新优化点是什么
5. DeepSeek的这些论文挑一篇论文给我大概讲解一下吧
6. Transformer框架原理
7. 最近看了什么论文,稍微讲一下

---

## 优化点

* RoPE, AliBi ==> Position Encoding 
* RMSNorm ==> LayerNorm
* 稀疏注意力 (Sparse Attention) ==>  Self-Attention
* Grouped-Query Attention (GQA) ==> Multi-Head Attention (MHA)
* 深度前馈网络 (Deep FFN)  ==> 传统的 FFN
* MoE 通过多个子网络（专家）并行处理，每步动态选择部分专家计算 ==>  FFN
* SwiGLU/GeGLU ==>  ReLU


## 1. 怎么做专家moe的负载均衡?

每个专家都是如此:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x):
        return self.net(x)
```

* **专家层 (Experts)：** 通常是多个结构相同的 FFN（前馈神经网络）。在训练后，这些专家会自发产生分工（比如专家 A 擅长数学，专家 B 擅长代码）。
* **门控网络/路由 (Router/Gating Network)：** 这是 MoE 的“大脑”。它接收输入 Token，通过计算（通常是简单的线性映射 + Softmax），决定将这个 Token 发送给哪几个专家。
* **路由策略 (Routing Strategy)：** 最常用的是 **Top-K 路由**。例如 Top-2 路由意味着每个 Token 只会被分配给得分最高的 2 个专家处理。


#### 辅助损失函数 (Auxiliary Loss) —— 最经典
这是 DeepSeek、Switch Transformer 等模型常用的方案。在主损失（如预测下一个词的准确率）之外，增加一个惩罚项：
* **原理：** 监控每个专家的“重要性”和“负载频率”。如果某个专家被选中的比例远超平均值，Loss 就会变大，迫使 Router 将流量分给其他专家。
* **公式示意：** $$L_{aux} = \alpha \cdot N \sum_{i=1}^N f_i \cdot P_i$$
    * $f_i$ 是分配到专家 $i$ 的 Token 比例。
    * $P_i$ 是路由给专家 $i$ 的概率权重之和。
    * 通过最小化这个乘积，可以促使 $f_i$ 和 $P_i$ 趋于均匀分布。

#### 2. 无损负载均衡 (Auxiliary-Loss-Free Load Balancing) —— 最新趋势
DeepSeek-V3 等模型提出了一种更优雅的方法。辅助 Loss 有个缺点：它会干扰主任务的学习。
* **原理：** 给每个专家设置一个**偏置值（Bias）**。
* **操作：** 如果专家 A 负载太高，系统就调低它的偏置值；负载太低，就调高。在 Router 做 Top-K 选择时，用“路由分数 + 偏置值”来选人。
* **好处：** 这种调整发生在训练过程中，但不直接改变模型权重，避免了干扰梯度，模型性能更好。

#### 3. 专家容量限制 (Expert Capacity)
* **原理：** 强行规定每个专家在一个 Batch 中最多只能处理 $C$ 个 Token。
* **结果：** 如果专家 A 的名额满了，即便它最合适，多出来的 Token 也会被丢弃或强制分给排名第二、第三的专家。

#### 4. 噪声路由 (Noisy Top-K Gating)
* **原理：** 在路由分数中加入随机噪声。
* **作用：** 尤其在训练初期，给那些“表现一般”的专家一点露脸的机会，防止“马太效应”导致的强者恒强。

#### 5. 例子

假设一个 Token（比如单词 "Bank"）进入了模型。Router 认为专家 1（擅长金融）和专家 2（擅长地理）最合适。

1.  **专家 1** 输出一个向量：$E_1(x)$
2.  **专家 2** 输出一个向量：$E_2(x)$
3.  **Router** 给出两个权重（得分）：比如专家 1 拿到了 **0.7**，专家 2 拿到了 **0.3**。

**最终的输出结果是这两个向量的“加权平均值”：**
$$Output = 0.7 \cdot E_1(x) + 0.3 \cdot E_2(x)$$

## 2. 位置编码

- 正弦余弦位置编码公式，：

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\end{aligned}
$$

其中：
*   $pos$ 表示词在序列中的位置。
*   $i$ 表示维度索引，范围为 $0 \le i < d_{model}/2$。
*   $d_{model}$ 为模型的隐藏层维度（在您提到的场景中为 512）。

RoPE（Rotary Positional Embedding）

传统的 Positional Encoding 中提出的基于正弦和余弦函数的绝对位置编码）有以下局限性：
- **缺乏相对位置信息**：绝对位置编码为每个 token 的位置分配一个固定的向量，无法直接捕捉 token 之间的相对位置关系，而相对位置在 NLP 任务中非常重要。
- **序列长度限制**：正弦/余弦编码在训练时对序列长度有固定假设，难以泛化到更长的序列（例如，训练时最大长度为 2K，推理时处理 65K 会表现不佳）。
- **计算复杂性**：对于长序列，传统编码可能需要额外的计算或存储。


#### **RoPE 的工作原理**
- **核心思想**：RoPE 将 token 的嵌入向量（query 和 key 向量）视为高维空间中的点，通过位置相关的旋转矩阵对其进行变换。旋转角度依赖于 token 的绝对位置，但计算注意力分数时，RoPE 能自然表达 token 之间的相对位置。
- **数学表达**：
  
  - 对于一个 token 在位置 $m $ 的嵌入向量 $x_m = [x_m^{(1)}, x_m^{(2)}, \dots, x_m^{(d)}] $，RoPE 将其分为 $d/2 $ 个二维平面，每对特征 $(x_m^{(2i-1)}, x_m^{(2i)}) $ 通过旋转矩阵 $R_{\Theta, m} $ 变换：
    $$
    \text{RoPE}(x_m^{(2i-1)}, x_m^{(2i)}, m) = \begin{pmatrix}
    \cos(m\theta_i) & -\sin(m\theta_i) \\
    \sin(m\theta_i) & \cos(m\theta_i)
    \end{pmatrix} \begin{pmatrix}
    x_m^{(2i-1)} \\
    x_m^{(2i)}
    \end{pmatrix}
    $$
    其中，$\theta_i = 10000^{-2i/d} $ 是频率参数，控制不同维度的旋转速度。

  - 在自注意力机制中，query 和 key 向量经过 RoPE 变换后，点积注意力分数会自然依赖于相对位置 $m - n $：
    $$
    \langle \text{RoPE}(q_m, m), \text{RoPE}(k_n, n) \rangle = \text{function}(m - n)
    $$
    这使得 RoPE 能有效捕捉相对位置信息。


## 3. mha,mqa,gqa,mla细讲
(注意力机制:mha,mqa,gqa,mla细讲,最好配合ascll图和公式和代码)

点积注意力（Scaled Dot-Product Attention）：
$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$


- MHA
```text
[MHA: 每个 Q 对应独立的 K 和 V]

Head 1:  Q1 ────────> K1, V1
Head 2:  Q2 ────────> K2, V2
Head 3:  Q3 ────────> K3, V3
Head 4:  Q4 ────────> K4, V4
```

- MQA

为了解决 MHA 的 KV Cache 过大问题，MQA 提出：所有的 Q 头共享同一个 K 头和同一个 V 头。

```text
[MQA: 所有 Q 共享一组 K 和 V]

Head 1:  Q1 ─┐
Head 2:  Q2 ─┼──────> K_shared, V_shared
Head 3:  Q3 ─┼
Head 4:  Q4 ─┘
```
- GQA

MHA 性能好但太耗显存，MQA 省显存但性能差。

```text
[GQA: 组内共享 K 和 V]

Group 1: 
  Head 1:  Q1 ─┐
  Head 2:  Q2 ─┴────> K_group1, V_group1

Group 2:
  Head 3:  Q3 ─┐
  Head 4:  Q4 ─┴────> K_group2, V_group2
```

- MLA

MLA 是通过**将庞大的 KV 信息联合压缩成一个极低维度的“隐向量 (Latent Vector) $c_t$”**。推理时，显存里**只存这个低维的 $c_t$**，计算 Attention 时再将 $c_t$ 动态解压缩（投影）回各个头的 K 和 V。

```text
[MLA: 缓存压缩的 Latent，计算时动态投影解压]

                     ┌──> 动态解压(W_UK) ──> 多头 K (K1, K2... Kh)
Input X ──> 投影压缩 ─┼
           (W_DKV)   └──> 动态解压(W_UV) ──> 多头 V (V1, V2... Vh)
                     
       ▲
       │
【只有这个中间的 c_t 被保存在 KV Cache 中，体积极小！】
```

> 多头变体
> ![](/using_files/img/article/MHA.png)
> ![](/using_files/img/article/MQA.png)
> ![](/using_files/img/article/GQA.png)
> 稀疏注意力
> ![](/using_files/img/article/sparse_attention1.png)
> ![](/using_files/img/article/sparse_attention2.png)
> ![](/using_files/img/article/sparse_attention3.png)

## 6. Transformer框架原理
xx

## 7. 最近看了什么论文,稍微讲一下


