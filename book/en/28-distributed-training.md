# Chapter 28. Distributed Training and Mixed Precision

> **Part 6 — Training at Scale** · Level: Advanced
>
> **In this chapter you will learn:** where the memory of a training run goes and why one GPU is not enough · how fp16 and bf16 differ and how mixed precision training stays stable · how data, tensor and pipeline parallelism and ZeRO sharding split the work across many GPUs

Take a modest modern model with $N = 7 \cdot 10^9$ parameters and ask two simple questions. Does it fit into one GPU with 80 GB of memory for training? And how long would one GPU need to train it on $10^{12}$ tokens?

For the second question we can use the formula from Chapter 27: $C \approx 6ND = 6 \cdot 7 \cdot 10^9 \cdot 10^{12} = 4.2 \cdot 10^{22}$ FLOPs. At a realistic $3 \cdot 10^{14}$ FLOP/s, that is $1.4 \cdot 10^8$ seconds — about four and a half years. For the first question, we must count bytes. This chapter shows that the answer is "no" as well, and then explains the two families of techniques that solve both problems: using fewer bytes per number (**mixed precision**) and using many GPUs at once (**distributed training**).

## Where the memory goes

During training, a GPU must hold four kinds of data:

1. **Parameters** — the weights of the model.
2. **Gradients** — one number per parameter, computed by backpropagation (Chapter 9).
3. **Optimizer state** — for Adam, two moving averages $m$ and $v$ per parameter (Chapter 9).
4. **Activations** — the intermediate outputs of every layer, saved in the forward pass because the backward pass needs them.

In standard mixed precision training with Adam, the accounting per parameter is: 2 bytes for the bf16 (or fp16) weights, 2 bytes for the gradients, and 12 bytes for the optimizer: an fp32 "master" copy of the weights (4), $m$ (4) and $v$ (4). In total

$$
\text{memory} \approx 16 N \text{ bytes} \quad (\text{without activations}).
$$

For $N = 7 \cdot 10^9$ this is 112 GB — more than one 80 GB GPU, before we store a single activation. Activation memory grows with the batch size, the sequence length and the number of layers, and can be even larger. A standard way to reduce it is **activation checkpointing**: keep only the inputs of some layers, and recompute the rest during the backward pass. This costs about one extra forward pass (roughly a third more compute) and saves most of the activation memory — a typical trade of compute for memory.

## Mixed precision: fewer bits per number

A floating-point number has a sign, an exponent (which sets the *range*) and a mantissa (which sets the *precision*). Three formats matter for training:

| Format | Sign / exponent / mantissa bits | Largest value | Precision near 1 |
|---|---|---|---|
| fp32 | 1 / 8 / 23 | $\approx 3.4 \cdot 10^{38}$ | $2^{-23} \approx 1.2 \cdot 10^{-7}$ |
| fp16 | 1 / 5 / 10 | $65\,504$ | $2^{-10} \approx 9.8 \cdot 10^{-4}$ |
| bf16 | 1 / 8 / 7 | $\approx 3.4 \cdot 10^{38}$ | $2^{-7} \approx 7.8 \cdot 10^{-3}$ |

Half-precision formats halve the memory and, more importantly, run much faster on the matrix units of modern GPUs (tensor cores). But each has a weakness. **fp16** has good precision but a narrow range: very small numbers **underflow** to zero, and numbers above 65,504 **overflow** to infinity. **bf16** ("brain float") keeps the full 8-bit exponent of fp32, so its range is the same, but it has only about three significant decimal digits.

```python
import torch

x = torch.tensor([1e-8, 1e-3, 1.0001, 70000.0])
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    print(f"{str(dtype):15}", x.to(dtype).tolist())

# A tiny gradient underflows in fp16, but survives with loss scaling
g = torch.tensor(1e-8)
scale = 2.0**16
print("fp16 without scaling:", g.half().item())
print("fp16 with scaling:   ", (g * scale).half().item() / scale)
```

Output:

```text
torch.float32   [9.99999993922529e-09, 0.0010000000474974513, 1.000100016593933, 70000.0]
torch.float16   [0.0, 0.0010004043579101562, 1.0, inf]
torch.bfloat16  [1.0011717677116394e-08, 0.00099945068359375, 1.0, 70144.0]
fp16 without scaling: 0.0
fp16 with scaling:    9.997165761888027e-09
```

Even fp32 stores the numbers only approximately, but its errors are tiny. In the half formats every weakness is visible. In fp16, $10^{-8}$ becomes 0 and 70,000 becomes infinity. In bf16 both survive, but 70,000 is rounded to 70,144. In both half formats, $1.0001$ becomes exactly $1.0$: the difference is smaller than the spacing of representable numbers near 1.

The last observation explains the **master weights**. An update such as $w \leftarrow w - \eta g$ with $w = 1$ and $\eta g = 10^{-4}$ would be lost completely in bf16. So the optimizer keeps an fp32 copy of the weights and applies updates there; a bf16 copy is made for the forward and backward passes. Operations that are sensitive to rounding — sums over many elements, softmax, normalization, the loss — are also computed in fp32.

For fp16 there is one more trick, **loss scaling**. Gradients are often tiny and would underflow. We multiply the loss by a large factor $S$ (for example $2^{16}$) before backpropagation; by the chain rule all gradients are multiplied by $S$ and move into the representable range. Before the optimizer step, the gradients are divided by $S$ again in fp32. Dynamic loss scaling lowers $S$ whenever an overflow (an `inf` or `nan`) appears and slowly raises it otherwise. With bf16, whose range equals fp32, loss scaling is usually unnecessary — one reason why bf16 has become the default for LLM training on hardware that supports it.

## Data parallelism

Even with mixed precision, one GPU is too slow. The simplest way to use $p$ GPUs is **data parallelism**: every GPU holds a full copy of the model, receives a different part of the batch, and computes gradients on its part. The GPUs then average their gradients and all apply the same update, so the copies stay identical.

Why is this correct? The loss of a batch is a mean over examples, and the gradient of a mean is the mean of the gradients. If the batch is split into $p$ equal shards, the full-batch gradient is exactly the average of the shard gradients:

$$
\nabla_\theta \frac{1}{B}\sum_{i=1}^{B} \ell_i = \frac{1}{p}\sum_{k=1}^{p} \nabla_\theta \frac{1}{B/p}\sum_{i \in \text{shard}_k} \ell_i .
$$

The code checks this with four simulated workers:

```python
import torch
torch.manual_seed(0)

model = torch.nn.Linear(4, 1)
X, y = torch.randn(16, 4), torch.randn(16, 1)
loss_fn = torch.nn.MSELoss()

# 1) Gradient on the full batch of 16 examples (one device)
model.zero_grad()
loss_fn(model(X), y).backward()
full_grad = model.weight.grad.clone()

# 2) Data parallelism: 4 "GPUs", each with its own shard of 4 examples
shard_grads = []
for Xs, ys in zip(X.chunk(4), y.chunk(4)):
    model.zero_grad()
    loss_fn(model(Xs), ys).backward()
    shard_grads.append(model.weight.grad.clone())

# All-reduce = sum the gradients of all workers and divide by their number
avg_grad = torch.stack(shard_grads).mean(dim=0)
print(torch.allclose(full_grad, avg_grad, atol=1e-6))   # True
```

The averaging step is a collective operation called **all-reduce**: every worker contributes a tensor and every worker receives the sum (or mean). Efficient implementations such as *ring all-reduce* arrange the GPUs in a ring and pass chunks around it; each GPU sends and receives about $2\frac{p-1}{p}$ times the size of the gradients, which is almost independent of $p$. In PyTorch this is what `DistributedDataParallel` does, overlapping communication with the backward pass.

The same idea works on a single GPU in time instead of space: **gradient accumulation** runs several small micro-batches, adds their gradients, and makes one optimizer step, giving the effect of a larger batch.

## Sharding and model parallelism

Data parallelism speeds training up, but it does not solve the memory problem: every GPU still stores all $16N$ bytes. Three further ideas split the model itself.

**ZeRO** (Rajbhandari et al., 2020) observes that keeping the optimizer state, gradients and parameters on every data-parallel GPU is redundant. It **shards** them: with $p$ GPUs, each one stores and updates only $1/p$ of the optimizer state (stage 1), also of the gradients (stage 2), and also of the parameters (stage 3). Parameters needed for a layer are gathered just in time and freed afterwards. At stage 3 the memory per GPU is about $16N/p$: for our 7B model on 8 GPUs, $112 / 8 = 14$ GB. PyTorch's **FSDP** (Fully Sharded Data Parallel) implements the same idea.

**Tensor parallelism** (as in Megatron-LM) splits the matrices inside each layer. In the feed-forward block $Y = \mathrm{GELU}(XW_1)W_2$ (Chapter 18), $W_1$ is split by columns and $W_2$ by rows across GPUs; each GPU computes its part independently, and one all-reduce at the end sums the partial results. Attention heads are split in the same way. Tensor parallelism needs communication in every layer, so it is used inside one server, where GPUs are connected by a very fast interconnect.

**Pipeline parallelism** gives different layers to different GPUs: GPU 1 holds layers 1–8, GPU 2 layers 9–16, and so on. A naive pipeline would leave most GPUs idle while one works, so the batch is split into $m$ micro-batches that flow through the stages one after another. Idle time remains at the start and the end — the **pipeline bubble**. In the simple GPipe schedule with $p$ stages its share is

$$
\text{bubble fraction} = \frac{p - 1}{m + p - 1},
$$

so with $p = 4$ stages and $m = 16$ micro-batches about $3/19 \approx 16\%$ of the time is lost. More micro-batches make the bubble smaller.

Large runs combine all of these — data parallelism (often with ZeRO/FSDP) across servers, tensor parallelism inside a server, pipeline parallelism across groups of servers. This is called **3D parallelism**. Mixture-of-Experts models (Chapter 29) add a fourth dimension, expert parallelism.

At this scale, engineering details become research questions. Runs over thousands of GPUs must save **checkpoints** regularly, survive hardware failures, and deal with **loss spikes** — sudden jumps of the loss that may require restarting from a checkpoint with a lower learning rate or a different data order.

## Summary

- Mixed-precision Adam training needs about $16N$ bytes per model for weights, gradients and optimizer state, plus activation memory; activation checkpointing trades extra compute for less memory.
- fp16 has a narrow range (underflow and overflow) and needs loss scaling; bf16 has the range of fp32 but low precision; fp32 master weights keep small updates from being lost.
- Data parallelism replicates the model, splits the batch and averages gradients with all-reduce; this gives exactly the full-batch gradient.
- ZeRO/FSDP shards optimizer state, gradients and parameters across data-parallel GPUs, reducing memory to about $16N/p$.
- Tensor parallelism splits matrices inside layers, pipeline parallelism splits the layers into stages; together with data parallelism they form 3D parallelism.

## Exercises

1. Explain in your own words why bf16 usually does not need loss scaling while fp16 does.
2. Compute the memory (without activations) for training a 13B-parameter model with mixed-precision Adam: (a) on one GPU, (b) with ZeRO stage 3 on 16 GPUs.
3. In the data parallelism code, change the loss to `torch.nn.MSELoss(reduction="sum")`. Is `full_grad` still equal to the *mean* of the shard gradients? What must replace the mean?
4. For a pipeline with $p = 8$ stages, how many micro-batches are needed to keep the bubble fraction below 10%?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| distributed training | распределённое обучение | 분산 학습 |
| mixed precision | смешанная точность | 혼합 정밀도 |
| half precision | половинная точность | 반정밀도 |
| loss scaling | масштабирование функции потерь | 손실 스케일링 |
| optimizer state | состояние оптимизатора | 옵티마이저 상태 |
| activation checkpointing | пересчёт активаций | 활성화 체크포인팅 |
| data parallelism | параллелизм по данным | 데이터 병렬화 |
| all-reduce | операция all-reduce | 올리듀스 |
| gradient accumulation | накопление градиентов | 기울기 누적 |
| sharding | шардирование | 샤딩 |
| tensor parallelism | тензорный параллелизм | 텐서 병렬화 |
| pipeline parallelism | конвейерный параллелизм | 파이프라인 병렬화 |
| pipeline bubble | пузырь конвейера | 파이프라인 버블 |
| interconnect | межсоединение | 인터커넥트 |
| checkpoint | контрольная точка | 체크포인트 |

---
[← Previous: Scaling Laws and Compute-Optimal Training](27-scaling-laws.md) · [Contents](README.md) · [Next: Mixture of Experts →](29-mixture-of-experts.md)
