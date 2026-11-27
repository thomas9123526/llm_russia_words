# Chapter 32. Parameter-Efficient Fine-tuning: LoRA and QLoRA

> **Part 7 — Adapting and Aligning Models** · Level: Advanced
>
> **In this chapter you will learn:** why full fine-tuning is expensive and why it is often unnecessary · how LoRA adds a trainable low-rank update to frozen weights · how QLoRA combines LoRA with 4-bit quantization to fine-tune large models on a single GPU

## Do we really need to change all the weights?

In Chapter 31 we fine-tuned every parameter of a model. Consider what that means for a model with 7 billion parameters. With Adam in mixed precision (Chapter 28), each parameter needs roughly 16 bytes: the weight and its gradient in 16-bit format, plus a 32-bit master copy and two 32-bit Adam states. That is about $7 \cdot 10^9 \cdot 16 \approx 112$ GB before we even count activations, which is more than one ordinary GPU has. And if you fine-tune for ten different tasks, you have to store ten full copies of the model.

Yet the change we want is usually small. We are not teaching the model Russian from scratch; we are teaching it a format, a style, or a new set of labels. Experiments on the **intrinsic dimension** of fine-tuning (Aghajanyan et al.) found that many tasks can be learned by optimising a surprisingly small number of parameters, projected into the full weight space by a fixed random matrix. This suggests a hypothesis: the *update* $\Delta W$ that fine-tuning applies to a weight matrix has a low "effective rank". **Parameter-efficient fine-tuning** (PEFT) methods exploit this: they freeze the pre-trained model and train only a small number of extra parameters.

## Low-rank matrices in one minute

Recall from Chapter 6 that the **rank** of a matrix is the number of linearly independent rows (or columns). A matrix $M \in \mathbb{R}^{d \times k}$ of rank $r$ can always be written as a product

$$M = B A, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k}.$$

If $r$ is much smaller than $d$ and $k$, the factors are much cheaper to store than $M$ itself: $r(d + k)$ numbers instead of $dk$. For example, with $d = k = 4096$ and $r = 8$:

- full matrix: $4096 \cdot 4096 = 16{,}777{,}216$ numbers;
- low-rank factors: $8 \cdot (4096 + 4096) = 65{,}536$ numbers, which is about $0.39\%$.

A small worked example makes the idea concrete. Take $B = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ and $A = \begin{pmatrix} 1 & 0 & -1 \end{pmatrix}$. Their product is a $3 \times 3$ matrix whose rows are all multiples of $(1, 0, -1)$:

$$BA = \begin{pmatrix} 1 & 0 & -1 \\ 2 & 0 & -2 \\ 3 & 0 & -3 \end{pmatrix}.$$

Nine numbers, but only six were needed to describe them, and the matrix has rank 1. Low-rank matrices can only express "simple" changes, but the LoRA hypothesis is that simple changes are enough for fine-tuning.

## LoRA: Low-Rank Adaptation

**LoRA** (Hu et al., 2021) applies this idea to each chosen weight matrix of the transformer. Let $W_0 \in \mathbb{R}^{d \times k}$ be a frozen pre-trained weight (for example, the query projection $W_Q$ from Chapter 15). Instead of learning a full update, LoRA learns two small matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ and computes

$$h = W_0 x + \frac{\alpha}{r} B A x,$$

where $x \in \mathbb{R}^k$ is the input, $h \in \mathbb{R}^d$ is the output, $r$ is the **rank** of the adapter (typical values are 4 to 64), and $\alpha$ is a constant scaling factor. Dividing by $r$ means that when you change $r$, you do not have to retune the learning rate as much.

Three details matter:

1. **Initialisation.** $A$ is initialised with small random values and $B$ with zeros. Therefore $BA = 0$ at the start, and the model behaves *exactly* like the pre-trained model. Training starts from the pre-trained solution, as in ordinary fine-tuning.
2. **Only $A$ and $B$ are trained.** $W_0$ receives no gradient, so we need no optimizer state for it. Memory for gradients and Adam states shrinks by orders of magnitude.
3. **Merging.** After training, we can compute $W = W_0 + \frac{\alpha}{r} BA$ once and replace $W_0$ with it. The merged model has the same shape and the same speed as the original: LoRA adds **no inference latency**. Alternatively, we keep one base model and many small adapters, one per task, and switch between them.

In the original paper, LoRA was applied to the attention projections (especially $W_Q$ and $W_V$). In practice, applying it to all linear layers, including the feed-forward network (Chapter 18), often works better.

Here is a complete LoRA layer in PyTorch:

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():          # freeze W0 and its bias
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # B = 0 at start
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    def merge(self):                              # W = W0 + (alpha/r) * B A
        return self.base.weight + self.scale * self.B @ self.A

torch.manual_seed(0)
base = nn.Linear(512, 512)
lora = LoRALinear(base, r=8)
x = torch.randn(4, 512)
print(torch.allclose(lora(x), base(x)))           # True: identical at initialisation
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(trainable, base.weight.numel())              # 8192 262144  (about 3%)
with torch.no_grad():
    lora.B.normal_()                               # pretend B has been trained
    merged = lora.merge()
print(torch.allclose(lora(x), x @ merged.T + base.bias, atol=1e-4))  # True
```

The last line checks the merging property: the adapter path and the single merged matrix give the same output. For the $512 \times 512$ layer the adapter has $8 \cdot (512 + 512) = 8192$ parameters, about $3\%$ of the $262{,}144$ weights. For larger layers the fraction is even smaller, as in the $4096 \times 4096$ example above.

## QLoRA: fine-tuning a quantized model

LoRA removes most of the memory for gradients and optimizer states, but the frozen base weights still have to be stored. In 16-bit format a 7-billion-parameter model takes about 14 GB, and larger models need much more. **QLoRA** (Dettmers et al., 2023) solves this by storing the frozen base model in 4 bits and training LoRA adapters on top of it. The adapters and the computation stay in 16-bit precision (for example bfloat16); the 4-bit weights are **dequantized** on the fly, block by block, when they are needed in a matrix multiplication. Gradients flow *through* the quantized weights to the adapters, but the quantized weights themselves are never updated.

**Quantization** means representing numbers with fewer bits; we will study it in detail for inference in Chapter 36. The simplest scheme is *absmax* quantization in blocks: split the weights into blocks of, say, 64 numbers, find the largest absolute value $m$ in each block, and map every weight $w$ to an integer $q = \mathrm{round}(7 \cdot w / m)$ in the range $[-7, 7]$. To dequantize, compute $\hat w = q \cdot m / 7$. Only the small integers and one scale per block are stored.

```python
import torch

torch.manual_seed(0)
w = torch.randn(4096)                           # weights of one layer (flattened)
blocks = w.view(-1, 64)                         # blocks of 64 weights
m = blocks.abs().max(dim=1, keepdim=True).values  # one scale per block
q = torch.round(7 * blocks / m).to(torch.int8)  # 4-bit range: integers -7..7
w_hat = (q.float() * m / 7).view(-1)            # dequantize

print(q.min().item(), q.max().item())           # -7 7
print(f"mean abs error: {(w - w_hat).abs().mean():.3f}")  # about 0.094
```

(For simplicity the integers are kept in `int8`; a real implementation packs two 4-bit values into each byte.) QLoRA introduced three refinements of this idea:

- **4-bit NormalFloat (NF4).** Pre-trained weights are roughly normally distributed, so instead of 16 equally spaced levels, NF4 places its levels at quantiles of a normal distribution. Each level is then used about equally often, which reduces the error.
- **Double quantization.** The per-block scales $m$ are themselves quantized, saving a further fraction of a bit per parameter.
- **Paged optimizers.** Optimizer states can be moved automatically between GPU and CPU memory to survive short memory spikes, for example on very long sequences.

With these techniques the QLoRA authors fine-tuned a 65-billion-parameter model on a single 48 GB GPU, with quality close to 16-bit full fine-tuning in their experiments. For a PhD student with limited hardware, this is often the most practical way to adapt a large model.

## Other PEFT methods and practical advice

LoRA is the most popular PEFT method, but not the only one:

- **Adapters** (Houlsby et al., 2019) insert small bottleneck networks (down-projection, nonlinearity, up-projection) inside each transformer block. Unlike LoRA, they cannot be merged and add a little latency.
- **Prefix tuning** and **prompt tuning** learn a few continuous vectors ("virtual tokens") that are prepended to the keys and values or to the input embeddings, while the whole model stays frozen.
- **BitFit** trains only the bias terms.

Practical guidelines for LoRA:

- Start with $r = 8$ or $16$ and $\alpha = 2r$; increase $r$ if the task is very different from pre-training.
- Apply LoRA to all linear layers of attention and the FFN rather than only to $W_Q$ and $W_V$.
- Use a larger learning rate than for full fine-tuning (for example $10^{-4}$ to $2 \cdot 10^{-4}$), since only the small matrices are trained and $B$ starts at zero.
- Because the base model is frozen, LoRA tends to forget less than full fine-tuning (Chapter 31), but it may also learn less on tasks that require large changes.

## Summary

- Full fine-tuning of large models needs a lot of memory (weights, gradients, optimizer states) and a full copy per task.
- The update that fine-tuning makes is often well approximated by a low-rank matrix; PEFT methods train only a small number of parameters.
- LoRA computes $h = W_0 x + \frac{\alpha}{r} BAx$ with frozen $W_0$, zero-initialised $B$, and trainable $A, B$; after training the update can be merged, so inference is not slowed down.
- QLoRA stores the frozen base model in 4-bit NF4 with double quantization, trains 16-bit LoRA adapters, and uses paged optimizers, making it possible to fine-tune very large models on one GPU.
- Other PEFT methods include adapters, prefix and prompt tuning, and BitFit.

## Exercises

1. Explain in your own words why $B$ is initialised to zero in LoRA. What would go wrong if both $A$ and $B$ were random?
2. A transformer has 32 layers. In each layer you apply LoRA with $r = 16$ to four $4096 \times 4096$ attention matrices. How many trainable parameters does that give in total?
3. Estimate the memory needed to store the weights of a 13-billion-parameter model in 16-bit format and in 4-bit format (ignore the block scales).
4. In the quantization code, change the block size from 64 to 1024 and compare the mean absolute error. Explain the result.

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| parameter-efficient fine-tuning (PEFT) | параметрически эффективное дообучение | 매개변수 효율적 미세 조정 |
| intrinsic dimension | внутренняя размерность | 내재 차원 |
| rank | ранг | 랭크(계수) |
| low-rank matrix | матрица низкого ранга | 저랭크 행렬 |
| LoRA (low-rank adaptation) | LoRA, низкоранговая адаптация | LoRA, 저랭크 적응 |
| adapter | адаптер | 어댑터 |
| scaling factor | масштабный коэффициент | 스케일링 계수 |
| weight merging | слияние весов | 가중치 병합 |
| trainable parameters | обучаемые параметры | 학습 가능한 파라미터 |
| quantization | квантование | 양자화 |
| dequantization | деквантование | 역양자화 |
| 4-bit NormalFloat (NF4) | 4-битный NormalFloat (NF4) | 4비트 NormalFloat(NF4) |
| double quantization | двойное квантование | 이중 양자화 |
| paged optimizer | страничный оптимизатор | 페이지드 옵티마이저 |
| prefix tuning / prompt tuning | префиксная настройка / настройка промптов | 프리픽스 튜닝 / 프롬프트 튜닝 |

---
[← Previous: Fine-tuning and Transfer Learning](31-fine-tuning.md) · [Contents](README.md) · [Next: Instruction Tuning →](33-instruction-tuning.md)
