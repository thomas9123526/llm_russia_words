# Chapter 29. Mixture of Experts

> **Part 6 — Training at Scale** · Level: Advanced
>
> **In this chapter you will learn:** how a Mixture-of-Experts layer separates the number of parameters from the compute per token · how a router chooses the top-$k$ experts and how their outputs are combined · why load balancing is necessary and what MoE costs in memory and communication

Consider three tokens that a language model might process in the same batch: a digit in "2 + 2 =", a keyword in a line of Python code, and a word in a Russian poem. In a dense Transformer (Chapter 18) all three pass through exactly the same feed-forward weights, and every parameter takes part in every computation. Is that necessary? A human expert team would send each question to the person who knows that topic best.

This is the idea of **conditional computation**: let each token use only a part of the network. The most successful form of it in LLMs is the **Mixture of Experts (MoE)**. In Chapter 27 we saw that a dense model's training cost is $C \approx 6ND$ — compute is tied to the number of parameters. MoE breaks this link: a model can have many more parameters than it uses for any single token.

## Experts and a router

An MoE Transformer keeps attention as it is, but replaces the feed-forward network (FFN) in some or all blocks with an MoE layer. The layer contains:

- $n$ **experts** $E_1, \dots, E_n$ — usually ordinary FFNs of the same shape as the one they replace;
- a **router** (also called a **gating network**) — a small linear layer that decides which experts each token should visit.

For each token, only $k$ of the $n$ experts are computed, with typical values like $k = 1$ or $k = 2$ and $n$ from 8 to a few hundred. So the layer holds $n$ times the FFN parameters but costs only about $k$ times the FFN compute per token. We therefore distinguish **total parameters** (everything stored) from **active parameters** (what is used per token). A model built this way is called a **sparse model**, in contrast with a **dense model**, where all parameters are active.

The idea is old. Jacobs, Jordan, Nowlan and Hinton proposed "adaptive mixtures of local experts" in 1991. Shazeer et al. (2017) made it work at large scale with a sparsely gated MoE layer between LSTM layers. GShard and the **Switch Transformer** (Fedus et al., 2021, with $k = 1$) brought it into Transformers. A well-known open example is **Mixtral 8x7B** (Mistral AI): 8 experts per layer and top-2 routing, about 47B total parameters but only about 13B active per token — fewer than the name suggests, because attention and embeddings are shared and are not multiplied by 8.

## How routing works

Let $x \in \mathbb{R}^d$ be the hidden vector of one token, and let $W_r \in \mathbb{R}^{d \times n}$ be the router's weight matrix. The router computes one score per expert and turns the scores into probabilities with softmax:

$$
p = \mathrm{softmax}(x W_r), \qquad p \in \mathbb{R}^n, \quad \sum_{i=1}^{n} p_i = 1 .
$$

Let $T$ be the set of the $k$ experts with the largest $p_i$. The output of the layer is the weighted sum of the chosen experts' outputs:

$$
y = \sum_{i \in T} g_i \, E_i(x), \qquad g_i = \frac{p_i}{\sum_{j \in T} p_j} .
$$

The weights $g_i$ are the router probabilities renormalized over the chosen experts, so they sum to 1. Experts outside $T$ are not computed at all — this is where the saving comes from.

**A worked example.** Take $n = 4$ experts, $k = 2$, and router scores $x W_r = (2.0,\ 1.0,\ 0.5,\ -1.0)$. Softmax gives

$$
p \approx (0.61,\ 0.22,\ 0.14,\ 0.03).
$$

The top two are experts 1 and 2. Renormalizing, $g_1 = 0.61 / 0.83 \approx 0.73$ and $g_2 \approx 0.27$. So $y = 0.73\,E_1(x) + 0.27\,E_2(x)$, and experts 3 and 4 do no work for this token.

How does the router learn? The choice of $T$ is discrete and has no gradient, but the weights $g_i$ depend smoothly on $W_r$. If expert 1 gives a useful output, the gradient of the loss increases $g_1$ through $W_r$, and similar tokens become more likely to choose expert 1 in the future. The router and the experts are trained together, end to end, by ordinary backpropagation.

## Keeping the experts balanced

There is a danger in this feedback loop. Early in training, some expert is slightly better by chance; the router sends it more tokens; it trains faster and becomes even better; soon most tokens go to a few experts while the others are idle. This **collapse** wastes parameters, and on many GPUs it also creates a traffic jam, because the overloaded experts become the bottleneck.

The standard remedy is an **auxiliary loss** that rewards **load balancing**. In the Switch Transformer form, for a batch of $B$ tokens,

$$
\mathcal{L}_{\text{aux}} = n \sum_{i=1}^{n} f_i \, P_i ,
$$

where $f_i$ is the fraction of tokens whose first choice is expert $i$, and $P_i$ is the average router probability for expert $i$ over the batch. If routing is perfectly uniform, $f_i = P_i = 1/n$ and $\mathcal{L}_{\text{aux}} = n \cdot n \cdot \frac{1}{n^2} = 1$, its minimum value. If one expert takes everything, $f$ and $P$ concentrate on it and the loss rises towards $n$. The counts $f_i$ are not differentiable, but $P_i$ is, so the gradient pushes the probabilities of overloaded experts down. The auxiliary loss is added to the language modeling loss with a small coefficient (values around $10^{-2}$ are common).

Two further mechanisms are used in practice:

- **Expert capacity.** For efficient batched computation, each expert accepts at most $\text{capacity} = c \cdot B / n$ tokens per batch, where $c$ (the *capacity factor*, e.g. 1.25) gives some slack. Tokens beyond the capacity are **dropped** from that layer: they skip the expert and continue through the residual connection (Chapter 18) unchanged.
- **Router z-loss.** A small penalty on the size of the router logits (introduced in ST-MoE) keeps them from growing too large, which improves numerical stability — important in bf16 (Chapter 28).

Newer designs refine the experts themselves: for example, DeepSeekMoE splits experts into many smaller ones and adds **shared experts** that every token always uses, so that common knowledge does not have to be copied into every expert.

## A minimal MoE layer

The code below implements top-$k$ routing and the auxiliary loss. For clarity it loops over experts; real implementations group tokens by expert and use specialized kernels.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, d_model=16, d_ff=32, n_experts=4, k=2):
        super().__init__()
        self.k, self.n_experts = k, n_experts
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(n_experts)])

    def forward(self, x):                                  # x: (tokens, d_model)
        probs = F.softmax(self.router(x), dim=-1)          # (tokens, n_experts)
        top_p, top_i = probs.topk(self.k, dim=-1)          # k best experts per token
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)    # renormalize the k weights
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            tok, slot = (top_i == e).nonzero(as_tuple=True)  # tokens routed to expert e
            out[tok] += top_p[tok, slot, None] * self.experts[e](x[tok])
        # Load-balancing loss (Switch Transformer): n * sum_i f_i * P_i
        f = F.one_hot(top_i[:, 0], self.n_experts).float().mean(dim=0)  # top-1 shares
        P = probs.mean(dim=0)                              # mean router probabilities
        aux_loss = self.n_experts * (f * P).sum()
        return out, aux_loss

torch.manual_seed(0)
y, aux = MoE()(torch.randn(10, 16))
print(y.shape, f"aux_loss={aux.item():.3f}")   # aux_loss ≈ 1 when balanced
```

It prints `torch.Size([10, 16]) aux_loss=1.044`: the output has the same shape as the input, as an FFN replacement must, and the freshly initialized router is close to balanced. Each of the 4 experts has as many parameters as a dense FFN, but each token is processed by only 2 of them.

## What MoE costs

MoE is not free. Its main costs are:

- **Memory.** All $n$ experts must be stored, even though few are used per token. Mixtral needs memory for 47B parameters while computing like a 13B model. For inference on a single device, memory, not compute, is often the limit.
- **Communication.** With **expert parallelism**, different experts live on different GPUs. Each MoE layer then needs an **all-to-all** exchange: every GPU sends its tokens to the GPUs of their chosen experts and receives the results back. This adds to the parallelism dimensions of Chapter 28.
- **Training stability and fine-tuning.** Sparse models have historically been more prone to instabilities and to overfitting during fine-tuning (Chapter 31), which motivated tools such as the z-loss.

What do the experts learn? One might hope for "a physics expert" and "a Russian expert", but analyses (for example, in the Mixtral paper) found little clear specialization by topic; routing depends more on syntax and on token types, and consecutive tokens often go to the same expert. For a multilingual model — the subject of the next chapter — whether experts specialize by language is an interesting research question.

The reason for the popularity of MoE is the scaling picture of Chapter 27: for the same training compute, a sparse model with more total parameters usually reaches a lower loss than a dense model. It buys capacity with memory instead of FLOPs.

## Summary

- A Mixture-of-Experts layer replaces the FFN with $n$ expert FFNs and a router; each token uses only $k$ of them, so total parameters grow while compute per token stays almost constant.
- The router computes $p = \mathrm{softmax}(xW_r)$, selects the top-$k$ experts and combines their outputs with renormalized weights $g_i$; it is trained by backpropagation through these weights.
- Without balancing, routing collapses onto a few experts; the auxiliary loss $n \sum_i f_i P_i$, expert capacity and the router z-loss keep training efficient and stable.
- Examples include the Switch Transformer (top-1) and Mixtral 8x7B (8 experts, top-2, about 47B total and 13B active parameters).
- MoE trades compute for memory and communication: all experts must be stored, and expert parallelism requires all-to-all exchange.

## Exercises

1. Explain in your own words the difference between total and active parameters, and why MoE models are described with both numbers.
2. For router scores $(0.0,\ 1.5,\ 1.5,\ -0.5)$ and $k = 2$, compute the softmax probabilities, the chosen experts and their weights $g_i$.
3. Compute $\mathcal{L}_{\text{aux}}$ for $n = 4$ when all tokens choose expert 1 first and the average probabilities are $P = (0.7, 0.1, 0.1, 0.1)$. Compare with the balanced value.
4. Modify the code to count how many tokens each expert receives (for example, using `torch.bincount(top_i.flatten(), minlength=4)`). Run it on 1000 random tokens. Is the routing balanced at initialization?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| mixture of experts (MoE) | смесь экспертов | 전문가 혼합 |
| conditional computation | условные вычисления | 조건부 연산 |
| expert network | экспертная сеть | 전문가 네트워크 |
| router | маршрутизатор | 라우터 |
| gating network | управляющая сеть | 게이팅 네트워크 |
| top-k routing | маршрутизация top-k | top-k 라우팅 |
| sparse model | разреженная модель | 희소 모델 |
| dense model | плотная модель | 밀집 모델 |
| active parameters | активные параметры | 활성 매개변수 |
| load balancing | балансировка нагрузки | 부하 분산 |
| auxiliary loss | вспомогательная функция потерь | 보조 손실 |
| expert capacity | ёмкость эксперта | 전문가 용량 |
| token dropping | отбрасывание токенов | 토큰 드롭 |
| expert parallelism | экспертный параллелизм | 전문가 병렬화 |
| all-to-all communication | коммуникация all-to-all | 올투올 통신 |

---
[← Previous: Distributed Training and Mixed Precision](28-distributed-training.md) · [Contents](README.md) · [Next: Multilingual Models and Cross-Lingual Transfer →](30-multilingual-models.md)
