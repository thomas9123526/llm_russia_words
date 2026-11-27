# Chapter 36. Efficient Inference: KV Cache, Quantization, Speculative Decoding

> **Part 8 — LLMs in the Real World** · Level: Advanced
>
> **In this chapter you will learn:** why generation is slow and where the time goes · how the KV cache and quantization save compute and memory · how speculative decoding produces several tokens per step without changing the output distribution

## Why is generating text so slow?

Suppose you have trained a 7-billion-parameter model and ask it to write a 500-token answer. Training processed millions of tokens per second across many GPUs, yet your answer appears token by token, at perhaps a few dozen tokens per second. Why?

Recall from Chapter 24 that generation is **autoregressive**: to produce token $t+1$, the model needs tokens $1, \dots, t$. We cannot compute token 500 before token 499 exists. During training, all positions of a sequence are processed in parallel because the whole text is known in advance (Chapter 21). During generation, the sequence grows one step at a time, so we run the full network once for every new token.

It is useful to split inference into two phases:

- **Prefill.** The prompt is known, so all its tokens are processed in one parallel forward pass. This phase is limited mostly by arithmetic: large matrix–matrix multiplications that keep the GPU busy. It determines the **time to first token**.
- **Decoding.** Each new token needs one forward pass that handles only a single position. Now every weight matrix is multiplied by a single vector. The GPU must read all the weights from memory to do very little arithmetic with them.

The second point is the key to this chapter. A 7B model in 16-bit precision has about 14 GB of weights. To generate one token, the GPU must read all 14 GB from its memory. If the **memory bandwidth** is, say, 1 TB/s, that alone takes about 14 ms, which limits us to roughly 70 tokens per second, no matter how fast the arithmetic units are. We say decoding is **memory-bound**, not compute-bound.

This gives us three strategies, each a section below:

1. Do not recompute what we already computed (the KV cache).
2. Make the weights smaller, so there is less to read (quantization).
3. Get more than one token out of each expensive pass over the weights (speculative decoding, batching).

## The KV cache

Look at self-attention from Chapter 15. At step $t$, the new token's query $q_t$ is compared with the keys $k_1, \dots, k_t$ of all tokens so far, and the result is a weighted sum of their values $v_1, \dots, v_t$:

$$
o_t = \sum_{i=1}^{t} \operatorname{softmax}_i\!\left(\frac{q_t \cdot k_i}{\sqrt{d}}\right) v_i
$$

Here $d$ is the head dimension. In a decoder with a causal mask (Chapter 16), the key and value of token $i$ depend only on tokens $1, \dots, i$. They never change when later tokens arrive. So it is wasteful to recompute $k_1, \dots, k_{t-1}$ and $v_1, \dots, v_{t-1}$ at every step. We compute them once, store them, and at step $t$ compute only $q_t, k_t, v_t$ for the new token. This store is the **KV cache**.

The code below checks, on a single attention head, that the cached computation gives exactly the same outputs as full recomputation:

```python
import torch

torch.manual_seed(0)
d, T = 8, 5                       # embedding size, number of tokens
x = torch.randn(T, d)             # hidden states of 5 tokens
Wq, Wk, Wv = (torch.randn(d, d) for _ in range(3))

def attend(q, K, V):
    w = torch.softmax(q @ K.T / d**0.5, dim=-1)
    return w @ V

# Without a cache: recompute K and V for the whole prefix at every step
full = [attend(x[t] @ Wq, x[:t+1] @ Wk, x[:t+1] @ Wv) for t in range(T)]

# With a KV cache: compute K and V only for the new token and append them
K_cache, V_cache, cached = [], [], []
for t in range(T):
    K_cache.append(x[t] @ Wk)
    V_cache.append(x[t] @ Wv)
    K, V = torch.stack(K_cache), torch.stack(V_cache)
    cached.append(attend(x[t] @ Wq, K, V))

print(torch.allclose(torch.stack(full), torch.stack(cached), atol=1e-6))  # True
```

The cache turns the cost of producing each new token's keys and values from $O(t)$ into $O(1)$. But it is not free: it costs memory. For every layer, every key–value head and every token we store two vectors (a key and a value). The size is

$$
\text{KV memory} = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot T \cdot b,
$$

where $L$ is the number of layers, $H_{kv}$ the number of key–value heads, $d_{head}$ the head dimension, $T$ the number of tokens, and $b$ the bytes per number. Take a model with $L = 32$, $H_{kv} = 8$, $d_{head} = 128$, 16-bit numbers ($b = 2$) and a context of $T = 4096$ tokens:

$$
2 \cdot 32 \cdot 8 \cdot 128 \cdot 4096 \cdot 2 = 536{,}870{,}912 \text{ bytes} = 512 \text{ MiB}.
$$

That is 128 KiB per token, for one sequence. With 32 users at once, the cache needs 16 GiB, more than the weights of a quantized 7B model. Now you can see why grouped-query attention (Chapter 25) matters: with 32 key–value heads instead of 8, the cache would be four times larger. Long-context serving is often limited by KV-cache memory, not by weights.

Serving systems manage this memory carefully. **PagedAttention**, introduced in the vLLM system, stores the cache in small fixed-size blocks, like pages in an operating system, so that memory is not wasted on reserved but unused space. **Continuous batching** adds new requests to the running batch as soon as other requests finish. Batching helps because of the memory-bound nature of decoding: one read of the weights can serve many sequences at once, so the total **throughput** grows almost for free, even though the **latency** of each request stays the same or grows slightly.

## Quantization

The second strategy is to store numbers with fewer bits. **Quantization** maps a floating-point weight $w$ to a small integer $q$ using a scale factor $s$:

$$
q = \operatorname{round}\!\left(\frac{w}{s}\right), \qquad \hat{w} = s \cdot q, \qquad s = \frac{\max |w|}{127}.
$$

This is **absmax** quantization to 8-bit integers (INT8): the largest absolute value is mapped to 127, and $\hat{w}$ is the reconstructed (dequantized) weight. The difference $w - \hat{w}$ is the rounding error. INT8 halves memory compared to 16 bits; INT4 divides it by four, so a 7B model fits into about 3.5 GB.

```python
import numpy as np

rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, size=(256, 256)).astype(np.float32)  # a weight matrix

def quantize_int8(w):
    scale = np.abs(w).max() / 127          # absmax: largest value maps to 127
    q = np.round(w / scale).astype(np.int8)
    return q, scale

def dequantize(q, scale):
    return q.astype(np.float32) * scale

q, s = quantize_int8(W)
W_hat = dequantize(q, s)
print("bytes fp32:", W.nbytes, " bytes int8:", q.nbytes)   # 262144 vs 65536
print("mean abs error:", np.abs(W - W_hat).mean())

W[0, 0] = 1.0                                # one outlier, 50x larger than typical
q, s = quantize_int8(W)
print("error with outlier:", np.abs(W - dequantize(q, s)).mean())
```

The last three lines show the main difficulty. A single **outlier** sets the scale for the whole matrix, so all the ordinary small weights are squeezed into a few integer levels, and the average error grows about tenfold. Real LLMs do have such outliers, especially in activations. Practical methods fight this in several ways:

- **Finer granularity.** Use a separate scale for each row, column or small group of, say, 64 or 128 weights, so that one outlier affects only its group.
- **Keep outliers in high precision.** LLM.int8() handles the few outlier feature dimensions in 16 bits and the rest in 8 bits.
- **Error-aware rounding.** GPTQ quantizes weights one column at a time and adjusts the remaining weights to compensate for the error, using a small calibration dataset. AWQ observes which weights meet large activations and protects them by rescaling.

These are **post-training quantization** methods: the model is trained in high precision and compressed afterwards, without retraining. In Chapter 32 you saw QLoRA, which keeps a 4-bit base model frozen and trains LoRA adapters on top of it. Quantization reduces both the memory footprint and, because decoding is memory-bound, the time per token. The price is some loss of quality, usually small at 8 bits and noticeable at 3 bits or fewer. The KV cache can be quantized too.

## Speculative decoding

The third strategy attacks the sequential bottleneck itself. The idea: a small, fast **draft model** guesses the next $k$ tokens, and the large **target model** checks all $k$ guesses in a single forward pass. Checking $k$ tokens in parallel costs about as much as generating one, because decoding is memory-bound: the weights are read once either way.

How do we check? Let $q(x)$ be the draft model's probability for token $x$ and $p(x)$ the target model's. For each proposed token, going from left to right:

1. Accept the token with probability $\min\!\left(1, \dfrac{p(x)}{q(x)}\right)$.
2. If it is rejected, sample a replacement from the corrected distribution $\propto \max(0, p(x) - q(x))$, and discard all later draft tokens.
3. If all $k$ tokens are accepted, the target pass also gives us one extra token for free.

This rule, a form of rejection sampling, has a remarkable property, proved in the original papers (Leviathan et al.; Chen et al., both 2023): the accepted tokens are distributed **exactly** as if they were sampled from the target model alone. Speculative decoding is not an approximation; it changes only the speed.

How much faster is it? If each draft token is accepted with probability $\alpha$ (the **acceptance rate**), the expected number of tokens produced per target pass is

$$
\frac{1 - \alpha^{k+1}}{1 - \alpha}.
$$

With $\alpha = 0.8$ and $k = 4$: $(1 - 0.8^5)/(1 - 0.8) = (1 - 0.32768)/0.2 \approx 3.36$ tokens per pass instead of 1. The real speedup is smaller, because the draft model also takes time, but factors of two to three are common. The acceptance rate is high on predictable text (code, repeated phrases) and low on creative text. Variants avoid a separate draft model: Medusa adds extra prediction heads to the target model, and some methods draft from n-grams found in the prompt.

Other techniques complete the picture. **FlashAttention** computes attention in tiles that stay in fast on-chip memory, reducing memory traffic. **Knowledge distillation** trains a small model to imitate a large one. **Pruning** removes unimportant weights. Mixture-of-experts models (Chapter 29) read only a fraction of their weights per token. All of these share one goal: fewer bytes moved, fewer operations, more useful tokens per second.

## Summary

- Generation is autoregressive: prefill processes the prompt in parallel, while decoding produces one token per forward pass and is usually memory-bound.
- The KV cache stores keys and values of past tokens so they are never recomputed; its memory grows linearly with layers, KV heads, head size and context length.
- Paged memory management and continuous batching let a server use one read of the weights for many requests, raising throughput.
- Quantization stores weights (and sometimes activations or the KV cache) in 8 or 4 bits; outliers are the main source of error, handled by group-wise scales and error-aware methods like GPTQ and AWQ.
- Speculative decoding lets a draft model propose tokens and a target model verify them in one pass; the acceptance rule keeps the output distribution exactly the same.

## Exercises

1. Explain in your own words why decoding a single sequence is memory-bound, while the prefill phase is compute-bound.
2. Compute the KV-cache size for a model with $L = 40$, $H_{kv} = 8$, $d_{head} = 128$, 16-bit numbers and a context of 8192 tokens. How does it change if the cache is stored in INT8?
3. Modify the quantization code to use a separate scale for each row of `W` (hint: `np.abs(w).max(axis=1, keepdims=True)`). Compare the error with the outlier to the per-matrix version.
4. For $k = 4$, compute the expected tokens per target pass for $\alpha = 0.5$ and $\alpha = 0.9$. What does this tell you about choosing a draft model?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| inference | инференс / вывод модели | 추론 |
| prefill | предзаполнение | 프리필 |
| time to first token (TTFT) | время до первого токена | 첫 토큰까지의 시간 |
| memory bandwidth | пропускная способность памяти | 메모리 대역폭 |
| KV cache | KV-кэш | KV 캐시 |
| latency | задержка | 지연 시간 |
| throughput | пропускная способность | 처리량 |
| continuous batching | непрерывный батчинг | 연속 배칭 |
| quantization | квантование | 양자화 |
| outliers | выбросы | 이상치 |
| post-training quantization | квантование после обучения | 학습 후 양자화 |
| speculative decoding | спекулятивное декодирование | 추측 디코딩 |
| draft model | черновая модель | 초안 모델 |
| acceptance rate | доля принятых токенов | 수락률 |
| knowledge distillation | дистилляция знаний | 지식 증류 |

---
[← Previous: Reasoning and Chain-of-Thought](35-reasoning.md) · [Contents](README.md) · [Next: Retrieval-Augmented Generation (RAG) →](37-retrieval-augmented-generation.md)
