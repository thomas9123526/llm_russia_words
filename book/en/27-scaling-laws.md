# Chapter 27. Scaling Laws and Compute-Optimal Training

> **Part 6 — Training at Scale** · Level: Advanced
>
> **In this chapter you will learn:** why the loss of a language model falls as a power law in model size, data and compute · how to count training FLOPs with $C \approx 6ND$ · how the Chinchilla analysis splits a fixed compute budget between parameters and tokens

Suppose your laboratory gives you a fixed budget: a certain number of GPU-hours, and not one more. You could train a large model on relatively few tokens, or a smaller model on many more tokens. Both runs cost the same. Which one gives the better model?

A few years ago this question was answered by intuition and habit. Today it has a quantitative answer, based on **scaling laws**: empirical formulas that predict the loss of a model from its size, the amount of data and the compute used. Scaling laws are one of the most practical tools in LLM research, because they let us run cheap experiments with small models and extrapolate to expensive ones.

## Loss falls as a power law

Train a family of Transformers (Chapter 18) that differ only in size, each on enough data, and plot the final test loss — the cross-entropy per token from Chapter 8 — against the number of parameters $N$. Kaplan et al. (2020, "Scaling Laws for Neural Language Models") found that the points lie on a smooth curve of the form

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N},
$$

where $N_c$ is a constant and $\alpha_N$ is a small exponent; for their setup $\alpha_N \approx 0.076$. Similar laws hold for the dataset size $D$ (in tokens), when the model is large enough, and for the compute $C$, when $N$ and $D$ are chosen well.

A relation $y = a x^{-\alpha}$ is called a **power law**. Taking logarithms gives

$$
\log L = \alpha_N \log N_c - \alpha_N \log N,
$$

a straight line with slope $-\alpha_N$. That is why scaling results are always drawn on **log-log plots**: a power law becomes a line, and a line is easy to fit and to extend.

What does the exponent mean in practice? Doubling $N$ multiplies the loss by $2^{-0.076} \approx 0.949$, so each doubling removes about 5% of the loss. The gains never stop, but they become more and more expensive: to get the same improvement again, you must double the size again.

Two remarks keep this picture honest. First, these laws are empirical: they are fitted to experiments over a certain range and hold for a given architecture, tokenizer and data distribution. Changing the data changes the constants. Second, the loss cannot fall to zero. Natural text has inherent uncertainty — even a perfect model cannot know exactly what word comes next — so a realistic law includes a constant term, the **irreducible loss**, as we will see below.

## Counting compute: $C \approx 6ND$

To compare runs, we need a unit of cost. The standard one is the number of floating-point operations, **FLOPs**, needed for training. For a dense Transformer with $N$ parameters trained on $D$ tokens, a good approximation is

$$
C \approx 6 N D .
$$

Where does the 6 come from? In the forward pass, each parameter takes part in about one multiplication and one addition per token: $2N$ FLOPs per token. The backward pass (Chapter 9) computes gradients with respect to both activations and weights and costs about twice the forward pass: $4N$ FLOPs per token. Together: $6N$ per token, times $D$ tokens. The attention over the context adds a further term that grows with sequence length, but for large models and moderate contexts it is small compared with $6ND$.

A worked example. The Chinchilla model (Hoffmann et al., 2022) has $N = 7 \cdot 10^{10}$ parameters and was trained on $D = 1.4 \cdot 10^{12}$ tokens:

$$
C \approx 6 \cdot 7 \cdot 10^{10} \cdot 1.4 \cdot 10^{12} \approx 5.9 \cdot 10^{23} \text{ FLOPs}.
$$

If a GPU sustains $3 \cdot 10^{14}$ FLOP/s in practice, this is about $2 \cdot 10^9$ GPU-seconds, or roughly 550,000 GPU-hours. This is why the question from the beginning of the chapter matters: a wrong choice wastes a very large amount of money and time.

## Compute-optimal training: the Chinchilla analysis

Now fix $C$ and ask: which $N$ (and therefore which $D = C / 6N$) gives the lowest loss? Hoffmann et al. trained over 400 models of different sizes on different numbers of tokens and fitted a loss of the form

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} .
$$

Each term has a clear meaning:

- $E$ is the irreducible loss — the entropy of natural text, which no model can remove;
- $A / N^{\alpha}$ is the error caused by a model that is too small: it vanishes as $N \to \infty$;
- $B / D^{\beta}$ is the error caused by too little data: it vanishes as $D \to \infty$.

Their fitted values were approximately $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$.

The intuition is a trade-off. If we make the model huge, $A/N^\alpha$ is small, but $D = C/6N$ becomes tiny and $B/D^\beta$ explodes. If we train a tiny model on endless data, the opposite happens. The optimum is in between.

To find it, substitute $D = C / (6N)$:

$$
L(N) = E + A N^{-\alpha} + B \left(\frac{C}{6}\right)^{-\beta} N^{\beta}.
$$

Set the derivative to zero:

$$
\frac{dL}{dN} = -\alpha A N^{-\alpha-1} + \beta B \left(\frac{C}{6}\right)^{-\beta} N^{\beta-1} = 0
\quad\Longrightarrow\quad
N_{\text{opt}} \propto C^{\frac{\beta}{\alpha+\beta}}, \qquad
D_{\text{opt}} \propto C^{\frac{\alpha}{\alpha+\beta}} .
$$

The exponents add up to 1, as they must, because $N D \propto C$. If $\alpha$ and $\beta$ are close, both exponents are close to $0.5$: **when the compute grows, the model size and the number of tokens should grow in roughly equal proportion.** The paper reached this conclusion with three different fitting methods, which agree on the exponents being near $0.5$ but differ in details. Its popular summary is a rule of thumb: about **20 tokens per parameter**.

This contradicted the earlier recommendation of Kaplan et al., who had suggested growing the model much faster than the data. Part of the difference comes from experimental details: in the earlier work, the learning-rate schedule was not adjusted to the length of each run, which made short runs look relatively better. The practical demonstration was convincing: Chinchilla (70B parameters, 1.4T tokens) outperformed the four-times-larger Gopher model (280B parameters, about 300B tokens) trained with a similar budget. Many large models of that time had been **undertrained**: too many parameters, too few tokens.

## Finding the optimum numerically

The code below takes the fitted Chinchilla loss, and for each budget $C$ searches over model sizes, giving each one the tokens it can afford.

```python
import numpy as np

# Parametric loss fit from the Chinchilla paper (Hoffmann et al., 2022)
E, A, B, alpha, beta = 1.69, 406.4, 410.7, 0.34, 0.28

def loss(N, D):
    return E + A / N**alpha + B / D**beta

def best_split(C):
    """For a budget of C FLOPs, find N that minimizes the loss when D = C / (6N)."""
    N = np.logspace(7, 12, 2000)          # candidate model sizes: 10M ... 1T
    D = C / (6 * N)                       # tokens we can afford for each N
    L = loss(N, D)
    i = np.argmin(L)
    return N[i], D[i], L[i]

for C in [1e19, 1e21, 1e23]:
    N, D, L = best_split(C)
    print(f"C={C:.0e}: N={N:.2e} params, D={D:.2e} tokens, "
          f"D/N={D / N:5.1f}, loss={L:.3f}")

# Rule of thumb: D ≈ 20N and C ≈ 6ND  =>  N ≈ sqrt(C / 120)
C = 1e23
N = np.sqrt(C / 120)
print(f"rule of thumb for C=1e23: N={N:.2e}, D={20 * N:.2e}")
```

Output:

```text
C=1e+19: N=2.28e+08 params, D=7.31e+09 tokens, D/N= 32.0, loss=2.986
C=1e+21: N=1.82e+09 params, D=9.14e+10 tokens, D/N= 50.1, loss=2.329
C=1e+23: N=1.46e+10 params, D=1.14e+12 tokens, D/N= 78.3, loss=2.005
rule of thumb for C=1e23: N=2.89e+10, D=5.77e+11
```

Notice two things. Every hundredfold increase of compute increases the optimal $N$ by a factor of about 8 and $D$ by about 12 — both close to $\sqrt{100} = 10$, as the analysis predicts. And with these particular constants $\alpha > \beta$, so the tokens-per-parameter ratio slowly grows with compute instead of staying at 20. This is a good lesson: the fitted function and the rule of thumb agree in spirit, but the exact numbers depend on the fit. Later re-analyses of the same data have also suggested corrections to these constants, so treat them as illustrative, not as laws of nature.

## Beyond compute-optimal

"Compute-optimal" means optimal *for training cost only*. In reality, a successful model is then used for millions of requests, and the cost of inference grows with $N$, not with $D$. It is therefore often rational to **overtrain** a smaller model: to use many more than 20 tokens per parameter, paying more during training to get a model that is cheaper to run. The LLaMA models from Chapter 25 follow this idea — their smaller versions were trained on far more tokens than the Chinchilla rule would suggest, and their loss was still falling.

A second limit is data. For some languages and domains there are simply not enough unique tokens. Muennighoff et al. (2023) studied **data-constrained scaling** and found that repeating data for a few epochs is almost as good as fresh data, while many more repetitions bring quickly diminishing returns. For a model of Russian or Korean text, this question is very real (Chapter 30).

Finally, scaling laws describe the *loss*. Abilities on downstream tasks sometimes seem to appear suddenly at a certain scale — the so-called **emergent abilities**. Part of this effect depends on the metric: an all-or-nothing score such as exact match can jump even when the underlying loss improves smoothly. In research, the most reliable use of scaling laws is modest: fit them on a series of small runs, then predict the loss of a larger run, choose its size, or compare data mixtures and architectures by how their curves bend.

## Summary

- The test loss of language models falls approximately as a power law in parameters $N$, tokens $D$ and compute $C$; power laws appear as straight lines on log-log plots.
- Training compute for a dense Transformer is $C \approx 6ND$ FLOPs: $2N$ per token forward, $4N$ backward.
- The Chinchilla loss $L = E + A/N^\alpha + B/D^\beta$ separates irreducible loss, model-size error and data error.
- Minimizing it under fixed compute gives $N_{\text{opt}} \propto C^{\beta/(\alpha+\beta)}$ and $D_{\text{opt}} \propto C^{\alpha/(\alpha+\beta)}$, both near $C^{0.5}$; the rule of thumb is about 20 tokens per parameter.
- Inference costs and limited data change the optimum: smaller models are often overtrained, and repeated data has diminishing value.

## Exercises

1. Explain in your own words why the optimal model size cannot be "as large as possible" when the compute is fixed.
2. Using $C \approx 6ND$ and the rule $D \approx 20N$, compute the compute-optimal $N$ and $D$ for $C = 10^{22}$ FLOPs.
3. In the code, change $\beta$ to $0.34$ (equal to $\alpha$). What happens to the ratio $D/N$ across the three budgets? Explain the result using the exponents $\beta/(\alpha+\beta)$ and $\alpha/(\alpha+\beta)$.
4. Kaplan et al. found $\alpha_N \approx 0.076$. By what factor must $N$ grow to reduce the loss by 10% (to $0.9$ of its value)?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| scaling laws | законы масштабирования | 스케일링 법칙 |
| power law | степенной закон | 거듭제곱 법칙 |
| exponent | показатель степени | 지수 |
| log-log plot | график в логарифмическом масштабе | 로그-로그 그래프 |
| FLOPs | FLOPs (операции с плавающей точкой) | FLOPs(부동소수점 연산 수) |
| compute budget | вычислительный бюджет | 연산 예산 |
| parameter count | число параметров | 매개변수 수 |
| dataset size | размер датасета | 데이터셋 크기 |
| irreducible loss | неустранимая ошибка | 줄일 수 없는 손실 |
| compute-optimal training | вычислительно оптимальное обучение | 연산 최적 학습 |
| Chinchilla scaling | закон Chinchilla | 친칠라 스케일링 |
| tokens per parameter | число токенов на параметр | 매개변수당 토큰 수 |
| data-constrained scaling | масштабирование при ограниченных данных | 데이터 제약 스케일링 |
| emergent abilities | эмерджентные способности | 창발적 능력 |

---
[← Previous: Training Data: Collection, Cleaning and Deduplication](26-training-data.md) · [Contents](README.md) · [Next: Distributed Training and Mixed Precision →](28-distributed-training.md)
