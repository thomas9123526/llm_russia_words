# Chapter 39. Evaluation, Benchmarks and Hallucination

> **Part 8 — LLMs in the Real World** · Level: Advanced
>
> **In this chapter you will learn:** how to measure an LLM with automatic metrics, benchmarks, humans and other models · why benchmark scores are often misleading · what hallucination is, why it happens and how to detect it

## Is model B really better?

Model A answers 70% of the questions in a benchmark correctly, model B answers 72%. The benchmark has 1,000 questions. Is B better?

Treat each question as a coin flip that the model gets right with probability $p$. The **standard error** of the measured accuracy is

$$
\text{SE} = \sqrt{\frac{p(1-p)}{n}},
$$

where $n$ is the number of questions. With $p = 0.7$ and $n = 1000$, $\text{SE} = \sqrt{0.21 / 1000} \approx 0.0145$. A 95% confidence interval is about $\pm 1.96 \cdot \text{SE} \approx \pm 2.8$ percentage points. The two-point difference between A and B is well inside the noise. (Because both models answer the same questions, a paired test is more sensitive, but the lesson stays.) Many published "improvements" are of this size.

This small example shows the spirit of this chapter. **Evaluation** looks easy — run the model, count the right answers — but almost every step hides a choice: what to measure, on which data, with which prompt, how to score free text, and how sure we are. A PhD in this field will spend much of its time on such questions, because a method is only as convincing as the evaluation behind it.

We will go from automatic metrics to benchmarks, then to human and model judges, and end with the most visible failure of LLMs: hallucination.

## Automatic metrics

We already met the most basic intrinsic metric in Chapter 4: **perplexity**, the exponential of the average negative log-likelihood per token. It tells how well a model predicts text, is cheap to compute, and is useful for comparing models with the *same tokenizer* during pre-training. But it says little about whether the model is useful, and it cannot be compared between tokenizers, because a model that splits Russian words into more tokens faces an easier per-token prediction task (Chapter 30).

For tasks, we compare the model's output with a reference answer:

- **Accuracy** for multiple-choice questions: the fraction of correct choices.
- **Exact match** (EM) for short answers: after normalization (lowercase, remove punctuation), is the prediction identical to the reference?
- **Token F1**: a softer version that gives partial credit. With precision $P$ = shared tokens / predicted tokens and recall $R$ = shared tokens / reference tokens, $F_1 = 2PR/(P+R)$.
- **BLEU** and **ROUGE** for translation and summarization: n-gram overlap with one or more references. They correlate only moderately with human judgments, especially for morphologically rich languages like Russian, where a correct translation can use a different word form.
- **pass@k** for code: generate $k$ programs; the task counts as solved if at least one passes the unit tests.

The naive estimate of pass@k (generate exactly $k$ samples once) has high variance. Chen et al. (2021) proposed generating $n \geq k$ samples, counting the $c$ correct ones, and computing the probability that a random subset of $k$ contains at least one correct program:

$$
\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

The code below implements these metrics:

```python
from collections import Counter
from math import comb

def normalize(s):
    return s.lower().replace(".", "").replace(",", "").split()

def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))

def token_f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    overlap = sum((Counter(p) & Counter(g)).values())   # shared tokens
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)

def pass_at_k(n, c, k):
    """Unbiased pass@k: n samples generated, c of them correct."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

print(exact_match("Moscow.", "moscow"))                      # 1.0
print(round(token_f1("the city of Moscow", "Moscow"), 3))    # 0.4
print(round(pass_at_k(10, 3, 1), 4), round(pass_at_k(10, 3, 5), 4))  # 0.3 0.9167
```

In the F1 example, one of four predicted tokens is shared, so $P = 0.25$, $R = 1$ and $F_1 = 0.5 / 1.25 = 0.4$. For pass@k with $n = 10$, $c = 3$: pass@1 is simply $3/10$, while pass@5 $= 1 - \binom{7}{5}/\binom{10}{5} = 1 - 21/252 \approx 0.917$.

## Benchmarks and their traps

A **benchmark** is a fixed dataset with a fixed metric, so that different models can be compared. Well-known examples are MMLU (multiple-choice questions over dozens of academic subjects), GSM8K (grade-school math word problems), HumanEval (Python functions checked by unit tests), HellaSwag (commonsense sentence completion) and TruthfulQA (questions on which people often hold false beliefs). For Russian there are suites such as Russian SuperGLUE and MERA, which collect many tasks into one score, and multilingual sets that translate English benchmarks into many languages. Benchmarks are indispensable, but they have well-known traps:

- **Contamination.** Benchmark questions are published on the internet, and the internet is the training data (Chapter 26). If a model has seen the test set, its score measures memory, not ability. This **benchmark contamination**, a form of **data leakage**, is hard to rule out for models whose training data is not public. Researchers check n-gram overlap with the training data, compare performance on old and newly written questions, or keep private **held-out sets**.
- **Saturation.** When top models reach near-perfect scores, a benchmark stops distinguishing them, and part of the remaining errors are often label mistakes in the dataset itself.
- **Prompt sensitivity.** The score can change by several points with the prompt format, the number of few-shot examples or the way answers are extracted. Comparisons are fair only under identical settings.
- **Goodhart's law.** "When a measure becomes a target, it ceases to be a good measure." Optimizing for a leaderboard can improve the number without improving the model.

For open-ended tasks such as writing or dialogue, there is no single correct answer. The gold standard is **human evaluation**, often as pairwise comparisons: a person sees two answers to the same prompt and chooses the better one. Many such votes can be turned into a ranking with the **Elo rating** or the Bradley–Terry model, in which the probability that model $i$ beats model $j$ is

$$
P(i \succ j) = \frac{1}{1 + 10^{(R_j - R_i)/400}},
$$

with $R_i, R_j$ the models' ratings. A difference of 100 points means the stronger model wins about 64% of comparisons. Human evaluation is slow and expensive, and people disagree; the **inter-annotator agreement** must be reported.

A cheaper substitute is **LLM-as-a-judge**: a strong model compares or grades answers. Zheng et al. (2023) found that such judges agree with humans about as often as humans agree with each other, but they also documented clear biases: **position bias** (preferring the first or second answer), **verbosity bias** (preferring longer answers) and **self-preference** (preferring answers in their own style). Standard precautions are to evaluate both orders, control for length and validate the judge against human labels on a sample.

## Hallucination

A **hallucination** is generated content that is fluent and confident but not supported by facts or by the given source. Ask a model for references on a narrow topic and it may produce a perfectly formatted citation to a paper that does not exist. It is useful to distinguish:

- **Factuality** errors: the statement contradicts world knowledge ("Moscow is on the Volga").
- **Faithfulness** errors: the output contradicts or adds to the provided input, for example a summary that mentions a number absent from the document. This matters most in RAG (Chapter 37).

Why do models hallucinate? The pre-training objective (Chapter 21) rewards producing the most plausible continuation, not the true one, and a plausible-looking author name and year is exactly what fits the pattern of a citation. Facts that appeared rarely in training are stored weakly in the weights. Instruction tuning and RLHF (Chapters 33–34) may reward answers that sound helpful and confident, and evaluation that gives zero points to "I don't know" pushes in the same direction. Sampling with high temperature (Chapter 24) can further move the model away from what it knows.

How can we detect hallucinations without knowing the answer?

- **Consistency.** Sample several answers to the same question. If the model knows the fact, the answers agree; if it is guessing, they scatter. SelfCheckGPT (Manakul et al., 2023) uses this idea. **Semantic entropy** (Farquhar et al., 2024) refines it: cluster the sampled answers by meaning (so "Moscow" and "the city of Moscow" count as one) and compute the entropy over clusters. High entropy signals likely confabulation.
- **Verification.** Split the answer into atomic claims and check each claim against retrieved sources, the approach behind many factuality metrics.
- **Calibration.** A model is **calibrated** if, among all answers it gives with 80% confidence, about 80% are correct. The **expected calibration error** groups predictions into confidence bins $B_b$ and averages the gap:

$$
\text{ECE} = \sum_{b} \frac{|B_b|}{n} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|,
$$

where $|B_b|$ is the number of predictions in bin $b$, $n$ the total number, $\text{acc}$ the accuracy in the bin and $\text{conf}$ the average confidence. A well-calibrated model can be trusted to say "I am not sure", which is the basis for **abstention**: refusing to answer when confidence is low.

Mitigation combines several tools: grounding answers in retrieved sources with citations, training the model to abstain, decoding conservatively for factual tasks, and verifying claims afterwards. None eliminates hallucination completely, which is why it remains one of the central open problems we discuss in Chapter 40.

## Summary

- Always estimate uncertainty: with $n$ test items, accuracy has standard error $\sqrt{p(1-p)/n}$, so small differences between models are often noise.
- Perplexity measures prediction quality; task metrics include accuracy, exact match, token F1, BLEU/ROUGE and the unbiased pass@k for code.
- Benchmarks suffer from contamination, saturation, prompt sensitivity and Goodhart's law; held-out and freshly written test sets help.
- Human pairwise judgments, aggregated with Elo or Bradley–Terry, are the gold standard for open-ended tasks; LLM judges are cheaper but have position, verbosity and self-preference biases.
- Hallucinations are fluent but unsupported outputs; they can be detected with consistency checks, semantic entropy, claim verification and calibration, and reduced but not removed by grounding and abstention.

## Exercises

1. Explain in your own words the difference between a factuality error and a faithfulness error, with one example of each.
2. A benchmark has 400 questions. Model A scores 60%. Compute the standard error and an approximate 95% interval. Would a 3-point improvement be convincing?
3. Use the code to compute pass@1, pass@5 and pass@10 for $n = 20$ and $c = 2$. Explain why pass@k grows so fast with $k$.
4. Two models have Elo ratings 1200 and 1300. What is the expected win rate of the stronger model? Why might an LLM judge with verbosity bias distort such a ranking?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| evaluation | оценивание | 평가 |
| exact match | точное совпадение | 정확 일치 |
| F1 score | F1-мера | F1 점수 |
| pass@k | метрика pass@k | pass@k |
| benchmark | бенчмарк | 벤치마크 |
| benchmark contamination | загрязнение бенчмарка | 벤치마크 오염 |
| held-out set | отложенная выборка | 홀드아웃 세트 |
| human evaluation | оценка людьми | 인간 평가 |
| Elo rating | рейтинг Эло | Elo 레이팅 |
| inter-annotator agreement | согласованность разметчиков | 평가자 간 일치도 |
| LLM-as-a-judge | LLM в роли судьи | 심판으로서의 LLM |
| hallucination | галлюцинация | 환각 |
| factuality | фактическая точность | 사실성 |
| semantic entropy | семантическая энтропия | 의미 엔트로피 |
| calibration | калибровка | 보정 |

---
[← Previous: Agents and Tool Use](38-agents-and-tools.md) · [Contents](README.md) · [Next: Safety, Interpretability and Open Research Problems →](40-safety-and-open-problems.md)
