# Chapter 35. Reasoning and Chain-of-Thought

> **Part 7 — Adapting and Aligning Models** · Level: Advanced
>
> **In this chapter you will learn:** why letting a model "think out loud" improves multi-step problem solving · how self-consistency, verifiers and search spend more computation at inference time · how reasoning models are trained with reinforcement learning on verifiable rewards

## Answering at once versus step by step

Consider a simple word problem:

> Masha had 3 boxes with 12 pencils in each. She gave 7 pencils to her brother. How many pencils does she have left?

If we ask a language model to give only the number, it must produce the answer in its very first output token. A model that answers "36" has done half the work and forgotten the subtraction. Now ask it to write out the steps:

> 3 boxes × 12 pencils = 36 pencils. 36 − 7 = 29. Answer: 29.

The model is much more likely to be correct. Writing intermediate steps before the final answer is called **chain-of-thought** (CoT) reasoning. Wei et al. (2022) showed that adding a few worked examples with written-out reasoning to the prompt (few-shot CoT prompting, compare Chapter 23) strongly improved large models on arithmetic, commonsense and symbolic reasoning benchmarks such as GSM8K, a dataset of grade-school maths word problems. Kojima et al. (2022) found that even without examples, simply adding "Let's think step by step" to the prompt (zero-shot CoT) gave large improvements. As with instruction tuning (Chapter 33), the effect was strong for large models and weak or absent for small ones.

## Why thinking out loud helps

There is a simple computational argument. A transformer (Chapter 18) performs a fixed amount of computation per generated token: one pass through $L$ layers. If a problem requires more sequential steps than the network can perform in one pass, the model cannot solve it reliably in a single token. But each generated token is appended to the context, and the next token can attend to it (Chapter 16). The generated text therefore works as a **scratchpad**: an external working memory where the model writes intermediate results and reads them back. Generating $k$ reasoning tokens gives the model roughly $k$ extra passes of computation before the answer. Theoretical work on transformers supports this intuition: allowing intermediate steps increases the class of problems a fixed-size transformer can solve.

A second view is probabilistic. Let $x$ be the question, $z$ a reasoning chain and $a$ the final answer. The model generates $z$ and then $a$, so the probability of an answer is a sum over all possible chains:

$$p(a \mid x) = \sum_{z} p(z \mid x)\, p(a \mid x, z).$$

Each individual step in a chain is easy and has high probability of being right, while jumping directly from $x$ to $a$ is hard. CoT decomposes one difficult prediction into many easy ones. The same formula suggests the next idea: since there are many possible chains, we can sample several of them.

## Spending more computation at inference time

**Self-consistency** (Wang et al., 2022) samples $N$ different reasoning chains with a temperature above zero (Chapter 24), extracts the final answer from each, and returns the most frequent answer by **majority voting**. This approximates the sum over $z$ above: answers that many different chains lead to are more likely to be correct. The simulation below makes the effect concrete. Each chain is correct with probability only 0.4, but wrong chains disagree with each other, so the correct answer usually wins the vote.

```python
import random
from collections import Counter

random.seed(0)
CORRECT = 29
WRONG = [27, 31, 33, 36, 41]          # typical mistakes, spread over several values

def sample_chain():
    # One reasoning chain: correct with probability 0.4, otherwise a random mistake
    return CORRECT if random.random() < 0.4 else random.choice(WRONG)

def self_consistency(n):
    answers = [sample_chain() for _ in range(n)]
    return Counter(answers).most_common(1)[0][0]   # majority vote

for n in [1, 5, 11, 21]:
    trials = 5000
    acc = sum(self_consistency(n) == CORRECT for _ in range(trials)) / trials
    print(f"N = {n:2d} chains: accuracy {acc:.2f}")
# N =  1 chains: accuracy 0.39
# N =  5 chains: accuracy 0.58
# N = 11 chains: accuracy 0.76
# N = 21 chains: accuracy 0.90
```

Here each wrong answer has probability $0.6 / 5 = 0.12$, much less than $0.4$, so with more chains the vote becomes reliable. If the model made the *same* mistake most of the time, voting would not help: self-consistency improves accuracy only when errors are scattered.

Voting is one way of using **test-time compute** — extra computation at inference rather than in training. Others include:

- **Best-of-N with a verifier.** Sample $N$ solutions and choose the one with the highest score from a separate **verifier** model. An **outcome reward model** (ORM) scores only the final answer; a **process reward model** (PRM) scores each step. Lightman et al. ("Let's Verify Step by Step", 2023) found that process supervision gave a more reliable verifier for mathematics than outcome supervision.
- **Search.** Tree of Thoughts (Yao et al., 2023) treats partial reasoning chains as nodes of a tree, generates several continuations of each, evaluates them, and explores the promising branches, backtracking when a branch looks wrong.

All of these trade inference cost for accuracy. A useful mental model: a smaller model that "thinks longer" can sometimes match a larger model that answers immediately, but the cost per question grows with $N$ and with the length of the chains.

## Training models to reason

Prompting only *elicits* reasoning a model already can do. Modern **reasoning models** are trained to produce long, useful chains of thought. Two ideas are central.

**Learning from your own correct solutions.** In STaR (Zelikman et al., 2022), the model generates rationales for many problems; those that reach the correct answer are kept and used for supervised fine-tuning, and the process is repeated. The model bootstraps its own reasoning data. This is closely related to rejection sampling: generate many solutions, keep only the good ones.

**Reinforcement learning with verifiable rewards.** In Chapter 34 the reward came from a learned reward model, which can be hacked. For mathematics and programming we can often do better: a program can check whether the final answer equals the reference answer, or whether generated code passes unit tests. Such a **verifiable reward** (for example 1 for correct, 0 for wrong) is cheap and hard to fool. The model is then trained with reinforcement learning to maximise this reward. A widely used algorithm is **GRPO** (group relative policy optimisation, introduced in the DeepSeekMath work, Shao et al., 2024), a simplification of PPO without a value network. For each problem, GRPO samples a group of $G$ answers, computes their rewards $r_1, \dots, r_G$, and uses the normalised reward as the **advantage** of each answer:

$$A_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G)}.$$

Answers better than the group average get a positive advantage and become more probable; worse answers become less probable. For a group of four answers with rewards $(1, 0, 0, 1)$, the mean is $0.5$ and the standard deviation is $0.5$:

```python
import torch
rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])   # 4 answers to one problem: 1 = correct
adv = (rewards - rewards.mean()) / rewards.std(unbiased=False)
print(adv)                                      # tensor([ 1., -1., -1.,  1.])
```

(Implementations differ in details, such as which standard deviation is used and how a KL penalty to the reference model is added.) Reports such as DeepSeek-R1 (2025) described how reinforcement learning with rule-based rewards led models to produce longer chains of thought on their own, including checking and correcting earlier steps. Strong reasoning models can also be used as teachers: training a smaller model by SFT on their reasoning traces is called **reasoning distillation**.

## Limits and open questions

- **Faithfulness.** A chain of thought is text the model generates; it is not guaranteed to describe the computation that actually produced the answer. Turpin et al. (2023) showed that models can be influenced by hidden biases in the prompt while giving explanations that never mention them. CoT is useful for performance and partly for transparency, but should not be treated as a reliable explanation.
- **Errors propagate.** One wrong step early in a chain can lead the rest of the reasoning astray, and a fluent chain can make a wrong answer look convincing.
- **Cost.** Long chains multiply the number of generated tokens and therefore the inference cost and latency (Chapter 36). Models may also "overthink" easy questions.
- **Verifiable domains are easier.** RL with verifiable rewards works best where answers can be checked automatically, such as mathematics and code. How to train similar reasoning for open-ended questions, where correctness is hard to check, is an active research question.

## Summary

- Chain-of-thought means generating intermediate reasoning steps before the answer; it can be elicited with few-shot examples or with a phrase like "Let's think step by step".
- CoT helps because each generated token adds computation and the context acts as a scratchpad; one hard prediction is split into many easy ones.
- Self-consistency samples many chains and takes a majority vote; verifiers (ORM, PRM) and tree search are other ways to spend test-time compute.
- Reasoning models are trained on their own correct solutions (STaR, rejection sampling) and with reinforcement learning on verifiable rewards, for example with GRPO's group-normalised advantages.
- Chains of thought are not necessarily faithful explanations, errors can propagate, and long reasoning is expensive.

## Exercises

1. Explain in your own words why a transformer may fail to solve a multi-step problem when it must output the answer in the first token, but succeed when it writes out the steps.
2. In the simulation, change the probability of a correct chain from 0.4 to 0.25. Does majority voting still help? What happens if `WRONG` contains only one value?
3. Compute the GRPO advantages for a group of rewards $(1, 0, 0, 0)$. What happens if all four answers are correct? Why might such groups be skipped in training?
4. Write a few-shot chain-of-thought prompt in Russian for two-step word problems, with two worked examples and one new question.

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| reasoning | рассуждение | 추론 |
| chain-of-thought (CoT) | цепочка рассуждений | 사고의 사슬 |
| step-by-step reasoning | пошаговое рассуждение | 단계별 추론 |
| scratchpad | черновик | 스크래치패드 |
| self-consistency | самосогласованность | 자기 일관성 |
| majority voting | голосование большинством | 다수결 투표 |
| test-time compute | вычисления во время инференса | 테스트 시점 연산 |
| verifier | верификатор | 검증기 |
| outcome reward model (ORM) | модель вознаграждения за результат | 결과 보상 모델 |
| process reward model (PRM) | модель вознаграждения за процесс | 과정 보상 모델 |
| Tree of Thoughts | дерево мыслей | 생각의 나무 |
| verifiable reward | проверяемая награда | 검증 가능한 보상 |
| advantage | преимущество | 이점(advantage) |
| reasoning model | рассуждающая модель | 추론 모델 |
| reasoning distillation | дистилляция рассуждений | 추론 증류 |

---
[← Previous: Alignment: RLHF and DPO](34-alignment-rlhf-dpo.md) · [Contents](README.md) · [Next: Efficient Inference: KV Cache, Quantization, Speculative Decoding →](36-efficient-inference.md)
