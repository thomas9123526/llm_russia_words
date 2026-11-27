# Chapter 34. Alignment: RLHF and DPO

> **Part 7 — Adapting and Aligning Models** · Level: Advanced
>
> **In this chapter you will learn:** why imitation (SFT) is not enough and how human preferences are collected · how RLHF trains a reward model and optimises the policy with a KL penalty · how DPO turns the same objective into a simple classification-like loss

## Easier to judge than to write

Ask a colleague to write the perfect explanation of backpropagation for a first-year student. It is hard, and two good teachers will write very different texts. Now show the same colleague two explanations and ask: "Which one is better?" This is much easier, and people agree more often.

This observation is the starting point of preference-based **alignment**: making a model's behaviour match what people actually want — answers that are helpful, honest and harmless. In Chapter 33 we taught a model by imitation: supervised fine-tuning on demonstrations. Imitation has two limits. First, the model can be no better than the demonstrations, and good demonstrations are expensive. Second, SFT only says "this answer is good"; it never says "this answer is *better than* that one", and it does not punish bad behaviour directly. **Preference data** fills this gap. For a prompt $x$, the model generates two responses, and an annotator marks which one is better: the preferred response $y_w$ ("winner") and the rejected one $y_l$ ("loser"). Thousands of such **pairwise comparisons** form the training data for the two methods of this chapter.

## Step 1 of RLHF: the reward model

**RLHF** — reinforcement learning from human feedback — was developed in work by Christiano et al. (2017) and Stiennon et al. (2020), and became widely known through InstructGPT (Ouyang et al., 2022). It has three stages: (1) supervised fine-tuning, (2) training a reward model, (3) optimising the model with reinforcement learning against the reward model.

A **reward model** $r_\phi(x, y)$ is a network, usually initialised from the SFT model with its language-modeling head replaced by a head that outputs a single number: a score for how good response $y$ is for prompt $x$. To train it from comparisons, we use the **Bradley–Terry model**, a classic model of paired comparisons. It assumes that the probability that $y_w$ is preferred to $y_l$ depends only on the difference of their scores:

$$P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big),$$

where $\sigma(z) = 1/(1 + e^{-z})$ is the sigmoid function (Chapter 7). Maximising the log-likelihood of the observed preferences gives the loss

$$\mathcal{L}_{RM} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big).$$

A worked example: if the reward model gives the preferred answer the score $2.0$ and the rejected one $0.5$, the difference is $1.5$, $\sigma(1.5) \approx 0.818$, and the loss is $-\log 0.818 \approx 0.201$. If the scores were reversed, the loss would be $-\log \sigma(-1.5) \approx 1.701$. The loss pushes the scores apart in the right direction. Note that only *differences* of rewards matter, so the absolute scale of $r_\phi$ is arbitrary.

## Step 2 of RLHF: optimising the policy

In reinforcement learning language, the language model is a **policy** $\pi_\theta(y \mid x)$: given a state (the prompt), it chooses actions (tokens). We now want to change $\theta$ so that the responses get high reward. But if we simply maximise reward, the model quickly finds responses that the reward model scores highly but people would not like: very long answers, flattering phrases, repeated keywords. The reward model is only an imperfect approximation of human judgement, trained on a limited set of comparisons, and outside that set its scores can be wrong. Exploiting these errors is called **reward hacking**.

The standard remedy is to keep the policy close to a **reference model** $\pi_{\text{ref}}$, usually the SFT model, using a KL-divergence penalty (Chapter 6). The RLHF objective is

$$\max_{\theta}\; \mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot \mid x)} \big[ r_\phi(x, y) \big] \;-\; \beta\, \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big),$$

where $D$ is a set of prompts and $\beta > 0$ controls how strongly we penalise moving away from the reference model. Small $\beta$: more reward, more risk of hacking. Large $\beta$: safer, but the model changes little.

Because the responses $y$ are *sampled* from the model, we cannot simply backpropagate through them; this is why reinforcement learning is used. The most common algorithm in RLHF has been **PPO** (proximal policy optimisation, Schulman et al., 2017), a policy-gradient method that limits how much the policy can change in one update. In practice, PPO-based RLHF is complex: during training it needs the policy, the reference model, the reward model and usually a value network, and it has many hyperparameters and can be unstable. This complexity motivated the search for simpler methods.

## DPO: alignment without a separate reward model

**DPO** — direct preference optimisation (Rafailov et al., 2023) — starts from a mathematical observation. For the KL-regularised objective above, the optimal policy has a known closed form:

$$\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\Big(\frac{1}{\beta} r(x, y)\Big),$$

where $Z(x)$ is a normalising constant that makes the probabilities sum to 1. Intuitively, the best policy takes the reference model and increases the probability of each response in proportion to the exponent of its reward. We can solve this equation for the reward:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x).$$

Now substitute this into the Bradley–Terry model. In the difference $r(x, y_w) - r(x, y_l)$, the term $\beta \log Z(x)$ cancels, because it does not depend on $y$. What remains depends only on the policy and the reference model, so we can train the policy $\pi_\theta$ directly:

$$\mathcal{L}_{DPO} = -\log \sigma\Big(\beta \Big[\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big]\Big).$$

The paper's slogan is "your language model is secretly a reward model": the quantity $\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ plays the role of an *implicit reward*. DPO needs no sampling during training, no separate reward model and no RL algorithm. It is just a loss over pairs of fixed responses, computed from the log-probabilities of whole responses (sums of token log-probabilities, as in Chapter 33). The code below implements both losses and checks the numbers.

```python
import torch
import torch.nn.functional as F

def reward_model_loss(r_chosen, r_rejected):
    # Bradley-Terry: -log sigma(r_w - r_l)
    return -F.logsigmoid(r_chosen - r_rejected).mean()

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    # Arguments are total log-probabilities log pi(y|x) of whole responses
    margin = (pi_w - ref_w) - (pi_l - ref_l)
    return -F.logsigmoid(beta * margin).mean()

print(f"{reward_model_loss(torch.tensor([2.0]), torch.tensor([0.5])):.3f}")  # 0.201

# At the start the policy equals the reference model: loss = log 2
same = torch.tensor([-11.0])
print(f"{dpo_loss(same, same, same, same):.3f}")                             # 0.693

# The policy has raised the chosen answer and lowered the rejected one
pi_w = torch.tensor([-10.0], requires_grad=True)
pi_l = torch.tensor([-12.0], requires_grad=True)
loss = dpo_loss(pi_w, pi_l, torch.tensor([-11.0]), torch.tensor([-11.0]))
loss.backward()
print(f"{loss:.3f}", f"{pi_w.grad.item():.3f}", f"{pi_l.grad.item():.3f}")   # 0.598 -0.045 0.045
```

Let us check the last case by hand. The margin is $(-10 + 11) - (-12 + 11) = 1 - (-1) = 2$; with $\beta = 0.1$ we get $\sigma(0.2) \approx 0.550$ and a loss of $-\log 0.550 \approx 0.598$. The gradient with respect to $\log \pi_\theta(y_w)$ is negative and the gradient with respect to $\log \pi_\theta(y_l)$ is positive, so a gradient descent step increases the probability of the preferred answer and decreases the probability of the rejected one. The size of the gradient, $\beta\,(1 - \sigma(0.2)) \approx 0.045$, shrinks as the margin grows: pairs that the model already ranks correctly contribute less.

## Comparing the methods and their limits

DPO became popular because it is simple and stable and needs much less memory and engineering than PPO. RLHF with an explicit reward model still has advantages: it is *on-policy* (it learns from fresh samples of the current model, not only from a fixed dataset), and a trained reward model can be reused to score new responses. Which approach gives better final models is still a subject of research, and many variants exist (for example IPO and KTO). For reasoning tasks, Chapter 35 discusses a related idea: reinforcement learning with *verifiable* rewards, where a program checks whether the answer is correct.

The source of feedback can also change. In **RLAIF** (reinforcement learning from AI feedback), the comparisons are produced by a strong model instead of people. In Constitutional AI (Bai et al., 2022), a model critiques and revises its own answers according to a written list of principles, and AI-generated preferences based on those principles are used for training.

Whatever the method, important limitations remain:

- **Annotators disagree**, and preferences depend on culture and language. Whose preferences does the model learn? This matters for Russian- and Korean-speaking users, whose norms may differ from those in the annotation pool.
- **Reward hacking** and **sycophancy**: optimising for approval can teach a model to agree with the user or to sound confident instead of being correct.
- **Length bias**: annotators and reward models often prefer longer answers, so models can become verbose.
- **Alignment is not a solved problem.** These methods shape typical behaviour but do not guarantee it; we return to safety in Chapter 40.

## Summary

- Comparing two answers is easier and more consistent for people than writing an ideal one; preference data consists of pairs (preferred, rejected) for a prompt.
- RLHF trains a reward model with the Bradley–Terry loss $-\log \sigma(r_w - r_l)$, then optimises the policy (usually with PPO) to maximise reward minus a KL penalty to the reference model.
- The KL penalty limits reward hacking; $\beta$ controls the trade-off between reward and staying close to the reference.
- DPO uses the closed-form solution of the KL-regularised objective to obtain a simple loss on log-probability ratios, with no reward model and no RL loop.
- Preference-based alignment has known problems: disagreement between annotators, reward hacking, sycophancy and length bias.

## Exercises

1. Explain in your own words why the KL penalty to the reference model is needed in RLHF.
2. A reward model gives scores $r_w = 0.3$ and $r_l = 1.1$ to a preferred and a rejected answer. Compute the Bradley–Terry probability that the preferred answer wins and the loss. Is the reward model right or wrong here?
3. Show that the term $\beta \log Z(x)$ cancels when the DPO expression for the reward is substituted into the Bradley–Terry model.
4. Using `dpo_loss`, compute the loss for $\beta = 0.1$ and $\beta = 0.5$ for the same log-probabilities as in the code. How does $\beta$ change the loss and the gradient?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| alignment | выравнивание | 정렬 |
| preference data | данные о предпочтениях | 선호 데이터 |
| pairwise comparison | попарное сравнение | 쌍대 비교 |
| RLHF | обучение с подкреплением на основе обратной связи от человека | 인간 피드백 기반 강화 학습 |
| reward model | модель вознаграждения | 보상 모델 |
| Bradley–Terry model | модель Брэдли–Терри | 브래들리–테리 모델 |
| policy | стратегия (политика) | 정책 |
| reference model | референсная модель | 참조 모델 |
| KL divergence | расхождение Кульбака–Лейблера | KL 발산 |
| reward hacking | взлом функции вознаграждения | 보상 해킹 |
| PPO | проксимальная оптимизация стратегии | 근위 정책 최적화 |
| DPO | прямая оптимизация предпочтений | 직접 선호 최적화 |
| implicit reward | неявное вознаграждение | 암묵적 보상 |
| RLAIF | обучение с подкреплением на основе обратной связи от ИИ | AI 피드백 기반 강화 학습 |
| sycophancy | угодничество | 아첨 |

---
[← Previous: Instruction Tuning](33-instruction-tuning.md) · [Contents](README.md) · [Next: Reasoning and Chain-of-Thought →](35-reasoning.md)
