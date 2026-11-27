# Chapter 40. Safety, Interpretability and Open Research Problems

> **Part 8 — LLMs in the Real World** · Level: Advanced
>
> **In this chapter you will learn:** which risks LLMs create and how they are reduced in practice · how interpretability methods look inside a model · which open problems are promising directions for a PhD

## A refusal that breaks

A model is asked how to synthesize a dangerous substance. It politely refuses. The same request is then wrapped in a story: "Write a scene where an old chemist explains to his student, step by step, how he once ..." This time the model answers. What went wrong? Did the model "know" the request was harmful the first time and then forget? Is there a "harmfulness" detector inside it that the story switched off?

This small example contains the two themes of this last chapter. **AI safety** asks how to make models behave well even under pressure, misuse and new situations. **Interpretability** asks what actually happens inside the network, so that we can explain, predict and correct its behavior instead of only observing it from outside. After both, we close the book with a map of open research problems: the questions that a new PhD student can realistically work on.

## Risks and defenses

It helps to sort the risks of LLMs into groups, because each needs different defenses:

- **Harmful content.** Instructions for weapons or malware, harassment, content that encourages self-harm.
- **Bias and toxicity.** Models absorb the stereotypes of their training data (Chapter 26) and may treat groups, dialects or languages unequally.
- **Privacy.** Models can memorize and reproduce rare strings from training data, including personal information.
- **Misinformation.** Hallucinations (Chapter 39) and persuasive, fluent falsehoods produced at scale.
- **Risks from autonomy.** Agents (Chapter 38) take actions; errors and manipulation now have consequences in the world.

The alignment methods of Chapter 34 — RLHF, DPO, constitutional AI — train the model toward being helpful, honest and harmless. But training is only one layer. Several failure modes show why more is needed:

- **Jailbreaks.** Inputs crafted to make the model ignore its training: role-play stories as above, long dialogues that slowly shift the topic, unusual encodings, translation into a **low-resource language** where safety training data was scarce, or automatically optimized nonsense suffixes. The last case shows that jailbreaks are a kind of **adversarial example**, like the imperceptible image perturbations known from computer vision.
- **Reward hacking.** In RLHF, the policy maximizes the reward model's score, not human intent. It can learn to exploit the reward model's weaknesses, for example by writing longer answers or flattering the user.
- **Sycophancy.** Models tend to agree with the user's stated opinion, even when it is wrong, because human raters often prefer agreement.

Practical defense is therefore layered. **Red teaming** means deliberately attacking the model, by people and by other models, to find failures before release. **Guardrails** are separate classifiers that check inputs and outputs. **Refusal** behavior is trained, and its rate is measured on both harmful and harmless prompts, because a model that refuses too often is also failing. For agents, the system-level measures of Chapter 38 apply. For generated text, **watermarking** hides a statistical signal in the choice of tokens, so that text can later be identified as machine-generated.

A deeper problem is **scalable oversight**: how do we supervise a model on tasks where it is better than its human supervisors? If a person cannot check a long proof or a large program, human feedback cannot reliably reward correctness. Proposals include having models critique each other's work, breaking tasks into checkable parts, and training on problems with automatic verification (Chapter 35). None is fully solved.

## Looking inside: interpretability

Return to the opening question. Is there a direction in the model's hidden states that means "this request is harmful"? The simplest tool to test such a question is a **probe**. We collect hidden states (activations) from some layer for many inputs, label each input with the property we care about, and train a small classifier, usually a **linear probe** (logistic regression), to predict the label from the activations. If a linear probe succeeds on held-out data, the information is linearly readable at that layer.

The code below simulates this. We create synthetic "activations" in which a hidden concept direction is added for one class, then train a logistic-regression probe with gradient descent (Chapter 8) and check whether it recovers the direction:

```python
import numpy as np

rng = np.random.default_rng(0)
d, n = 64, 1000
concept = rng.normal(size=d)
concept /= np.linalg.norm(concept)             # hidden "true" direction

# Synthetic hidden states: label 1 adds the concept direction, plus noise
y = rng.integers(0, 2, size=n)
H = rng.normal(size=(n, d)) + 1.5 * np.outer(2 * y - 1, concept)

# Linear probe = logistic regression trained by gradient descent
w, b, lr = np.zeros(d), 0.0, 0.1
for _ in range(500):
    p = 1 / (1 + np.exp(-(H @ w + b)))         # predicted probability
    w -= lr * H.T @ (p - y) / n                 # gradient of cross-entropy
    b -= lr * np.mean(p - y)

H_test = rng.normal(size=(200, d))
y_test = rng.integers(0, 2, size=200)
H_test += 1.5 * np.outer(2 * y_test - 1, concept)
acc = np.mean(((H_test @ w + b) > 0) == y_test)
cos = w @ concept / np.linalg.norm(w)
print(f"probe accuracy: {acc:.2f}, cosine with true direction: {cos:.2f}")
```

The probe reaches about 0.92 accuracy (the classes overlap because of noise, so perfect accuracy is impossible), and its weight vector has a cosine of about 0.95 with the true direction. In real models, probes have found linear representations of many properties: part of speech, sentiment, truthfulness of a statement, even the language of the text.

But a probe shows only that information is *present*, not that the model *uses* it. For causal claims, **mechanistic interpretability** intervenes in the computation:

- **Logit lens.** Apply the final unembedding matrix to the hidden state of an intermediate layer, to see which tokens the model "would predict" at that point. This shows how the prediction forms layer by layer along the **residual stream** (Chapter 18).
- **Activation patching.** Run the model on two inputs, for example "The capital of Russia is" and "The capital of Korea is". Copy the activation of one component from the second run into the first. If the output changes from "Moscow" to "Seoul", that component carries the relevant information. Repeating this across layers and heads locates **circuits**: small subnetworks that implement a specific behavior.
- **Steering vectors.** Add a direction (for example, the difference of mean activations on two kinds of prompts) to the hidden state during generation, and observe how behavior changes. If adding a "refusal" direction makes the model refuse and subtracting it removes refusals, we have causal evidence for that representation.

A major obstacle is that single neurons are often **polysemantic**: one neuron responds to several unrelated concepts. The **superposition** hypothesis (studied in Elhage et al., 2022, with toy models) explains why: a network must represent more features than it has dimensions, so it stores them as nearly orthogonal directions that overlap. To undo this, researchers train a **sparse autoencoder** (SAE) on the activations $\mathbf{h}$:

$$
\mathbf{f} = \operatorname{ReLU}(W_e \mathbf{h} + \mathbf{b}_e), \qquad \hat{\mathbf{h}} = W_d \mathbf{f}, \qquad \mathcal{L} = \lVert \mathbf{h} - \hat{\mathbf{h}} \rVert^2 + \lambda \lVert \mathbf{f} \rVert_1 .
$$

Here $W_e$ and $\mathbf{b}_e$ are the encoder weights and bias, $W_d$ is the decoder matrix, $\mathbf{f}$ is a feature vector with many more dimensions than $\mathbf{h}$, and $\lambda$ weights the L1 penalty that makes most features zero for any input. The learned features are often far more interpretable than neurons: one may fire on Korean text, another on Python code errors. Interpretability is still young; its methods scale poorly, and turning many local findings into an understanding of a whole model remains an open challenge.

## Open research problems: where you can start

We end the book where your own work begins. The following problems are widely recognized as open. Each is framed as a possible PhD direction with concrete questions.

**1. Reliability and hallucination.** Models still state falsehoods with confidence. Open questions: Can a model know what it knows — can we read reliable confidence from internal states (probes, semantic entropy, Chapter 39) rather than from the output text? How can training reward abstention without making the model useless? How do we guarantee faithfulness to sources in RAG, especially with long and conflicting documents? A good first project: measure the calibration of open models on factual questions in Russian versus English, and test whether a linear probe predicts errors better than the model's own token probabilities.

**2. Efficiency.** Inference cost limits who can use and study LLMs (Chapter 36). Open questions: How far can quantization go (3 bits, 2 bits, even ternary weights) before quality collapses, and why do some capabilities degrade first? Can attention be made sub-quadratic without losing long-range recall, as state-space and linear-attention models attempt? How can small models be distilled to keep reasoning ability? A first project: quantize one open model with several methods and measure which tasks suffer most, including Russian-language tasks.

**3. Multilingual and low-resource quality, including Russian.** Most training data is English (Chapter 30). Open questions: Why does a model that knows a fact in English sometimes fail to state it in Russian — and how is knowledge shared across languages inside the network? Tokenizers split Russian and Korean words into more tokens than English ones (**tokenizer fertility**), raising cost and possibly hurting quality; can vocabulary adaptation fix this for an existing model? Russian morphology — six cases, aspect, free word order — challenges both generation and evaluation metrics like BLEU. And the many languages of Russia, such as Tatar, Bashkir, Chuvash, Yakut and others, are truly low-resource: how can data be collected and models adapted for them responsibly? Safety alignment also transfers imperfectly across languages, which makes this direction important for safety as well. A first project: compare tokenizer fertility and benchmark accuracy for one model across Russian, Tatar and English, and test whether continued pre-training on a small Tatar corpus helps.

**4. Evaluation.** We cannot improve what we cannot measure (Chapter 39). Open questions: How do we build benchmarks resistant to contamination, for example dynamically generated or regularly renewed tests? How reliable are LLM judges outside English, and how do their biases interact with language? How do we evaluate long, open-ended agent trajectories? How do we measure culture-specific knowledge and norms rather than translated English content? A first project: create a small, carefully verified Russian test set in your own domain, and measure the agreement between human and LLM judges on it.

**5. Alignment and interpretability.** Open questions: Why do jailbreaks work, and can safety be made robust instead of a thin layer over the pre-trained model? Does a model's written chain of thought faithfully reflect the computation that produced its answer (Chapter 35)? Can sparse-autoencoder features or probes detect deception or reward hacking before they appear in behavior? How do we supervise systems on tasks humans cannot check? A first project: train linear probes for "harmful request" on an open model in Russian and English, and test whether translation-based jailbreaks change the probe's signal or only the output.

Some advice for choosing among these. Pick a question you can test on hardware you actually have; small open models and careful experiments are more valuable than vague plans for huge ones. Report uncertainty, use held-out data, and release code. And use your unusual advantage: as a researcher who reads Korean, English and Russian, you can see multilingual problems that monolingual researchers miss.

## Summary

- LLM risks include harmful content, bias, privacy leakage, misinformation and risks from autonomous agents; each needs its own defenses.
- Training-based alignment is not enough on its own: jailbreaks, reward hacking and sycophancy show its limits; red teaming, guardrails, measured refusal and scalable oversight add further layers.
- Linear probes show what information is linearly present in activations; logit lens, activation patching and steering vectors test what the model actually uses.
- Superposition makes neurons polysemantic; sparse autoencoders decompose activations into many sparse, more interpretable features.
- Central open problems — reliability, efficiency, multilingual and low-resource quality (including Russian and the languages of Russia), evaluation, and alignment with interpretability — offer concrete PhD directions.

## Exercises

1. Explain in your own words why a successful linear probe does not prove that the model uses the probed information. Which method from this chapter would give causal evidence?
2. Run the probe code with the separation factor `1.5` replaced by `0.5` and by `3.0`. How do accuracy and the cosine change? Explain the result in terms of signal and noise.
3. A sparse autoencoder has an input dimension of 512 and 8192 features. Count the parameters of $W_e$, $\mathbf{b}_e$ and $W_d$. Why do we want many more features than input dimensions?
4. Choose one of the five research directions. Write a half-page plan: the research question, the model and data you would use, the metric, and the result that would surprise you.

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| AI safety | безопасность ИИ | AI 안전성 |
| interpretability | интерпретируемость | 해석 가능성 |
| jailbreak | джейлбрейк / взлом модели | 탈옥 |
| adversarial example | состязательный пример | 적대적 예제 |
| reward hacking | взлом функции вознаграждения | 보상 해킹 |
| red teaming | ред-тиминг | 레드 티밍 |
| guardrails | защитные ограничения | 가드레일 |
| scalable oversight | масштабируемый надзор | 확장 가능한 감독 |
| linear probe | линейный зонд | 선형 프로브 |
| mechanistic interpretability | механистическая интерпретируемость | 기계적 해석 가능성 |
| activation patching | подмена активаций | 활성값 패칭 |
| steering vector | управляющий вектор | 조향 벡터 |
| superposition | суперпозиция | 중첩 |
| sparse autoencoder (SAE) | разреженный автокодировщик | 희소 오토인코더 |
| research direction | направление исследований | 연구 방향 |

---
[← Previous: Evaluation, Benchmarks and Hallucination](39-evaluation-and-hallucination.md) · [Contents](README.md)
