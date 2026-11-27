# Chapter 31. Fine-tuning and Transfer Learning

> **Part 7 — Adapting and Aligning Models** · Level: Advanced
>
> **In this chapter you will learn:** why knowledge learned in pre-training transfers to new tasks · the main fine-tuning strategies, from a frozen backbone to full fine-tuning · how to avoid catastrophic forgetting and overfitting on small datasets

## A model that knows a lot, but not your task

Suppose your laboratory wants to classify Russian scientific abstracts into five fields: physics, biology, chemistry, mathematics and computer science. You have 2,000 labelled abstracts. That is far too little to train a transformer from scratch: in Chapter 27 we saw that a good language model needs billions of tokens. But a pre-trained model has already read a huge amount of text, including Russian. It already "knows" Russian morphology, which words are related, and roughly what a physics paper sounds like. Your 2,000 examples only need to teach it one new thing: how to map what it already knows onto your five labels.

This is the central idea of **transfer learning**: knowledge acquired while solving one task (the *source* task, here next-token prediction on a large corpus) is reused for another task (the *target* task, here classification). **Fine-tuning** is the most common way to do it: we take the pre-trained weights as a starting point and continue training them on data for the target task.

Transfer learning is not specific to language. In computer vision it became standard practice to start from a network pre-trained on ImageNet. In NLP, the big change came around 2018. ULMFiT (Howard and Ruder) showed that a pre-trained language model could be fine-tuned for text classification with very few labels, and BERT (Chapter 22) made "pre-train once, fine-tune for each task" the default recipe. Today almost every applied LLM project starts from a pre-trained checkpoint.

## Why transfer works

Why should weights learned for predicting the next token help with classifying abstracts? The intuition is that pre-training forces the model to build general-purpose **representations**. To predict the next word well, a model must track syntax, agreement, topic, facts about the world and much more. These features are useful for almost any language task.

Research on probing (training small classifiers on hidden states) suggests a rough picture: lower layers capture more local and surface information (word pieces, morphology), middle layers more syntax, and upper layers more task-specific and semantic information. The picture is not perfectly clean, but it explains a practical rule: lower layers are usually more general and transfer better, while upper layers are more specialised to the pre-training objective and need more change.

We can also look at transfer through optimisation. Training from random weights starts at an arbitrary point of the loss landscape (Chapter 8). Fine-tuning starts from a point that is already very good for a related problem. From there, a small number of gradient steps with a small learning rate is often enough to reach a good solution for the new task. In this sense, the pre-trained weights act like a strong prior: they tell the optimiser where good solutions are likely to be.

## Fine-tuning strategies

There is a spectrum of strategies, from changing nothing in the pre-trained model to changing everything.

**1. Feature extraction (frozen backbone).** We **freeze** all pre-trained weights and use the model only to compute representations, for example the hidden state of the `[CLS]` token in BERT or the mean of all token states. On top we train a small new **classification head**, often just one linear layer. This is cheap, stable and hard to overfit, but the representations cannot adapt to the task.

**2. Partial fine-tuning.** We unfreeze only the top $k$ layers plus the head. Because upper layers are the most task-specific, this often gives most of the benefit of full fine-tuning at a lower cost.

**3. Full fine-tuning.** All parameters are updated. This usually gives the best quality when you have enough data, but it is the most expensive: for a model with $N$ parameters trained with Adam, you must store the weights, the gradients and two optimizer states per parameter, roughly $4N$ numbers (plus activations), as discussed in Chapter 28. You also get a full copy of the model for each task.

**4. Parameter-efficient fine-tuning.** We freeze the model and train a small number of new parameters inserted into it. This is the subject of Chapter 32 (LoRA and QLoRA).

The head depends on the task. For classification with $C$ classes, the head maps a hidden vector $h \in \mathbb{R}^d$ to logits:

$$z = W h + b, \qquad W \in \mathbb{R}^{C \times d},\; b \in \mathbb{R}^C,$$

and we train with cross-entropy loss (Chapter 8). For token-level tasks such as named entity recognition, the same head is applied to every token's hidden state. For generation tasks with a decoder model, no new head is needed: we keep the language-modeling head and simply continue training on (input, output) text pairs. That is the basis of instruction tuning in Chapter 33.

The code below shows the first two strategies on a toy "pre-trained" network. We freeze the backbone, add a new head, and then check how many parameters are trainable.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
d, num_classes = 64, 5
# A stand-in for a pre-trained backbone: three layers
backbone = nn.Sequential(nn.Linear(32, d), nn.ReLU(),
                         nn.Linear(d, d), nn.ReLU(),
                         nn.Linear(d, d), nn.ReLU())
head = nn.Linear(d, num_classes)          # new, randomly initialised head
model = nn.Sequential(backbone, head)

# Strategy 1: freeze the whole backbone
for p in backbone.parameters():
    p.requires_grad = False
# Strategy 2: unfreeze only the top layer of the backbone
for p in backbone[4].parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable {trainable} of {total}")   # trainable 4485 of 10757

# Only trainable parameters are given to the optimiser
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
x, y = torch.randn(8, 32), torch.randint(0, num_classes, (8,))
loss = nn.functional.cross_entropy(model(x), y)
loss.backward()
opt.step()
print(backbone[0].weight.grad is None)        # True: frozen layer gets no gradient
```

Let us check the numbers. The first layer has $32 \cdot 64 + 64 = 2112$ parameters, each of the other two backbone layers has $64 \cdot 64 + 64 = 4160$, and the head has $64 \cdot 5 + 5 = 325$. The total is $2112 + 2 \cdot 4160 + 325 = 10{,}757$. The trainable part is the top layer plus the head: $4160 + 325 = 4485$.

## Catastrophic forgetting and how to avoid it

Fine-tuning has a well-known danger. If we train too hard on a narrow dataset, the model can lose the general abilities it had. This is called **catastrophic forgetting**: neural networks trained sequentially on task A and then task B tend to overwrite what they learned for A. A model fine-tuned aggressively on legal documents may become worse at ordinary conversation, or forget knowledge it had before.

With small datasets there is a second danger: **overfitting** (Chapter 10). A model with hundreds of millions of parameters can easily memorise 2,000 examples. Several practical techniques help with both problems:

- **Small learning rate.** Fine-tuning uses much smaller learning rates than pre-training. For BERT-sized models, the original paper recommended values like $2\cdot10^{-5}$ to $5\cdot10^{-5}$. We want to move the weights a little, not destroy them.
- **Few epochs.** Often 2–4 passes over the data are enough. Monitor the loss on a **validation set** and stop early when it stops improving.
- **Warmup and decay.** A short learning rate warmup prevents large, noisy updates in the first steps, when the new head is still random and its gradients are large.
- **Discriminative (layer-wise) learning rates.** ULMFiT proposed giving lower layers smaller learning rates than upper layers, because lower layers are more general. A common rule is to multiply the learning rate by a factor $\xi < 1$ for each layer going down: layer $l$ gets $\eta_l = \eta \cdot \xi^{L-l}$, where $L$ is the top layer and $\eta$ is the learning rate of the top layer.
- **Gradual unfreezing.** Also from ULMFiT: train the head first, then unfreeze layers one by one from the top.
- **Mixing in general data.** When fine-tuning a generative model, adding a small share of general pre-training data or general instructions reduces forgetting.
- **Weight decay towards the starting point.** Some methods penalise the distance $\lVert \theta - \theta_0 \rVert^2$ between the current weights $\theta$ and the pre-trained weights $\theta_0$, which keeps the model close to where it started.

In PyTorch, layer-wise learning rates are implemented with *parameter groups*. Continuing the example above:

```python
# Unfreeze everything and give each layer its own learning rate
for p in model.parameters():
    p.requires_grad = True

base_lr, xi = 1e-3, 0.5
groups = [
    {"params": head.parameters(),        "lr": base_lr},
    {"params": backbone[4].parameters(), "lr": base_lr * xi},
    {"params": backbone[2].parameters(), "lr": base_lr * xi**2},
    {"params": backbone[0].parameters(), "lr": base_lr * xi**3},
]
opt = torch.optim.AdamW(groups, weight_decay=0.01)
print([g["lr"] for g in opt.param_groups])   # [0.001, 0.0005, 0.00025, 0.000125]
```

## When to fine-tune, and with what

Fine-tuning is not always the right tool. Before training anything, it is worth asking three questions.

**Can prompting solve it?** Large decoder models can often do a task with only an instruction and a few examples in the prompt (in-context learning, Chapter 23). This costs no training at all. Fine-tuning becomes attractive when you need consistent output formats, a specialised domain, lower inference cost (a small fine-tuned model can replace a large prompted one), or when you have many labelled examples.

**Is the problem knowledge or behaviour?** Fine-tuning is good at teaching *behaviour*: style, format, the labels of a classification task, how to follow instructions. It is a less reliable way to add many new *facts*; models may learn them only partially and may hallucinate. For fresh or private knowledge, retrieval (Chapter 37) is often better.

**How much data and compute do you have?** With a few hundred examples, a frozen backbone with a linear head, or a parameter-efficient method, is safer. With tens of thousands of examples and enough GPU memory, full fine-tuning can give the best results.

A related idea is **domain adaptation**: before fine-tuning on the labelled task, you continue pre-training (with the original language-modeling objective) on unlabelled text from the target domain, for example a large collection of Russian physics papers. This "continued pre-training" step helps the model learn domain vocabulary and style, and the labelled fine-tuning afterwards becomes easier.

## Summary

- Transfer learning reuses knowledge from a source task (pre-training) for a target task; fine-tuning continues training the pre-trained weights on target data.
- It works because pre-training builds general representations; lower layers tend to be more general, upper layers more task-specific.
- Strategies range from feature extraction with a frozen backbone, through partial and full fine-tuning, to parameter-efficient methods (Chapter 32).
- Catastrophic forgetting and overfitting are the main risks; small learning rates, few epochs, warmup, early stopping, layer-wise learning rates and gradual unfreezing reduce them.
- Fine-tuning is best for teaching behaviour and format; for new facts, retrieval is often more reliable.

## Exercises

1. Explain in your own words why lower layers of a pre-trained model are usually given smaller learning rates than upper layers.
2. A BERT-base model has a hidden size $d = 768$. How many parameters does a linear classification head for 5 classes have? What fraction is this of a 110-million-parameter model?
3. Modify the first code example so that the top *two* backbone layers are trainable. How many trainable parameters do you get? Check your answer by hand.
4. You have 300 labelled medical questions in Russian and access to a 7-billion-parameter model on one GPU. Which strategy from this chapter would you choose first, and why?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| transfer learning | перенос обучения | 전이 학습 |
| fine-tuning | дообучение | 미세 조정 |
| pre-trained model | предобученная модель | 사전 학습된 모델 |
| source task / target task | исходная задача / целевая задача | 원천 과제 / 목표 과제 |
| feature extraction | извлечение признаков | 특징 추출 |
| frozen weights | замороженные веса | 고정된 가중치 |
| classification head | классификационная голова | 분류 헤드 |
| full fine-tuning | полное дообучение | 전체 미세 조정 |
| catastrophic forgetting | катастрофическое забывание | 파국적 망각 |
| layer-wise learning rate | послойная скорость обучения | 층별 학습률 |
| gradual unfreezing | постепенное размораживание | 점진적 고정 해제 |
| learning rate warmup | прогрев скорости обучения | 학습률 워밍업 |
| domain adaptation | доменная адаптация | 도메인 적응 |
| checkpoint | контрольная точка | 체크포인트 |

---
[← Previous: Multilingual Models and Cross-Lingual Transfer](30-multilingual-models.md) · [Contents](README.md) · [Next: Parameter-Efficient Fine-tuning: LoRA and QLoRA →](32-lora-and-peft.md)
