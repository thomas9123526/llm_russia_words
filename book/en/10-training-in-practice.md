# Chapter 10. Training in Practice: Batches, Overfitting and Regularization

> **Part 2 — Neural Network Foundations** · Level: Beginner
>
> **In this chapter you will learn:** why we train on mini-batches and what an epoch is · how to detect overfitting with a validation set · how weight decay, dropout and early stopping help a model generalize

## The student who memorized the answers

Imagine a student preparing for an exam with a book of 200 solved problems. One student understands the methods. Another student memorizes every answer by heart. On the practice problems both get 100%. On the real exam, with new problems, only the first student succeeds.

Neural networks can be either student. A network with many parameters can simply memorize its training data, including its noise and accidents, and then fail on new inputs. This is the central practical problem of machine learning. What we really want is **generalization**: good performance on data the model has never seen. In Chapters 8 and 9 we learned how to make the training loss small. This chapter is about the practical side of training: how to feed data efficiently, how to notice memorization, and how to prevent it.

## Batches, iterations and epochs

The loss in Chapter 8 was an average over the *whole* training set. Computing its exact gradient means passing every example through the network before making a single step. For a data set of millions of documents this is far too slow.

Instead, we estimate the gradient on a small random group of examples, a **mini-batch** (often just called a **batch**). If the batch $B$ contains $|B|$ examples, the estimate is

$$\mathbf{g}_B = \frac{1}{|B|} \sum_{i \in B} \nabla_\theta\, \ell(f_\theta(x_i),\ y_i).$$

On average, $\mathbf{g}_B$ equals the true gradient, but each individual estimate is noisy. This is what "stochastic" means in stochastic gradient descent (Chapter 9). The noise is not only a cost: it shakes the parameters a little and can help them escape poor flat regions.

Three words describe the progress of training:
- The **batch size** is the number of examples in one mini-batch.
- One **iteration** (or step) is one update of the parameters, computed from one mini-batch.
- One **epoch** is one full pass through the training set.

For example, with 10,000 training examples and batch size 32, one epoch contains $\lceil 10000 / 32 \rceil = 313$ iterations (the last batch is smaller). Before each epoch the data are **shuffled**, so that batches differ from epoch to epoch.

How large should a batch be? Larger batches give less noisy gradients and use parallel hardware such as GPUs efficiently, but each step costs more memory, and beyond some size the extra precision stops paying off. Smaller batches are noisier but give more updates for the same compute. When the batch size changes, the learning rate usually has to be tuned again. For language models, the batch size is usually counted in *tokens*, not sentences, and large models often see most of their huge training data only once or a few times, so the idea of an epoch matters less there (Chapter 27).

## Generalization: training, validation and test sets

To measure generalization we must evaluate the model on data that it did not train on. The standard practice is to split the data into three parts:
- the **training set**, used to compute gradients and update the parameters;
- the **validation set** (also called the development set), used to compare models, choose hyperparameters such as the learning rate, and decide when to stop training;
- the **test set**, used only once at the end, to report the final quality.

Why three and not two? Every time we choose something by looking at the validation score, we fit the validation set a little. After many such choices the validation score becomes optimistic. The test set stays untouched and gives an honest estimate. In Chapter 39 we will see the same problem in a modern form: test data of public benchmarks can leak into the training data of language models.

By comparing the training loss and the validation loss, we can diagnose two opposite problems:
- **Underfitting**: both losses are high. The model is too simple or has not trained long enough. It cannot even fit the training data.
- **Overfitting**: the training loss is low but the validation loss is much higher, and it often *rises* while the training loss keeps falling. The model memorizes the training data.

A classical illustration is polynomial fitting. We take 15 noisy points from the curve $y = \sin 3x$ and fit a polynomial of degree 12, which has 13 coefficients, almost as many as there are points. We add a penalty $\lambda \|\mathbf{w}\|^2$ on the size of the coefficients (explained in the next section) and look at what different $\lambda$ do:

```python
import numpy as np

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(-1, 1, 15))              # 15 training points
y = np.sin(3 * x) + 0.2 * rng.normal(size=15)    # noisy targets
xv = rng.uniform(-1, 1, 100)                     # 100 validation points
yv = np.sin(3 * xv) + 0.2 * rng.normal(size=100)

def poly(x):                                     # features 1, x, x^2, ..., x^12
    return np.vander(x, 13, increasing=True)

A, Av = poly(x), poly(xv)
for lam in [0.0, 1e-3, 1e-1, 10.0]:              # strength of the L2 penalty
    # minimize ||A w - y||^2 + lam * ||w||^2 (closed-form ridge solution)
    w = np.linalg.solve(A.T @ A + lam * np.eye(13), A.T @ y)
    train = np.mean((A @ w - y) ** 2)
    val = np.mean((Av @ w - yv) ** 2)
    print(f"lambda={lam:<6} train {train:.3f}  val {val:.3f}  |w| {np.linalg.norm(w):.1f}")
```

Without the penalty, the training error is almost zero (about 0.001), but the validation error is enormous (tens), and the norm of the coefficient vector is above ten thousand: the curve passes through every noisy point and swings wildly between them. With $\lambda = 0.001$ the training error is a little higher (about 0.009), but the validation error falls to about 0.14. With $\lambda = 10$ the model is forced to be too simple and underfits: both errors are high (about 0.2 and 0.46). Good generalization lies between the two extremes.

## Regularization: weight decay, dropout, early stopping

Any technique that reduces overfitting without simply making the model smaller is called **regularization**. Four are used everywhere.

**Weight decay (L2 regularization).** We add a penalty on large weights to the loss:

$$L_{\text{total}}(\theta) = L(\theta) + \frac{\lambda}{2}\, \|\theta\|^2.$$

Here $\lambda \ge 0$ controls the strength, and $\|\theta\|^2$ is the sum of squares of all weights. The gradient of the penalty is $\lambda \theta$, so one step of gradient descent becomes

$$\theta \leftarrow \theta - \eta(\nabla L + \lambda\theta) = (1 - \eta\lambda)\,\theta - \eta\, \nabla L.$$

Before every step the weights are multiplied by a number slightly below 1: they "decay" towards zero unless the data keep pushing them. Small weights give smoother functions, which is exactly what we saw with the polynomial. For Adam, adding $\lambda\theta$ to the gradient works poorly, because Adam rescales it. **AdamW** (Loshchilov and Hutter) instead applies the decay directly to the weights, separately from the adaptive step. This "decoupled" weight decay is the standard choice for Transformers.

**Dropout.** During training, each activation in a layer is set to zero with probability $p$ (for example $p = 0.1$), independently at every step, and the surviving activations are multiplied by $1/(1 - p)$ so that their expected value does not change. The network cannot rely on any single neuron, because any neuron may disappear, so it learns redundant, more robust features. At evaluation time dropout is switched off. In PyTorch, `model.train()` and `model.eval()` switch between the two modes; forgetting `model.eval()` is a classic bug.

**Early stopping.** We measure the validation loss regularly during training, remember the parameters with the best validation loss, and stop when it has not improved for a while (the **patience**). This simple method is very effective, because overfitting usually grows with training time.

**More data and data augmentation.** The most reliable cure for overfitting is more data. When new data are expensive, we can create modified copies of existing examples: in images, small rotations and crops; in text, for example, paraphrases or back-translation. For large language models the training set is so big that classical overfitting is less of a problem, but repeated data and memorized test sets still matter (Chapters 26 and 39).

## A complete training loop

The code below puts everything together: a training set and a separate validation set, mini-batches with shuffling, the AdamW optimizer with weight decay, dropout, and early stopping. The model is deliberately too large for only 20 noisy training points.

```python
import copy, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(1)
X = torch.rand(220, 1) * 6 - 3                     # inputs in [-3, 3]
Y = torch.sin(X) + 0.3 * torch.randn_like(X)       # noisy sine wave
train = DataLoader(TensorDataset(X[:20], Y[:20]), batch_size=5, shuffle=True)
Xva, Yva = X[20:], Y[20:]                          # 200 validation points

model = nn.Sequential(nn.Linear(1, 256), nn.ReLU(), nn.Dropout(0.1),
                      nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
best, best_epoch, patience = float("inf"), 0, 50

for epoch in range(1, 1001):
    model.train()                                  # dropout on
    for xb, yb in train:                           # one mini-batch per step
        loss = nn.functional.mse_loss(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()                                   # dropout off
    with torch.no_grad():
        val = nn.functional.mse_loss(model(Xva), Yva).item()
    if val < best:                                 # remember the best model
        best, best_epoch = val, epoch
        best_state = copy.deepcopy(model.state_dict())
    elif epoch - best_epoch >= patience:           # early stopping
        break

model.load_state_dict(best_state)
print(f"stopped at epoch {epoch}, best epoch {best_epoch}, val MSE {best:.3f}")
```

With this seed the best validation loss is reached very early (around epoch 10); after that the training loss keeps falling while the validation loss slowly rises, so training stops 50 epochs later and the best parameters are restored. The noise in the targets has variance $0.3^2 = 0.09$, so no model can reach a validation MSE much below 0.09.

Two more tools appear in almost every real training run, and we will use them later in the book. A **learning rate schedule** changes $\eta$ during training: typically a short **warmup**, where $\eta$ grows from near zero, followed by a slow decay, for example along a cosine curve. And **gradient clipping** rescales the gradient whenever its norm exceeds a threshold $c$, $\mathbf{g} \leftarrow \mathbf{g} \cdot c / \|\mathbf{g}\|$, which protects against exploding gradients (Chapter 9); in PyTorch this is `torch.nn.utils.clip_grad_norm_`. Finally, a practical habit: before training a new model on all data, check that it can overfit one small batch almost perfectly. If it cannot, there is a bug.

## Summary
- We estimate the gradient on random mini-batches; an iteration is one update, an epoch is one pass over the training set.
- Generalization is measured on data not used for training: the validation set guides decisions, the test set gives the final honest score.
- Underfitting means high training and validation loss; overfitting means low training loss but much higher validation loss.
- Weight decay shrinks weights at every step; AdamW applies it separately from Adam's adaptive step.
- Dropout randomly removes activations during training only; early stopping keeps the parameters with the best validation loss.
- Learning rate warmup and decay, and gradient clipping, make real training runs stable.

## Exercises
1. Explain in your own words the difference between the validation set and the test set, and why using the test set to choose hyperparameters is a mistake.
2. A data set has 50,000 examples. With batch size 64, how many iterations are in one epoch? How many iterations are in 10 epochs?
3. With weight decay, one step multiplies the weights by $(1 - \eta\lambda)$. For $\eta = 0.01$ and $\lambda = 0.1$, by what factor does a weight shrink after 1,000 steps if the loss gradient is zero? (Hint: $(1 - a)^n \approx e^{-an}$.)
4. In the training loop, remove dropout and set `weight_decay=0.0`, then set `patience` to 1000 so that training never stops early. Print the training and validation loss every 100 epochs and describe the curves.

## Key terms
| English | Русский | 한국어 |
|---|---|---|
| generalization | обобщение | 일반화 |
| mini-batch | мини-пакет (батч) | 미니배치 |
| batch size | размер батча | 배치 크기 |
| epoch | эпоха | 에포크 |
| training set | обучающая выборка | 훈련 데이터셋 |
| validation set | валидационная выборка | 검증 데이터셋 |
| test set | тестовая выборка | 테스트 데이터셋 |
| overfitting | переобучение | 과적합 |
| underfitting | недообучение | 과소적합 |
| regularization | регуляризация | 정규화 |
| weight decay | затухание весов | 가중치 감쇠 |
| dropout | дропаут | 드롭아웃 |
| early stopping | ранняя остановка | 조기 종료 |
| data augmentation | аугментация данных | 데이터 증강 |
| gradient clipping | обрезка градиентов | 그래디언트 클리핑 |

---
[← Previous: Backpropagation and Optimizers (SGD, Momentum, Adam)](09-backpropagation-and-optimizers.md) · [Contents](README.md) · [Next: Recurrent Neural Networks →](11-recurrent-networks.md)
