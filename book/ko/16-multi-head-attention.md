# 제16장. 스케일드 닷-프로덕트 어텐션, 멀티헤드 어텐션과 마스킹

> **제4부 — 트랜스포머** · 수준: 중급
>
> **이 장에서 배울 내용:** 어텐션 점수를 왜 $\sqrt{d_k}$로 나누는가 · 여러 어텐션 헤드가 어떻게 문장을 병렬로 바라보는가 · 마스크가 어떻게 토큰이 미래나 패딩을 보지 못하게 하는가

## 하나의 질문에서 여러 질문으로

*"The cat that the dog chased ran away."*(개가 쫓던 고양이가 달아났다)라는 문장을 생각해 봅시다. 모델이 *ran*(달아났다)이라는 단어를 처리할 때는 적어도 두 가지 서로 다른 정보가 필요합니다. 문법적으로는 주어를 찾아야 합니다. 누가 달아났는가? 답은 다섯 단어 앞의 *cat*입니다. 의미적으로는 바로 앞에 무슨 일이 있었는지도 알고 싶을 수 있습니다. *chased*, *dog*입니다. 이것은 같은 문장에 대한 서로 다른 두 질문입니다.

제15장에서 우리는 셀프 어텐션(self-attention)을 만들었습니다. 각 토큰은 쿼리 $q$, 키 $k$, 값 $v$를 만듭니다. 쿼리를 모든 키와 비교하고, 점수를 softmax로 가중치로 바꾼 뒤, 값들의 가중 평균을 출력합니다. 이런 어텐션 연산 하나는 각 토큰에 **하나의** 가중 평균만 주므로, 한 번에 대략 하나의 "질문"에만 답할 수 있습니다. 주어와 최근 동작이 모두 중요하다면 하나의 가중치 집합은 둘 사이에서 타협해야 합니다.

Vaswani 등의 트랜스포머("Attention Is All You Need", 2017)는 이 문제를 간단하게 해결합니다. 각자 고유한 투영을 가진 여러 어텐션 연산을 병렬로 실행하고 그 결과를 합칩니다. 이것이 **멀티헤드 어텐션(multi-head attention)**입니다. 이를 만들기 전에 단일 헤드 어텐션의 수치적 세부 사항 하나(스케일링)와 실용적 세부 사항 하나(마스크)를 정리해야 합니다. 이 세 가지 아이디어가 이 책에 나오는 모든 트랜스포머의 핵심입니다.

## 왜 $\sqrt{d_k}$로 나누는가

제15장의 어텐션 공식을 다시 떠올려 봅시다. 이제 정식 이름인 **스케일드 닷-프로덕트 어텐션(scaled dot-product attention)**으로 부르겠습니다.

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

여기서 $Q$는 쿼리 행렬(토큰당 한 행), $K$는 키 행렬, $V$는 값 행렬이고, $d_k$는 각 쿼리 벡터와 키 벡터의 길이입니다. 곱 $QK^\top$에는 모든 쌍의 내적(dot product) $q_i \cdot k_j$가 들어 있습니다. 행 $i$는 토큰 $i$의 쿼리가 각 키와 얼마나 잘 맞는지를 보여 줍니다.

왜 $\sqrt{d_k}$로 나눌까요? 먼저 직관입니다. 내적은 $d_k$개 항의 합 $q_1k_1 + q_2k_2 + \dots$입니다. 더하는 항이 많을수록 합의 전형적인 크기도 커집니다. $q$와 $k$의 각 성분이 평균 0, 분산 1인 서로 독립적인 난수라고 가정합시다. 그러면 각 곱 $q_m k_m$의 평균은 0, 분산은 1이고, 독립적인 항들의 합의 분산(variance)은 분산들의 합입니다.

$$
\operatorname{Var}(q \cdot k) = \sum_{m=1}^{d_k} \operatorname{Var}(q_m k_m) = d_k .
$$

따라서 점수의 표준편차는 $\sqrt{d_k}$처럼 커집니다. $d_k = 64$이면 전형적인 점수는 약 $\pm 8$이고, $d_k = 512$이면 약 $\pm 22$입니다. 이것은 softmax에 문제가 됩니다. 큰 수의 softmax는 거의 원-핫 벡터가 됩니다. $\text{softmax}(8, 0, 0) \approx (0.9993, 0.0003, 0.0003)$입니다. 이런 분포는 **포화(saturation)** 상태입니다. 가중치 하나는 거의 1이고 나머지는 거의 0이며, 작은 가중치를 통과하는 기울기는 0에 가깝습니다(제11장의 기울기 소실을 떠올려 보십시오). 학습이 느리고 불안정해집니다.

$\sqrt{d_k}$로 나누면 차원과 상관없이 표준편차가 약 1로 돌아옵니다. 수치로 확인해 봅시다.

```python
import torch

torch.manual_seed(0)
for d in (4, 64, 512):
    q, k = torch.randn(10000, d), torch.randn(10000, d)  # 분산 1인 무작위 벡터
    s = (q * k).sum(-1)                                  # 내적 10,000개
    print(d, round(s.std().item(), 2), round((s / d ** 0.5).std().item(), 2))
```

출력은 대략 `4 1.99 1.0`, `64 8.01 1.0`, `512 22.49 0.99`입니다. 원래 표준편차는 $\sqrt{d}$(2, 8, 22.6)를 따르지만, 스케일링한 값은 1에 머뭅니다. 이것이 "스케일드"라는 단어가 붙은 이유의 전부입니다.

## 멀티헤드 어텐션

이제 핵심 아이디어입니다. 폭이 $d_{\text{model}}$(**모델 차원(model dimension)**, 예를 들어 512)인 벡터에 어텐션 연산을 하나 적용하는 대신, 더 작은 연산 $h$개를 실행합니다. 이들을 **헤드(head)**라고 부르며, 각 헤드는 폭이 $d_k = d_{\text{model}} / h$인 벡터(예: $512 / 8 = 64$)에서 동작합니다. 헤드 $i$는 크기가 $d_{\text{model}} \times d_k$인 고유한 투영 행렬(projection matrix) $W_i^Q, W_i^K, W_i^V$를 가집니다.

$$
\text{head}_i = \text{Attention}(X W_i^Q,\; X W_i^K,\; X W_i^V)
$$

여기서 $X$는 입력 토큰 벡터의 행렬(토큰당 한 행)입니다. 각 헤드는 같은 입력을 자기만의 작은 **부분 공간(subspace)**으로 투영하고 자기만의 어텐션 가중치를 계산합니다. 어떤 헤드는 동사와 주어를 연결하는 법을 배우고, 다른 헤드는 이전 토큰을 보고, 또 다른 헤드는 따옴표를 따라갈 수 있습니다. 아무도 헤드에게 무엇을 하라고 알려 주지 않습니다. 전문화는 학습 과정에서 저절로 생겨나며, 어텐션 맵을 시각화하는 연구자들(제40장)은 해석 가능한 헤드와 해석하기 어려운 헤드를 모두 발견합니다.

그다음 모든 헤드의 출력을 폭 $h \cdot d_k = d_{\text{model}}$인 하나의 벡터로 **연결(concatenation)**하고, 크기가 $d_{\text{model}} \times d_{\text{model}}$인 출력 행렬 $W^O$로 섞습니다.

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O
$$

이 설계가 왜 매력적인지 두 가지를 짚어 보겠습니다.

첫째, 비용입니다. 각 헤드는 $h$배 작은 공간에서 동작하므로 전체 파라미터 수와 연산량은 큰 헤드 하나와 거의 같습니다. 크기 $d_{\text{model}} \times d_k$인 행렬 $h$개를 모으면 크기 $d_{\text{model}} \times d_{\text{model}}$인 행렬 하나가 됩니다. 여러 개의 "질문"을 거의 공짜로 얻는 셈입니다.

둘째, 구현입니다. 코드에서는 $h$개의 별도 층을 만들지 않습니다. 모든 쿼리, 키, 값을 한 번에 만드는 큰 선형층 하나를 적용한 다음 결과의 **모양을 바꿉니다(reshape)**. (배치, 토큰, $d_{\text{model}}$) 모양의 텐서가 (배치, 헤드, 토큰, $d_k$)가 됩니다. 어텐션이 끝나면 모양을 되돌리는데, 이것이 바로 연결이며, 그 뒤에 $W^O$를 적용합니다.

## 마스킹: 토큰이 무엇을 볼 수 있는가

어텐션은 모든 토큰이 다른 모든 토큰을 볼 수 있게 합니다. 이것이 잘못된 경우도 있으며, 그때는 일부 연결을 금지하는 **어텐션 마스크(attention mask)**가 필요합니다. 흔한 경우가 두 가지 있습니다.

**인과 마스크.** 텍스트를 왼쪽에서 오른쪽으로 생성하는 언어 모델(제1장, 자세한 내용은 제23장)은 토큰 $1, \dots, t$만으로 토큰 $t+1$을 예측해야 합니다. 학습할 때는 효율을 위해 문장 전체를 한 번에 넣기 때문에, 마스크가 없으면 토큰 3이 토큰 4를 그냥 "보고" 답을 베낄 수 있습니다. **인과 마스크(causal mask)**는 뒤쪽 위치에 대한 어텐션을 금지합니다. 이는 하삼각 행렬입니다. 위치 $i$는 $j \le i$일 때만 위치 $j$에 어텐션할 수 있습니다.

연결은 어떻게 금지할까요? softmax 이후에 가중치를 0으로 만들면 행의 합이 더 이상 1이 아니게 됩니다. 대신 softmax 이전에 *점수*를 $-\infty$로 설정합니다. $e^{-\infty} = 0$이므로 금지된 위치의 가중치는 정확히 0이 되고, 나머지 가중치는 자동으로 다시 정규화됩니다.

토큰 3개로 된 작은 예를 봅시다. 세 번째 토큰의 스케일링된 점수가 $(2, 1, 0)$이라고 합시다. softmax는 $e^2 : e^1 : e^0 = 7.389 : 2.718 : 1$, 즉 $(0.665, 0.245, 0.090)$을 줍니다. 이제 같은 점수 $(2, 1, 0)$을 가진 두 번째 토큰을 봅시다. 위치 3은 이 토큰의 미래이므로 마스크가 점수를 $(2, 1, -\infty)$로 바꾸고, softmax는 $7.389 : 2.718 : 0$, 즉 $(0.731, 0.269, 0)$을 줍니다. 첫 번째 토큰은 자기 자신만 볼 수 있으므로 가중치는 $(1, 0, 0)$입니다.

**패딩 마스크.** 배치 안의 문장들은 길이가 다르므로 짧은 문장은 특수한 패딩 토큰으로 채웁니다(제10장). 이 위치들은 의미가 없으므로 실제 토큰이 여기에 어텐션해서는 안 됩니다. **패딩 마스크(padding mask)**도 같은 방식으로 이들의 점수를 $-\infty$로 설정합니다. 실제로는 인과 마스크와 패딩 마스크를 논리 AND로 결합하는 경우가 많습니다. 한 가지 주의할 점이 있습니다. 어떤 행이 *완전히* 마스킹되면 softmax가 0을 0으로 나누어 NaN을 반환합니다. 라이브러리는 이 경우를 따로 처리하며, 직접 작성한 코드도 이를 피해야 합니다.

다음은 이 장의 모든 내용을 담은 간결한 구현입니다.

```python
import torch
import torch.nn.functional as F

def attention(q, k, v, mask=None):
    # q, k, v: (batch, heads, seq, d_k); mask: True = "볼 수 있음"
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5  # (batch, heads, seq, seq)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)               # 각 행의 합은 1
    return weights @ v, weights
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)  # W_Q, W_K, W_V를 한 행렬에
        self.out = torch.nn.Linear(d_model, d_model)      # W_O
    def forward(self, x, mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        # (B, T, D) -> (B, h, T, d_k): 폭을 h개의 헤드로 나눔
        q, k, v = [t.view(B, T, self.h, self.d_k).transpose(1, 2) for t in (q, k, v)]
        y, w = attention(q, k, v, mask)
        y = y.transpose(1, 2).reshape(B, T, D)            # 헤드들을 연결
        return self.out(y), w

torch.manual_seed(0)
x = torch.randn(1, 4, 8)                                  # 문장 1개, 토큰 4개, d_model = 8
causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))   # 하삼각 마스크
y, w = MultiHeadAttention(8, 2)(x, causal)
print(y.shape, w.shape, w[0, 0].detach().round(decimals=2), sep="\n")  # 헤드 0: 대각선 위는 0
```

출력 모양은 `torch.Size([1, 4, 8])`와 `torch.Size([1, 2, 4, 4])`입니다. 출력은 입력과 모양이 같고, 헤드마다 $4 \times 4$ 가중치 행렬이 하나씩 있습니다. 출력된 헤드 0의 행렬은 대각선 위가 0이고 각 행의 합은 1입니다. 첫 행은 `[1, 0, 0, 0]`입니다. 첫 번째 토큰은 자기 자신에게만 어텐션할 수 있습니다.

PyTorch에는 같은 계산을 훨씬 빠르게 하는 내장 함수 `torch.nn.functional.scaled_dot_product_attention`도 있으며, 제20장에서 이를 사용합니다. 그래도 한 번은 직접 작성해 보는 것이 이해하는 가장 좋은 방법입니다.

## 어텐션의 비용

또 하나 주목할 성질이 있습니다. 토큰 $n$개로 된 시퀀스에서 점수 행렬 $QK^\top$의 크기는 모든 헤드, 모든 층에서 $n \times n$입니다. 따라서 문맥 길이를 두 배로 늘리면 이 부분의 시간과 메모리 비용은 네 배가 됩니다. 이 **이차 복잡도(quadratic complexity)**는 긴 문맥이 어려운 주된 이유이며, 이 책 뒷부분의 많은 아이디어의 동기가 됩니다. KV 캐시와 효율적인 어텐션 커널(제36장), 그리고 헤드들이 키와 값을 공유하는 그룹 쿼리 어텐션(제25장)이 그 예입니다.

어텐션이 *모르는* 것도 눈여겨보십시오. 바로 토큰의 순서입니다. 입력 행을 섞으면 출력도 똑같이 섞일 뿐 그 밖에는 변하지 않습니다. 어텐션에게 "the dog chased the cat"과 "the cat chased the dog"은 같은 토큰 집합입니다. 인과 마스크가 순서 정보를 조금 더해 주지만 충분하지 않습니다. 각 토큰이 어디에 있는지 모델에게 알려 주는 방법이 다음 장의 주제입니다.

## 요약

- 스케일드 닷-프로덕트 어텐션은 점수를 $\sqrt{d_k}$로 나눕니다. 무작위 벡터 내적의 분산은 $d_k$에 비례해 커지고, 큰 점수는 softmax를 포화시키기 때문입니다.
- 멀티헤드 어텐션은 $h$개의 어텐션 헤드를 병렬로 실행합니다. 각 헤드는 폭 $d_{\text{model}}/h$인 자기만의 부분 공간에서 동작하며, 결과를 연결한 뒤 $W^O$로 섞습니다.
- 여러 헤드의 비용은 넓은 헤드 하나와 거의 같지만, 모델이 서로 다른 관계에 동시에 어텐션할 수 있게 합니다.
- 마스크는 softmax 전에 금지된 점수를 $-\infty$로 설정합니다. 인과 마스크는 미래 토큰을, 패딩 마스크는 패딩을 가립니다.
- 어텐션의 비용은 시퀀스 길이에 대해 이차적이며, 어텐션 자체는 토큰 순서를 고려하지 않습니다.

## 연습 문제

1. 마스킹된 점수를 softmax *이후에* 가중치를 0으로 만드는 대신 softmax *이전에* $-\infty$로 설정하는 이유를 자신의 말로 설명해 보십시오.
2. $(4, 0)$의 softmax와 $(4/\sqrt{16}, 0) = (1, 0)$의 softmax를 계산해 보십시오. 스케일링이 분포를 어떻게 바꿉니까?
3. 어떤 모델의 $d_{\text{model}} = 768$이고 헤드는 12개입니다. $d_k$는 얼마입니까? $W^Q, W^K, W^V, W^O$의 파라미터 수를 모두 합하면(편향 제외) 얼마입니까?
4. 코드를 수정하여 길이가 4와 2인 두 문장(4로 패딩)의 배치에 대한 패딩 마스크를 만들고, 이를 인과 마스크와 결합한 뒤 두 번째 문장의 가중치를 출력해 보십시오.

## 핵심 용어

| 한국어 | English | Русский | 러시아어 발음 |
|---|---|---|---|
| 스케일드 닷-프로덕트 어텐션 | scaled dot-product attention | внимание на основе масштабированного скалярного произведения | vnimániye na osnóve masshtabírovannogo skalyárnogo proizvedéniya |
| 내적 | dot product | скалярное произведение | skalyárnoye proizvedéniye |
| 멀티헤드 어텐션 | multi-head attention | многоголовое внимание | mnogogolóvoye vnimániye |
| 어텐션 헤드 | attention head | голова внимания | golová vnimániya |
| 모델 차원 | model dimension | размерность модели | razmérnost' modéli |
| 분산 | variance | дисперсия | dispérsiya |
| 포화 | saturation | насыщение | nasyshchéniye |
| 부분 공간 | subspace | подпространство | podprostránstvo |
| 연결 | concatenation | конкатенация | konkatenátsiya |
| 투영 행렬 | projection matrix | матрица проекции | mátritsa proyéktsii |
| 어텐션 마스크 | attention mask | маска внимания | máska vnimániya |
| 인과 마스크 | causal mask | каузальная маска | kauzál'naya máska |
| 패딩 마스크 | padding mask | маска паддинга | máska páddinga |
| 이차 복잡도 | quadratic complexity | квадратичная сложность | kvadratíchnaya slózhnost' |

---
[← 이전: 셀프 어텐션: 쿼리, 키, 값](15-self-attention.md) · [목차](README.md) · [다음: 위치 정보: 사인파 인코딩, 학습형 인코딩, RoPE, ALiBi →](17-positional-encoding.md)
