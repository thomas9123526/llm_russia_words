# 제29장. 전문가 혼합

> **제6부 — 대규모 학습** · 수준: 고급
>
> **이 장에서 배울 내용:** 전문가 혼합 층은 어떻게 매개변수 수와 토큰당 연산량을 분리하는가 · 라우터는 어떻게 top-$k$ 전문가를 고르고 그 출력을 어떻게 합치는가 · 부하 분산이 왜 필요하며 MoE는 메모리와 통신 측면에서 어떤 비용을 치르는가

언어 모델이 한 배치에서 함께 처리할 수 있는 토큰 세 개를 생각해 봅시다. "2 + 2 ="의 숫자, 파이썬 코드 한 줄의 키워드, 러시아어 시의 한 단어입니다. 밀집 트랜스포머(제18장)에서는 세 토큰 모두 똑같은 피드포워드 가중치를 통과하며, 모든 매개변수가 모든 계산에 참여합니다. 꼭 그래야 할까요? 사람으로 이루어진 전문가 팀이라면 각 질문을 그 주제를 가장 잘 아는 사람에게 보낼 것입니다.

이것이 **조건부 연산(conditional computation)**, 즉 각 토큰이 네트워크의 일부만 사용하게 하자는 아이디어입니다. LLM에서 가장 성공한 형태는 **전문가 혼합(Mixture of Experts, MoE)** 입니다. 제27장에서 밀집 모델의 학습 비용이 $C \approx 6ND$, 즉 연산량이 매개변수 수에 묶여 있음을 보았습니다. MoE는 이 연결을 끊습니다. 모델은 개별 토큰에 사용하는 것보다 훨씬 많은 매개변수를 가질 수 있습니다.

## 전문가와 라우터

MoE 트랜스포머는 어텐션은 그대로 두고, 일부 또는 모든 블록의 피드포워드 네트워크(FFN)를 MoE 층으로 바꿉니다. 이 층에는 다음이 들어 있습니다.

- $n$개의 **전문가(expert network)** $E_1, \dots, E_n$ — 보통 원래 FFN과 같은 모양의 평범한 FFN입니다.
- **라우터(router)**, 또는 **게이팅 네트워크(gating network)** — 각 토큰이 어떤 전문가에게 가야 할지 정하는 작은 선형 층입니다.

각 토큰에 대해 $n$개 전문가 중 $k$개만 계산합니다. 전형적인 값은 $k = 1$ 또는 $k = 2$이고, $n$은 8에서 수백 개까지입니다. 따라서 이 층은 FFN 매개변수를 $n$배 담고 있지만, 토큰당 연산은 FFN의 약 $k$배만 듭니다. 그래서 **전체 매개변수**(저장된 모든 것)와 **활성 매개변수(active parameters)**(토큰 하나에 쓰이는 것)를 구분합니다. 이렇게 만든 모델을 **희소 모델(sparse model)** 이라 하고, 모든 매개변수가 활성인 **밀집 모델(dense model)** 과 대비합니다.

이 아이디어는 오래되었습니다. Jacobs, Jordan, Nowlan, Hinton이 1991년에 "국소 전문가들의 적응적 혼합"을 제안했습니다. Shazeer 등(2017)은 LSTM 층 사이에 희소 게이팅 MoE 층을 넣어 대규모에서 작동하게 만들었습니다. GShard와 **Switch Transformer**(Fedus 등, 2021, $k = 1$)는 이를 트랜스포머로 가져왔습니다. 잘 알려진 공개 사례는 **Mixtral 8x7B**(Mistral AI)입니다. 층마다 전문가 8개와 top-2 라우팅을 사용하며, 전체 매개변수는 약 470억 개이지만 토큰당 활성 매개변수는 약 130억 개뿐입니다. 이름에서 짐작되는 것보다 적은 이유는 어텐션과 임베딩이 공유되어 8배가 되지 않기 때문입니다.

## 라우팅은 어떻게 작동하는가

토큰 하나의 은닉 벡터를 $x \in \mathbb{R}^d$, 라우터의 가중치 행렬을 $W_r \in \mathbb{R}^{d \times n}$이라고 합시다. 라우터는 전문가마다 점수를 하나씩 계산하고 softmax로 확률로 바꿉니다.

$$
p = \mathrm{softmax}(x W_r), \qquad p \in \mathbb{R}^n, \quad \sum_{i=1}^{n} p_i = 1 .
$$

$p_i$가 가장 큰 $k$개 전문가의 집합을 $T$라고 합시다. 층의 출력은 선택된 전문가 출력의 가중합입니다.

$$
y = \sum_{i \in T} g_i \, E_i(x), \qquad g_i = \frac{p_i}{\sum_{j \in T} p_j} .
$$

가중치 $g_i$는 선택된 전문가들에 대해 다시 정규화한 라우터 확률이므로 합이 1입니다. $T$ 밖의 전문가는 아예 계산하지 않습니다. 절약은 바로 여기에서 생깁니다. 이런 선택 방식을 **top-k 라우팅(top-k routing)** 이라고 합니다.

**예제로 살펴봅시다.** 전문가 $n = 4$개, $k = 2$, 라우터 점수 $x W_r = (2.0,\ 1.0,\ 0.5,\ -1.0)$이라고 합시다. softmax를 적용하면

$$
p \approx (0.61,\ 0.22,\ 0.14,\ 0.03)
$$

입니다. 상위 두 개는 전문가 1과 2입니다. 다시 정규화하면 $g_1 = 0.61 / 0.83 \approx 0.73$, $g_2 \approx 0.27$입니다. 따라서 $y = 0.73\,E_1(x) + 0.27\,E_2(x)$이고, 전문가 3과 4는 이 토큰을 위해 아무 일도 하지 않습니다.

라우터는 어떻게 학습할까요? $T$의 선택은 이산적이어서 기울기가 없지만, 가중치 $g_i$는 $W_r$에 매끄럽게 의존합니다. 전문가 1이 유용한 출력을 내면 손실의 기울기가 $W_r$를 통해 $g_1$을 키우고, 비슷한 토큰들이 앞으로 전문가 1을 고를 가능성이 높아집니다. 라우터와 전문가들은 평범한 역전파로 처음부터 끝까지 함께 학습됩니다.

## 전문가 간 균형 유지하기

이 피드백 고리에는 위험이 숨어 있습니다. 학습 초기에 어떤 전문가가 우연히 조금 더 낫습니다. 라우터는 그 전문가에게 더 많은 토큰을 보냅니다. 그 전문가는 더 빨리 학습되어 더욱 나아집니다. 곧 대부분의 토큰이 소수의 전문가에게 가고 나머지는 놀게 됩니다. 이런 **붕괴(collapse)** 는 매개변수를 낭비하며, 여러 GPU에서는 과부하된 전문가가 병목이 되어 정체까지 일으킵니다.

표준 해법은 **부하 분산(load balancing)** 을 장려하는 **보조 손실(auxiliary loss)** 입니다. Switch Transformer 방식에서는 토큰 $B$개로 된 배치에 대해

$$
\mathcal{L}_{\text{aux}} = n \sum_{i=1}^{n} f_i \, P_i
$$

이며, 여기서 $f_i$는 전문가 $i$를 첫 번째로 선택한 토큰의 비율이고 $P_i$는 배치 전체에서 전문가 $i$에 대한 라우터 확률의 평균입니다. 라우팅이 완벽히 균등하면 $f_i = P_i = 1/n$이고 $\mathcal{L}_{\text{aux}} = n \cdot n \cdot \frac{1}{n^2} = 1$로 최솟값이 됩니다. 한 전문가가 모든 것을 가져가면 $f$와 $P$가 그 전문가에 몰려 손실은 $n$ 쪽으로 커집니다. 비율 $f_i$는 미분할 수 없지만 $P_i$는 미분할 수 있으므로, 기울기는 과부하된 전문가의 확률을 낮춥니다. 보조 손실은 작은 계수(흔히 $10^{-2}$ 정도)를 곱해 언어 모델링 손실에 더합니다.

실제로는 두 가지 장치를 더 사용합니다.

- **전문가 용량(expert capacity).** 효율적인 배치 연산을 위해 각 전문가는 배치당 최대 $\text{capacity} = c \cdot B / n$개의 토큰만 받습니다. 여기서 $c$(*용량 계수*, 예: 1.25)는 약간의 여유를 줍니다. 용량을 넘는 토큰은 그 층에서 **드롭(token dropping)** 됩니다. 즉 전문가를 건너뛰고 잔차 연결(제18장)을 통해 변하지 않은 채 다음으로 넘어갑니다.
- **라우터 z-손실(router z-loss).** 라우터 로짓의 크기에 작은 벌점을 주어(ST-MoE에서 도입) 로짓이 너무 커지지 않게 하며, 수치 안정성을 높입니다. 이는 bf16(제28장)에서 중요합니다.

더 새로운 설계는 전문가 자체를 다듬습니다. 예를 들어 DeepSeekMoE는 전문가를 더 작은 여러 전문가로 쪼개고, 모든 토큰이 항상 사용하는 **공유 전문가(shared experts)** 를 추가하여 공통 지식을 모든 전문가에 복사하지 않아도 되게 합니다.

## 최소한의 MoE 층

아래 코드는 top-$k$ 라우팅과 보조 손실을 구현합니다. 이해하기 쉽도록 전문가에 대해 반복문을 돌리지만, 실제 구현은 토큰을 전문가별로 묶고 전용 커널을 사용합니다.

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

    def forward(self, x):                                  # x: (토큰, d_model)
        probs = F.softmax(self.router(x), dim=-1)          # (토큰, n_experts)
        top_p, top_i = probs.topk(self.k, dim=-1)          # 토큰마다 상위 k개 전문가
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)    # k개 가중치를 다시 정규화
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            tok, slot = (top_i == e).nonzero(as_tuple=True)  # 전문가 e로 보낸 토큰
            out[tok] += top_p[tok, slot, None] * self.experts[e](x[tok])
        # 부하 분산 손실(Switch Transformer): n * sum_i f_i * P_i
        f = F.one_hot(top_i[:, 0], self.n_experts).float().mean(dim=0)  # top-1 비율
        P = probs.mean(dim=0)                              # 라우터 확률의 평균
        aux_loss = self.n_experts * (f * P).sum()
        return out, aux_loss

torch.manual_seed(0)
y, aux = MoE()(torch.randn(10, 16))
print(y.shape, f"aux_loss={aux.item():.3f}")   # 균형이 맞으면 aux_loss ≈ 1
```

이 코드는 `torch.Size([10, 16]) aux_loss=1.044`를 출력합니다. FFN을 대체하는 층답게 출력은 입력과 같은 모양이고, 막 초기화된 라우터는 균형에 가깝습니다. 전문가 4개 각각은 밀집 FFN 하나만큼의 매개변수를 갖지만, 각 토큰은 그중 2개만 거칩니다.

## MoE의 비용

MoE는 공짜가 아닙니다. 주요 비용은 다음과 같습니다.

- **메모리.** 토큰당 일부만 사용하더라도 $n$개 전문가를 모두 저장해야 합니다. Mixtral은 470억 매개변수분의 메모리가 필요하지만 130억 매개변수 모델처럼 계산합니다. 장치 하나에서 추론할 때는 연산이 아니라 메모리가 한계가 되는 경우가 많습니다.
- **통신.** **전문가 병렬화(expert parallelism)** 에서는 서로 다른 전문가가 서로 다른 GPU에 있습니다. 그러면 MoE 층마다 **올투올(all-to-all) 통신**이 필요합니다. 모든 GPU가 자기 토큰을 선택된 전문가의 GPU로 보내고 결과를 다시 받습니다. 이는 제28장의 병렬화 차원에 하나를 더합니다.
- **학습 안정성과 미세 조정.** 희소 모델은 역사적으로 불안정성에 더 취약했고 미세 조정(제31장) 시 과적합되기 쉬웠습니다. 이 때문에 z-손실 같은 도구가 나왔습니다.

전문가들은 무엇을 배울까요? "물리 전문가"나 "러시아어 전문가"를 기대할 수도 있지만, 분석 결과(예: Mixtral 논문)에서는 주제별로 뚜렷한 전문화가 거의 발견되지 않았습니다. 라우팅은 구문과 토큰 유형에 더 많이 좌우되고, 연속된 토큰은 흔히 같은 전문가에게 갑니다. 다음 장의 주제인 다국어 모델에서 전문가가 언어별로 전문화되는지는 흥미로운 연구 문제입니다.

MoE가 인기 있는 이유는 제27장의 스케일링 양상 때문입니다. 같은 학습 연산량에서, 전체 매개변수가 더 많은 희소 모델은 대개 밀집 모델보다 낮은 손실에 도달합니다. MoE는 FLOPs 대신 메모리로 용량을 사는 셈입니다.

## 요약

- 전문가 혼합 층은 FFN을 $n$개의 전문가 FFN과 라우터로 바꿉니다. 각 토큰은 그중 $k$개만 사용하므로, 전체 매개변수는 늘어나도 토큰당 연산은 거의 일정합니다.
- 라우터는 $p = \mathrm{softmax}(xW_r)$를 계산하고 top-$k$ 전문가를 골라 다시 정규화한 가중치 $g_i$로 출력을 합칩니다. 라우터는 이 가중치를 통한 역전파로 학습됩니다.
- 균형을 잡지 않으면 라우팅이 소수 전문가로 붕괴합니다. 보조 손실 $n \sum_i f_i P_i$, 전문가 용량, 라우터 z-손실이 학습을 효율적이고 안정적으로 유지합니다.
- 예로는 Switch Transformer(top-1)와 Mixtral 8x7B(전문가 8개, top-2, 전체 약 470억·활성 약 130억 매개변수)가 있습니다.
- MoE는 연산을 메모리와 통신으로 맞바꿉니다. 모든 전문가를 저장해야 하고, 전문가 병렬화에는 올투올 통신이 필요합니다.

## 연습 문제

1. 전체 매개변수와 활성 매개변수의 차이, 그리고 MoE 모델을 두 숫자로 함께 설명하는 이유를 자신의 말로 설명하십시오.
2. 라우터 점수가 $(0.0,\ 1.5,\ 1.5,\ -0.5)$이고 $k = 2$일 때, softmax 확률, 선택된 전문가, 그 가중치 $g_i$를 계산하십시오.
3. $n = 4$에서 모든 토큰이 전문가 1을 첫 번째로 고르고 평균 확률이 $P = (0.7, 0.1, 0.1, 0.1)$일 때 $\mathcal{L}_{\text{aux}}$를 계산하십시오. 균형 상태의 값과 비교하십시오.
4. 각 전문가가 받는 토큰 수를 세도록 코드를 수정하십시오(예: `torch.bincount(top_i.flatten(), minlength=4)` 사용). 무작위 토큰 1000개로 실행해 보십시오. 초기화 시점에 라우팅은 균형을 이룹니까?

## 핵심 용어

| 한국어 | English | Русский | 러시아어 발음 |
|---|---|---|---|
| 전문가 혼합 | mixture of experts (MoE) | смесь экспертов | smes' ekspértov |
| 조건부 연산 | conditional computation | условные вычисления | uslóvnyye vychisléniya |
| 전문가 네트워크 | expert network | экспертная сеть | ekspértnaya set' |
| 라우터 | router | маршрутизатор | marshrutizátor |
| 게이팅 네트워크 | gating network | управляющая сеть | upravlyáyushchaya set' |
| top-k 라우팅 | top-k routing | маршрутизация top-k | marshrutizátsiya top-ká |
| 희소 모델 | sparse model | разреженная модель | razrezhyónnaya módel' |
| 밀집 모델 | dense model | плотная модель | plótnaya módel' |
| 활성 매개변수 | active parameters | активные параметры | aktívnyye paramétry |
| 부하 분산 | load balancing | балансировка нагрузки | balansiróvka nagrúzki |
| 보조 손실 | auxiliary loss | вспомогательная функция потерь | vspomogátel'naya fúnktsiya potér' |
| 전문가 용량 | expert capacity | ёмкость эксперта | yómkost' ekspérta |
| 토큰 드롭 | token dropping | отбрасывание токенов | otbrásyvaniye tokénov |
| 전문가 병렬화 | expert parallelism | экспертный параллелизм | ekspértnyy parallelízm |
| 올투올 통신 | all-to-all communication | коммуникация all-to-all | kommunikátsiya ol-tu-ól |

---
[← 이전: 분산 학습과 혼합 정밀도](28-distributed-training.md) · [목차](README.md) · [다음: 다국어 모델과 교차 언어 전이 →](30-multilingual-models.md)
