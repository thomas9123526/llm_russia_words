# 제34장. 정렬: RLHF와 DPO

> **제7부 — 모델의 적응과 정렬** · 수준: 고급
>
> **이 장에서 배울 내용:** 모방(SFT)만으로 충분하지 않은 이유와 인간 선호를 수집하는 방법 · RLHF가 보상 모델을 학습하고 KL 벌점과 함께 정책을 최적화하는 방식 · DPO가 같은 목표를 분류와 비슷한 단순한 손실로 바꾸는 방법

## 쓰는 것보다 판단하는 것이 쉽다

동료에게 1학년 학생을 위한 역전파의 완벽한 설명을 써 달라고 부탁해 보십시오. 어려운 일이며, 좋은 교사 두 명은 매우 다른 글을 쓸 것입니다. 이제 같은 동료에게 두 개의 설명을 보여 주고 "어느 쪽이 더 낫습니까?"라고 물어보십시오. 이것은 훨씬 쉽고, 사람들의 의견도 더 자주 일치합니다.

이 관찰이 선호 기반 **정렬**(alignment)의 출발점입니다. 정렬이란 모델의 행동을 사람들이 실제로 원하는 것, 즉 도움이 되고 정직하며 무해한 답변에 맞추는 것입니다. 33장에서 우리는 모방, 즉 시범 응답에 대한 지도 미세 조정으로 모델을 가르쳤습니다. 모방에는 두 가지 한계가 있습니다. 첫째, 모델은 시범 응답보다 나을 수 없으며, 좋은 시범 응답은 비쌉니다. 둘째, SFT는 "이 답이 좋다"고만 말할 뿐 "이 답이 저 답*보다 낫다*"고는 결코 말하지 않으며, 나쁜 행동을 직접 벌하지도 않습니다. **선호 데이터**(preference data)가 이 공백을 메웁니다. 프롬프트 $x$에 대해 모델이 두 개의 응답을 생성하면, 레이블러가 더 나은 쪽을 표시합니다. 선호된 응답 $y_w$("승자")와 거부된 응답 $y_l$("패자")입니다. 이런 **쌍대 비교**(pairwise comparison) 수천 개가 이 장에서 다루는 두 방법의 학습 데이터가 됩니다.

## RLHF 1단계: 보상 모델

**RLHF**, 즉 인간 피드백 기반 강화 학습(reinforcement learning from human feedback)은 Christiano 외(2017)와 Stiennon 외(2020)의 연구에서 개발되었고, InstructGPT(Ouyang 외, 2022)를 통해 널리 알려졌습니다. 이 방법은 세 단계로 이루어집니다. (1) 지도 미세 조정, (2) 보상 모델 학습, (3) 보상 모델에 대한 강화 학습으로 모델 최적화입니다.

**보상 모델**(reward model) $r_\phi(x, y)$는 보통 SFT 모델에서 초기화되는 네트워크로, 언어 모델링 헤드를 숫자 하나를 출력하는 헤드로 바꾼 것입니다. 이 숫자는 프롬프트 $x$에 대해 응답 $y$가 얼마나 좋은지를 나타내는 점수입니다. 비교 데이터로 이를 학습시키기 위해 쌍대 비교의 고전적 모델인 **브래들리–테리 모델**(Bradley–Terry model)을 사용합니다. 이 모델은 $y_w$가 $y_l$보다 선호될 확률이 두 점수의 차이에만 의존한다고 가정합니다.

$$P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big),$$

여기서 $\sigma(z) = 1/(1 + e^{-z})$는 시그모이드 함수(7장)입니다. 관찰된 선호의 로그 가능도를 최대화하면 다음 손실을 얻습니다.

$$\mathcal{L}_{RM} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big).$$

수치 예제를 들어 봅시다. 보상 모델이 선호된 답에 $2.0$, 거부된 답에 $0.5$의 점수를 준다면 차이는 $1.5$이고, $\sigma(1.5) \approx 0.818$, 손실은 $-\log 0.818 \approx 0.201$입니다. 점수가 반대였다면 손실은 $-\log \sigma(-1.5) \approx 1.701$이 되었을 것입니다. 이 손실은 점수들을 올바른 방향으로 벌려 놓습니다. 보상의 *차이*만 중요하므로 $r_\phi$의 절대적 척도는 임의적이라는 점에 유의하십시오.

## RLHF 2단계: 정책 최적화

강화 학습의 언어로 말하면 언어 모델은 **정책**(policy) $\pi_\theta(y \mid x)$입니다. 상태(프롬프트)가 주어지면 행동(토큰)을 선택합니다. 이제 응답이 높은 보상을 받도록 $\theta$를 바꾸고 싶습니다. 그러나 보상을 단순히 최대화하면, 모델은 보상 모델은 높게 평가하지만 사람들은 좋아하지 않을 응답을 금세 찾아냅니다. 매우 긴 답변, 아첨하는 문구, 반복되는 키워드 같은 것들입니다. 보상 모델은 제한된 비교 데이터로 학습된 인간 판단의 불완전한 근사일 뿐이며, 그 범위 밖에서는 점수가 틀릴 수 있습니다. 이런 오류를 악용하는 것을 **보상 해킹**(reward hacking)이라고 합니다.

표준적인 해결책은 KL 발산(KL divergence, 6장) 벌점을 사용해 정책을 **참조 모델**(reference model) $\pi_{\text{ref}}$, 보통 SFT 모델 가까이에 묶어 두는 것입니다. RLHF의 목표는 다음과 같습니다.

$$\max_{\theta}\; \mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot \mid x)} \big[ r_\phi(x, y) \big] \;-\; \beta\, \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big),$$

여기서 $D$는 프롬프트 집합이고, $\beta > 0$은 참조 모델에서 멀어지는 것에 얼마나 강하게 벌점을 줄지 조절합니다. $\beta$가 작으면 보상은 커지지만 해킹 위험이 커지고, $\beta$가 크면 더 안전하지만 모델이 거의 바뀌지 않습니다.

응답 $y$는 모델에서 *샘플링*되므로 이를 통과하여 단순히 역전파할 수 없습니다. 이것이 강화 학습을 사용하는 이유입니다. RLHF에서 가장 흔히 쓰여 온 알고리즘은 **PPO**(proximal policy optimisation, 근위 정책 최적화, Schulman 외, 2017)로, 한 번의 갱신에서 정책이 얼마나 바뀔 수 있는지를 제한하는 정책 경사 방법입니다. 실제로 PPO 기반 RLHF는 복잡합니다. 학습 중에 정책, 참조 모델, 보상 모델, 그리고 보통 가치 네트워크까지 필요하고, 하이퍼파라미터가 많으며 불안정할 수 있습니다. 이 복잡성 때문에 더 단순한 방법을 찾게 되었습니다.

## DPO: 별도의 보상 모델 없는 정렬

**DPO**, 즉 직접 선호 최적화(direct preference optimisation, Rafailov 외, 2023)는 하나의 수학적 관찰에서 출발합니다. 위의 KL 정규화 목표에 대해 최적 정책은 알려진 닫힌 형태를 가집니다.

$$\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\Big(\frac{1}{\beta} r(x, y)\Big),$$

여기서 $Z(x)$는 확률의 합이 1이 되게 하는 정규화 상수입니다. 직관적으로 최선의 정책은 참조 모델을 가져와 각 응답의 확률을 보상의 지수에 비례하여 높입니다. 이 식을 보상에 대해 풀 수 있습니다.

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x).$$

이제 이것을 브래들리–테리 모델에 대입해 봅시다. 차이 $r(x, y_w) - r(x, y_l)$에서 항 $\beta \log Z(x)$는 $y$에 의존하지 않으므로 상쇄됩니다. 남는 것은 정책과 참조 모델에만 의존하므로 정책 $\pi_\theta$를 직접 학습시킬 수 있습니다.

$$\mathcal{L}_{DPO} = -\log \sigma\Big(\beta \Big[\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big]\Big).$$

논문의 표어는 "여러분의 언어 모델은 사실 보상 모델이다"입니다. 양 $\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$가 **암묵적 보상**(implicit reward)의 역할을 합니다. DPO는 학습 중 샘플링도, 별도의 보상 모델도, RL 알고리즘도 필요하지 않습니다. 고정된 응답 쌍에 대한 손실일 뿐이며, 전체 응답의 로그 확률(33장에서처럼 토큰 로그 확률의 합)로 계산됩니다. 아래 코드는 두 손실을 모두 구현하고 숫자를 확인합니다.

```python
import torch
import torch.nn.functional as F

def reward_model_loss(r_chosen, r_rejected):
    # 브래들리-테리: -log sigma(r_w - r_l)
    return -F.logsigmoid(r_chosen - r_rejected).mean()

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    # 인자는 전체 응답의 로그 확률 합 log pi(y|x)
    margin = (pi_w - ref_w) - (pi_l - ref_l)
    return -F.logsigmoid(beta * margin).mean()

print(f"{reward_model_loss(torch.tensor([2.0]), torch.tensor([0.5])):.3f}")  # 0.201

# 처음에는 정책이 참조 모델과 같음: 손실 = log 2
same = torch.tensor([-11.0])
print(f"{dpo_loss(same, same, same, same):.3f}")                             # 0.693

# 정책이 선택된 답의 확률은 높이고 거부된 답의 확률은 낮춘 경우
pi_w = torch.tensor([-10.0], requires_grad=True)
pi_l = torch.tensor([-12.0], requires_grad=True)
loss = dpo_loss(pi_w, pi_l, torch.tensor([-11.0]), torch.tensor([-11.0]))
loss.backward()
print(f"{loss:.3f}", f"{pi_w.grad.item():.3f}", f"{pi_l.grad.item():.3f}")   # 0.598 -0.045 0.045
```

마지막 경우를 손으로 확인해 봅시다. 마진은 $(-10 + 11) - (-12 + 11) = 1 - (-1) = 2$이고, $\beta = 0.1$이면 $\sigma(0.2) \approx 0.550$, 손실은 $-\log 0.550 \approx 0.598$입니다. $\log \pi_\theta(y_w)$에 대한 그래디언트는 음수이고 $\log \pi_\theta(y_l)$에 대한 그래디언트는 양수이므로, 경사 하강 한 단계는 선호된 답의 확률을 높이고 거부된 답의 확률을 낮춥니다. 그래디언트의 크기 $\beta\,(1 - \sigma(0.2)) \approx 0.045$는 마진이 커질수록 줄어듭니다. 모델이 이미 올바르게 순위를 매기는 쌍은 덜 기여합니다.

## 방법 비교와 한계

DPO는 단순하고 안정적이며 PPO보다 훨씬 적은 메모리와 엔지니어링을 요구하기 때문에 인기를 얻었습니다. 그래도 명시적 보상 모델을 쓰는 RLHF에는 장점이 있습니다. *온-폴리시*(on-policy) 방식이어서 고정된 데이터셋만이 아니라 현재 모델의 새로운 샘플로 학습하며, 학습된 보상 모델은 새로운 응답을 채점하는 데 재사용할 수 있습니다. 어느 접근이 더 나은 최종 모델을 만드는지는 여전히 연구 주제이며, 많은 변형(예: IPO, KTO)이 있습니다. 추론 과제에 대해서는 35장에서 관련된 아이디어인 *검증 가능한* 보상을 이용한 강화 학습을 다룹니다. 이 방식에서는 프로그램이 답의 정답 여부를 확인합니다.

피드백의 출처도 바뀔 수 있습니다. **RLAIF**(reinforcement learning from AI feedback, AI 피드백 기반 강화 학습)에서는 사람 대신 강력한 모델이 비교를 수행합니다. 헌법적 AI(Constitutional AI, Bai 외, 2022)에서는 모델이 글로 적힌 원칙 목록에 따라 자신의 답을 비판하고 수정하며, 그 원칙에 기반하여 AI가 생성한 선호를 학습에 사용합니다.

어떤 방법이든 중요한 한계는 남아 있습니다.

- **레이블러들의 의견이 다르며**, 선호는 문화와 언어에 따라 달라집니다. 모델은 누구의 선호를 배우는 것일까요? 이는 규범이 레이블링 인력과 다를 수 있는 러시아어 및 한국어 사용자에게 중요한 문제입니다.
- **보상 해킹**과 **아첨**(sycophancy): 승인을 얻도록 최적화하면 모델이 옳은 답을 내기보다 사용자에게 동의하거나 자신 있게 들리도록 배울 수 있습니다.
- **길이 편향**: 레이블러와 보상 모델은 더 긴 답을 선호하는 경우가 많아서 모델이 장황해질 수 있습니다.
- **정렬은 해결된 문제가 아닙니다.** 이 방법들은 전형적인 행동을 형성하지만 보장하지는 않습니다. 안전성은 40장에서 다시 다룹니다.

## 요약

- 사람에게는 이상적인 답을 쓰는 것보다 두 답을 비교하는 것이 더 쉽고 일관적입니다. 선호 데이터는 프롬프트에 대한 (선호, 거부) 쌍으로 이루어집니다.
- RLHF는 브래들리–테리 손실 $-\log \sigma(r_w - r_l)$로 보상 모델을 학습한 뒤, 보상에서 참조 모델에 대한 KL 벌점을 뺀 값을 최대화하도록 정책을 (보통 PPO로) 최적화합니다.
- KL 벌점은 보상 해킹을 제한하며, $\beta$는 보상과 참조 모델에 가까이 머무는 것 사이의 균형을 조절합니다.
- DPO는 KL 정규화 목표의 닫힌 형태 해를 이용해, 보상 모델도 RL 루프도 없이 로그 확률 비율에 대한 단순한 손실을 얻습니다.
- 선호 기반 정렬에는 레이블러 간 불일치, 보상 해킹, 아첨, 길이 편향 같은 알려진 문제가 있습니다.

## 연습 문제

1. RLHF에서 참조 모델에 대한 KL 벌점이 필요한 이유를 자신의 말로 설명하십시오.
2. 보상 모델이 선호된 답과 거부된 답에 각각 $r_w = 0.3$, $r_l = 1.1$의 점수를 줍니다. 브래들리–테리 모델에 따른 선호된 답의 승리 확률과 손실을 계산하십시오. 여기서 보상 모델은 맞습니까, 틀립니까?
3. 보상에 대한 DPO 식을 브래들리–테리 모델에 대입하면 항 $\beta \log Z(x)$가 상쇄됨을 보이십시오.
4. `dpo_loss`를 사용하여 코드와 같은 로그 확률에 대해 $\beta = 0.1$과 $\beta = 0.5$일 때의 손실을 계산하십시오. $\beta$는 손실과 그래디언트를 어떻게 바꿉니까?

## 핵심 용어

| 한국어 | English | Русский | 러시아어 발음 |
|---|---|---|---|
| 정렬 | alignment | выравнивание | vyrávnivaniye |
| 선호 데이터 | preference data | данные о предпочтениях | dánnye o predpochténiyakh |
| 쌍대 비교 | pairwise comparison | попарное сравнение | popárnoye sravnéniye |
| 강화 학습 | reinforcement learning | обучение с подкреплением | obuchéniye s podkrepléniyem |
| 보상 모델 | reward model | модель вознаграждения | modél' voznagrazhdéniya |
| 브래들리–테리 모델 | Bradley–Terry model | модель Брэдли–Терри | modél' Brédli-Térri |
| 정책 | policy | стратегия | stratégiya |
| 참조 모델 | reference model | референсная модель | referénsnaya modél' |
| KL 발산 | KL divergence | расхождение Кульбака–Лейблера | raskhozhdéniye Kul'báka-Léyblera |
| 보상 해킹 | reward hacking | взлом функции вознаграждения | vzlom fúnktsii voznagrazhdéniya |
| 직접 선호 최적화 | direct preference optimisation (DPO) | прямая оптимизация предпочтений | pryamáya optimizátsiya predpochténiy |
| 암묵적 보상 | implicit reward | неявное вознаграждение | neyávnoye voznagrazhdéniye |
| 아첨 | sycophancy | угодничество | ugódnichestvo |
| 레이블러 | annotator | разметчик | razmétchik |

---
[← 이전: 지시 튜닝](33-instruction-tuning.md) · [목차](README.md) · [다음: 추론과 사고의 사슬 →](35-reasoning.md)
