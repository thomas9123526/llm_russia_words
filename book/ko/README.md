# LLM: 쉬운 것부터 어려운 것까지

텍스트와 토큰에서 에이전트와 열린 연구 문제까지, 40개 장으로 구성된 대규모 언어 모델 입문서입니다.

각 장은 요약, 연습 문제, 그리고 한국어·영어·러시아어 핵심 용어 표로 끝납니다.

## 목차

### 제1부 — 텍스트와 표현

- [제1장. 언어 모델이란 무엇인가?](01-what-is-a-language-model.md)
- [제2장. 데이터로서의 텍스트: 문자, 단어, 인코딩](02-text-as-data.md)
- [제3장. 토큰화와 서브워드(BPE, WordPiece, SentencePiece)](03-tokenization.md)
- [제4장. 단어 세기: 단어 주머니, N-그램, 퍼플렉서티](04-counting-words.md)
- [제5장. 임베딩: 벡터로서의 단어(Word2Vec, GloVe)](05-embeddings.md)

### 제2부 — 신경망의 기초

- [제6장. 필요한 수학: 벡터, 행렬, 확률](06-math-you-need.md)
- [제7장. 뉴런, 층, 활성화 함수](07-neurons-and-layers.md)
- [제8장. 손실 함수와 경사 하강법](08-loss-and-gradient-descent.md)
- [제9장. 역전파와 옵티마이저 (SGD, 모멘텀, Adam)](09-backpropagation-and-optimizers.md)
- [제10장. 실전 학습: 배치, 과적합, 정규화](10-training-in-practice.md)

### 제3부 — 시퀀스와 어텐션

- [제11장. 순환 신경망](11-recurrent-networks.md)
- [제12장. LSTM과 GRU](12-lstm-and-gru.md)
- [제13장. 시퀀스-투-시퀀스와 인코더–디코더](13-sequence-to-sequence.md)
- [제14장. 어텐션의 아이디어(바다나우, 루옹)](14-the-idea-of-attention.md)
- [제15장. 셀프 어텐션: 쿼리, 키, 값](15-self-attention.md)

### 제4부 — 트랜스포머

- [제16장. 스케일드 닷-프로덕트 어텐션, 멀티헤드 어텐션과 마스킹](16-multi-head-attention.md)
- [제17장. 위치 정보: 사인파 인코딩, 학습형 인코딩, RoPE, ALiBi](17-positional-encoding.md)
- [제18장. 트랜스포머 블록: 잔차 연결, LayerNorm/RMSNorm, FFN, GLU](18-transformer-block.md)
- [제19장. 인코더, 디코더, 인코더–디코더 아키텍처](19-transformer-architectures.md)
- [제20장. PyTorch로 작은 트랜스포머 만들기](20-tiny-transformer.md)

### 제5부 — 실전 언어 모델

- [제21장. 사전 학습 목표: 인과적 언어 모델링과 마스크 언어 모델링](21-pretraining-objectives.md)
- [제22장. BERT와 인코더 모델](22-bert-and-encoders.md)
- [제23장. GPT와 디코더 전용 모델, 문맥 내 학습](23-gpt-and-decoders.md)
- [제24장. 텍스트 생성: 탐욕적 디코딩, 빔 서치, 온도, Top-k, Top-p](24-text-generation.md)
- [제25장. 현대의 오픈 모델: LLaMA, GQA, Mistral, Qwen](25-modern-open-models.md)

### 제6부 — 대규모 학습

- [제26장. 학습 데이터: 수집, 정제, 중복 제거](26-training-data.md)
- [제27장. 스케일링 법칙과 연산 최적 학습](27-scaling-laws.md)
- [제28장. 분산 학습과 혼합 정밀도](28-distributed-training.md)
- [제29장. 전문가 혼합](29-mixture-of-experts.md)
- [제30장. 다국어 모델과 교차 언어 전이](30-multilingual-models.md)

### 제7부 — 모델의 적응과 정렬

- [제31장. 미세 조정과 전이 학습](31-fine-tuning.md)
- [제32장. 매개변수 효율적 미세 조정: LoRA와 QLoRA](32-lora-and-peft.md)
- [제33장. 지시 튜닝](33-instruction-tuning.md)
- [제34장. 정렬: RLHF와 DPO](34-alignment-rlhf-dpo.md)
- [제35장. 추론과 사고의 사슬](35-reasoning.md)

### 제8부 — 현실 세계의 LLM

- [제36장. 효율적인 추론: KV 캐시, 양자화, 추측 디코딩](36-efficient-inference.md)
- [제37장. 검색 증강 생성(RAG)](37-retrieval-augmented-generation.md)
- [제38장. 에이전트와 도구 사용](38-agents-and-tools.md)
- [제39장. 평가, 벤치마크, 환각](39-evaluation-and-hallucination.md)
- [제40장. 안전성, 해석 가능성, 열린 연구 문제](40-safety-and-open-problems.md)
