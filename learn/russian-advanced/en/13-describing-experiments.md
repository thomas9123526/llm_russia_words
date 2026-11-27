# Lesson 13. Describing Experiments, Data and Setup

> **Advanced Russian for Researchers** · Level: C1 · **You will learn:** four ways to say who did what in a method section, and when to use each · the prepositions and fixed phrases of an experimental setup (в качестве, при, по, на, размером) · verbs for describing a dataset: состоять из, включать, насчитывать, составлять

## Text

Minho is writing his first article for a Russian journal. Ivan Petrovich has read the draft and said: «Эксперимент описан так, что его никто не повторит» ("The experiment is described so that nobody could repeat it"). Below is the rewritten section 3. The dataset RuSciQA, the models and all numbers are invented for this textbook.

**3. Экспериментальная часть**

**3.1. Данные.** Для оценки предложенного метода был использован набор данных RuSciQA, собранный авторами в рамках настоящей работы. Набор включает 6 200 пар «вопрос — ответ», составленных на основе аннотаций русскоязычных статей по информатике и прикладной математике. Исходный корпус насчитывал около 9 000 пар; после удаления дубликатов и фильтрации некорректных примеров было оставлено 6 200 пар. Разметка выполнялась тремя аннотаторами независимо друг от друга; спорные случаи рассматривались совместно. Согласованность разметки, измеренная с помощью коэффициента каппа Коэна, составила 0,81. Данные были случайным образом разделены на обучающую, валидационную и тестовую выборки в соотношении 80 : 10 : 10.

*3.1. Data. To evaluate the proposed method, we used the RuSciQA dataset, collected by the authors within the present work. The dataset includes 6,200 question–answer pairs compiled on the basis of abstracts of Russian-language papers in computer science and applied mathematics. The original corpus numbered about 9,000 pairs; after removing duplicates and filtering out incorrect examples, 6,200 pairs were kept. Annotation was carried out by three annotators independently of each other; disputed cases were discussed jointly. Inter-annotator agreement, measured with Cohen's kappa, was 0.81. The data were randomly split into training, validation and test sets in the ratio 80 : 10 : 10.*

**3.2. Модели.** В качестве базовых моделей использовались две открытые языковые модели размером 7 и 13 млрд параметров. Обе модели были предварительно обучены на многоязычных корпусах, однако не дообучались на научных текстах. Для сравнения мы также оценили качество базовых моделей без дообучения (в режиме zero-shot) и с полным дообучением всех параметров.

*3.2. Models. Two open language models with 7 and 13 billion parameters were used as base models. Both models were pre-trained on multilingual corpora but had not been fine-tuned on scientific texts. For comparison, we also evaluated the base models without fine-tuning (zero-shot) and with full fine-tuning of all parameters.*

**3.3. Параметры обучения.** Дообучение проводилось с помощью адаптеров низкого ранга (LoRA) с рангом r = 16. Обучение длилось три эпохи при размере батча 32 и скорости обучения 2·10⁻⁴. Оптимальные значения гиперпараметров подбирались по валидационной выборке. Все эксперименты проводились на одном графическом процессоре с 80 ГБ памяти; одна эпоха обучения модели размером 7 млрд параметров занимала около двух часов.

*3.3. Training parameters. Fine-tuning was performed with low-rank adapters (LoRA) of rank r = 16. Training lasted three epochs with a batch size of 32 and a learning rate of 2·10⁻⁴. Optimal hyperparameter values were selected on the validation set. All experiments were run on a single GPU with 80 GB of memory; one training epoch of the 7-billion-parameter model took about two hours.*

**3.4. Метрики оценки.** Качество ответов оценивалось по двум метрикам: доле точных совпадений (exact match, EM) и F1-мере на уровне токенов. Чтобы обеспечить воспроизводимость результатов, каждый эксперимент повторялся с тремя различными начальными значениями генератора случайных чисел; в таблицах приводятся средние значения и стандартные отклонения. Код и данные будут опубликованы в открытом доступе после завершения рецензирования.

*3.4. Evaluation metrics. Answer quality was evaluated with two metrics: exact match rate (EM) and token-level F1. To ensure reproducibility, each experiment was repeated with three different random seeds; the tables report means and standard deviations. The code and data will be made publicly available after the review process is completed.*

## New words

| Russian | Pronunciation | English |
|---|---|---|
| эксперимента́льная часть | eksperimentál'naya chast' | experimental section (ж.) |
| вы́борка | výborka | sample; data split (ж.) |
| обуча́ющая / валидацио́нная / тесто́вая вы́борка | obucháyushchaya / validatsiónnaya / testóvaya výborka | training / validation / test set |
| разме́тка | razmétka | annotation, labelling (ж.) |
| аннота́тор | annotátor | annotator (м.) |
| согласо́ванность | soglasóvannost' | agreement (between annotators) (ж.) |
| дублика́т | dublikát | duplicate (м.) |
| соотноше́ние | sootnoshénie | ratio, proportion (ср.) |
| ба́зовая моде́ль | bázovaya modél' | base model, baseline model (ж.) |
| гиперпара́метр | giperparámetr | hyperparameter (м.) |
| ско́рость обуче́ния | skórost' obuchéniya | learning rate (ж.) |
| разме́р ба́тча | razmér bátcha | batch size (м.) |
| эпо́ха | epókha | epoch (ж.) |
| графи́ческий проце́ссор | grafícheskiy protséssor | GPU (м.) |
| ме́трика | métrika | metric (ж.) |
| до́ля то́чных совпаде́ний | dólya tóchnykh sovpadéniy | exact match rate (ж.) |
| воспроизводи́мость | vosproizvodímost' | reproducibility (ж.) |
| станда́ртное отклоне́ние | standártnoe otklonénie | standard deviation (ср.) |
| насчи́тывать | nashchítyvat' | to number, to count (so many items) |
| в ка́честве (+ gen.) | v káchestve | as, in the role of |

Note the pronunciation of **насчи́тывать**: the letters сч are pronounced like щ, as in счёт and сча́стье.

## Grammar

### 1. Who did it? Four ways to describe a procedure

An experimental section answers the questions *what was done, with what, and how*. Russian has four constructions for this, and they are not interchangeable.

| Construction | Example | Use it for |
|---|---|---|
| был / была́ / бы́ло / бы́ли + short passive participle (perfective) | Данные были разделены на три выборки. | one completed action with a visible result |
| imperfective verb + -ся | Разметка выполнялась тремя аннотаторами. | a process, a procedure, a repeated step |
| мы + verb | Мы также оценили качество базовых моделей. | your own decision or contribution ("author's we") |
| 3rd person plural without a subject | Данные разделили на три выборки. | speech, blogs; avoid in a paper |

The choice between the first two follows aspect, which you already know: a perfective verb gives the participle (разделить → разделены, удалить → удалены, собрать → собран), an imperfective verb takes -ся (выполнять → выполнялась, подбирать → подбирались, проводить → проводились). The performer, if you name it, goes into the **instrumental**: тремя аннотаторами, авторами.

- Набор данных был собран авторами. — *The dataset was collected by the authors.*
- Дубликаты были удалены. — *Duplicates were removed.*
- Гиперпараметры подбирались по валидационной выборке. — *Hyperparameters were selected on the validation set.*
- Каждый эксперимент повторялся три раза. — *Each experiment was repeated three times.*

**Trap: perfective + -ся is not a passive.** Модель обучилась means "the model learned", as if by itself, and Данные разделились means "the data split up" on their own. When you mean that *you* trained the model, write Модель была обучена or Модель обучалась.

**Мы in Russian papers.** The author's мы is normal in Russian scientific writing, especially to mark your own choices: Мы также оценили…, Мы выбрали ранг r = 16, поскольку… But a method section written entirely with мы sounds like a lab diary. A good rhythm is mostly passive and -ся, with мы for the decisions you want to stand out.

**Tense.** Describe what you did in the past tense (были разделены, проводилось). Use the present for what is true of the paper itself: в таблицах приводятся средние значения, в качестве метрики используется F1-мера. The future is only for promises: код будет опубликован.

### 2. Prepositions and fixed phrases of a setup

Most of the information in a setup paragraph sits in a few prepositional phrases. Learn them as ready-made blocks with their case.

| Phrase | Example | Meaning |
|---|---|---|
| в ка́честве + gen. | В качестве базовых моделей использовались… | as (in the role of) |
| на + prep. | дообучать на корпусе; на одном графическом процессоре | on (data, hardware) |
| по + dat. | оценивать по двум метрикам; подбирать по валидационной выборке | by, on the basis of |
| при + prep. | при размере батча 32; при скорости обучения 2·10⁻⁴ | with (a parameter value) |
| с + instr. | LoRA с рангом r = 16; с тремя начальными значениями | with |
| с по́мощью + gen. | с помощью адаптеров LoRA; с помощью коэффициента каппа Коэна | by means of |
| в тече́ние + gen. | в течение трёх эпох | for (a duration) |
| в соотноше́нии | в соотношении 80 : 10 : 10 | in the ratio |
| разме́ром / объёмом + quantity | модель размером 7 млрд параметров; корпус объёмом 2 млн токенов | of size |

Two remarks:

- **размером, объёмом** are instrumentals of measure. In conversation people say модель на 7 миллиардов (параметров), but in a paper write модель размером 7 млрд параметров.
- Duration can be given without a preposition: Обучение длилось три эпохи. With an imperfective verb of process, use в течение: Модель обучалась в течение трёх эпох.

More examples:
- В качестве метрики использовалась F1-мера. — *F1 was used as the metric.*
- Модель дообучалась на корпусе научных статей. — *The model was fine-tuned on a corpus of scientific papers.*
- Обучение проводилось при размере батча 16. — *Training was carried out with a batch size of 16.*
- Корпус объёмом 2 млн токенов был собран из открытых источников. — *A corpus of 2 million tokens was collected from open sources.*

### 3. Describing a dataset: состоять из, включать, содержать, насчитывать, составлять

These verbs look similar but have different government and meaning.

| Verb | Government | Example | Meaning |
|---|---|---|---|
| состоя́ть | из + gen. | Набор данных состоит из двух частей. | consists of (parts) |
| включа́ть | + acc. | Набор включает 6 200 пар. | includes |
| содержа́ть | + acc. | Корпус содержит тексты статей. | contains |
| насчи́тывать | + acc. (a number) | Корпус насчитывал около 9 000 пар. | numbers, counts |
| составля́ть | + acc. (a value) | Согласованность составила 0,81. | amounts to, is |
| охва́тывать | + acc. | Набор охватывает две научные области. | covers |

**Составлять** is the most useful verb of the whole section: it introduces any value, share or result. Вопросы по информатике составляют 60 % набора данных. Точность составила 74,2 %. Do not confuse it with состоять: *составляет из* does not exist. Also remember the preposition after состоять: состоит из 6 200 пар, not *состоит 6 200 пар*.

Numbers: after a compound number, the noun agrees with the last word: 6 200 (шесть тысяч двести) пар, 21 (двадцать одна) пара, 32 (тридцать два) примера. Decimals are read with це́лых and the fraction: 0,81 = ноль целых восемьдесят одна сотая.

## Study tip

Before you send a method section to your supervisor, check it against a short reproducibility list, written in the phrases of this lesson: **данные** (откуда, сколько, как размечены, как разделены), **модели** (что использовалось в качестве базовых, размером сколько параметров), **обучение** (с помощью чего, сколько эпох, при каком размере батча и скорости обучения, на каком оборудовании), **оценка** (по каким метрикам, сколько запусков, что приводится в таблицах). If a reader cannot answer one of these questions from your text, add one sentence. Russian reviewers often write exactly this remark: «Не указаны условия проведения эксперимента».

## Exercises

**Exercise 1.** Answer the questions about the text.
1. Из чего состоит набор данных RuSciQA и на основе чего он составлен?
2. Сколько пар было в исходном корпусе и почему их стало меньше?
3. Как выполнялась разметка?
4. Какие модели использовались в качестве базовых?
5. Сколько эпох длилось обучение и каким был размер батча?
6. Как авторы обеспечили воспроизводимость результатов?

**Exercise 2.** Rewrite each sentence without мы. Use был + short participle for a perfective verb and -ся for an imperfective verb.
1. Мы удалили дубликаты.
2. Мы собрали корпус из открытых источников.
3. Мы подбирали гиперпараметры по валидационной выборке.
4. Мы оценивали качество ответов по двум метрикам.
5. Мы разделили данные на три выборки.
6. Мы проводили все эксперименты на одном графическом процессоре.

**Exercise 3.** Fill in the gaps. Use each phrase once: в качестве, при, по, на, с помощью, в течение.
1. ___ базовой модели использовалась открытая языковая модель.
2. Обучение проводилось ___ размере батча 16.
3. Гиперпараметры подбирались ___ валидационной выборке.
4. Модель дообучалась ___ корпусе научных статей.
5. Согласованность разметки измерялась ___ коэффициента каппа Коэна.
6. Модель обучалась ___ трёх эпох.

**Exercise 4.** Fill in the gaps with a verb from the list in the right form: состоять, включать, насчитывать, составить, составлять.
1. Набор данных ___ из двух частей.
2. Согласованность разметки ___ 0,81.
3. Тестовая выборка ___ вопросы трёх типов: фактические, числовые и открытые.
4. Корпус ___ около 9 000 статей.
5. Вопросы по информатике ___ 60 % набора данных.

**Exercise 5.** Find and correct the mistake in each sentence.
1. Модель обучилась на корпусе научных статей. (Её обучали авторы.)
2. Набор данных состоит 6 200 пар.
3. Обучение проводилось при размер батча 32.
4. В качестве базовая модель использовалась модель размером 13 млрд параметров.
5. Качество оценивалось по метрикой F1.
6. Данные были разделены на обучающую, валидационную и тестовую выборку.

**Exercise 6.** Rewrite in scientific style. The sentences come from Minho's lab chat.
1. Мы взяли открытую модель на 7 миллиардов параметров и дообучили её с LoRA.
2. Три человека размечали данные, а потом вместе спорили о сложных примерах.
3. Мы выкинули все повторы и плохие примеры, осталось 6 200.
4. Мы гоняли каждый эксперимент три раза с разными сидами и потом посчитали среднее.

**Exercise 7.** Write the "experimental setup" paragraph (5–7 sentences) from these notes.
- модель: открытая, 13 млрд параметров
- данные: 4 000 вопросов по физике; 90 % / 10 % (обучение / тест)
- метод: LoRA, r = 8
- обучение: 2 эпохи, батч 16, скорость обучения 1·10⁻⁴, 4 графических процессора
- метрика: точность (accuracy); 5 запусков

## Answer key

**Exercise 1.**
1. Из 6 200 пар «вопрос — ответ»; он составлен на основе аннотаций русскоязычных статей по информатике и прикладной математике.
2. Около 9 000 пар; дубликаты и некорректные примеры были удалены.
3. Разметка выполнялась тремя аннотаторами независимо друг от друга; спорные случаи рассматривались совместно.
4. Две открытые языковые модели размером 7 и 13 млрд параметров.
5. Три эпохи; размер батча был 32.
6. Каждый эксперимент повторялся с тремя различными начальными значениями генератора случайных чисел; в таблицах приводятся средние значения и стандартные отклонения; код и данные будут опубликованы в открытом доступе.

**Exercise 2.**
1. Дубликаты были удалены.
2. Корпус был собран из открытых источников.
3. Гиперпараметры подбирались по валидационной выборке.
4. Качество ответов оценивалось по двум метрикам.
5. Данные были разделены на три выборки.
6. Все эксперименты проводились на одном графическом процессоре.

**Exercise 3.** 1. В качестве 2. при 3. по 4. на 5. с помощью 6. в течение

**Exercise 4.**
1. состоит
2. составила
3. включает (также возможно: содержит)
4. насчитывает (также возможно: содержит, включает)
5. составляют

**Exercise 5.**
1. Модель была обучена на корпусе научных статей. (Также верно: Модель обучалась на корпусе научных статей.)
2. Набор данных состоит из 6 200 пар.
3. Обучение проводилось при размере батча 32.
4. В качестве базовой модели использовалась модель размером 13 млрд параметров.
5. Качество оценивалось по метрике F1.
6. Данные были разделены на обучающую, валидационную и тестовую выборки.

**Exercise 6.** Model answers (other correct variants are possible):
1. В качестве базовой модели была использована открытая языковая модель размером 7 млрд параметров, дообученная с помощью адаптеров LoRA.
2. Разметка выполнялась тремя аннотаторами; спорные случаи рассматривались совместно.
3. После удаления дубликатов и некорректных примеров в наборе данных было оставлено 6 200 пар.
4. Каждый эксперимент повторялся три раза с различными начальными значениями генератора случайных чисел; приводятся средние значения.

**Exercise 7.** Model answer:
В качестве базовой модели использовалась открытая языковая модель размером 13 млрд параметров. Для дообучения был использован набор данных, включающий 4 000 вопросов по физике; данные были разделены на обучающую и тестовую выборки в соотношении 90 : 10. Дообучение проводилось с помощью адаптеров LoRA с рангом r = 8. Обучение длилось две эпохи при размере батча 16 и скорости обучения 1·10⁻⁴; все эксперименты проводились на четырёх графических процессорах. Качество ответов оценивалось по точности (accuracy). Каждый эксперимент повторялся пять раз; приводятся средние значения и стандартные отклонения.

Checklist: (1) mostly passive and -ся forms, no colloquial words (взяли, гоняли, сиды); (2) perfective → был + participle, imperfective → -ся; (3) в качестве + genitive, при + prepositional, размером + number; (4) every note from the list is in the paragraph; (5) numbers agree: 4 000 вопросов, две эпохи, на четырёх графических процессорах.

## Review

- You can describe data, models, training and evaluation in the standard four-part structure of a Russian experimental section.
- You choose between был + participle (perfective), -ся (imperfective) and the author's мы, and you avoid the perfective -ся trap (модель обучилась).
- You use the setup phrases в качестве, на, по, при, с помощью, в течение, в соотношении, размером with the right case.
- You describe a dataset with состоять из, включать, содержать, насчитывать and introduce values with составлять.
- You can check a method section for reproducibility and turn lab-chat notes into scientific prose.

---
[← Previous: Writing a Literature Review (обзор литературы)](12-literature-review.md) · [Contents](README.md) · [Next: Describing Tables, Figures and Results →](14-tables-figures-results.md)
