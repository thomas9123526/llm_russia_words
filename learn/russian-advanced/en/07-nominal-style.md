# Lesson 7. Nominal Style: Verbal Nouns and Noun Chains

> **Advanced Russian for Researchers** · Level: B2 · **You will learn:** how verbal nouns are formed and what case they take · how to turn a verb phrase into a noun phrase (light verbs, при / в ходе / с целью) · noun chains: when they help and how to break them

## Text

Minho is preparing an article for a Russian journal. Below is part of the "Data and training" section and a note his supervisor wrote in the margin. Read the Russian first, then check the translation after each paragraph.

**2. Подготовка данных и обучение модели**

Для решения поставленной задачи был сформирован корпус русскоязычных научных текстов объёмом около двух миллиардов токенов (здесь и далее все числа приводятся в качестве примера). Формирование корпуса включало несколько этапов: сбор текстов из открытых источников, удаление дубликатов, фильтрацию по качеству и нормализацию разметки.

*To solve the task at hand, a corpus of Russian-language scientific texts of about two billion tokens was built (here and below, all numbers are given as examples). Building the corpus involved several stages: collecting texts from open sources, removing duplicates, quality filtering and normalising the markup.*

На этапе фильтрации осуществлялась проверка каждого документа по ряду критериев. Удалению подвергались тексты с высокой долей неалфавитных символов, а также документы длиной менее ста слов. В результате фильтрации объём корпуса сократился примерно на треть.

*At the filtering stage, each document was checked against a number of criteria. Texts with a high share of non-alphabetic characters were removed, as were documents shorter than one hundred words. As a result of filtering, the corpus shrank by about a third.*

Токенизация проводилась с помощью токенизатора, обученного на том же корпусе. Выбор размера словаря обусловлен компромиссом между скоростью обработки и качеством представления редких слов. Увеличение словаря приводит к сокращению длины последовательностей, однако сопровождается ростом числа параметров во входном слое модели.

*Tokenization was performed with a tokenizer trained on the same corpus. The choice of vocabulary size reflects a trade-off between processing speed and the quality of rare-word representations. A larger vocabulary shortens the sequences but is accompanied by growth in the number of parameters in the model's input layer.*

Дообучение базовой модели выполнялось в два этапа. На первом этапе проводилось дополнительное предобучение на собранном корпусе, на втором — настройка модели на инструкциях. В ходе обучения осуществлялся контроль значения функции потерь на отложенной выборке. При отсутствии улучшения на трёх контрольных точках подряд обучение прекращалось.

*The base model was fine-tuned in two stages. The first stage was additional pre-training on the collected corpus, the second was instruction tuning. During training, the loss on a held-out set was monitored. If there was no improvement for three checkpoints in a row, training was stopped.*

Оценка качества модели проводилась на двух наборах данных. Для обеспечения сопоставимости результатов все модели оценивались в одинаковых условиях. Сравнение с исходной моделью позволяет определить вклад каждого этапа дообучения в итоговое качество.

*The model's quality was evaluated on two datasets. To make the results comparable, all models were evaluated under identical conditions. Comparison with the original model makes it possible to determine the contribution of each fine-tuning stage to the final quality.*

**Комментарий научного руководителя на полях**

«Минхо, стиль в целом правильный, научный. Но посмотрите на фразу из введения: „Целью работы является исследование влияния изменения размера словаря токенизатора на эффективность обучения моделей“. Здесь пять родительных падежей подряд, и читатель теряется. Лучше так: „Цель работы — исследовать, как изменение размера словаря токенизатора влияет на эффективность обучения моделей“. И не злоупотребляйте словом „осуществлялся“: в одном разделе его лучше не повторять. Иван Петрович»

*"Minho, the style is on the whole correct and scientific. But look at this sentence from the introduction: 'The aim of the work is a study of the influence of the change of the size of the vocabulary of the tokenizer on the efficiency of the training of models.' There are five genitives in a row here, and the reader gets lost. Better: 'The aim of the work is to study how changing the tokenizer's vocabulary size affects the efficiency of model training.' And don't overuse the word 'осуществлялся': better not to repeat it within one section. Ivan Petrovich"*

## New words

| Russian | Pronunciation | English |
|---|---|---|
| отглаго́льное существи́тельное | otglagól'noe sushchestvítel'noe | verbal noun (ср.) |
| именно́й стиль | imennóy stil' | nominal style (м.) |
| формирова́ние ко́рпуса | formirovánie kórpusa | building a corpus (ср.) |
| ко́рпус (те́кстов) | kórpus (tékstov) | (text) corpus (м.) |
| удале́ние дублика́тов | udalénie dublikátov | removal of duplicates, deduplication (ср.) |
| подверга́ться (+ dat.) | podvergát'sya | to be subjected to |
| обусло́вленный (+ instr.) | obuslóvlennyy | determined by, due to |
| сопровожда́ться (+ instr.) | soprovozhdát'sya | to be accompanied by |
| компроми́сс ме́жду (+ instr.) | kompromíss mézhdu | trade-off between (м.) |
| предобуче́ние | predobuchénie | pre-training (ср.) |
| дообуче́ние | doobuchénie | fine-tuning (ср.) |
| настро́йка на инстру́кциях | nastróyka na instrúktsiyakh | instruction tuning (ж.) |
| фу́нкция поте́рь | fúnktsiya potér' | loss function (ж.) |
| отло́женная вы́борка | otlózhennaya výborka | held-out set (ж.) |
| контро́льная то́чка | kontról'naya tóchka | checkpoint (ж.) |
| исхо́дный | iskhódnyy | original, initial |
| вклад (в + acc.) | vklad | contribution (to) (м.) |
| осуществля́ть(ся) | osushchestvlyát'(sya) | to carry out (to be carried out) |
| канцеляри́т | kantselyarít | bureaucratese, officialese (м.) |
| злоупотребля́ть (+ instr.) | zloupotreblyát' | to overuse, to abuse |

## Grammar

Scientific Russian likes to name actions with nouns: instead of *we cleaned the data* it says *очистка данных* (the cleaning of the data), instead of *the accuracy grows* it says *рост точности* (the growth of accuracy). This is called **nominal style** (именной стиль). In moderation it makes a text compact and impersonal; in excess it produces chains of genitives that nobody can read. This lesson shows you both sides.

### 1. Verbal nouns: how they are formed

A verbal noun (отглагольное существительное) names an action or process. You already know many of them: обучение, исследование, решение. Here are the main patterns:

| Suffix | Gender | Examples |
|---|---|---|
| -ние / -ение / -ание | ср. | обучать → обучение; использовать → использование; создать → создание; описать → описание; сократить → сокращение |
| -тие | ср. | развить → развитие; открыть → открытие |
| -ка | ж. | проверить → проверка; оценить → оценка; обработать → обработка; разработать → разработка; разметить → разметка; настроить → настройка |
| -ция (verbs in -ировать / -овать) | ж. | токенизировать → токенизация; генерировать → генерация; оптимизировать → оптимизация; фильтровать → фильтрация; нормализовать → нормализация |
| no suffix | м. | расти → рост; выбрать → выбор; анализировать → анализ; отобрать → отбор; перейти → переход; сдвинуть → сдвиг |

Points to remember:

- **Consonant changes** follow the same alternations as in the verb: сократить → сокращение (т → щ), снизить → снижение (з → ж), повысить → повышение (с → ш), ускорить → ускорение, увеличить → увеличение.
- **Aspect disappears.** Сокращение словаря can mean the process (*while the vocabulary is being reduced*) or the result (*after it has been reduced*). Context or a preposition (в ходе, после) tells you which.
- Sometimes there are **two nouns** from one aspect pair: оценивание (the process of evaluating) and оценка (evaluation, also a grade or an estimate). In papers оценка is far more common.
- Many verbal nouns have a second, concrete meaning: разметка = *the annotation process* and *the annotation itself*; предложение = *proposing* and *a sentence / a proposal*.

### 2. From verb phrase to noun phrase

When you turn a verb into a noun, the words around it change too. These rules cover almost every case:

| Verb phrase | Noun phrase | Rule |
|---|---|---|
| обучать **модель** | обучение **модели** | direct object (acc.) → genitive |
| **точность** растёт | рост **точности** | subject of an intransitive verb → genitive |
| **эксперты** размечают данные | разметка данных **экспертами** | agent → instrumental (the object takes the genitive) |
| влиять **на качество** | влияние **на качество** | a preposition stays |
| зависеть **от размера** | зависимость **от размера** | a preposition stays |
| соответствовать **требованиям** | соответствие **требованиям** | dative stays |
| управлять **процессом** | управление **процессом** | instrumental stays |
| **быстро** растёт | **быстрый** рост | adverb → adjective |
| **значительно** сократить | **значительное** сокращение | adverb → adjective |

Only one genitive "slot" is natural after a verbal noun. If a verb has both a subject and an object, the object takes the genitive and the agent goes into the instrumental: эксперты оценили ответы → оценка ответов экспертами.

**Light verbs.** A verbal noun usually needs a verb with very little meaning of its own. These pairs are worth learning as units:

| Light verb + noun | Meaning | Example |
|---|---|---|
| проводить / провести + acc. | to carry out | провести анализ, эксперимент, сравнение |
| осуществлять / осуществить + acc. | to carry out (more formal) | осуществлять контроль, проверку |
| выполнять / выполнить + acc. | to perform | выполнять обучение, оценку |
| оказывать / оказать влияние на + acc. | to influence | оказывать влияние на качество |
| принимать / принять участие в + prep. | to take part in | принять участие в разметке |
| вносить / внести вклад в + acc. | to contribute to | внести вклад в изучение моделей |
| подвергаться + dat. | to undergo | подвергаться обработке, удалению, анализу |
| приводить / привести к + dat. | to lead to | приводить к росту, к снижению |
| сопровождаться + instr. | to be accompanied by | сопровождаться ростом затрат |
| обеспечивать / обеспечить + acc. | to ensure | обеспечить сопоставимость результатов |

**Prepositional phrases instead of clauses.** A whole subordinate clause can be compressed into a preposition + verbal noun. This is one of the most typical features of the scientific register:

| Nominal phrase | Clause it replaces |
|---|---|
| **при** обучении модели (prep.) | когда модель обучается |
| **в ходе** экспериментов (gen.) | пока проводились эксперименты |
| **после** завершения обучения (gen.) | после того как обучение завершилось |
| **до** начала обучения (gen.) / **перед** обучением (instr.) | до того как / перед тем как начали обучение |
| **в результате** фильтрации (gen.) | так как данные отфильтровали, … |
| **с целью** / **для** повышения точности (gen.) | чтобы повысить точность |
| **за счёт** сокращения словаря (gen.) | так как словарь сократили (the mechanism of a gain) |
| **путём** расширения словаря (gen.) | тем, что расширили словарь (the method) |
| **при отсутствии** улучшения (gen.) | если улучшения нет |
| **по завершении** обучения (gen.) | когда обучение завершилось |

Note the spelling **по завершении** (по + prepositional, ending -и), a frequent mistake even for native speakers.

### 3. Noun chains and how to break them

Each verbal noun brings a genitive, and that genitive may itself be a verbal noun with its own genitive. The result is a **noun chain**:

- исследование влияния изменения размера словаря токенизатора — *a study of the influence of the change of the size of the vocabulary of the tokenizer*

Two or three genitives in a row are normal in Russian (размер словаря токенизатора). From four on, the reader has to hold too much in memory. There is also a second danger, **ambiguity**:

- Оценка ответов модели экспертов… — whose answers? Whose evaluation? It could be *the experts' model* or *evaluation by experts*.

Five ways to break a chain:

| Method | Before | After |
|---|---|---|
| a verb instead of a noun | Целью является исследование влияния размера словаря на скорость | Цель — исследовать, как размер словаря влияет на скорость |
| an adjective instead of a genitive | качество перевода текстов машиной | качество машинного перевода |
| a preposition | алгоритм поиска ошибок разметки данных | алгоритм поиска ошибок в разметке данных |
| a clause with который | анализ ошибок модели перевода научных текстов | анализ ошибок, которые модель допускает при переводе научных текстов |
| agent with a participle | оценка ответов модели экспертов | оценка ответов модели, проведённая экспертами |

A practical rule: **nominalise the steps, keep verbs for the claims.** Stages of a method (сбор, фильтрация, токенизация, оценка) are naturally nouns, because you talk *about* them. Your findings and aims are better expressed with verbs: Мы показываем, что…; Цель работы — исследовать, как…

## Culture

**Канцелярит.** The writer Korney Chukovsky coined the word *канцелярит* in the 1960s, on the model of names of diseases, for the bureaucratic style that had spread from official documents into everyday Russian. Its symptoms are exactly the excesses of nominal style: осуществление проведения проверки (*the carrying out of the conducting of a check*), имеет место снижение (*a decrease takes place*), является + noun in every sentence. Russian editors and supervisors use the word as a warning, and Ivan Petrovich's note is a typical example. Scientific style is not канцелярит: a good Russian paper uses verbal nouns where they save words and drops them where they only add weight. If you can say the same thing shorter with a verb, do it.

## Exercises

**Exercise 1.** Form the verbal noun.
1. проверить
2. оценить
3. сократить
4. увеличить
5. применить
6. разработать
7. выбрать
8. расти
9. разметить
10. токенизировать

**Exercise 2.** Turn the verb phrase into a noun phrase.
1. обучать модель
2. точность растёт
3. влиять на качество
4. резко снижать скорость
5. эксперты размечают данные
6. сравнивать модели друг с другом
7. зависеть от размера словаря
8. быстро генерировать ответы

**Exercise 3.** Fill in the gaps. Use each word once: оказывает, проведён, подвергались, приводит, приняли, вносит.
1. Размер словаря ___ существенное влияние на качество токенизации.
2. Был ___ сравнительный анализ трёх моделей.
3. Все тексты ___ предварительной обработке.
4. Увеличение длины контекста ___ к росту затрат памяти.
5. В разметке данных ___ участие пять экспертов.
6. Данная работа ___ вклад в изучение многоязычных моделей.

**Exercise 4.** Rewrite in scientific style. Replace the clause in bold with a nominal phrase (при, после, с целью / для, за счёт, в ходе, в результате).
1. **Когда модель обучалась,** использовались два графических процессора.
2. **После того как эксперименты завершились,** был проведён анализ ошибок.
3. **Чтобы повысить точность,** в корпус были добавлены новые тексты.
4. Скорость выросла, **потому что словарь был сокращён.**
5. **Пока проводились эксперименты,** была обнаружена ошибка в разметке.
6. **Модель дообучили, и в результате** качество перевода повысилось.

**Exercise 5.** Remove the канцелярит. Rewrite each sentence simply and clearly.
1. Осуществление проверки корректности разметки производилось экспертами.
2. Нами было произведено осуществление сбора текстов из открытых источников.
3. Имеет место снижение точности при увеличении длины документа.

**Exercise 6.** Break the noun chain or remove the ambiguity.
1. Целью работы является разработка метода повышения качества оценки ответов моделей.
2. Оценка ответов модели экспертов показала высокое качество.
3. Исследуется влияние изменения длины контекста обучения модели на точность.

**Exercise 7.** Answer the questions about the text.
1. Какие этапы включало формирование корпуса?
2. Какие тексты подвергались удалению?
3. Чем обусловлен выбор размера словаря?
4. Когда прекращалось обучение?
5. Что не понравилось Ивану Петровичу во фразе из введения?

**Exercise 8.** Write a paragraph (5–6 sentences) about the data preparation for an experiment of your own. Use nominal style, but avoid канцелярит.

## Answer key

**Exercise 1.**
1. проверка
2. оценка (также: оценивание)
3. сокращение
4. увеличение
5. применение
6. разработка
7. выбор
8. рост
9. разметка
10. токенизация

**Exercise 2.**
1. обучение модели
2. рост точности
3. влияние на качество
4. резкое снижение скорости
5. разметка данных экспертами
6. сравнение моделей друг с другом
7. зависимость от размера словаря
8. быстрая генерация ответов

**Exercise 3.**
1. оказывает
2. проведён
3. подвергались
4. приводит
5. приняли
6. вносит

**Exercise 4.**
1. При обучении модели использовались два графических процессора.
2. После завершения экспериментов был проведён анализ ошибок.
3. С целью повышения точности (Для повышения точности) в корпус были добавлены новые тексты.
4. Скорость выросла за счёт сокращения словаря.
5. В ходе экспериментов была обнаружена ошибка в разметке.
6. В результате дообучения модели качество перевода повысилось.

**Exercise 5.** (model answers; other simple versions are possible)
1. Корректность разметки проверяли эксперты. / Разметку проверили эксперты.
2. Мы собрали тексты из открытых источников. / Тексты были собраны из открытых источников.
3. При увеличении длины документа точность снижается.

**Exercise 6.** (model answers)
1. Цель работы — разработать метод, который позволяет точнее оценивать ответы моделей.
2. Оценка ответов модели, проведённая экспертами, показала высокое качество. / Эксперты высоко оценили ответы модели.
3. Исследуется, как изменение длины контекста при обучении модели влияет на точность.

**Exercise 7.**
1. Сбор текстов из открытых источников, удаление дубликатов, фильтрацию по качеству и нормализацию разметки.
2. Тексты с высокой долей неалфавитных символов и документы длиной менее ста слов.
3. Компромиссом между скоростью обработки и качеством представления редких слов.
4. При отсутствии улучшения на трёх контрольных точках подряд.
5. В ней пять родительных падежей подряд, и читатель теряется.

**Exercise 8.** Model answer:

Для обучения модели был сформирован корпус диалогов на русском языке. Данные собирались из открытых источников. На этапе предварительной обработки проводились удаление дубликатов и фильтрация коротких реплик. Токенизация выполнялась с помощью исходного токенизатора модели. Для повышения качества ответов часть данных была дополнительно размечена экспертами. В результате обработки объём корпуса сократился примерно на 20 % (значение приводится в качестве примера).

Checklist:
- at least three verbal nouns naming the steps (сбор, обработка, фильтрация, токенизация…);
- at least one light-verb construction (проводиться, выполняться + verbal noun);
- at least one prepositional nominal phrase (для / с целью, в результате, при, в ходе);
- no chain of more than three or four genitives in a row;
- осуществлять / осуществляться used no more than once;
- impersonal or passive forms instead of «я».

## Review

- You can form verbal nouns in -ние, -тие, -ка, -ция and without a suffix, and you know their gender.
- You can turn a verb phrase into a noun phrase: object and intransitive subject → genitive, agent → instrumental, prepositions stay, adverbs become adjectives.
- You know the main light verbs: проводить анализ, оказывать влияние на, вносить вклад в, подвергаться обработке, приводить к.
- You can compress a clause into при / в ходе / после / с целью / за счёт / путём + verbal noun, and you spell по завершении correctly.
- You can spot a noun chain or an ambiguous genitive and break it with a verb, an adjective, a preposition or a participle.
- You know what канцелярит is and how to avoid it.

---
[← Previous: The Scientific Register: Formal, Neutral and Colloquial Russian](06-scientific-register.md) · [Contents](README.md) · [Next: Connectors and Logical Structure →](08-connectors-and-logic.md)
