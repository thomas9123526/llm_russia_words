# Lesson 15. Writing an Abstract and Conclusions

> **Advanced Russian for Researchers** · Level: C1 · **You will learn:** the moves, tenses and "do-nots" of a Russian abstract (аннотация) · how to build a conclusion section: results, significance, limitations, future work · how to turn typical English abstract phrases into natural Russian

## Text

The article from Lessons 13 and 14 is almost ready. Ivan Petrovich: «Аннотацию пишут последней, а читают первой» ("The abstract is written last and read first"). Below are Minho's abstract and the final section of the paper. The dataset, the results and the journal are invented for this textbook.

**Дообучение больших языковых моделей для ответов на вопросы по русскоязычным научным текстам: набор данных RuSciQA и сравнение методов**

**Аннотация.** Настоящая работа посвящена адаптации больших языковых моделей (БЯМ) к ответам на вопросы по русскоязычным научным текстам. Несмотря на быстрый прогресс в этой области, качество работы БЯМ с научными текстами на русском языке остаётся недостаточно изученным, а открытые наборы данных для его оценки практически отсутствуют. В работе представлен новый набор данных RuSciQA, содержащий 6 200 пар «вопрос — ответ», составленных на основе аннотаций статей по информатике и прикладной математике. На этом наборе проведено сравнение полного дообучения и дообучения с помощью адаптеров низкого ранга (LoRA) для двух открытых моделей размером 7 и 13 млрд параметров. Показано, что LoRA уступает полному дообучению не более чем на 1,1 процентного пункта по доле точных совпадений, сокращая при этом число обучаемых параметров более чем в 150 раз. Наилучший результат (79,0 % по F1-мере) достигнут моделью размером 13 млрд параметров, дообученной с помощью LoRA. Анализ ошибок показал, что основные трудности у моделей вызывают вопросы, требующие числовых рассуждений. Набор данных и код будут опубликованы в открытом доступе.

*Fine-tuning large language models for question answering on Russian-language scientific texts: the RuSciQA dataset and a comparison of methods*

*Abstract. This paper addresses the adaptation of large language models (LLMs) to answering questions about Russian-language scientific texts. Despite rapid progress in the field, the performance of LLMs on scientific texts in Russian remains insufficiently studied, and open datasets for evaluating it are practically non-existent. We present RuSciQA, a new dataset of 6,200 question–answer pairs compiled from abstracts of papers in computer science and applied mathematics. On this dataset we compare full fine-tuning and fine-tuning with low-rank adapters (LoRA) for two open models with 7 and 13 billion parameters. We show that LoRA is at most 1.1 percentage points behind full fine-tuning in exact match while reducing the number of trainable parameters more than 150-fold. The best result (79.0% F1) is achieved by the 13-billion-parameter model fine-tuned with LoRA. Error analysis shows that the main difficulties for the models are questions requiring numerical reasoning. The dataset and code will be made publicly available.*

**Ключевые слова:** большие языковые модели, дообучение, LoRA, ответы на вопросы, набор данных, русский язык.

*Keywords: large language models, fine-tuning, LoRA, question answering, dataset, Russian language.*

**5. Заключение**

В работе рассмотрена задача ответов на вопросы по русскоязычным научным текстам с помощью больших языковых моделей. Основные результаты работы состоят в следующем.

1. Создан набор данных RuSciQA, включающий 6 200 пар «вопрос — ответ» с согласованностью разметки 0,81 по коэффициенту каппа Коэна.
2. Проведено систематическое сравнение полного дообучения и дообучения с помощью LoRA. Показано, что при сопоставимом качестве LoRA требует на два порядка меньше обучаемых параметров.
3. Установлено, что при ранге адаптера выше 16 качество выходит на плато; следовательно, дальнейшее увеличение ранга нецелесообразно.
4. Выявлено, что почти половина ошибок приходится на числовые вопросы.

*5. Conclusion. The paper has considered the task of answering questions about Russian-language scientific texts with large language models. The main results are as follows. 1. The RuSciQA dataset has been created, containing 6,200 question–answer pairs with an annotation agreement of 0.81 (Cohen's kappa). 2. A systematic comparison of full fine-tuning and LoRA fine-tuning has been carried out. It has been shown that, at comparable quality, LoRA requires two orders of magnitude fewer trainable parameters. 3. It has been established that above adapter rank 16 quality reaches a plateau; hence a further increase of the rank is not advisable. 4. It has been found that almost half of the errors fall on numerical questions.*

Практическая значимость работы состоит в том, что предложенный подход позволяет адаптировать языковую модель к научной предметной области на одном графическом процессоре. Полученные результаты могут быть использованы при разработке систем поиска и анализа научной информации на русском языке.

*The practical significance of the work is that the proposed approach makes it possible to adapt a language model to a scientific domain on a single GPU. The results can be used in developing systems for searching and analysing scientific information in Russian.*

Вместе с тем работа имеет ряд ограничений. Во-первых, набор данных охватывает лишь две научные области. Во-вторых, оценка проводилась только с помощью автоматических метрик, которые не всегда отражают качество развёрнутых ответов.

*At the same time, the work has a number of limitations. First, the dataset covers only two scientific fields. Second, the evaluation used only automatic metrics, which do not always reflect the quality of extended answers.*

В дальнейшем планируется расширить набор данных за счёт статей по гуманитарным и естественным наукам, а также дополнить автоматическую оценку экспертной. Перспективным направлением представляется также исследование методов, повышающих точность ответов на числовые вопросы.

*In future work we plan to extend the dataset with papers in the humanities and natural sciences and to complement the automatic evaluation with expert evaluation. Another promising direction is the study of methods that improve the accuracy of answers to numerical questions.*

## New words

| Russian | Pronunciation | English |
|---|---|---|
| аннота́ция | annotátsiya | abstract (ж.) |
| заключе́ние | zaklyuchénie | conclusion (final section) (ср.) |
| вы́вод | vývod | conclusion, finding (м.) |
| ключевы́е слова́ | klyuchevýe slová | keywords |
| посвящённый (+ dat.) | posvyashchyónnyy | devoted to |
| сопостави́мый | sopostavímyy | comparable |
| поря́док | poryádok | order (of magnitude) (м.) |
| нецелесообра́зный | netselesoobráznyy | not advisable, pointless |
| вы́явить (pf.) | výyavit' | to reveal, to identify |
| значи́мость | znachímost' | significance (ж.) |
| ограниче́ние | ogranichénie | limitation (ср.) |
| охва́тывать | okhvátyvat' | to cover |
| перспекти́вный | perspektívnyy | promising |
| направле́ние | napravlénie | direction (of research) (ср.) |
| в дальне́йшем | v dal'néyshem | in future, later |
| экспе́ртная оце́нка | ekspértnaya otsénka | expert (human) evaluation (ж.) |
| развёрнутый | razvyórnutyy | detailed, extended |
| рассужде́ние | rassuzhdénie | reasoning (ср.) |
| вме́сте с тем | vméste s tem | at the same time, however |
| в откры́том до́ступе | v otkrýtom dóstupe | publicly available |

## Grammar

### 1. The abstract: moves, tenses and what to leave out

A Russian abstract is a compressed version of the whole paper. It usually follows six moves, each with its own marker phrases.

| Move | Marker phrases | In the text |
|---|---|---|
| 1. Topic | Работа посвящена + dat.; В статье рассматривается… | Настоящая работа посвящена адаптации… |
| 2. Problem | Несмотря на…; однако; остаётся недостаточно изученным; практически отсутствуют | Несмотря на быстрый прогресс… |
| 3. Contribution | В работе представлен / предлагается / разработан… | В работе представлен новый набор данных… |
| 4. Method, data | проведено сравнение / исследование…; на материале… | На этом наборе проведено сравнение… |
| 5. Results | Показано, что…; Установлено, что…; достигнут…; Анализ показал, что… | Показано, что LoRA уступает… |
| 6. Significance, availability | Полученные результаты могут быть использованы…; будут опубликованы | Набор данных и код будут опубликованы… |

**Tenses and forms.** What is true of the paper is in the present (работа посвящена, в работе предлагается). What has been done is expressed with perfective short participles, usually without a subject and with the verb first: Проведено сравнение…, Показано, что…, Достигнут результат… The impersonal **Показано, что…** is the standard Russian equivalent of *We show that…*.

**What to leave out.** A Russian abstract must be understandable without the paper, so it usually contains:
- no references ([12, 15]);
- no references to tables or figures (как видно из табл. 2);
- no unexplained abbreviations: write the full term first, then the abbreviation in brackets — большие языковые модели (БЯМ);
- no empty promises such as В статье будут рассмотрены некоторые вопросы… — say *what* was found.

Concrete numbers, on the other hand, are welcome: на 1,1 процентного пункта, 79,0 % по F1-мере. Length requirements and the exact name of the section (аннотация, реферат, резюме) depend on the journal, so always check the journal's rules for authors.

**Compression.** To keep the abstract short, turn clauses into participial phrases and verbal nouns (Lessons 2 and 7):
- Мы собрали набор данных. Он содержит 6 200 пар. → Представлен набор данных, содержащий 6 200 пар.
- LoRA уступает полному дообучению на 1,1 п. п. При этом LoRA сокращает число параметров в 150 раз. → LoRA уступает полному дообучению на 1,1 п. п., сокращая при этом число параметров в 150 раз.

### 2. The conclusion section: results, significance, limitations, future work

The final section is called **Заключение** (in a dissertation and in many papers) or **Выводы** (often a numbered list). Unlike the abstract, it may refer to the sections of the paper and should name limitations. A standard plan:

| Part | Phrases |
|---|---|
| What was done | В работе рассмотрена задача…; В работе исследован… |
| Main results | Основные результаты работы состоят в следующем.; В работе получены следующие результаты: |
| Significance | Практическая значимость работы состоит в том, что…; Полученные результаты могут быть использованы при + prep. |
| Limitations | Вместе с тем работа имеет ряд ограничений.; К ограничениям работы относится…; Ограничение работы состоит в том, что… |
| Future work | В дальнейшем планируется + inf.; Перспективным направлением представляется + nom.; Представляет интерес + nom. |

**Numbered results** usually open with a short passive participle that agrees with the noun that follows:

| Form | Example |
|---|---|
| masculine | Создан набор данных… / Разработан метод… |
| feminine | Предложена модель… / Разработана методика… |
| neuter | Проведено сравнение… / Проведено исследование… |
| plural | Получены результаты… / Выявлены закономерности… |
| impersonal (neuter) + что | Показано, что… / Установлено, что… / Выявлено, что… |

**Parallelism.** Items joined by и must have the same form. Mixing a verbal noun and an infinitive is a typical mistake: *планируется расширение набора данных и проверить метод*. Write either two infinitives — планируется расширить набор данных и проверить метод — or two nouns — планируются расширение набора данных и проверка метода.

**Limitations are normal.** Russian reviewers and dissertation councils expect them. Name them calmly with во-первых, во-вторых and, if possible, link each limitation to future work.

### 3. From an English abstract to a Russian one

Many journals ask for both a Russian and an English abstract. Do not translate word for word; use the Russian conventions.

| English | Russian |
|---|---|
| This paper addresses / is devoted to… | Работа посвящена + dat. (no preposition!) |
| In this paper, we propose… | В работе предлагается… / В статье предложен… |
| We show that… | Показано, что… |
| a novel method | новый метод |
| outperforms state-of-the-art approaches | превосходит существующие подходы / лучшие известные результаты |
| significantly improves | существенно повышает (статистически значимо — only if you tested it) |
| achieves 79.0 F1 | достигает 79,0 % по F1-мере |
| To the best of our knowledge, … | Насколько нам известно, … |
| Our code and data are publicly available. | Код и данные находятся в открытом доступе. |
| Future work will… | В дальнейшем планируется… |

Watch three details: **посвящена** takes the dative without a preposition (посвящена оценке, not *посвящена к оценке*); **превосходить** takes the accusative without над (*превосходит над* is a calque of "superior over"); and decimals take a comma (79,0 %).

## Study tip

Write the abstract last, from the conclusion. Take each numbered result from Заключение, compress it to one clause and put it into move 5; take the first sentence of Заключение for move 1 and the significance sentence for move 6. Then check the abstract against the list in section 1: no references, no tables, no unexplained abbreviations, at least one concrete number. Finally, read the Russian and the English abstracts side by side: they must say the same thing, but each in its own language's conventions.

## Exercises

**Exercise 1.** Answer the questions about the text.
1. Чему посвящена работа?
2. Какой новый ресурс представлен в работе?
3. Какие методы сравниваются?
4. Какой результат достигнут лучшей моделью?
5. Какие ограничения работы называют авторы?
6. Что планируется сделать в дальнейшем?

**Exercise 2.** Which move of the abstract is each sentence? Write A (topic), B (problem), C (contribution), D (results) or E (availability).
1. Набор данных и код будут опубликованы в открытом доступе.
2. Показано, что LoRA уступает полному дообучению не более чем на 1,1 процентного пункта по доле точных совпадений, сокращая при этом число обучаемых параметров более чем в 150 раз.
3. Настоящая работа посвящена адаптации больших языковых моделей (БЯМ) к ответам на вопросы по русскоязычным научным текстам.
4. В работе представлен новый набор данных RuSciQA, содержащий 6 200 пар «вопрос — ответ».
5. Несмотря на быстрый прогресс в этой области, качество работы БЯМ с научными текстами на русском языке остаётся недостаточно изученным.
6. Анализ ошибок показал, что основные трудности у моделей вызывают вопросы, требующие числовых рассуждений.

**Exercise 3.** Write the verb in brackets as a short passive participle in the correct form.
1. (создать) ___ набор данных RuSciQA.
2. (провести) ___ сравнение двух методов дообучения.
3. (предложить) ___ новая метрика оценки.
4. (получить) ___ следующие результаты.
5. (показать) ___, что LoRA требует меньше обучаемых параметров.
6. (разработать) ___ программный комплекс для разметки данных.

**Exercise 4.** Translate these English abstract sentences into Russian.
1. In this paper, we propose a new fine-tuning method.
2. We show that the method outperforms existing approaches.
3. To the best of our knowledge, this is the first Russian-language dataset of this kind.
4. The best model achieves 79.0% F1.
5. The code and data are publicly available.
6. This paper is devoted to the evaluation of language models.

**Exercise 5.** Find and correct the mistake in each sentence.
1. Работа посвящена к оценке языковых моделей.
2. В работе предлагается новельный метод дообучения.
3. Показано, что модель превосходит над базовыми методами.
4. Результаты могут быть использованы при разработку поисковых систем.
5. В дальнейшем планируется расширение набора данных и проверить метод на других языках.
6. Ограничением работы является в том, что оценка проводилась автоматически.

**Exercise 6.** Rewrite these sentences from Minho's first draft of the conclusion in scientific style.
1. Мы сделали новый датасет, и он получился довольно качественный.
2. Минус нашей работы — мы проверили всё только на двух областях.
3. Потом хотим ещё попробовать людей-экспертов для оценки.
4. Это может пригодиться тем, кто делает системы научного поиска.

**Exercise 7.** Write an abstract (100–150 words) from these notes.
- тема: перевод научных терминов с английского языка на русский с помощью БЯМ
- проблема: модели часто переводят термины непоследовательно
- вклад: словарь из 3 000 терминов по информатике; метод: подстановка терминов из словаря в запрос
- эксперимент: 3 открытые модели, 1 500 предложений из научных статей
- результат: доля правильно переведённых терминов выросла с 71,2 до 89,5 %; качество перевода (BLEU) не снизилось
- доступность: словарь будет опубликован

## Answer key

**Exercise 1.**
1. Адаптации больших языковых моделей к ответам на вопросы по русскоязычным научным текстам.
2. Набор данных RuSciQA, содержащий 6 200 пар «вопрос — ответ».
3. Полное дообучение и дообучение с помощью адаптеров низкого ранга (LoRA).
4. 79,0 % по F1-мере (модель размером 13 млрд параметров, дообученная с помощью LoRA).
5. Набор данных охватывает лишь две научные области; оценка проводилась только с помощью автоматических метрик.
6. Расширить набор данных за счёт статей по гуманитарным и естественным наукам, дополнить автоматическую оценку экспертной и исследовать методы, повышающие точность ответов на числовые вопросы.

**Exercise 2.** 1. E 2. D 3. A 4. C 5. B 6. D

**Exercise 3.** 1. Создан 2. Проведено 3. Предложена 4. Получены 5. Показано 6. Разработан

**Exercise 4.**
1. В работе предлагается новый метод дообучения.
2. Показано, что метод превосходит существующие подходы.
3. Насколько нам известно, это первый русскоязычный набор данных такого типа.
4. Лучшая модель достигает 79,0 % по F1-мере.
5. Код и данные находятся в открытом доступе.
6. Работа посвящена оценке языковых моделей.

**Exercise 5.**
1. Работа посвящена оценке языковых моделей.
2. В работе предлагается новый метод дообучения.
3. Показано, что модель превосходит базовые методы.
4. Результаты могут быть использованы при разработке поисковых систем.
5. В дальнейшем планируется расширить набор данных и проверить метод на других языках. (Также верно: В дальнейшем планируются расширение набора данных и проверка метода на других языках.)
6. Ограничение работы состоит в том, что оценка проводилась автоматически. (Также верно: Ограничением работы является то, что оценка проводилась автоматически.)

**Exercise 6.** Model answers (other correct variants are possible):
1. Создан новый набор данных с высокой согласованностью разметки.
2. Ограничение работы состоит в том, что эксперименты проводились лишь на материале двух научных областей.
3. В дальнейшем планируется дополнить автоматическую оценку экспертной.
4. Полученные результаты могут быть использованы при разработке систем научного поиска.

**Exercise 7.** Model answer:
Настоящая работа посвящена переводу научных терминов с английского языка на русский с помощью больших языковых моделей (БЯМ). Несмотря на высокое общее качество машинного перевода, модели часто переводят термины непоследовательно, что затрудняет использование переводов в научной практике. В работе представлен словарь, включающий 3 000 терминов по информатике, и предлагается метод, основанный на подстановке терминов из словаря в запрос. Эксперименты проведены на трёх открытых языковых моделях и 1 500 предложениях из научных статей. Показано, что предложенный метод повышает долю правильно переведённых терминов с 71,2 до 89,5 %, при этом общее качество перевода по метрике BLEU не снижается. Полученные результаты могут быть использованы при разработке систем перевода научных текстов. Словарь будет опубликован в открытом доступе.

Checklist: (1) all six moves are present, in order; (2) the topic sentence uses посвящена + dative; (3) results start with Показано, что… and contain concrete numbers; (4) the abbreviation БЯМ is explained at its first use; (5) no references, no tables, no "we will consider some questions"; (6) no мы, no colloquial words (запрос is fine; avoid промпт in the Russian abstract unless the journal accepts it).

## Review

- You can write a Russian abstract in six moves: topic, problem, contribution, method, results, significance.
- You use the present for the paper (работа посвящена) and perfective short participles for what was done (проведено, показано, достигнут).
- You know what to leave out of an abstract: references, tables, unexplained abbreviations, empty promises.
- You can build a conclusion section with numbered results, practical significance, limitations and future work, keeping lists parallel.
- You turn English abstract phrases into Russian conventions and avoid calques such as *посвящена к*, *превосходит над*, *новельный*.

---
[← Previous: Describing Tables, Figures and Results](14-tables-figures-results.md) · [Contents](README.md) · [Next: The Candidate Dissertation: Structure and Key Parts (актуальность, научная новизна, положения, выносимые на защиту) →](16-dissertation-structure.md)
