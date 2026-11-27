# Lesson 12. Writing a Literature Review (обзор литературы)

> **Advanced Russian for Researchers** · Level: B2 · **You will learn:** how to organise a review by approaches, not by authors, and lead it to a research gap · the phrases for presenting, comparing and criticising earlier work · how to name authors in Russian: declension of Russian, foreign and Korean surnames, и др., agreement with the verb

## Text

Minho is writing the first chapter of his dissertation. Below is one section of the literature review from his draft, already corrected by Ivan Petrovich. All references [1]–[10] and the authors' names are invented examples. Read the Russian first, then check the translation after each paragraph.

**1.2. Методы снижения галлюцинаций в больших языковых моделях**

Проблема галлюцинаций, т. е. генерации правдоподобных, но недостоверных утверждений, привлекает всё больше внимания исследователей [1–3]. Существующие подходы к её решению можно условно разделить на три группы: методы, использующие внешние источники знаний, методы дообучения и методы проверки уже сгенерированного ответа.

*1.2. Methods for reducing hallucinations in large language models. The problem of hallucinations, i.e. the generation of plausible but false statements, is attracting more and more attention from researchers [1–3]. Existing approaches to solving it can be roughly divided into three groups: methods that use external knowledge sources, fine-tuning methods, and methods that check an answer that has already been generated.*

Наиболее распространённым направлением является генерация с опорой на поиск (retrieval-augmented generation, RAG). Этот подход был впервые предложен в работе [4]: перед ответом модель получала фрагменты документов из внешней базы. Позднее он получил развитие в работах [5, 6], авторы которых показали, что качество ответов во многом зависит от качества поиска. Однако, как отмечают Белова и др. [6], при неудачном поиске модель может повторять ошибки найденных документов.

*The most widespread line of research is retrieval-augmented generation (RAG). This approach was first proposed in [4]: before answering, the model received fragments of documents from an external database. Later it was developed further in [5, 6], whose authors showed that the quality of the answers depends largely on the quality of retrieval. However, as Belova et al. [6] note, when retrieval fails, the model may repeat the errors of the retrieved documents.*

Вторая группа методов основана на дообучении модели. Так, в работе [7] модель дообучалась на ответах, проверенных экспертами, что позволило снизить долю недостоверных утверждений. Близкий результат был получен Кимом [8] на материале корейского языка. Недостатком таких методов является потребность в больших объёмах размеченных данных, получение которых требует значительных затрат.

*The second group of methods is based on fine-tuning the model. For example, in [7] the model was fine-tuned on answers checked by experts, which made it possible to reduce the share of false statements. A similar result was obtained by Kim [8] on Korean data. The drawback of such methods is the need for large amounts of labelled data, which are costly to obtain.*

Наконец, ряд авторов предлагает проверять ответ после генерации [9, 10]. В отличие от методов первых двух групп, такие методы не требуют изменения самой модели, однако существенно увеличивают время получения ответа.

*Finally, a number of authors propose checking the answer after generation [9, 10]. Unlike the methods of the first two groups, such methods do not require changing the model itself, but they considerably increase the time it takes to get an answer.*

Анализ литературы показывает, что большинство работ выполнено на материале английского языка, а для русского языка подобные исследования единичны. Кроме того, вопрос о том, как сочетать методы разных групп, остаётся практически неизученным. Настоящая работа направлена на восполнение этого пробела: в ней предлагается и исследуется комбинированный подход для русскоязычных научных текстов.

*The analysis of the literature shows that most studies have been carried out on English data, while for Russian such studies are few and far between. In addition, the question of how to combine methods from different groups remains practically unstudied. The present work aims to fill this gap: it proposes and investigates a combined approach for Russian-language scientific texts.*

## New words

| Russian | Pronunciation | English |
|---|---|---|
| обзо́р литерату́ры | obzór literatúry | literature review (м.) |
| направле́ние | napravlénie | line of research, direction (ср.) |
| галлюцина́ция | gallyutsinátsiya | hallucination (ж.) |
| правдоподо́бный | pravdopodóbnyy | plausible |
| недостове́рный | nedostovérnyy | false, unreliable |
| усло́вно | uslóvno | roughly, for convenience |
| с опо́рой на (+ acc.) | s opóroy na | relying on, grounded in |
| получи́ть разви́тие | poluchít' razvítie | to be developed further |
| на материа́ле (+ gen.) | na materiále | on data from, using material of |
| разме́ченные да́нные | razméchennye dánnye | labelled data (pl.) |
| затра́ты | zatráty | costs, expenditure (pl.) |
| недоста́ток | nedostátok | drawback, shortcoming (м.) |
| потре́бность в (+ prep.) | potrébnost' v | need for (ж.) |
| пробе́л | probél | gap (м.) |
| восполне́ние | vospolnénie | filling (a gap) (ср.) |
| едини́чный | ediníchnyy | isolated, very few |
| ряд (+ gen.) | ryad | a number of (м.) |
| и др. (и други́е) | i drugíe | et al. |
| отмеча́ть / отме́тить | otmechát' / otmétit' | to note, to point out |
| ука́зывать / указа́ть на (+ acc.) | ukázyvat' / ukazát' na | to point to, to indicate |

## Grammar

The literature review (обзор литературы) is part of the first chapter of a Russian dissertation and of the introduction of most papers. Its job is not to prove that you have read a lot, but to show the state of the field and to lead the reader to the question your work answers. In Korean terms, this is the 선행 연구 검토 — and the logic is the same in both languages.

### 1. Structure: from a catalogue to a map

The most common weakness of a beginner's review is the **catalogue**: one sentence per paper, in the order the papers were read.

> ✗ Иванов [1] предложил метод A. Петрова [2] предложила метод B. Смит [3] исследовал метод A на английском языке. Ли [4] тоже исследовал метод A.

A good review is a **map**: works are grouped by approach (or by problem), compared and evaluated. The text above follows a typical plan:

| Part | What it does | In the text |
|---|---|---|
| Opening paragraph | the problem, why it matters, the groups of approaches | Проблема галлюцинаций… привлекает всё больше внимания…; можно условно разделить на три группы |
| One paragraph per group | topic sentence → first work → development → evaluation | Наиболее распространённым направлением является… → впервые предложен в работе [4] → получил развитие в работах [5, 6] → Однако, как отмечают… |
| Comparison between groups | how the groups differ | В отличие от методов первых двух групп… |
| Final paragraph | the gap and the step to your own work | Анализ литературы показывает…; остаётся практически неизученным; Настоящая работа направлена на восполнение этого пробела |

The review works like a funnel: from the general problem to the groups of methods, from the groups to their limitations, and from the limitations to the gap that your dissertation fills. Within a group you may go chronologically (впервые… → позднее…), but the review as a whole is organised by ideas.

### 2. Phrases for each move

| Move | Phrases |
|---|---|
| introduce the topic | Проблема X привлекает всё больше внимания исследователей [1–3]. В последние годы активно исследуется… Существующие подходы можно (условно) разделить на N групп. |
| present works | Этот подход был впервые предложен в работе [4]. Позднее он получил развитие в работах [5, 6]. Так, в работе [7]… Ряд авторов предлагает… [9, 10]. |
| compare | В отличие от… Аналогичный подход используется в работе [5]. Близкий результат был получен… Этот вывод подтверждается в работах [5, 6]. |
| evaluate, criticise | Недостатком таких методов является… Однако эти методы не учитывают… Как отмечают Белова и др. [6], … |
| show the gap | Анализ литературы показывает, что… X остаётся практически неизученным. Подобные исследования единичны. Вопрос о… недостаточно изучен. |
| step to your work | Настоящая работа направлена на восполнение этого пробела. В связи с этим целью работы является… |

**Reporting verbs and their government.** показывать, что…; отмечать, что…; подчёркивать, что…; указывать **на** + acc. (авторы указывают на ограничения метода); предлагать + acc. or + infinitive (предлагают метод / предлагают проверять ответ); рассматривать + acc.; критиковать + acc. **за** + acc. (критикуют подход за высокую стоимость).

**Tense and aspect.**
- **Past perfective** for what a specific work did once: был предложен, показали, был получен, позволило снизить.
- **Past imperfective** to describe how a work was set up: модель дообучалась, модель получала фрагменты.
- **Present** for the state of the field and for views that are still valid: привлекает внимание, отмечают, предлагает, остаётся неизученным.

**Polite criticism.** Russian academic style criticises approaches and results, not people. Write *Недостатком метода является…* or *Однако метод не учитывает…*, not *Авторы ошиблись*. A drawback is usually introduced with **однако**, often together with a source that confirms it (как отмечают Белова и др. [6]). Note the punctuation in the text: *Однако, как отмечают Белова и др. [6], при неудачном поиске…* — the commas here do not belong to однако (see Lesson 10) but set off the inserted clause как отмечают….

### 3. Naming authors: surnames, и др., agreement

Russian papers usually refer to works by number: в работе [3], в работах [5, 6], [1–3]. A surname is added when the author matters for your argument: как отмечают Белова и др. [6]; результат был получен Кимом [8]. Then the surname must be in the right case — and surnames decline by their own rules:

| Surname | Gender | Genitive (в работе…) | Instrumental (получен…) |
|---|---|---|---|
| Орлов (Russian, -ов/-ин) | male | Орлова | Орловым |
| Белова (Russian, -ова/-ина) | female | Беловой | Беловой |
| Смит (foreign, ends in a consonant) | male | Смита | Смитом |
| Смит | female | Смит | Смит |
| Ким, Пак (Korean, consonant) | male | Кима, Пака | Кимом, Паком |
| Ким, Пак | female | Ким, Пак | Ким, Пак |
| Ли, Моро (end in a vowel such as -и, -о) | any | Ли, Моро | Ли, Моро |

In short: a male surname ending in a consonant declines like a noun; a female foreign surname ending in a consonant does not decline; surnames ending in -и, -о, -у, -е, -ю do not decline at all. Russian female surnames in -ова/-ина decline like adjectives (Беловой). Before the surname, initials stay unchanged: в работе И. П. Смирнова.

**и др. and agreement.** **и др.** (= и другие) corresponds to *et al.* With it, and with two authors joined by и, the verb is plural: Белова и др. [6] отмечают…; Орлов и Белова [5] показали… With **ряд** and **большинство**, the verb is normally singular (neuter in the past or a short participle): ряд авторов предлагает…; большинство работ выполнено…

**Where the number goes.** Put the reference right after the name or at the end of the clause, before the full stop: …повторять ошибки найденных документов [6]. A range uses a dash: [1–3]; separate works use a comma: [5, 6].

## Study tip

Before writing, put your paper cards (Lesson 11) into a table: rows are papers, columns are approach, data, language, main result, limitation. The groups and the gap often become visible in the table itself — an empty cell such as "Russian data" may be exactly the gap your dissertation fills. Then write one paragraph per group, and write the opening paragraph last.

## Exercises

**Exercise 1.** Answer the questions about the text.
1. На сколько групп автор делит существующие подходы?
2. Какое направление является наиболее распространённым?
3. От чего, по данным работ [5, 6], зависит качество ответов?
4. Какой недостаток есть у методов дообучения?
5. Чем методы третьей группы отличаются от методов первых двух групп?
6. Какой пробел в литературе находит автор?

**Exercise 2.** Put the surname into the correct case.
1. в работе (Смит — мужчина)
2. в работе (Анна Смит)
3. Метод был предложен (Орлов).
4. по мнению (Белова)
5. как показано в работе (Ли)
6. Результат был получен (Ким — мужчина).
7. по мнению (Пак — женщина)
8. в статье (И. П. Смирнов)

**Exercise 3.** What does each sentence do? Write введение темы, представление работ, сравнение, критика, пробел or переход к своей работе.
1. Существующие подходы можно условно разделить на три группы.
2. Позднее этот подход получил развитие в работах [5, 6].
3. Недостатком таких методов является потребность в размеченных данных.
4. Подобные исследования для русского языка единичны.
5. Настоящая работа направлена на восполнение этого пробела.
6. В отличие от методов первых двух групп, такие методы не требуют изменения модели.

**Exercise 4.** Fill in the missing preposition.
1. Авторы указывают … ограничения метода.
2. Этот подход получил развитие … работах [5, 6].
3. … отличие от работы [4], мы используем обучаемый порог.
4. Близкий результат был получен … материале корейского языка.
5. Вопрос … сочетании методов остаётся открытым.
6. Настоящая работа направлена … восполнение этого пробела.
7. Генерация … опорой на поиск — наиболее распространённое направление.

**Exercise 5.** Put the word in brackets into the correct form.
1. Белова и др. [6] (отмечать, наст. вр.), что модель может повторять ошибки.
2. Большинство работ (выполнен, кр. форма) на материале английского языка.
3. Ряд авторов (предлагать, наст. вр.) проверять ответ после генерации.
4. Орлов и Белова [5] (показать, прош. вр.), что метод эффективен.
5. Вопрос о сочетании методов (оставаться, наст. вр.) открытым.
6. Подобные исследования (единичный, кр. форма).

**Exercise 6.** Rewrite in the passive, with the author in the instrumental.
1. Смит предложил этот метод [3].
2. Белова получила близкий результат [6].
3. Ким исследовал эту проблему [8].
4. Авторы работы [7] создали новый набор данных.

**Exercise 7.** Turn each "catalogue" into a synthesis (one or two sentences).
1. Иванов [2] использовал поиск по внешней базе. Петрова [3] тоже использовала поиск по внешней базе. Ли [4] использовал поиск по внешней базе и показал, что качество ответов зависит от качества поиска.
2. Смит [5] дообучал модель на ответах экспертов. Ким [6] делал то же самое на материале корейского языка. Для этого нужно много размеченных данных.

**Exercise 8.** Write a paragraph of a literature review (5–7 sentences) from these notes. Group the works, evaluate them and end with the gap.

- [1] Орлов: квантование модели, точность снижается на 3 %.
- [2] Ли: квантование + дообучение, точность снижается на 1 %.
- [3] Белова и др.: квантование на материале русского языка, только задачи классификации.
- Пробел: генерация текста на русском языке не изучена.

## Answer key

**Exercise 1.**
1. На три группы.
2. Генерация с опорой на поиск (RAG).
3. От качества поиска.
4. Потребность в больших объёмах размеченных данных.
5. Они не требуют изменения самой модели, но существенно увеличивают время получения ответа.
6. Для русского языка подобные исследования единичны, а вопрос о сочетании методов разных групп практически не изучен.

**Exercise 2.**
1. в работе Смита
2. в работе Анны Смит
3. Метод был предложен Орловым.
4. по мнению Беловой
5. как показано в работе Ли
6. Результат был получен Кимом.
7. по мнению Пак
8. в статье И. П. Смирнова

**Exercise 3.** 1. введение темы 2. представление работ 3. критика 4. пробел 5. переход к своей работе 6. сравнение

**Exercise 4.** 1. на 2. в 3. В 4. на 5. о 6. на 7. с

**Exercise 5.**
1. отмечают
2. выполнено
3. предлагает (*also:* предлагают — possible, but the singular is the norm in scientific texts)
4. показали
5. остаётся
6. единичны

**Exercise 6.**
1. Этот метод был предложен Смитом [3].
2. Близкий результат был получен Беловой [6].
3. Эта проблема была исследована Кимом [8].
4. Новый набор данных был создан авторами работы [7].

**Exercise 7.** Model answers:
1. Поиск по внешней базе используется в работах [2–4]; при этом в работе [4] показано, что качество ответов зависит от качества поиска. (*also:* Ряд авторов использует поиск по внешней базе [2–4]. Как показал Ли [4], качество ответов зависит от качества поиска.)
2. Дообучение модели на ответах экспертов предлагается в работах [5, 6], в том числе на материале корейского языка [6]. Недостатком этого подхода является потребность в больших объёмах размеченных данных.

**Exercise 8.** Model answer:
Одним из основных способов уменьшения размера языковых моделей является квантование. Этот подход был предложен в работе Орлова [1], где точность модели снизилась на 3 %. Позднее он получил развитие в работе Ли [2]: сочетание квантования с дообучением позволило сократить потерю точности до 1 %. На материале русского языка квантование исследовалось Беловой и др. [3], однако авторы рассматривали только задачи классификации. Таким образом, применение квантования к генерации текста на русском языке остаётся практически неизученным. Настоящая работа направлена на восполнение этого пробела.

Checklist: the paragraph opens with a topic sentence, not with an author; the works are linked (впервые / позднее / получил развитие), not listed; every work has a reference number; surnames are declined correctly (Орлова, Беловой и др.; Ли does not change); a limitation is introduced with однако; the last sentences name the gap and lead to your own work.

## Review

- You organise a review by approaches and lead it like a funnel to a research gap.
- You can introduce a topic, present and compare works, criticise politely and state the gap with standard phrases.
- You choose past perfective for what a work did, imperfective for how it was set up, and present for the state of the field.
- You decline Russian, foreign and Korean surnames correctly and know which ones do not change.
- You use и др. with a plural verb and ряд / большинство with a singular verb.
- You can turn a catalogue of papers into a synthesis.

---
[← Previous: Reading a Full Research Paper: Strategy and Practice](11-reading-full-paper.md) · [Contents](README.md) · [Next: Describing Experiments, Data and Setup →](13-describing-experiments.md)
