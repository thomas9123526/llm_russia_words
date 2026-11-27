# Lesson 2. Long Sentences: Participial and Verbal Adverb Phrases

> **Advanced Russian for Researchers** · Level: B2 · **You will learn:** where to put a participial phrase and when it needs commas · the same-subject rule for verbal adverb phrases, its typical violations and its real exceptions · how to unpack a long scientific sentence when reading and how to build one when writing

## Text

Minho is writing the Methods section of his article. Here is a fragment of his draft after Ivan Petrovich's corrections. (The text was written for this textbook; the method and the numbers are invented examples.)

**3. Предлагаемый метод**

**3.1. Общая схема**

Предлагаемый в настоящей работе метод сжатия, основанный на дистилляции знаний, состоит из трёх последовательных этапов. На первом этапе исходная модель, содержащая около семи миллиардов параметров, обрабатывает корпус текстов, собранный из открытых источников. Получая на вход очередной фрагмент текста, она порождает распределение вероятностей следующего токена. Это распределение сохраняется и в дальнейшем используется как «мягкие» метки.

На втором этапе обучается модель-ученик, имеющая в пять раз меньше параметров. Минимизируя расхождение между своими предсказаниями и сохранёнными распределениями, модель-ученик постепенно перенимает поведение исходной модели (модели-учителя). В отличие от работ, опирающихся только на итоговые предсказания учителя, мы используем также промежуточные представления, извлечённые из скрытых слоёв. Это позволяет передать ученику информацию, не отражённую в выходном распределении.

На третьем этапе ученик дообучается на небольшом наборе размеченных данных, относящихся к решаемой прикладной задаче. Завершив дообучение, мы сравниваем полученную модель с исходной по точности и скорости генерации.

**3.2. Выбор слоёв**

Отдельного обсуждения заслуживает выбор слоёв, используемых для передачи промежуточных представлений. Проведя предварительные эксперименты, мы установили, что наилучшие результаты дают слои, расположенные в средней части сети. Слои, находящиеся ближе к выходу, по-видимому, слишком тесно связаны с задачей предобучения, тогда как нижние слои кодируют главным образом поверхностные признаки текста.

Следует подчеркнуть, что, сопоставляя слои учителя и ученика, необходимо учитывать разницу в их количестве. Не имея возможности установить взаимно однозначное соответствие между слоями, мы ставим в соответствие каждому слою ученика группу соседних слоёв учителя, усредняя их представления.

*Translation:*

*3. The proposed method*

*3.1. General scheme*

*The compression method proposed in the present paper, based on knowledge distillation, consists of three successive stages. At the first stage the original model, containing about seven billion parameters, processes a corpus of texts collected from open sources. Receiving the next text fragment as input, it produces a probability distribution of the next token. This distribution is saved and later used as "soft" labels.*

*At the second stage a student model with five times fewer parameters is trained. By minimizing the divergence between its own predictions and the saved distributions, the student model gradually takes over the behaviour of the original model (the teacher model). Unlike works that rely only on the teacher's final predictions, we also use intermediate representations extracted from the hidden layers. This makes it possible to pass to the student information that is not reflected in the output distribution.*

*At the third stage the student is fine-tuned on a small set of labelled data related to the applied task being solved. Having completed the fine-tuning, we compare the resulting model with the original one in accuracy and generation speed.*

*3.2. Choice of layers*

*The choice of the layers used to transfer intermediate representations deserves separate discussion. Having conducted preliminary experiments, we found that the best results are given by layers located in the middle part of the network. The layers closer to the output are apparently too closely tied to the pre-training task, whereas the lower layers mainly encode surface features of the text.*

*It should be stressed that when matching the layers of the teacher and the student, one must take into account the difference in their number. Since we cannot establish a one-to-one correspondence between the layers, we map each student layer to a group of neighbouring teacher layers, averaging their representations.*

## New words

| Russian | Pronunciation | English |
|---|---|---|
| дистилля́ция зна́ний | distillyátsiya znániy | knowledge distillation (ж.) |
| исхо́дный | iskhódnyy | original, initial |
| очередно́й | ocherednóy | next (in a series) |
| порожда́ть (impf.) / породи́ть (pf.) | porozhdát' / porodít' | to produce, to generate |
| распределе́ние вероя́тностей | raspredelénie veroyátnostey | probability distribution (ср.) |
| ме́тка | métka | label (ж.) |
| моде́ль-учени́к / моде́ль-учи́тель | modél'-ucheník / modél'-uchítel' | student model / teacher model (ж.) |
| расхожде́ние | raskhozhdénie | divergence, discrepancy (ср.) |
| перенима́ть (impf.) / переня́ть (pf.) | perenimát' / perenyát' | to adopt, to take over |
| опира́ться (impf.) на + acc. | opirát'sya na | to rely on, to be based on |
| промежу́точное представле́ние | promezhútochnoe predstavlénie | intermediate representation (ср.) |
| скры́тый слой | skrýtyy sloy | hidden layer (м.) |
| разме́ченный | razméchennyy | labelled, annotated |
| прикладно́й | prikladnóy | applied |
| заслу́живать (impf.) + gen. | zaslúzhivat' | to deserve |
| по-ви́димому | po-vídimomu | apparently |
| пове́рхностный при́знак | povérkhnostnyy príznak | surface feature (м.) |
| взаи́мно однозна́чное соотве́тствие | vzaímno odnoznáchnoe sootvétstvie | one-to-one correspondence (ср.) |
| ста́вить в соотве́тствие + acc. + dat. | stávit' v sootvétstvie | to map something to something |
| усредня́ть (impf.) | usrednyát' | to average |

## Grammar

You formed participles in Lessons 13–14 and verbal adverbs in Lesson 15 of the Intermediate book. Scientific Russian uses them constantly, because they let the writer pack several facts into one sentence. The price is long sentences that are hard to read and easy to get wrong. This lesson is about the phrases built around these forms: the participial phrase (причастный оборот) and the verbal adverb phrase (деепричастный оборот).

### 1. The participial phrase: position, commas and agreement

A participial phrase can stand **after** or **before** the noun it describes. The punctuation depends on the position:

| Position | Commas | Example |
|---|---|---|
| after the noun | yes, on both sides (unless the sentence ends) | Метод, предложенный в работе [3], работает быстрее. |
| before the noun | no | Предложенный в работе [3] метод работает быстрее. |

Both orders are correct and common in scientific texts. A practical rule: a short phrase (one or two words) sits comfortably before the noun (*предлагаемый в настоящей работе метод*); a long phrase is easier to read after it. Compare a heavy left-hand version: *выбор используемых для передачи промежуточных представлений слоёв* — correct, but the reader must hold four words in memory before learning that they are about *слоёв*. The text uses the lighter order: *выбор слоёв, используемых для передачи промежуточных представлений*.

**Agreement shows what the phrase belongs to.** The participle agrees in gender, number and case with its noun, and this is often the only signal. Compare:

- *корпус текстов, **собранный** из открытых источников* — the corpus was collected (masculine singular, agrees with *корпус*);
- *корпус текстов, **собранных** из открытых источников* — the texts were collected (genitive plural, agrees with *текстов*).

Both sentences are grammatical, but they say different things. When you read, check the ending of the participle and find the noun that matches it; when you write, check that the ending matches the noun you mean.

**Do not put the noun inside the phrase.** A phrase before the noun must be complete before the noun: ✓ *собранный из открытых источников корпус*; ✓ *корпус, собранный из открытых источников*; ✗ *собранный корпус из открытых источников* (this reads as "the collected corpus from open sources").

### 2. The verbal adverb phrase: the same-subject rule

A verbal adverb phrase is always separated by commas, wherever it stands: *Завершив дообучение, мы сравниваем…; Мы ставим…, усредняя их представления.* Reminder: an imperfective verbal adverb (*получая, минимизируя*) describes an action at the same time as the main verb; a perfective one (*проведя, завершив*) an action completed before it.

**The rule.** The action of the verbal adverb must be performed by the subject of the main clause. In *Проведя эксперименты, мы установили…*, the people who conducted the experiments (*мы*) are the ones who found. In *Минимизируя расхождение, модель-ученик перенимает…*, the student model both minimizes and takes over.

**The two most common violations in scientific writing:**

| ✗ Wrong | Why | ✓ Correct |
|---|---|---|
| Проведя эксперименты, была получена высокая точность. | the subject is *точность*, and accuracy did not conduct the experiments | Проведя эксперименты, мы получили высокую точность. |
| Прочитав статью, у меня появились вопросы. | the subject is *вопросы* | Когда я прочитал статью, у меня появились вопросы. |

The first type (verbal adverb + passive verb) is very widespread, even in published papers, but editors and reviewers mark it as a mistake. If the main clause is passive or impersonal with *у меня*, either make it active with a real subject (*мы получили*) or replace the verbal adverb with a clause (*Когда…, После того как…*) or a preposition (*В результате экспериментов была получена…*).

**Real exceptions.** A verbal adverb is acceptable:

- with an infinitive, whose implied subject does the action: *Сопоставляя слои учителя и ученика, необходимо учитывать разницу в их количестве.* (The one who compares is the one who must take into account.) Similarly *Следует…, нужно…, можно…, важно… + infinitive*.
- in fixed expressions that have become prepositions or introductory words and are no longer tied to a subject: *судя по* (judging by), *исходя из* (based on), *начиная с* (starting from), *учитывая* (given, considering), *благодаря* (thanks to; a preposition now): *Судя по результатам, средние слои важнее.*

### 3. Unpacking and building long sentences

**Reading.** Take the last sentence of the text:

*Не имея возможности установить взаимно однозначное соответствие между слоями, мы ставим в соответствие каждому слою ученика группу соседних слоёв учителя, усредняя их представления.*

1. Find the main subject and predicate: *мы ставим в соответствие* (we map).
2. Put the phrases in brackets: [*Не имея возможности установить…*] — a verbal adverb phrase (reason); [*усредняя их представления*] — a second one (manner).
3. For each participle, find its noun by the ending; here there are none.
4. Restate in short sentences: *Мы не можем установить взаимно однозначное соответствие. Поэтому мы ставим… Представления этих слоёв мы усредняем.*

**Writing.** Three guidelines that Russian editors give:

- No more than two participial or verbal adverb phrases in one sentence. If you have three, split the sentence.
- Do not put two participles of the same type one inside the other (*модель, обученная на данных, собранных из источников, отобранных…*): the reader loses the thread.
- Vary: one sentence with a participle, the next with *который*. *Который* is never wrong and is often clearer, especially when you need a tense or a subject different from the main clause.

**Replacing which with what.** Participial phrase ↔ *который*-clause; verbal adverb phrase ↔ *когда, после того как, так как, если*-clause:

- *модель, содержащая семь миллиардов параметров* = *модель, которая содержит…*
- *корпус, собранный нами* = *корпус, который мы собрали*
- *Проведя эксперименты, мы установили…* = *После того как мы провели эксперименты, мы установили…*
- *Не имея данных, мы…* = *Так как у нас не было данных, мы…*

## Study tip

Korean puts every modifier before the noun, however long: *오픈 소스에서 수집한 텍스트 코퍼스*. If you write Russian with the same habit, your sentences fill up with heavy pre-noun phrases. When a phrase is longer than two or three words, move it after the noun and put commas around it. And before you send a text, search it for every verbal adverb (forms in *-я, -а, -в, -вшись*) and ask "who does this action?" — the answer must be the subject of the clause.

## Exercises

**Exercise 1.** Write out each participial phrase and the noun it describes. Mark it действ. (active) or страд. (passive).
1. На первом этапе исходная модель, содержащая около семи миллиардов параметров, обрабатывает корпус текстов, собранный из открытых источников.
2. На втором этапе обучается модель-ученик, имеющая в пять раз меньше параметров.
3. Это позволяет передать ученику информацию, не отражённую в выходном распределении.
4. Отдельного обсуждения заслуживает выбор слоёв, используемых для передачи промежуточных представлений.

**Exercise 2.** Put in commas where they are needed.
1. Предложенный нами метод работает быстрее.
2. Метод предложенный нами работает быстрее.
3. Обучив модель мы проверили её на тестовой выборке.
4. Мы использовали данные собранные в 2023 году.
5. Собранные в 2023 году данные содержат ошибки.
6. Не меняя архитектуры модель можно ускорить в два раза.

**Exercise 3.** Replace the *который*-clause with a participial phrase.
1. Модель, которая содержит семь миллиардов параметров, требует много памяти.
2. Корпус, который мы собрали из открытых источников, содержит около миллиона документов.
3. Работы, которые опираются только на итоговые предсказания учителя, не используют скрытые слои.
4. Мы выбрали слои, которые расположены в средней части сети.
5. В статье, которую опубликовала Ковалёва, описан похожий подход.

**Exercise 4.** Replace the subordinate clause with a verbal adverb phrase.
1. Когда мы провели эксперименты, мы установили, что средние слои работают лучше.
2. Так как у нас не было размеченных данных, мы использовали синтетические тексты.
3. После того как модель завершает обучение, она сохраняет веса.
4. Когда ученик минимизирует расхождение, он постепенно перенимает поведение учителя.

**Exercise 5.** Fix the mistake.
1. Проведя эксперименты, была получена высокая точность.
2. Используя этот метод, скорость генерации выросла в два раза.
3. Прочитав статью, у меня появились вопросы.
4. Мы проанализировали ошибки, допущенных моделью.
5. Собранный корпус из открытых источников содержит ошибки.

**Exercise 6.** Combine the two sentences into one, using a participial or a verbal adverb phrase.
1. Мы собрали корпус. Корпус содержит 10 000 статей.
2. Мы проанализировали ошибки. Потом мы изменили архитектуру.
3. Модель обучалась на данных. Данные были размечены вручную.
4. Ученик использует промежуточные представления. Так он получает дополнительную информацию.

**Exercise 7.** Unpack each sentence from the text into two or three short sentences without participles or verbal adverbs.
1. Не имея возможности установить взаимно однозначное соответствие между слоями, мы ставим в соответствие каждому слою ученика группу соседних слоёв учителя, усредняя их представления.
2. Минимизируя расхождение между своими предсказаниями и сохранёнными распределениями, модель-ученик постепенно перенимает поведение исходной модели.

**Exercise 8.** Write a paragraph of 5–6 sentences describing the stages of a method from your own research (or an invented one). Use at least two participial phrases and two verbal adverb phrases.

## Answer key

**Exercise 1.**
1. содержащая около семи миллиардов параметров → модель (действ.); собранный из открытых источников → корпус (страд.)
2. имеющая в пять раз меньше параметров → модель-ученик (действ.)
3. не отражённую в выходном распределении → информацию (страд.)
4. используемых для передачи промежуточных представлений → слоёв (страд.)

**Exercise 2.**
1. Предложенный нами метод работает быстрее. (без запятых)
2. Метод, предложенный нами, работает быстрее.
3. Обучив модель, мы проверили её на тестовой выборке.
4. Мы использовали данные, собранные в 2023 году.
5. Собранные в 2023 году данные содержат ошибки. (без запятых)
6. Не меняя архитектуры, модель можно ускорить в два раза.

**Exercise 3.**
1. Модель, содержащая семь миллиардов параметров, требует много памяти.
2. Корпус, собранный нами из открытых источников, содержит около миллиона документов. (*also:* Корпус, собранный из открытых источников, …)
3. Работы, опирающиеся только на итоговые предсказания учителя, не используют скрытые слои.
4. Мы выбрали слои, расположенные в средней части сети.
5. В статье, опубликованной Ковалёвой, описан похожий подход.

**Exercise 4.**
1. Проведя эксперименты, мы установили, что средние слои работают лучше.
2. Не имея размеченных данных, мы использовали синтетические тексты.
3. Завершив обучение, модель сохраняет веса.
4. Минимизируя расхождение, ученик постепенно перенимает поведение учителя.

**Exercise 5.**
1. Проведя эксперименты, мы получили высокую точность.
2. Используя этот метод, мы увеличили скорость генерации в два раза. (*also:* Благодаря этому методу скорость генерации выросла в два раза.)
3. Когда я прочитал статью, у меня появились вопросы. (*also:* Прочитав статью, я записал несколько вопросов.)
4. Мы проанализировали ошибки, допущенные моделью.
5. Собранный из открытых источников корпус содержит ошибки. (*also:* Корпус, собранный из открытых источников, содержит ошибки.)

**Exercise 6.**
1. Мы собрали корпус, содержащий 10 000 статей.
2. Проанализировав ошибки, мы изменили архитектуру.
3. Модель обучалась на данных, размеченных вручную.
4. Используя промежуточные представления, ученик получает дополнительную информацию.

**Exercise 7.** Model answers:
1. Мы не можем установить взаимно однозначное соответствие между слоями. Поэтому мы ставим в соответствие каждому слою ученика группу соседних слоёв учителя. Представления этих слоёв мы усредняем.
2. Модель-ученик минимизирует расхождение между своими предсказаниями и сохранёнными распределениями. Так она постепенно перенимает поведение исходной модели.

**Exercise 8.** Model answer (your text will be different):

Предлагаемый подход состоит из двух этапов. На первом этапе мы отбираем из корпуса тексты, содержащие научную терминологию. Проанализировав их, мы составляем словарь терминов, встречающихся чаще всего. На втором этапе модель дообучается на отобранных текстах. Сравнивая её ответы с ответами исходной модели, мы оцениваем, насколько улучшилось качество работы с терминами.

Checklist:
- at least two participial phrases (here: *содержащие научную терминологию, встречающихся чаще всего*), each agreeing with its noun and set off by commas when it follows the noun;
- at least two verbal adverb phrases (here: *Проанализировав их, Сравнивая её ответы…*), always with commas;
- the subject of each clause with a verbal adverb is the one who performs the action (*мы*), and the main verb is not passive;
- no more than two such phrases in one sentence.

## Review

- You know that a participial phrase after the noun takes commas and one before the noun does not, and you place long phrases after the noun.
- You read the ending of a participle to find its noun: *корпус текстов, собранный…* vs *собранных…*.
- You apply the same-subject rule and can fix *Проведя эксперименты, была получена…* and *Прочитав статью, у меня…*.
- You know the real exceptions: verbal adverbs with an infinitive (*Сопоставляя…, необходимо учитывать*) and fixed expressions (*судя по, исходя из, начиная с, учитывая*).
- You can unpack a long scientific sentence into short ones and combine short sentences into a long one.

---
[← Previous: Aspect Nuances: Negation, Repetition, Imperatives and Politeness](01-aspect-nuances.md) · [Contents](README.md) · [Next: Word Order, Theme and Rheme, Emphasis →](03-word-order-and-emphasis.md)
