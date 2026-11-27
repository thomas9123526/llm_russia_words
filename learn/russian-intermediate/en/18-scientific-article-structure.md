# Lesson 18. The Structure of a Scientific Article: аннотация, введение, методы, результаты, выводы

> **Intermediate Russian** · Level: B1 · **You will learn:** the sections of a Russian scientific article and what goes into each · key constructions: являться + instrumental, impersonal -ся verbs, short passives · references inside the text: рис., табл., см., и др.

## Text

Minho and Ivan Petrovich have written a short article for a university conference. Read it and notice how each section begins.

**Дообучение языковой модели для ответов на вопросы студентов**

*Авторы: Минхо, И. П. Смирнов*

**Аннота́ция.** В статье рассматривается задача автоматических ответов на вопросы студентов о расписании и документах. Предлагается метод дообучения небольшой языковой модели на русскоязычных данных. Результаты показывают, что дообученная модель отвечает точнее, чем базовая модель.

**Ключевы́е слова́:** языковая модель, дообучение, вопро́сно-отве́тная система.

**1. Введе́ние.** Каждый год студенты задают в деканате одни и те же вопросы. Це́лью работы явля́ется создание системы, отвечающей на такие вопросы автоматически. Актуа́льность работы связана с ростом числа иностранных студентов.

**2. Обзо́р литературы.** Задача ответов на вопросы хорошо изучена для английского языка [1, 2]. Для русского языка таких работ меньше [3].

**3. Методы.** Для обучения были собраны 5000 пар «вопрос — ответ». Данные были разме́чены вручну́ю. Модель дообучалась в течение трёх эпо́х.

**4. Результаты.** Точность модели соста́вила 82 %. Это на 15 % больше, чем у базовой модели (табл. 1). Примеры ответов приведены́ на рис. 1.

**5. Обсужде́ние.** Модель иногда ошибается, если вопрос содержит редкие слова.

**6. Вы́воды.** Таки́м о́бразом, предложенный метод позволя́ет повы́сить точность ответов. В дальне́йшем планируется использовать систему в деканате.

**Благода́рности.** Авторы благодарят сотрудников деканата за помощь в сборе данных.

**Список литературы**
1. …
2. …
3. …

*Fine-Tuning a Language Model to Answer Students' Questions*
*Abstract. The article considers the task of automatically answering students' questions about the timetable and documents. A method of fine-tuning a small language model on Russian-language data is proposed. The results show that the fine-tuned model answers more accurately than the baseline model.*
*Keywords: language model, fine-tuning, question answering system.*
*1. Introduction. Every year students ask the same questions at the dean's office. The aim of the work is to create a system that answers such questions automatically. The relevance of the work is connected with the growing number of international students.*
*2. Literature review. The question answering task has been well studied for English [1, 2]. For Russian there are fewer such works [3].*
*3. Methods. For training, 5,000 question–answer pairs were collected. The data were labelled manually. The model was fine-tuned for three epochs.*
*4. Results. The accuracy of the model was 82%. That is 15% more than the baseline model (Table 1). Examples of answers are given in Fig. 1.*
*5. Discussion. The model sometimes makes mistakes if a question contains rare words.*
*6. Conclusions. Thus, the proposed method makes it possible to increase the accuracy of answers. In the future, it is planned to use the system at the dean's office.*
*Acknowledgements. The authors thank the staff of the dean's office for their help in collecting the data.*
*References.*

## New words

| Russian | Pronunciation | English |
|---|---|---|
| аннота́ция | annotátsiya | abstract (ж.) |
| ключевы́е слова́ | klyuchevýe slová | keywords (мн.) |
| введе́ние | vvedénie | introduction (ср.) |
| обзо́р литерату́ры | obzór literatúry | literature review, related work (м.) |
| обсужде́ние | obsuzhdénie | discussion (ср.) |
| вы́воды | vývody | conclusions (мн.; sg. вы́вод, м.) |
| заключе́ние | zaklyuchénie | conclusion (final section) (ср.) |
| благода́рности | blagodárnosti | acknowledgements (мн.) |
| приложе́ние | prilozhénie | appendix (ср.) |
| цель | tsel' | aim, goal (ж.) |
| явля́ться + instr. | yavlyát'sya | to be (formal) |
| актуа́льность | aktuál'nost' | relevance, topicality (ж.) |
| вопро́сно-отве́тная систе́ма | voprósno-otvétnaya sistéma | question answering system (ж.) |
| разме́тить (pf.; разме́чу) | razmétit' | to label, to annotate |
| вручну́ю | vruchnúyu | manually |
| эпо́ха | epókha | epoch (ж.) |
| соста́вить (pf.) | sostávit' | to amount to, to be (a value) |
| привести́ (pf.; приведу́) | privestí | to give, to cite (examples, results) |
| позволя́ть | pozvolyát' | to allow, to make possible |
| повы́сить (pf.) | povýsit' | to increase, to raise |
| таки́м о́бразом | takím óbrazom | thus |
| в дальне́йшем | v dal'néyshem | in the future, later |

## Grammar

### 1. The sections of an article

Most Russian articles in computer science follow the same plan as English ones (IMRaD). Learn the section names and the typical first sentence of each:

| Section | English | What it contains | Typical phrase |
|---|---|---|---|
| Аннота́ция | Abstract | the whole article in 100–250 words | В статье рассматривается… |
| Ключевы́е слова́ | Keywords | 3–7 terms | языковая модель, дообучение |
| Введе́ние | Introduction | problem, relevance, aim | Целью работы является… |
| Обзо́р литерату́ры | Related work | what others have done | Задача изучена в работах [1, 2]. |
| Постано́вка зада́чи | Problem statement | exact definition of the task | Пусть дано множество… (*Let … be given*) |
| Ме́тоды | Methods | data, model, training | Для обучения были собраны… |
| Экспериме́нты | Experiments | set-up of tests | Эксперименты проводились на… |
| Результа́ты | Results | numbers, tables, figures | Точность составила… |
| Обсужде́ние | Discussion | interpretation, limitations | Модель иногда ошибается, если… |
| Вы́воды / Заключе́ние | Conclusions | summary, future work | Таким образом, … В дальнейшем планируется… |
| Благода́рности | Acknowledgements | thanks, grants | Авторы благодарят… |
| Спи́сок литерату́ры | References | sources | Иванов А. А. … // … |
| Приложе́ние | Appendix | extra material | |

Вы́воды (plural) are usually a numbered or short list of findings; Заключе́ние is a final section in prose. Both translate as "Conclusion(s)".

### 2. Key constructions of the scientific style

**a) являться + instrumental = "to be".** In the present tense Russian normally drops "to be" (Цель работы — создание системы). In formal writing the verb **явля́ться** is used, and the noun after it goes into the **instrumental**:

- Це́лью рабо́ты явля́ется созда́ние систе́мы. — *The aim of the work is to create a system.*
- Основно́й пробле́мой явля́ется недоста́ток да́нных. — *The main problem is the lack of data.*
- Трансфо́рмер явля́ется ба́зовой архитекту́рой совреме́нных моде́лей. — *The transformer is the basic architecture of modern models.*

Note the word order in the first two sentences: the instrumental often comes **first**, the subject (создание, недостаток) **after** the verb. Явля́ется agrees with the subject: Це́лью рабо́ты явля́ются (pl.) созда́ние и тести́рование… when there are two subjects.

**b) Impersonal -ся verbs without "we".** Russian abstracts avoid naming the author. Instead of "we consider", they say "is considered" with a -ся verb (Lesson 9). The verb usually stands **before** its subject and agrees with it:

- В статье рассма́тривается зада́ча… (sg.) — *The article considers the problem…*
- В разде́ле 3 опи́сываются экспериме́нты. (pl.) — *Section 3 describes the experiments.*
- Предлага́ется но́вый ме́тод. — *A new method is proposed.*
- В табли́це 2 приво́дятся результа́ты. — *Table 2 gives the results.*

With an infinitive the verb is in the neuter singular: В дальне́йшем плани́руется испо́льзовать систе́му. — *In the future it is planned to use the system.*

**c) Short passives for completed work (Lesson 14):** Да́нные бы́ли со́браны. Результа́ты приведены́ в табл. 1. A very common pattern is the neuter short form + что: Пока́зано, что… (*It is shown that…*), Устано́влено, что… (*It has been established that…*).

**d) The "authorial we".** Many authors also write with мы: Мы предлага́ем ме́тод… This is acceptable, especially in computer science, but in an abstract the impersonal form is still more common. Do not mix both in one paragraph.

**e) Linking words.** These make your text sound academic:

| Russian | English |
|---|---|
| во-пе́рвых, во-вторы́х | firstly, secondly |
| кро́ме того́ | moreover, in addition |
| одна́ко | however |
| сле́довательно | therefore |
| таки́м о́бразом | thus, so |
| в то же вре́мя | at the same time |
| в дальне́йшем | in the future |

### 3. References inside the text

| Abbreviation | Full word | English |
|---|---|---|
| рис. 1 | рису́нок 1 | Fig. 1 |
| табл. 2 | табли́ца 2 | Table 2 |
| разд. 3 | разде́л 3 | Section 3 |
| см. | смотри́(те) | see |
| ср. | сравни́(те) | cf., compare |
| т. е. | то есть | i.e. |
| и т. д. | и так да́лее | etc. |
| напр. | наприме́р | e.g. |
| и др. | и други́е | et al. |
| с. 15 | страни́ца 15 | p. 15 |
| [1, 2] | | references 1 and 2 |

**Reading aloud.** The noun declines, but the number used as a label stays in the nominative: на рис. 1 — на рису́нке оди́н; в табл. 2 — в табли́це два; в разде́ле 3 — в разде́ле три (or во тре́тьем разде́ле, with an ordinal).

**Reference list.** Russian references often follow the GOST standard. The double slash separates the article from the journal; С. means "pages":
Иванов А. А. Вопросно-ответные системы // Вестник университета. 2021. № 2. С. 10–20.

## Culture

**Publishing as a Russian PhD student.** At the top of a Russian article you often see **УДК** — a library classification code (for example, УДК 004.8 for artificial intelligence). Most journals require the title, abstract and keywords in **both Russian and English**. To defend a кандида́тская диссерта́ция you need several articles in journals from the **Пе́речень ВАК** (the list of the Higher Attestation Commission), and your publications are counted in **РИНЦ**, the Russian Science Citation Index, on the eLIBRARY website. Ask your нау́чный руководи́тель early which journals your department recommends.

## Exercises

**Exercise 1.** Match the Russian section names (1–7) with their English equivalents (a–g).
1. аннота́ция
2. введе́ние
3. обзо́р литерату́ры
4. ме́тоды
5. результа́ты
6. вы́воды
7. спи́сок литерату́ры

a) conclusions · b) references · c) abstract · d) introduction · e) related work · f) methods · g) results

**Exercise 2.** In which section of Minho's article is each sentence?
1. Це́лью рабо́ты явля́ется созда́ние систе́мы, отвеча́ющей на таки́е вопро́сы автомати́чески.
2. Таки́м о́бразом, предло́женный ме́тод позволя́ет повы́сить то́чность отве́тов.
3. Для обуче́ния бы́ли со́браны 5000 пар «вопро́с — отве́т».
4. То́чность моде́ли соста́вила 82 %.
5. Зада́ча отве́тов на вопро́сы хорошо́ изу́чена для англи́йского языка́ [1, 2].
6. В статье́ рассма́тривается зада́ча автомати́ческих отве́тов на вопро́сы студе́нтов.

**Exercise 3.** Rewrite with явля́ться + instrumental.
Example: Цель рабо́ты — созда́ние систе́мы. → Це́лью рабо́ты явля́ется созда́ние систе́мы.
1. Основна́я зада́ча — разме́тка да́нных.
2. Гла́вный результа́т — но́вый ме́тод.
3. Э́та статья́ — пе́рвая рабо́та Минхо.
4. На́ша моде́ль — генерати́вная моде́ль.

**Exercise 4.** Choose the correct form.
1. В статье́ (рассма́тривается / рассма́триваются) зада́ча классифика́ции.
2. В разде́ле 3 (опи́сывается / опи́сываются) экспериме́нты.
3. (Предлага́ется / Предлага́ются) но́вый ме́тод.
4. В табли́це 2 (приво́дится / приво́дятся) результа́ты.
5. В дальне́йшем (плани́руется / плани́руются) испо́льзовать систе́му.

**Exercise 5.** Write the abbreviations in full.
1. см. рис. 2
2. в табл. 1
3. Ивано́в и др.
4. т. е.
5. на с. 15
6. и т. д.
7. напр.

**Exercise 6.** Translate into Russian. Use the impersonal style where possible.
1. The article considers the problem of text classification.
2. The aim of the work is to create a new method.
3. The results are given in Table 2.
4. Thus, the proposed method makes it possible to increase accuracy.
5. In the future it is planned to use a larger model.

**Exercise 7.** Answer the questions about the text.
1. Кака́я зада́ча рассма́тривается в статье́?
2. Что явля́ется це́лью рабо́ты?
3. Ско́лько пар «вопро́с — отве́т» бы́ло со́брано?
4. Каку́ю то́чность показа́ла моде́ль?
5. Когда́ моде́ль ошиба́ется?
6. Где плани́руется испо́льзовать систе́му?

## Answer key

**Exercise 1.** 1 — c · 2 — d · 3 — e · 4 — f · 5 — g · 6 — a · 7 — b

**Exercise 2.** 1. Введе́ние 2. Вы́воды 3. Ме́тоды 4. Результа́ты 5. Обзо́р литерату́ры 6. Аннота́ция

**Exercise 3.**
1. Основно́й зада́чей явля́ется разме́тка да́нных.
2. Гла́вным результа́том явля́ется но́вый ме́тод.
3. Э́та статья́ явля́ется пе́рвой рабо́той Минхо.
4. На́ша моде́ль явля́ется генерати́вной моде́лью.

**Exercise 4.** 1. рассма́тривается 2. опи́сываются 3. Предлага́ется 4. приво́дятся 5. плани́руется

**Exercise 5.**
1. смотри́(те) рису́нок 2
2. в табли́це 1
3. Ивано́в и други́е
4. то есть
5. на страни́це 15
6. и так да́лее
7. наприме́р

**Exercise 6.**
1. В статье́ рассма́тривается зада́ча классифика́ции те́кстов.
2. Це́лью рабо́ты явля́ется созда́ние но́вого ме́тода.
3. Результа́ты приведены́ в табли́це 2. (*also:* Результа́ты приво́дятся в табли́це 2.)
4. Таки́м о́бразом, предло́женный ме́тод позволя́ет повы́сить то́чность.
5. В дальне́йшем плани́руется испо́льзовать бо́лее кру́пную (бо́льшую) моде́ль.

**Exercise 7.**
1. Зада́ча автомати́ческих отве́тов на вопро́сы студе́нтов о расписа́нии и докуме́нтах.
2. Созда́ние систе́мы, отвеча́ющей на вопро́сы студе́нтов автомати́чески.
3. Пять ты́сяч (5000).
4. Во́семьдесят два проце́нта (82 %).
5. Е́сли вопро́с соде́ржит ре́дкие слова́.
6. В декана́те.

## Review

- You know the sections of a scientific article: аннота́ция, введе́ние, обзо́р литерату́ры, ме́тоды, результа́ты, обсужде́ние, вы́воды, спи́сок литерату́ры.
- You can state the aim of a study: Це́лью рабо́ты явля́ется…
- You can write impersonal sentences: В статье́ рассма́тривается…, Предлага́ется…, В дальне́йшем плани́руется…
- You understand and read aloud рис., табл., см., т. е., и др., с.
- You can read a short Russian article and find its aim, method and main result.

---
[← Previous: Word Formation: Prefixes, Suffixes and International Words (-ция, -ость, -ние)](17-word-formation.md) · [Contents](README.md) · [Next: Academic Phrases for Methods and Results (в работе предлагается, было показано) →](19-academic-phrases.md)
