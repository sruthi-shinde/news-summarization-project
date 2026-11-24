# 📰 News Summarization and Key Information Extraction Using NLP

A B.Tech (AI & Data Science) NLP mini project. The web app takes a long news article
(pasted text, `.txt` or `.pdf`) and automatically produces:

1. An **extractive summary** (TF-IDF + cosine similarity sentence ranking)
2. **Keywords** (TF-IDF)
3. **Named entities** – Person, Organization, Location, Date, Money, Event (spaCy)
4. **Article statistics** – word counts, compression ratio, etc.

Everything is free and runs offline: no paid APIs, no API keys, no model training.

---

## 1. Folder structure

```
news_summarization_project/
├── app.py                      # Streamlit dashboard (run this)
├── requirements.txt
├── README.md
├── data/
│   ├── sample_news.txt         # default sample (same as sample_technology.txt)
│   ├── sample_technology.txt
│   ├── sample_sports.txt
│   ├── sample_politics_business.txt
│   └── reference_summaries.json  # human summaries used for ROUGE evaluation
├── src/
│   ├── preprocessing.py        # cleaning, tokenization, stopwords, lemmatization
│   ├── summarizer.py           # TF-IDF extractive summarizer
│   ├── keyword_extractor.py    # TF-IDF keywords
│   ├── ner.py                  # spaCy named entity recognition
│   ├── pdf_extractor.py        # PDF -> text (pypdf)
│   ├── evaluation.py           # OPTIONAL: ROUGE-1/2/L
│   └── utils.py                # settings, validation, NLTK checks
├── docs/
│   ├── REPORT.md               # full project report content
│   ├── PPT_CONTENT.md          # 12-slide presentation content
│   └── VIVA_QA.md              # 30 viva questions and answers
└── assets/screenshots/         # put your screenshots here
```

All the sample articles are **original and fictional** (written for testing).

---

## 2. Installation on Windows (VS Code)

**Requirement:** Python **3.10, 3.11 or 3.12** (tick *"Add Python to PATH"* while installing).
Check with `python --version`.

1. Open **VS Code** → `File` → `Open Folder...` → select `news_summarization_project`.
2. Open the terminal: `Terminal` → `New Terminal` (PowerShell is fine).
3. Create a virtual environment:
   ```
   python -m venv venv
   ```
4. Activate it:
   ```
   venv\Scripts\activate
   ```
   > If PowerShell says *"running scripts is disabled"*, run this once and activate again:
   > `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
   > (or use **Command Prompt** instead, where `venv\Scripts\activate` always works).

   You should now see `(venv)` at the start of the line.
5. Install the packages:
   ```
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Download the NLTK resources:
   ```
   python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
   ```
7. Download the spaCy English model:
   ```
   python -m spacy download en_core_web_sm
   ```
8. Run the app:
   ```
   streamlit run app.py
   ```
   The browser opens at `http://localhost:8501`. Stop the app with `Ctrl + C`.

**Next time** you only need: `venv\Scripts\activate` then `streamlit run app.py`.

> The app also downloads missing NLTK data automatically the first time, but step 6 is
> recommended so that it works without internet later.

---

## 3. How to use

1. Choose an input method: **Paste text**, **Upload TXT**, **Upload PDF** or **Use a sample article**.
2. Choose the summary length (**Short / Medium / Long**).
3. Click **Analyze Article**.
4. Read the four result cards: Summary, Keywords, Named Entities, Statistics.
5. Open the *"How it worked"* expanders to show preprocessing steps and sentence scores
   (useful in the viva).

---

## 4. Testing

| Test | Steps | Expected result |
|---|---|---|
| Sample article | Choose *Use a sample article* → Analyze | Summary, ~10 keywords, ~17 entities, statistics shown |
| Summary lengths | Analyze same sample as Short, Medium, Long | Summary gets longer each time (4 → 6 → 8 sentences for the samples) |
| Empty input | Paste nothing → Analyze | "The input is empty..." |
| Very short text | Paste `This is a test.` → Analyze | "The article is too short..." |
| TXT upload | Upload `data/sample_sports.txt` | Works like pasted text |
| Wrong file | Upload a `.docx` as TXT/PDF | "Unsupported file type..." message |
| Invalid PDF | Rename any file to `x.pdf` and upload | "not a valid PDF or damaged" |
| Scanned PDF | Upload an image-only PDF | "No text could be extracted..." |
| Missing spaCy model | Run in an environment without the model | Summary and keywords still work; entities card shows how to install the model |
| ROUGE (command line) | `python -m src.evaluation` | ROUGE-1/2/L F1 for each sample and length |

### What was actually verified during development

These checks were run on **Linux with Python 3.12** (Streamlit 1.65, spaCy 3.8, NLTK 3.10, scikit-learn 1.8):

* all `.py` files compile; the real server starts (`/_stcore/health` returns `ok`, page returns HTTP 200, no errors in the log)
* app run headlessly with Streamlit's `AppTest`: 3 sample articles × Short/Medium/Long (4 / 6 / 8 sentences), all four result cards shown
* uploads through the real uploader widget: valid TXT, too-short TXT, empty TXT, valid text PDF, blank PDF (no text), invalid PDF bytes, and wrong file types (`.docx` as TXT, `.txt` as PDF)
* password-protected PDF, UTF-8 and Latin-1 text files, missing NLTK data (friendly error), missing spaCy model (summary and keywords still work, entities card explains the fix)

**Not verified:** running on Windows itself, and visual appearance in a real browser (no screenshots were taken). Please run the steps in section 2 on your PC and add your own screenshots to `assets/screenshots/`.

Command-line ROUGE output on the supplied samples (Medium summaries) looks like:

```
sample_technology.txt         Medium ROUGE-1: F1=0.383  ROUGE-2: F1=0.109  ROUGE-L: F1=0.283
sample_sports.txt             Medium ROUGE-1: F1=0.369  ROUGE-2: F1=0.177  ROUGE-L: F1=0.286
sample_politics_business.txt  Medium ROUGE-1: F1=0.360  ROUGE-2: F1=0.199  ROUGE-L: F1=0.275
```

---

## 5. Algorithm in brief

```
Article → clean → split into sentences → tokenize → remove stopwords → lemmatize
        → TF-IDF (each sentence = one document)
        → score = 0.45 × average TF-IDF of the sentence
                + 0.45 × cosine similarity with the whole-article vector
                + 0.10 × position bonus      (sentences < 6 words are penalised)
        → rank sentences → pick top 20% / 30% / 40% → show in ORIGINAL order
```

* **Keywords:** sum of TF-IDF weights of every word over the article + repeated
  two-word phrases (e.g. "Mumbai Strikers"). 5–15 keywords depending on article length.
* **NER:** spaCy `en_core_web_sm` + simple rules (names after *Dr./Professor*, money like
  *Rs 5,000 crore*, removing junk dates such as "day", merging "Joshi" into "Vikram Joshi").
* **Headline:** if the first paragraph is a short line without a full stop, it is shown as the
  headline and kept out of the analysis.

## 6. Known limitations

* Extractive: sentences are copied, not rewritten; summaries can feel slightly disconnected.
* spaCy's small model can still make mistakes (for example tagging a person as an organization).
* PDF text loses blank lines, so a headline may not be detected automatically.
* Scanned (image-only) PDFs need OCR, which is not included.
* English only.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `'streamlit' is not recognized` | Activate the venv first: `venv\Scripts\activate` |
| spaCy model error message in the Entities card | `python -m spacy download en_core_web_sm` |
| NLTK `LookupError` / resources missing | `python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4` |
| spaCy install fails | Use Python 3.10–3.12 (very new Python versions may lack wheels) |
| Port already in use | `streamlit run app.py --server.port 8502` |
