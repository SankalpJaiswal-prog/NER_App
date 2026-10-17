# Named Entity Recognition App

Finds **people, organizations and locations** in text using two models:

- **spaCy** (`en_core_web_sm`): fast baseline
- **Transformer** (`dslim/bert-base-NER`): BERT model fine-tuned for NER

The app highlights entities in colour, shows a results table (Entity | Type | Start | End), lets you download it as CSV, and can run both models side by side.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The transformer model (about 400 MB) downloads automatically the first time you use it, so do this once while you have internet.

## Run the app

```bash
streamlit run app.py
```

## Evaluate the models

```bash
python evaluate.py
```

Prints precision, recall and F1 for each model and entity type, and saves every mistake to `results/errors.csv`.

## Compare both models and make the team Excel file

```bash
python compare_models.py
```

Runs spaCy and the transformer on the answer-key paragraphs and writes `results/NER_Model_Comparison.xlsx` (scores, per-paragraph comparison, and a Mistakes sheet with an empty Cause column). Send this file to Yeshasvini. The first run downloads the transformer (about 400 MB), so you need internet.

## Files

| File | What it does |
|---|---|
| `ner_engine.py` | Extraction logic for both models, model comparison, highlighting |
| `app.py` | Streamlit interface |
| `config.py` | Highlight colours and labels (**Samrudhi: edit colours here**) |
| `samples.py` | Demo texts shown in the app |
| `evaluate.py` | Scores the models against the answer key |
| `compare_models.py` | Runs both models and writes the comparison Excel file |
| `data/samples.csv` | The 18 news paragraphs (from Yeshasvini); also shown in the app's sample picker |
| `data/gold_entities.csv` | Hand-labelled answer key: 30 entities in P2, P3, P4, P6, P10, P13 (from Yeshasvini) |
| `DEMO_SCRIPT.md` | Slide content for the approach and the live demo flow |

## Data format for the team

**`data/samples.csv`**: one row per paragraph

```
sample_id,text
S1,Sundar Pichai met officials from Google India in Hyderabad on Monday.
```

**`data/gold_entities.csv`**: one row per correct entity. `type` must be `PERSON`, `ORG` or `LOC`.

```
sample_id,entity,type
S1,Sundar Pichai,PERSON
S1,Google India,ORG
S1,Hyderabad,LOC
```

An entity counts as correct only if both the text and the type match (upper or lower case does not matter, and a leading "the" is ignored), so label entities exactly as they appear in the text. Each (entity, type) pair is counted once per paragraph, and only paragraphs that appear in the gold file are scored (currently P2, P3, P4, P6, P10, P13). A model that finds a correct entity the key does not list (for example a surname-only mention like "Kapoor") is counted as a false positive, so the key should list every mention.

## How labels are mapped

| Model label | Our label |
|---|---|
| spaCy `PERSON` / transformer `PER` | PERSON |
| `ORG` | ORG |
| spaCy `GPE`, `LOC` / transformer `LOC` | LOC |
| everything else (dates, money, MISC, ...) | ignored |

## Limits

- Input is limited to 20,000 characters.
- The transformer reads long text in chunks of about 1,000 characters.
- Models work best on English text.
