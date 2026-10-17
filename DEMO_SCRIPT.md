# Demo Script and Approach Slide Content

## Approach slide (task 9.2)

**Title: How the system works**

- **Input:** raw English text (pasted or typed)
- **Step 1, extraction:** two models read the text
  - spaCy `en_core_web_sm`: fast, rule-and-statistics baseline
  - BERT (`dslim/bert-base-NER`): transformer fine-tuned on the CoNLL-2003 news dataset
- **Step 2, mapping:** each model's labels are mapped to three types: Person, Organization, Location
- **Step 3, output:** colour-highlighted text plus a table of Entity | Type | Position, exportable as CSV
- **Evaluation:** precision, recall and F1 against a hand-labelled answer key

Suggested diagram: `Text -> [spaCy | BERT] -> label mapping -> highlighted text + table`

## Why two models (say this out loud)

"spaCy is quick and runs anywhere. The BERT model reads whole-sentence context, so it should handle ambiguous names better. We compare them instead of assuming."

## Live demo flow (about 3 minutes)

1. **News sample, spaCy.** Click *Extract*. Point out the colours and the table.
2. **Same text, Compare both.** Show the "Where the models agree" table, then point at any "only one model" rows.
3. **Corporate filing sample.** This is the relevant domain for company documents (directors, registered office, auditors).
4. **Tricky sample.** Show the ambiguity: "Washington" (person or place?), "Jordan", "Amazon". Say plainly where the models disagree or are wrong.
5. **Paste a new text from the audience** if time permits. Prepare one yourself first so you know the result.
6. **Download CSV** to show the output is usable.

## What to expect (checked with the spaCy baseline)

On the three built-in samples, spaCy:

- Tags "Sundar Pichai" as an **organization** instead of a person
- Tags "Google India" as a **location** instead of an organization
- Includes "The Board of Directors of" inside an organization name

These are good examples for the Limitations slide. Run the transformer on the same samples before the presentation and write down what it does differently. The results above are for spaCy only.

## Evaluation result so far (spaCy only, on the 30-entity answer key)

| Type | Precision | Recall | F1 |
|---|---|---|---|
| Person | 0.88 | 1.00 | 0.93 |
| Organization | 0.73 | 0.89 | 0.80 |
| Location | 0.92 | 0.79 | 0.85 |
| **All** | **0.84** | **0.87** | **0.85** |

Typical spaCy mistakes: Lyon tagged as an organization (should be a location), Hyderabad tagged as an organization, Greenpeace tagged as a person, and the surname "Paris" tagged as a location even though it refers to a person. Run `python evaluate.py` on your own machine to add the transformer's numbers, and use the comparison on the results slide.

## Backup plan

- Take screenshots of each demo step before the presentation.
- Keep the app already running and the transformer already loaded (first load is slow).
- If the live demo fails, switch to the screenshots and carry on.
- Have `results/errors.csv` open as proof of the evaluation.

## Likely viva questions

- **Why two models?** To compare a fast baseline against a context-aware transformer.
- **What does precision mean here?** Of the entities the model found, how many were correct.
- **What does recall mean?** Of the entities that really exist, how many the model found.
- **Why only three entity types?** Focus: the project targets names, places and organizations.
- **What are the limits?** English only, trained mostly on news text, ambiguous names, informal or lowercase text.
- **How would you improve it?** Fine-tune on company and legal documents, add more labelled data.
