# Legal Sentence Game

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ynrdq3wmdc/badge.svg)](https://ask.thili.ai/b/ynrdq3wmdc)

This project creates a unique CLI game designed for law students to hone their legal drafting skills. Players complete partial legal sentences using n-gram predictions, receiving instant feedback from a fine-tuned GPT-2 model on the clause types they've generated. It's an interactive way to practice legal language and understand common sentence structures in a game format.

## What you are building

**Goal.** You will build a Command Line Interface (CLI) game where law students practice drafting legal sentences by completing partial sentences using n-gram predictions, and receive instant feedback on their generated clause types from a fine-tuned GPT-2 model.

**Done when.** The CLI game successfully presents partial legal sentences for completion, uses n-gram predictions to assist, and then classifies the full completed sentence using the fine-tuned GPT-2 model, providing feedback on the identified clause type.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare a legal text corpus for n-gram model training.**  
      Hint: Use the text processing techniques from the 'Build a Next-Word Predictor from Scratch' drill to prepare your corpus.
- [ ] **2. Implement the n-gram next-word predictor.**  
      Hint: Apply the n-gram probability calculation and prediction logic from the 'Build a Next-Word Predictor from Scratch' drill.
- [ ] **3. Load the fine-tuned GPT-2 model for legal clause classification.**  
      Hint: Recall how to load and use a pre-trained or fine-tuned model from the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
- [ ] **4. Create a CLI game interface presenting partial legal sentences for completion.**  
      Hint: Focus on user input and output, integrating your n-gram model to present initial sentence fragments.
- [ ] **5. After user completion, classify the full sentence and provide feedback.**  
      Hint: Use the loaded GPT-2 model for inference, similar to the classification step in the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Implement a scoring system based on the accuracy of the GPT-2 classification of the user's completed sentence.
- Allow users to choose different difficulty levels, perhaps by varying the length of the partial sentences presented.
- Provide more granular feedback, such as suggesting alternative completions from the n-gram model or highlighting discrepancies in clause types.
