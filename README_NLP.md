# NLP Module

## Goal

This module parses Chinese map path-planning requests into structured JSON using color keys. It is designed for a map with seven color points:

`red`, `orange`, `yellow`, `green`, `cyan`, `blue`, `purple`

The NLP layer only extracts the requested route frame. It does not output coordinates, does not search for paths, and does not run CV or MDP logic. Backend CV/MDP modules should use `start`, `waypoints`, and `end` color keys to perform path planning.

## Interface

Use `HybridPathNLP.parse(text)` for normal inference.

Input:

```text
从蓝色点出发，经过青色点和紫色点，最后到绿色点
```

Output:

```json
{
  "intent": "navigation",
  "start": "blue",
  "waypoints": ["cyan", "purple"],
  "end": "green",
  "is_complete": true,
  "missing_slots": []
}
```

The final `parse(text)` output fields are fixed:

```json
{
  "intent": "navigation or unknown",
  "start": "color key or null",
  "waypoints": ["color key"],
  "end": "color key or null",
  "is_complete": true,
  "missing_slots": []
}
```

For diagnostics, use `HybridPathNLP.parse_with_debug(text)`. It wraps the same parse result with fallback metadata:

```json
{
  "result": {
    "intent": "navigation",
    "start": "blue",
    "waypoints": [],
    "end": "green",
    "is_complete": true,
    "missing_slots": []
  },
  "used_fallback": true,
  "fallback_reason": "missing_required_slots",
  "slot_source": "fallback"
}
```

## Color Keys

| Chinese | English key |
| --- | --- |
| 红/赤 | `red` |
| 橙 | `orange` |
| 黄 | `yellow` |
| 绿 | `green` |
| 青 | `cyan` |
| 蓝 | `blue` |
| 紫 | `purple` |

## Architecture

- `IntentClassifier`: TF-IDF character features with `LogisticRegression`.
- `BiLSTMCRFSlotTagger`: PyTorch character-level BiLSTM-CRF BIO sequence tagger.
- `ColorNormalizer`: maps aliases such as `蓝色点`, `蓝点`, and `蓝` to `blue`.
- `PathNLPParser`: rule-based parser used as fallback.
- `HybridPathNLP`: model-first parser. If intent is navigation but slot extraction is missing, invalid, or cannot be normalized to color keys, it falls back to `PathNLPParser`.

## Run Tests

```bash
python -m unittest
```

## Train Slot Tagger

```bash
python train_slot_tagger.py --epochs 20 --output slot_tagger.pkl
```

Small smoke test:

```bash
python train_slot_tagger.py --limit 5 --epochs 1 --output slot_tagger_test.pkl
```

## Generate Training Data

```bash
python generate_training_data.py --output training_data.jsonl
```

## Run Evaluation

```bash
python evaluate_hybrid_nlp.py
```

The report includes intent accuracy, slot accuracy, semantic frame accuracy, and fallback rate. Incorrect predictions are printed with expected output, predicted output, and the fields that differ.

## Scope

This NLP module does not output coordinates. It does not perform path search. It only outputs `start`, `waypoints`, and `end` as color keys. The backend CV/MDP path-planning module is responsible for using those color keys to compute actual movement plans.
