# Supplementary Material: "Say What You See" Benchmark & C3 Cleaned Dataset

**Paper:** *Say What You See: Data-Centric Verifiability Filtering for Synthetic Face-Video Supervision*  
**Workshop:** Data4VFM @ WACV 2027  
**License:** CC BY 4.0 (Annotations & Benchmark Metadata) / MIT License (Code)  
**Double-Blind Review Copy:** All author/institution identifiers have been anonymized.

---

## 1. Package Overview

This supplementary archive contains the complete dataset artifacts, benchmark splits, and evaluation code for reproducing the results presented in the paper:

| File | Size | Description |
|---|---|---|
| `cleaned_captions_c3.json` | 90.7 MB | Complete Condition C3 dataset (48,790 clean training clips, fluent rewrites, token-level masked captions, dropped/retained claim breakdown). |
| `val_gold.json` | 136 KB | Leak-free human gold validation benchmark (192 clean single-speaker clips, 634 claims). |
| `test_gold.json` | 137 KB | Held-out human gold test benchmark (192 clean single-speaker clips, 632 claims). |
| `pope_abstain_benchmark.json` | 410 KB | 734 grounded 3-way evaluation probes (296 YES, 296 NO, 142 UNOBSERVABLE). |
| `clean_dataset_captions.py` | 18.4 KB | Deterministic Stage 4.5 C3 cleaning script (spaCy dependency rules + lexical verifiability tables). |
| `evaluate_benchmark.py` | 4.2 KB | Standalone Python evaluation script to calculate Micro-F1, Macro-F1, and CHAIR metrics. |
| `datasheet.md` | 3.2 KB | Datasheet for Dataset following the Gebru et al. standard. |
| `croissant.json` | 1.7 KB | MLCommons Croissant metadata specification for dataset indexing. |

---

## 2. Dataset Schema (`cleaned_captions_c3.json`)

Each video record in `cleaned_captions_c3.json` follows this structured schema:

```json
{
  "video_id": "000000.mp4",
  "is_usable": true,
  "trust_tier": "gold",
  "original_caption": "A girl speaks fluently, keeping a faint self-compassion expression throughout, while selecting words carefully.",
  "cleaned_caption_fluent": "A girl speaks on camera.",
  "cleaned_caption_masked": "A girl speaks [UNVERIFIED], keeping a [UNVERIFIED] expression throughout, while selecting words carefully.",
  "clean_attributes": {
    "gender": {
      "value": "girl",
      "confidence": "certain",
      "span": "girl"
    }
  },
  "dropped_claims": [
    {
      "attribute": "emotion",
      "value": "faint self-compassion",
      "human_judgment": "Not Verifiable"
    },
    {
      "attribute": "style",
      "value": "fluently",
      "human_judgment": "Inferable (Name cue)"
    }
  ],
  "retained_claims": [
    {
      "attribute": "gender",
      "value": "girl",
      "human_judgment": "Supported"
    }
  ]
}
```

---

## 3. Video Asset Redistribution & Alignment Policy

To respect original upstream copyright and hosting terms, this benchmark releases **derivative annotations, claim scores, and cleaned captions**. Raw `.mp4` video files remain hosted by the original authors of Open3DFaceVid (TDMM-LM, Song et al., arXiv:2603.16936).

### Replicating the Environment:
1. Download Open3DFaceVid Part-1 MP4 files following the official repository instructions.
2. Verify that your downloaded video files match the canonical Part-1 container SHA-256 manifest hash:  
   `c493484f4457ecbb2c6a1523ea811af6d887ca593d8bf6b9c99eada28b3d5cb9`
3. Pair the video containers directly with `cleaned_captions_c3.json` via the matching `video_id` key (e.g., `000000.mp4` through `056186.mp4`).

---

## 4. Reproducing Downstream Evaluation

To compute Gold Micro-F1, Macro-F1, and CHAIR-Video metrics on model predictions:

```bash
python evaluate_benchmark.py --predictions path/to/predictions.jsonl --gold test_gold.json
```
