"""
Stage 4.5: Verifiability-Cleaned Captioner (Condition C3: Say What You See)
Applies claim-level verifiability filtering across all 56,187 videos in the dataset based on:
1. Human-annotator flagged visual anomalies (quarantine + QC exclusions).
2. Human-calibrated claim verifiability probabilities (p >= 0.50 supported, p < 0.50 ungrounded).

For each video, produces:
1. cleaned_caption_fluent: Syntactically fluent, natural English with ungrounded clauses dropped.
2. cleaned_caption_masked: Grounding audit caption with ungrounded attribute spans replaced by [UNVERIFIED].
3. clean_attributes: Dictionary retaining only visually verified attribute claims.
4. dropped_claims: List of removed hallucinated attribute claims.
5. trust_tier: Gold, High, Medium, or Unverified.

Output: outputs/cleaned_captions_c3.json
"""

import os
import re
import json
import yaml
import argparse
from tqdm import tqdm
from collections import Counter


_PROF_PHRASES_RE = (
    r"(?:customer\s+support\s+agent|office\s+worker|news\s+anchor|"
    r"doctor|engineer|vlogger|student|athlete|teacher|chef|gamer|interviewer|"
    r"dancer|singer|scientist|artist|lawyer|nurse|professor|musician|writer|"
    r"actor|actress|cashier|clerk|manager|pilot|barista|coach|trainer|speaker|host|anchor|presenter)"
)


_VISUAL_ACTION_TERMS = (
    "shake", "shakes", "nod", "nods", "tilt", "tilts", "blink", "blinks",
    "smile", "smiles", "smiling", "frown", "frowns", "gaze", "eyes", "brow",
    "eyebrow", "tear", "tears", "cheek", "cheeks", "dimple", "dimples",
    "head", "hand", "hands", "lip", "lips", "mouth", "chin", "shoulder",
)


def regularize_fluent_caption(raw_caption, ungrounded_spans, grounded_attrs):
    """
    Produces a syntactically clean, fluent English caption removing ungrounded clauses
    while strictly preserving grounded attributes (e.g., Supported emotion/age/gender)
    and visible physical motion cues.
    """
    if not raw_caption or not isinstance(raw_caption, str):
        return "A person appears on camera."
    text = raw_caption

    # 1. Subject noun phrase regularization when profession is ungrounded
    gender_val = grounded_attrs.get("gender", {}).get("value")
    age_val = grounded_attrs.get("age", {}).get("value")
    keep_profession = "profession" in grounded_attrs
    keep_emotion = "emotion" in grounded_attrs
    keep_style = "style" in grounded_attrs

    if gender_val == "girl" and not age_val:
        subj = "girl"
    elif gender_val == "boy" and not age_val:
        subj = "boy"
    elif gender_val in ["woman", "female", "girl", "lady"]:
        subj = f"{age_val} woman" if age_val else "woman"
    elif gender_val in ["man", "male", "boy", "gentleman"]:
        subj = f"{age_val} man" if age_val else "man"
    elif age_val:
        subj = f"{age_val} person"
    else:
        subj = "person"

    lead_art = "An" if subj[0].lower() in "aeiou" else "A"

    if not keep_profession:
        text = re.sub(
            r"\b(?:A|An)\s+(?:(?:female|male|elderly|young|old|middle-aged)\s+){0,2}" + _PROF_PHRASES_RE + r"\b",
            f"{lead_art} {subj}",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(A|An)\s+(?:female|male|elderly|young)\s+(?:news\s+)?(?!woman\b|man\b|girl\b|boy\b|person\b|lady\b|gentleman\b|kid\b|child\b)[a-z\-]+(?=\s+(?:speaks|talks|reads|introduces|adds|continues|displays|presents|gestures|looks|turns|smiles|keeps|shows|carries|maintains|sings|recounts|elaborates|finishes))",
            f"{lead_art} {subj}",
            text,
            flags=re.I,
        )

    # 2. Strip ungrounded emotion/demeanor clauses ONLY when emotion is NOT grounded
    _EMO_TAIL = r"(?:\s+(?:between\s+phrases|from\s+sentence\s+to\s+sentence|in\s+every\s+line|throughout|steady\s+on\s+the\s+face|on\s+the\s+face))?"
    _EMO_NOUN = r"(?:expression|demeanor|look|tone|face|countenance)"
    if not keep_emotion:
        text = re.sub(
            r"(?:[,;]\s*|\s+(?:in|and|as|by|with|while)\s+|\s+\b)(?:featuring|keeping|holding|letting|presenting|framing(?:\s+each\s+thought)?\s+with|"
            r"framed\s+by|marked\s+by|shaped\s+by|voice\s+carrying|carrying|bearing|wearing|showing|maintaining|sustaining|"
            r"revealing|paired\s+with)\s+(?:a|an)?\s*(?:[^,;.]*?,\s*)?[^,;.]*?"
            + _EMO_NOUN + _EMO_TAIL + r"[^,;.]*",
            ", ",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"[,;]?\s*\b(?:with|in|under|by)\s+(?:a|an)?\s*(?:[^,;.]*?,\s*)?[^,;.]*?" + _EMO_NOUN + _EMO_TAIL + r"[^,;.]*",
            ", ",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"[,;]\s*(?:a|an)\s+(?:[^,;.]*?,\s*)?[^,;.]*?" + _EMO_NOUN + _EMO_TAIL + r"[^,;.]*",
            ", ",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(displays|shows|keeps|wears|carries|maintains)\s+(?:a|an)\s+(?:[^,;.]*?,\s*)?[^,;.]*?" + _EMO_NOUN + _EMO_TAIL + r"[^,;.]*",
            r"\1",
            text,
            flags=re.I,
        )

    # 3. Strip ungrounded acoustic speech style patterns ONLY when style is NOT grounded
    if not keep_style:
        text = re.sub(r"\bkeeps\s+(?:the\s+)?voice\s+(?:steady|even)\s+as\b", "appears as", text, flags=re.I)
        text = re.sub(r"\bkeeps\s+(?:the\s+)?voice\s+(?:steady|even)\b", "appears", text, flags=re.I)
        text = re.sub(r"\blayering\s+strong\s+nods\s+over\s+a\s+steady\s+voice\b", "nodding noticeably", text, flags=re.I)
        text = re.sub(r"[,;]?\s*\b(?:over|with|in)\s+a\s+(?:steady|even|warm|calm)\s+voice\b", "", text, flags=re.I)
        text = re.sub(
            r"[,;]?\s*\b(?:voice\s+(?:colored(?:\s+by\s+[^,;.]+)?|bright(?:\s+on\s+the\s+high\s+notes)?|carrying\s+[^,;.]+|warm\s+and\s+steady|steady|even|quivering)|"
            r"(?:with\s+)?(?:clear\s+)?diction(?:\s+(?:precise\s+and\s+unhurried|clear|steady|even|measured))?)\b",
            ", ",
            text,
            flags=re.I,
        )
        text = re.sub(r"\bpresent(?:s)?(?=\s*(?:[,;.]|$))", "speaks on camera", text, flags=re.I)
        text = re.sub(r"\bhums\s+between\s+lines\s+and\b", "", text, flags=re.I)
        text = re.sub(r"[,;]\s*a\s+[^,;.]*?\s+tone\s+shaping\s+each\s+line", ", ", text, flags=re.I)
        text = re.sub(
            r"\b(?:in|with|at)\s+(?:a|an)\s+(?:warm\s+tone|matter-of-fact\s+(?:tone|way)|steady\s+pace|consistent\s+pace|"
            r"even\s+rhythm|confident\s+rhythm|natural\s+pace|lively\s+delivery|calm\s+cadence|quiet\s+beat|steady\s+beat|brief\s+fermata)\b",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"[,;]\s*(?:as\s+the\s+voice\s+stays\s+steady|voice\s+quivering|and\s+keeping\s+diction\s+clear|"
            r"as\s+the\s+delivery\s+remains\s+composed|without\s+rushing\s+any\s+line|as\s+the\s+sentences\s+flow\s+naturally|"
            r"with\s+brief\s+pauses\s+for\s+emphasis|with\s+emphasis(?:\s+where\s+it\s+matters)?|"
            r"and\s+pausing\s+briefly\s+to\s+breathe|shaping\s+the\s+phrasing|punctuating\s+phrases|"
            r"pacing\s+the\s+words\s+with\s+care)",
            ", ",
            text,
            flags=re.I,
        )
        text = re.sub(r"\btalks\s+plainly\b", "talks", text, flags=re.I)
        text = re.sub(r"\bkeeps\s+talking\s+steadily\b", "keeps talking", text, flags=re.I)
        text = re.sub(r"\b(?:while\s+)?(?:keeping\s+tone\s+even|selecting\s+words\s+carefully|choosing\s+words)\b", "", text, flags=re.I)
        text = re.sub(r"\b(?:fluently|steadily|smoothly|crisp)\b", "", text, flags=re.I)

    # 4. Strip remaining extracted ungrounded spans (protecting physical anatomical motion cues)
    for span in ungrounded_spans:
        sp_low = (span or "").lower()
        if any(tok in sp_low for tok in _VISUAL_ACTION_TERMS):
            continue
        if len(span) >= 3 and sp_low in text.lower():
            pattern = re.compile(re.escape(span), re.IGNORECASE)
            text = pattern.sub("", text)

    # Clean any bare leftover "(showing/maintaining/with) a expression/demeanor/look/tone" fragments
    if not keep_emotion:
        text = re.sub(
            r"(?:[,;]\s*|\b)(?:showing|keeping|carrying|holding|maintaining|wearing|bearing|revealing|sustaining|"
            r"framed\s+by|marked\s+by|shaped\s+by|with|in|under)\s+(?:a|an)\s+" + _EMO_NOUN + _EMO_TAIL + r"\b",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"[,;]\s*" + _EMO_NOUN + _EMO_TAIL + r"\s*(?=[,;. ]|$)", "", text, flags=re.I)

    text = re.sub(r"\b(?:displays|shows|keeps|wears|carries|maintains)\s*(?=[,;.])", "appears on camera", text, flags=re.I)
    text = re.sub(
        r"\b(speaks|talks|presents|appears|adds|continues|sings|reads|finishes|delivers)\s+(?:in|and|as|by|with|at|for|while|through)\s*(?=[,;. ]*(?:[,;.]|$))",
        r"\1",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"[,;]\s*(?:while|as|and|by|with|into|through|throughout|paired|framed(?:\s+by)?|marked(?:\s+by)?|sustaining|revealing|holding|keeping|letting|presenting|featuring|carrying|showing|maintaining)\s*(?=[,;. ]*(?:[,;.]|$))",
        "",
        text,
        flags=re.I,
    )

    # 5. Syntactic punctuation & whitespace cleanup
    text = re.sub(r"\s+([.,;])", r"\1", text)
    text = re.sub(r"([,;])(?:\s*[,;])+", r"\1", text)
    text = re.sub(r"\s*,\s*\.", ".", text)
    text = re.sub(r"\band\s*,", "and", text)
    text = re.sub(r",\s*and\s*(?=\.|$)", "", text)
    text = re.sub(r"([,;])(?:\s*[,;])+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    if text.startswith(",") or text.startswith(";"):
        text = text.lstrip(",; ").strip()
    if text.endswith(",") or text.endswith(";"):
        text = text[:-1].strip() + "."
    if text and not text.endswith("."):
        text += "."
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    words = text.split()
    if len(words) <= 3:
        has_speech_verb = bool(re.search(r"\b(?:speaks|talks|sings|recounts|elaborates|narrates|continues)\b", raw_caption, re.I))
        text = f"{lead_art} {subj} {'speaks' if has_speech_verb else 'appears'} on camera."

    return text


def mask_caption(raw_caption, ungrounded_spans):
    """
    Replaces ungrounded claim spans with [UNVERIFIED] token for causal grounding ablation.
    """
    if not raw_caption or not isinstance(raw_caption, str):
        return ""
    text = raw_caption
    # Sort spans by descending length to prevent partial sub-string collision
    sorted_spans = sorted(ungrounded_spans, key=len, reverse=True)
    for span in sorted_spans:
        if len(span) >= 3 and span.lower() in text.lower():
            pattern = re.compile(re.escape(span), re.IGNORECASE)
            text = pattern.sub('[UNVERIFIED]', text)
    return text


def run_clean_captions(config_path="configs/config.yaml", max_samples=None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    manifest_path = cfg["paths"]["manifest_json"]
    attrs_path = cfg["paths"]["attributes_json"]
    scores_path = os.path.join(cfg["paths"]["output_dir"], "dataset_verifiability_scores.json")
    qc_path = cfg["paths"]["qc_json"]
    quarantine_path = os.path.join(cfg["paths"]["output_dir"], "anomalous_quarantine.json")
    clean_verif_path = os.path.join(cfg["paths"]["output_dir"], "verification_clean.json")
    out_path = os.path.join(cfg["paths"]["output_dir"], "cleaned_captions_c3.json")

    print("=" * 70)
    print("STAGE 4.5: VERIFIABILITY CAPTION CLEANING (CONDITION C3)")
    print("=" * 70)

    if not os.path.exists(scores_path):
        print(f"Error: Claim scores file not found at {scores_path}. Run Stage 4.4 first.")
        return

    # Load Quarantined Anomalies
    quarantined_ids = set()
    if os.path.exists(quarantine_path):
        with open(quarantine_path, "r", encoding="utf-8") as f:
            q_data = json.load(f)
            quarantined_ids = set(item["video_id"] for item in q_data.get("quarantined_samples", []))
        print(f"Loaded {len(quarantined_ids)} quarantined anomalous videos.")

    # Load QC Exclusions
    hard_excludes = set()
    if os.path.exists(qc_path):
        with open(qc_path, "r", encoding="utf-8") as f:
            qc_data = json.load(f)
            for vid, info in qc_data.get("records", {}).items():
                if info.get("decision") == "hard_exclude":
                    hard_excludes.add(vid)
        print(f"Loaded {len(hard_excludes)} QC hard-excluded videos.")

    # Load Gold Clean Human Verifications
    gold_map = {}
    if os.path.exists(clean_verif_path):
        with open(clean_verif_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                gold_map[item["video_id"]] = item
        print(f"Loaded {len(gold_map)} clean human gold records.")

    # Load Manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_map = {m["video_id"]: m for m in json.load(f)["records"]}

    # Load Claim Verifiability Scores
    with open(scores_path, "r", encoding="utf-8") as f:
        scores_data = json.load(f)
        video_claims = scores_data.get("videos", {})

    # Load Original Attributes
    with open(attrs_path, "r", encoding="utf-8") as f:
        attrs_records = {item["video_id"]: item for item in json.load(f)["records"]}

    all_vids = list(manifest_map.keys())
    if max_samples:
        all_vids = all_vids[:max_samples]

    print(f"Processing causal cleaning across {len(all_vids)} videos...")

    cleaned_records = {}
    attr_dropped_counts = Counter()
    attr_retained_counts = Counter()
    usable_count = 0
    excluded_count = 0

    for vid in tqdm(all_vids, desc="Cleaning Captions"):
        man_item = manifest_map.get(vid, {})
        orig_caption = man_item.get("caption") or ""
        attr_item = attrs_records.get(vid, {})
        orig_attrs = attr_item.get("attributes", {})
        claim_info = video_claims.get(vid, {}).get("claims", [])

        # Check visual video gate
        is_quarantined = vid in quarantined_ids
        is_hard_exclude = vid in hard_excludes
        is_usable = not (is_quarantined or is_hard_exclude)

        if not is_usable:
            excluded_count += 1
            reason = "quarantined_anomaly" if is_quarantined else "hard_exclude_qc"
            cleaned_records[vid] = {
                "video_id": vid,
                "is_usable": False,
                "exclusion_reason": reason,
                "original_caption": orig_caption,
                "cleaned_caption_fluent": None,
                "cleaned_caption_masked": None,
                "clean_attributes": {},
                "dropped_claims": [],
                "retained_claims": []
            }
            continue

        usable_count += 1
        grounded_attrs = {}
        ungrounded_spans = []
        dropped_claims = []
        retained_claims = []

        # Claim-level verifiability partitioning
        for c in claim_info:
            attr_name = c["attribute"]
            val = c.get("value")
            span = c.get("span", val)
            prob = c.get("probability", 0.0)
            tier = c.get("tier", "unverified")

            if prob >= 0.50:
                # Grounded claim -> KEEP
                retained_claims.append(c)
                grounded_attrs[attr_name] = orig_attrs.get(attr_name, {"value": val})
                attr_retained_counts[attr_name] += 1
            else:
                # Ungrounded / hallucinated claim -> DROP / MASK
                dropped_claims.append(c)
                if span:
                    ungrounded_spans.append(span)
                attr_dropped_counts[attr_name] += 1

        # Generate Fluent Condition C3 caption ("Say What You See")
        fluent_caption = regularize_fluent_caption(orig_caption, ungrounded_spans, grounded_attrs)

        # Generate Masked Condition C3 caption
        masked_caption = mask_caption(orig_caption, ungrounded_spans)

        # Determine dominant trust tier
        if vid in gold_map:
            dominant_tier = "gold"
        elif any(c["tier"] == "high" for c in retained_claims):
            dominant_tier = "high"
        elif any(c["tier"] == "medium" for c in retained_claims):
            dominant_tier = "medium"
        else:
            dominant_tier = "unverified"

        cleaned_records[vid] = {
            "video_id": vid,
            "is_usable": True,
            "trust_tier": dominant_tier,
            "original_caption": orig_caption,
            "cleaned_caption_fluent": fluent_caption,
            "cleaned_caption_masked": masked_caption,
            "clean_attributes": grounded_attrs,
            "dropped_claims": dropped_claims,
            "retained_claims": retained_claims
        }

    summary = {
        "title": "Condition C3 Verifiability-Cleaned Dataset",
        "total_videos_audited": len(all_vids),
        "usable_clean_videos": usable_count,
        "excluded_anomalous_videos": excluded_count,
        "retention_rate": f"{usable_count / len(all_vids) * 100:.2f}%",
        "total_claims_retained_grounded": sum(attr_retained_counts.values()),
        "total_claims_dropped_ungrounded": sum(attr_dropped_counts.values()),
        "retained_by_attribute": dict(attr_retained_counts),
        "dropped_by_attribute": dict(attr_dropped_counts),
        "records": cleaned_records
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("VERIFIABILITY CLEANING (CONDITION C3) SUMMARY:")
    print(f"  Total Audited Videos:        {len(all_vids)}")
    print(f"  Clean Usable Human Videos:   {usable_count} ({summary['retention_rate']})")
    print(f"  Excluded Anomalous Videos:   {excluded_count}")
    print("-" * 70)
    print(f"  Grounded Claims Retained:    {sum(attr_retained_counts.values())}")
    print(f"  Ungrounded Claims Dropped:   {sum(attr_dropped_counts.values())}")
    print("-" * 70)
    for a in ["gender", "age", "emotion", "profession", "style"]:
        r = attr_retained_counts.get(a, 0)
        d = attr_dropped_counts.get(a, 0)
        tot = r + d
        pct = (r / tot * 100) if tot > 0 else 0.0
        print(f"  {a:<12}: Retained {r:<6} | Dropped {d:<6} ({pct:.1f}% Grounded)")
    print("=" * 70)
    print(f"Verifiability-cleaned Condition C3 dataset saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    run_clean_captions(args.config, args.max_samples)
