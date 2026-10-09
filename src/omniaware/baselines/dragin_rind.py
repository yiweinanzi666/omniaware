"""CPU numerical core for a one-decision DRAGIN-style Qwen adapter.

The runner supplies processed generation scores and draft-only last-layer
attention. This module never loads or forwards a language model.
"""

from bisect import bisect_left, bisect_right
from decimal import Decimal
import re

import numpy as np
from scipy.special import softmax
import torch
from transformers.convert_slow_tokenizer import bytes_to_unicode


REFERENCE_THRESHOLD = 1.2
CONTENT_POS = frozenset(("NOUN", "ADJ", "VERB", "PROPN", "NUM"))


def generation_entropy_nats(processed_scores):
    """HF outputs.scores: one [1,vocabulary] tensor per emitted draft token."""
    values = []
    for score in processed_scores:
        logits = score.detach().float().cpu().numpy()
        if (logits.ndim != 2 or logits.shape[0] != 1 or not logits.shape[1]
                or np.isnan(logits).any() or np.isposinf(logits).any() or not np.isfinite(logits).any()):
            raise ValueError("Need one finite-support processed vocabulary distribution per token")
        probability = softmax(logits, axis=-1)
        values.append(float(-np.sum(probability*np.log(probability+1e-10), axis=-1)[0]))
    return np.asarray(values, dtype=np.float64)


def received_attention(last_layer_attention, token_count):
    """Official default: mean_head(max_query A), including causal diagonal."""
    attention = last_layer_attention.detach().to(device="cpu", dtype=torch.float64).numpy()
    if (attention.ndim != 4 or attention.shape[0] != 1 or attention.shape[1] < 1
            or attention.shape[2:] != (token_count, token_count)
            or not np.isfinite(attention).all() or (attention < 0).any()):
        raise ValueError("Need finite nonnegative draft-only attention [1,heads,T,T]")
    if token_count == 0:
        return np.empty(0, dtype=np.float64)
    return attention[0].max(axis=1).mean(axis=0)


def align_qwen_tokens(tokenizer, draft_ids):
    """Map actual generated IDs to decoded character spans, retaining indices.

    Fast-tokenizer offsets cover canonical encodings. A valid noncanonical
    byte-BPE segmentation uses its actual token bytes, checked against decode.
    """
    special = set(tokenizer.all_special_ids)
    positions = [i for i, token in enumerate(draft_ids) if token not in special]
    visible_ids = [draft_ids[i] for i in positions]
    text = tokenizer.decode(visible_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    method = "tokenizer_offsets"
    if encoded["input_ids"] == visible_ids:
        offsets = encoded["offset_mapping"]
    else:
        decoder = {character:byte for byte, character in bytes_to_unicode().items()}
        pieces = [bytes(decoder[c] for c in token) for token in tokenizer.convert_ids_to_tokens(visible_ids)]
        if b"".join(pieces) != text.encode("utf-8"):
            raise ValueError("Generated Qwen byte tokens do not reconstruct the decoded draft")
        boundaries = [0]
        for character in text:
            boundaries.append(boundaries[-1]+len(character.encode("utf-8")))
        cursor, offsets = 0, []
        for piece in pieces:
            stop = cursor+len(piece)
            offsets.append((bisect_right(boundaries, cursor)-1, bisect_left(boundaries, stop)))
            cursor = stop
        method = "generated_byte_offsets_checked_against_decode"
    if len(offsets) != len(positions) or any(not 0 <= a <= b <= len(text) for a,b in offsets):
        raise ValueError("Draft token/character alignment is incomplete")
    return dict(text=text, positions=positions, offsets=offsets, offset_method=method,
                special_positions=[i for i in range(len(draft_ids)) if i not in positions])


def word_alignment(alignment, nlp, token_count):
    """Whitespace words within spaCy sentences; POS mapped by actual spans."""
    text = alignment["text"]
    doc = nlp(text)
    words = []
    for sentence_id, sentence in enumerate(doc.sents):
        content = [(t.idx, t.idx+len(t.text)) for t in sentence if t.pos_ in CONTENT_POS]
        for match in re.finditer(r"\S+", text[sentence.start_char:sentence.end_char]):
            start, end = sentence.start_char+match.start(), sentence.start_char+match.end()
            words.append(dict(text=text[start:end], start=start, end=end, sentence_id=sentence_id,
                content_word=any(max(start,a) < min(end,b) for a,b in content)))
    weights = np.zeros((len(words), token_count), dtype=np.float64)
    for position, (start,end) in zip(alignment["positions"], alignment["offsets"], strict=True):
        overlaps = np.asarray([max(0, min(end,w["end"])-max(start,w["start"])) for w in words], dtype=float)
        if overlaps.sum():
            weights[:,position] = overlaps/overlaps.sum()
        elif start < end and not text[start:end].strip():
            # A standalone Qwen space prefix belongs to its following word.
            following = next((i for i,w in enumerate(words) if w["start"] >= end
                              and not text[end:w["start"]].strip()), None)
            if following is not None:
                weights[following,position] = 1.
    return words, weights


def rind_word_scores(entropy, attention, token_weights, sentence_ids, content_mask):
    """Official word sum/mean and sentence scaling, followed by the T05 mask."""
    entropy, attention, weights = map(lambda x:np.asarray(x, dtype=np.float64), (entropy, attention, token_weights))
    sentence_ids, content_mask = np.asarray(sentence_ids), np.asarray(content_mask, dtype=bool)
    if (entropy.ndim != 1 or attention.shape != entropy.shape or weights.shape != (len(sentence_ids), len(entropy))
            or content_mask.shape != sentence_ids.shape or not all(np.isfinite(x).all() for x in (entropy,attention,weights))
            or any((x < 0).any() for x in (attention,weights))):
        raise ValueError("Inconsistent token scores, word weights or sentence/POS mapping")
    if (weights.sum(axis=0)>1+1e-12).any() or (weights.sum(axis=1)<=0).any():
        raise ValueError("Each word needs tokens and token mass cannot be counted twice")
    word_attention = weights@attention
    word_entropy = (weights@entropy)/weights.sum(axis=1)
    scores = np.zeros(len(sentence_ids), dtype=float)
    for sentence in np.unique(sentence_ids):
        selected = sentence_ids == sentence
        mass = word_attention[selected].sum()
        if mass <= 0:
            raise ValueError("Sentence attention mass must be positive")
        scores[selected] = selected.sum()*word_attention[selected]/mass*word_entropy[selected]
    return word_attention, word_entropy, scores, np.where(content_mask, scores, 0.)


def score_draft(draft_ids, processed_scores, last_layer_attention, tokenizer, nlp,
                *, threshold=REFERENCE_THRESHOLD):
    """Return serializable word diagnostics and the single post-mask decision."""
    ids = [int(token) for token in draft_ids]
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative")
    entropy = generation_entropy_nats(processed_scores)
    if len(entropy) != len(ids):
        raise ValueError("Generation scores must align one-to-one with emitted draft IDs")
    attention = received_attention(last_layer_attention, len(ids))
    alignment = align_qwen_tokens(tokenizer, ids)
    words, weights = word_alignment(alignment, nlp, len(ids))
    word_attention, word_entropy, scores, masked = rind_word_scores(entropy, attention, weights,
        [word["sentence_id"] for word in words], [word["content_word"] for word in words])
    for index, word in enumerate(words):
        positions = np.flatnonzero(weights[index]).tolist()
        word.update(token_indices=positions, token_weights=weights[index,positions].tolist(),
            received_attention=float(word_attention[index]), entropy_nats=float(word_entropy[index]),
            unmasked_score=float(scores[index]), masked_score=float(masked[index]),
            hit=bool(masked[index]>threshold))
    unmasked_max, masked_max = float(scores.max(initial=0.)), float(masked.max(initial=0.))
    return dict(draft_text=alignment["text"], draft_token_ids=ids, offset_method=alignment["offset_method"],
        token_character_spans=[dict(token_index=p,start=a,end=b) for p,(a,b) in
            zip(alignment["positions"], alignment["offsets"], strict=True)],
        special_token_positions=alignment["special_positions"],
        unassigned_token_positions=np.flatnonzero(weights.sum(axis=0)==0).tolist(),
        token_entropy_nats=entropy.tolist(), token_received_attention=attention.tolist(), words=words,
        unmasked_max_score=unmasked_max, masked_max_score=masked_max, threshold=float(threshold),
        retrieve=masked_max>threshold, pre_mask_trigger=unmasked_max>threshold,
        mask_removed_all_hits=unmasked_max>threshold and not masked_max>threshold,
        trigger_definition="one decision from max post-POS-mask word score; strict > threshold")


def calibrate_threshold(max_scores, *, target_rate=.785):
    """Nearest attainable strict-threshold rate; tied scores stay together."""
    scores = np.asarray(max_scores, dtype=np.float64)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or (scores<0).any() or not 0 <= target_rate <= 1):
        raise ValueError("Need nonempty finite nonnegative unlabeled scores and target rate in [0,1]")
    ordered = np.sort(scores)
    thresholds = np.unique(np.r_[0., ordered])
    counts = len(scores)-np.searchsorted(ordered, thresholds, side="right")
    target_count = Decimal(str(target_rate))*len(scores)
    index = min(range(len(thresholds)), key=lambda i:(
        abs(Decimal(int(counts[i]))-target_count), int(counts[i]), float(thresholds[i])))
    threshold, count = float(thresholds[index]), int(counts[index])
    return dict(threshold=threshold, reference_threshold=REFERENCE_THRESHOLD,
        calibration_questions=len(scores), target_rate=float(target_rate), retrieve_count=count,
        achieved_rate=count/len(scores), threshold_tie_count=int((scores==threshold).sum()),
        zero_score_count=int((scores==0).sum()),
        rule="minimize absolute count error; ties prefer fewer retrievals, then smaller threshold; strict >; never split equal scores",
        threshold_domain="nonnegative; zero/no-content scores never trigger", label_access=False)
