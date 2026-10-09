# WebQA global source metadata: revised protocol v1

Frozen before full-corpus retrieval and Phase1 answer generation. This specifies
reader inputs, separately from the frozen BGE/CLIP retrieval and ECDF selection.

1. Use the selected exact-text document or image ID. Text evidence is the exact
   Unicode body in `text_documents.jsonl`; retain every snippet-ID alias in the
   provenance record. Never choose an alias by target-question support membership.
2. An image source includes its pixels and its public captions. The official
   [WebQA task description](https://webqna.github.io/) treats the caption as part of
   an image source because it links names to visible entities.
3. For each image ID, collapse whitespace inside each caption, discard empty
   strings, deduplicate exact resulting strings, sort by Unicode code-point order,
   and join with one newline. Retain **all** resulting caption variants. Preserve
   the raw metadata file and the normalized list in provenance. The rule does not
   inspect the question, answers, support labels, category, retrieval score, or
   model output. No caption ranking, first/last-write choice, or per-question
   caption selection is performed.
4. Multiple captions still form one image source with one image slot. Existing
   source replication repeats that entire block. Input lengths and costs include
   every caption. Compute the common context cap from the actual final manifest
   and the planned permutations/replicas; report any coverage failure. Do not
   silently remove caption variants or questions to meet a previous sample's cap.
5. Bind pixels to the asset rows used by the CLIP index, including path, SHA256,
   dimensions, origin and encoding manifest. Reader and retriever must use the
   same asset version. Preserve the four metadata-only PNG repairs, three official
   tolerant JPEG decodes, and the separately documented public revision30172755.
6. Use identical source text, image versions, source order schedules and relevance
   values for static, replication and decode-only EAB. Keep actual U and its gates
   separate until the frozen initial-observation workload completes. A source
   bundle without U is not an executable Phase1 manifest.

The public pool contains12417 image IDs with multiple raw captions and10007 with
multiple whitespace-normalized captions; one common location-map image has248
variants. The union rule preserves available evidence without oracle selection,
but can expose unrelated or conflicting captions and increase input length.
These are measurable limitations, not evidence that any caption variant is useful.
Historical `imgid2caption.pkl` and its choice rule have not been recovered; this
explicit union representation is a **new protocol**, not an exact historical
reader reconstruction. Keep the original and revised protocol results separate.
