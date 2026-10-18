# V3.17 第一轮：Gemini 逐句修改建议与处理结果

模型：gemini-3.8-flash-high。每条建议都经过自动检查（引用、占位符、数字、禁用词是否改动）和人工审读。

后续为守住 8 页，又有少量措辞调整，以 `diff_V3.16_V3.17/diff.pdf` 为准。

## G00-00（Abstract，采纳）
- Gemini 的理由：Adopt the locked term format-specific and eliminate wordy phrasing.
- 原文：Existing multimodal RAG systems decide when and where to retrieve with uncertainty signals and rules designed for each format, which makes these decisions hard to coordinate across formats.
- Gemini：Existing multimodal RAG systems decide when and where to retrieve with format-specific uncertainty signals and rules, making these decisions hard to coordinate across formats.

## G00-01（Abstract，采纳）
- Gemini 的理由：Remove the duplicate occurrence of 'benchmarks' and streamline the sentence opening.
- 原文：On WebQA and on image, table, and short-video benchmarks, \method{} outperforms the strongest baseline, M2-RAG, on all five evaluation sets of four benchmarks (72.15 versus 68.90 on image-backed WebQA).
- Gemini：Across four benchmarks covering WebQA and image, table, and short-video settings, \method{} outperforms the strongest baseline, M2-RAG, on all five evaluation sets (72.15 versus 68.90 on image-backed WebQA).

## G01-00（Introduction，改写后采纳）
- Gemini 的理由：Cuts redundant topic restatement and merges track context into a single direct sentence.
- 原文：We study multimodal Web answering, where a question may arrive with or without supporting context. This is Web question answering with multimodal RAG (Search, Recommendation, and Retrieval-Augmented AI track): Web pages mix formats, separate engines retrieve them with incomparable scores, and answers must cite the pages used.
- Gemini：We study multimodal Web question answering (Search, Recommendation, and Retrieval-Augmented AI track), where a question may arrive with or without supporting context: Web pages mix formats, separate engines retrieve them with incomparable scores, and answers must cite the pages used.
- 最终：We study multimodal Web answering, a form of Web question answering with multimodal RAG (Search, Recommendation, and Retrieval-Augmented AI track): a question may arrive with or without supporting context, Web pages mix formats, and separate engines retrieve them with incomparable scores.

## G01-01（Introduction，采纳）
- Gemini 的理由：Replaces the ambiguous pronoun 'it' with 'accuracy' to specify what metric drops.
- 原文：Even a shared controller is affected: with raw instead of per-modality normalized retriever scores, it drops from 67.88 to 64.25 on OK-VQA (Section~\ref{sec:ablation}).
- Gemini：Even a shared controller is affected: with raw instead of per-modality normalized retriever scores, accuracy drops from 67.88 to 64.25 on OK-VQA (Section~\ref{sec:ablation}).

## G01-02（Introduction，采纳）
- Gemini 的理由：Replaces the synonym 'Repeating chunks' with the locked term 'Chunk replication'.
- 原文：Repeating chunks lowers the deviation only to 2.03 and spends input budget on duplicates.
- Gemini：Chunk replication lowers the deviation only to 2.03 and spends input budget on duplicates.

## G01-03（Introduction，改写后采纳）
- Gemini 的理由：Replaces vague phrasing with locked terminology ('triggers' and 'routes') and clarifies the routing target.
- 原文：For Challenge~1, an uncertainty score from predictive entropy and the dispersion of evidence attention decides whether to retrieve, and the attention each modality receives decides where.
- Gemini：For Challenge~1, an uncertainty score from predictive entropy and the dispersion of evidence attention triggers retrieval, and modality-level attention routes which modality to search.
- 最终：For Challenge~1, an uncertainty score from predictive entropy and the dispersion of evidence attention decides whether to retrieve, and the attention each modality receives decides which modality to search.

## G01-04（Introduction，采纳）
- Gemini 的理由：Merges repetitive consecutive sentences into a single concise description of EAB.
- 原文：For Challenge~2, Evidence Attention Bias (EAB) adds a bias to the attention logits of each retrieved chunk. The bias grows with the relevance of the chunk and with the uncertainty score, without duplicated input or parameter updates.
- Gemini：For Challenge~2, Evidence Attention Bias (EAB) adds a relevance-weighted bias to the attention logits of each retrieved chunk, scaled by the uncertainty score, without duplicated input or parameter updates.

## G02-00（Related Work，采纳）
- Gemini 的理由：Split a sentence that juggles three predicate clauses with multiple internal disjunctions.
- 原文：Multimodal RAG systems~\cite{abootorabi2025survey} extend retrieval to images, knowledge memories, or retrieval hierarchies~\cite{adjali2024multilevel,chen2022murag,hu2023reveal}, retrieve visual documents or generate multimodal answers~\cite{ma2025m2rag,wang2025vidorag,wu2025visualrag,yu2025visrag}, or refine selection by reranking, reflection, or adaptive visual selection~\cite{chen2025ragvl,zhai2025samrag,zhang2024mr2ag}.
- Gemini：Multimodal RAG systems~\cite{abootorabi2025survey} extend retrieval to images, knowledge memories, or retrieval hierarchies~\cite{adjali2024multilevel,chen2022murag,hu2023reveal}. Other systems retrieve visual documents or generate multimodal answers~\cite{ma2025m2rag,wang2025vidorag,wu2025visualrag,yu2025visrag}, or refine selection by reranking, reflection, or adaptive visual selection~\cite{chen2025ragvl,zhai2025samrag,zhang2024mr2ag}.

## G02-01（Related Work，采纳）
- Gemini 的理由：Replace the ambiguous pronoun it with an explicit noun phrase.
- 原文：MMA-RAG classifies hidden states of the input and retrieved visual context to decide whether to use it~\cite{du2026mmarag}, and UniversalRAG routes the query alone to modality- and granularity-specific corpora~\cite{yeo2026universalrag}.
- Gemini：MMA-RAG classifies hidden states of the input and retrieved visual context to decide whether to use that context~\cite{du2026mmarag}, and UniversalRAG routes the query alone to modality- and granularity-specific corpora~\cite{yeo2026universalrag}.

## G02-02（Related Work，改写后采纳）
- Gemini 的理由：Remove the banned set-up-and-knock-down contrast and express the distinction directly with locked terminology.
- 原文：These methods read generated text, the query, or pooled hidden states, not the attention to the evidence spans of each modality, which our controller reads within the same prefill and reuses to weight evidence.
- Gemini：These methods read generated text, the query, or pooled hidden states, whereas our controller reads evidence attention across modalities within the same prefill and reuses that attention to weight evidence.
- 最终：These methods read generated text, the query, or pooled hidden states; our controller reads the attention to the evidence spans of each modality within the same prefill and reuses it to weight evidence.

## G02-03（Related Work，采纳）
- Gemini 的理由：Enforce the locked term attention row and clarify whose logits are modified.
- 原文：PASTA emphasizes user-specified spans on profiled attention heads~\cite{zhang2024pasta}, which at a fixed row equals adding one constant to their logits, and PAI amplifies attention to all image tokens to reduce hallucination~\cite{liu2024pai}.
- Gemini：PASTA emphasizes user-specified spans on profiled attention heads~\cite{zhang2024pasta}, which at a fixed attention row equals adding a constant to span logits, and PAI amplifies attention to all image tokens to reduce hallucination~\cite{liu2024pai}.

## G02-04（Related Work，拒绝）
- Gemini 的理由：Replace the ambiguous pronoun it with the decoder.
- 原文：It gives each retrieved chunk its own bias, set by retriever relevance and the uncertainty score; it couples this weighting to the retrieval decision, which OpenDecoder and PASTA take as given; and it keeps the generator frozen, whereas OpenDecoder fine-tunes it.
- Gemini：It gives each retrieved chunk its own bias, set by retriever relevance and the uncertainty score; it couples this weighting to the retrieval decision, which OpenDecoder and PASTA take as given; and it keeps the generator frozen, whereas OpenDecoder fine-tunes the decoder.

## G03-00（Problem Definition，改写后采纳）
- Gemini 的理由：Group context statements before defining chunks and their retrieval to preserve old-to-new topic flow.
- 原文：Each chunk $d_j\in\mathcal{D}$ is the unit that can be packed into the input, and it keeps its source identifier and modality. The context may be empty, as in a text-only WebQA question. For a query,
- Gemini：The context $c$ may be empty, as in a text-only WebQA question. Each chunk $d_j\in\mathcal{D}$ is the unit packed into the input, retaining its source identifier and modality. For query $q$,
- 最终：The context $c$ may be empty, as in a text-only WebQA question. Each chunk $d_j\in\mathcal{D}$ is the unit that can be packed into the input and keeps its source identifier and modality. For a query,

## G03-01（Problem Definition，改写后采纳）
- Gemini 的理由：Define evidence span before introducing its mathematical notation, eliminating the nested appositive.
- 原文：We call $\mathrm{span}(d_j)$, the token positions that an encoded chunk occupies in $X$, its evidence span;
- Gemini：The token positions that an encoded chunk occupies in $X$ form its evidence span, $\mathrm{span}(d_j);
- 最终：The token positions that an encoded chunk occupies in $X$ form its evidence span $\mathrm{span}(d_j)$;

## G03-02（Problem Definition，采纳）
- Gemini 的理由：Replace the ambiguous pronoun 'it' with a direct reference to cited chunks.
- 原文：the chunks it cites
- Gemini：the cited chunks

## G04-00（Framework Overview，采纳）
- Gemini 的理由：Supplies a governing noun for the bare ordinal and sharpens the contrast between the decisions.
- 原文：The first two address Challenge~1 and EAB addresses Challenge~2.
- Gemini：The first two decisions address Challenge~1; EAB addresses Challenge~2.

## G04-01（Framework Overview，采纳）
- Gemini 的理由：Adds a comma between the two independent clauses to prevent a garden-path reading.
- 原文：Serialized tables keep header and row boundaries and video frames keep their temporal order.
- Gemini：Serialized tables keep header and row boundaries, and video frames keep their temporal order.

## G04-02（Framework Overview，采纳）
- Gemini 的理由：Replaces the non-native verb "draw" with "show" when referring to figures.
- 原文：Spans are not compressed; the figures draw them schematically.
- Gemini：Spans are not compressed; the figures show them schematically.

## G05-00（Uncertainty from Evidence and Prediction，改写后采纳）
- Gemini 的理由：Resolve ambiguous pronouns ('The two', 'Both') and replace imprecise phrasing ('observes two things', 'still spreads').
- 原文：To decide whether the current input suffices, the controller observes two things: how the generator distributes attention over the available evidence, and how uncertain its next prediction is. The two can disagree: attention can settle on one chunk while the next-token distribution still spreads. Both come from a zero-bias prefill, the forward pass over $X$ before generation.
- Gemini：To decide whether the current input suffices, the controller evaluates two signals: how the generator distributes attention over available evidence, and how uncertain its next prediction is. These signals can disagree: attention can settle on one chunk while the next-token distribution remains diffuse. Both signals come from a zero-bias prefill, the forward pass over $X$ before generation.
- 最终：To decide whether the current input suffices, the controller observes two signals: how the generator distributes attention over the available evidence, and how uncertain its next prediction is. These signals can disagree: attention can settle on one chunk while the next-token distribution remains diffuse. Both signals come from a zero-bias prefill, the forward pass over $X$ before generation.

## G05-01（Uncertainty from Evidence and Prediction，改写后采纳）
- Gemini 的理由：Use locked terminology ('attention row') and remove repetitive phrasing ('the row').
- 原文：We read row $\tend$ of the final layer $L$, the row whose output predicts the first answer token, average its $n_h$ heads, and normalize over the $N$ tokens in evidence spans:
- Gemini：We read attention row $\tend$ of the final layer $L$, whose output predicts the first answer token, average its $n_h$ heads, and normalize over the $N$ tokens in evidence spans:
- 最终：We read attention row $\tend$ of the final layer $L$, the row whose output predicts the first answer token, average its $n_h$ heads, and normalize over the $N$ tokens in evidence spans:

## G05-02（Uncertainty from Evidence and Prediction，采纳）
- Gemini 的理由：Specify the referent for 'A high value', replace colloquial 'settling on', and clarify that 'it' refers to the score rather than the bound.
- 原文：A high value means that attention is spread over many evidence tokens instead of settling on a few. Dividing by $\log N$ bounds $\hgrd$ in $[0,1]$, although span length and image-patch density can still affect it.
- Gemini：A high value of $\hgrd$ means that attention spreads across many evidence tokens instead of concentrating on a few. Dividing by $\log N$ bounds $\hgrd$ in $[0,1]$, though span length and image-patch density can still affect the score.

## G05-03（Uncertainty from Evidence and Prediction，拒绝）
- Gemini 的理由：Eliminate the repeated explanation of what parameter $\beta$ does across consecutive sentences.
- 原文：Here $\beta$ weights predictive entropy, IQR is the interquartile range, and $S$ is the set of pre-intervention scores of unlabeled calibration queries for a given benchmark and backbone. Because $\hpred$ is small after division by $\log|\mathcal{V}|$, $\beta$ sets its share of the sum; Norm maps the median of $S$ to 0.5 so that one threshold serves all benchmarks.
- Gemini：Here IQR is the interquartile range, and $S$ is the set of pre-intervention scores of unlabeled calibration queries for a given benchmark and backbone. Because $\hpred$ is small after division by $\log|\mathcal{V}|$, $\beta$ weights its share of the sum; Norm maps the median of $S$ to 0.5 so that one threshold serves all benchmarks.

## G05-04（Uncertainty from Evidence and Prediction，采纳）
- Gemini 的理由：Apply locked terminology ('triggers retrieval') and eliminate the bare pronoun 'It'.
- 原文：The controller retrieves if $\Uomni>\tau$ and otherwise answers directly. It keeps $\Uomni$ for the bias gate of Section~\ref{sec:eab}.
- Gemini：The controller triggers retrieval if $\Uomni>\tau$ and otherwise answers directly, retaining $\Uomni$ for the bias gate of Section~\ref{sec:eab}.

## G05-05（Uncertainty from Evidence and Prediction，采纳）
- Gemini 的理由：Clarify that $\beta>0$ scales the score and calibration statistics equally rather than the normalization function itself.
- 原文：The normalization scales the score, median, and IQR by the same factor $\beta>0$, so normalized predictive entropy alone triggers retrieval, and routing uses retriever-side scores.
- Gemini：Because $\beta>0$ scales the score, median, and IQR equally, normalized predictive entropy alone triggers retrieval, and routing uses retriever-side scores.

## G06-00（Modality Routing，改写后采纳）
- Gemini 的理由：Use locked terminology for trigger as a rule rather than an event.
- 原文：After a trigger, the controller selects the modalities to search.
- Gemini：When retrieval triggers, the controller selects the modalities to search.
- 最终：After retrieval is triggered, the controller selects the modalities to search.

## G06-01（Modality Routing，采纳）
- Gemini 的理由：Clarify that the attention distribution is concentrated rather than the modality itself.
- 原文：and a concentrated one has already located its evidence
- Gemini：and one with concentrated attention has already located its evidence

## G06-02（Modality Routing，采纳）
- Gemini 的理由：Fix faulty predication where modalities rather than retrieved chunks fill slots.
- 原文：both modalities are searched and fill the $k$ slots by \tbd{[[merge rule]]}.
- Gemini：both modalities are searched, and their chunks fill the $k$ slots by \tbd{[[merge rule]]}.

## G06-03（Modality Routing，采纳）
- Gemini 的理由：Eliminate awkward repetition of 'scored with ... scores' and improve concision.
- 原文：Modalities absent from the supplied context are scored with retriever-side scores, as at cold start;
- Gemini：Modalities absent from the supplied context receive retriever-side scores, as at cold start;

## G06-04（Modality Routing，采纳）
- Gemini 的理由：Make the phrasing more direct and concise.
- 原文：the cross-modal scores of the retrievers select the modality,
- Gemini：cross-modal retriever scores select the modality,

## G06-05（Modality Routing，采纳）
- Gemini 的理由：Replace informal 'stays' with precise academic phrasing.
- 原文：the supplied context stays,
- Gemini：the supplied context remains,

## G06-06（Modality Routing，改写后采纳）
- Gemini 的理由：Add the locked term 'uncertainty score' to the bare mathematical symbol.
- 原文：The initial $\Uomni$ is not recomputed.
- Gemini：The uncertainty score $\Uomni$ is not recomputed.
- 最终：The initial uncertainty score $\Uomni$ is not recomputed.

## G07-00（Evidence Attention Bias，采纳）
- Gemini 的理由：Removes the repeated word 'input', eliminates the bare pronoun 'It', and tightens two loosely coordinated sentences.
- 原文：EAB raises the influence of relevant chunks within the packed input. It takes the retrieved chunks $\{d_j\}_{j=1}^{k}$, their evidence spans, and the uncertainty score as input.
- Gemini：EAB raises the influence of relevant chunks, taking as input the retrieved chunks $\{d_j\}_{j=1}^{k}$, their evidence spans, and the uncertainty score.

## G07-01（Evidence Attention Bias，采纳）
- Gemini 的理由：Splits a three-clause colon sentence and uses 'triggered queries' consistently with the rest of the paper.
- 原文：\textit{Bias gate.} EAB is active on retrieved queries with $\Uomni>\taub$. The second threshold $\taub\ge\tau$ reserves the bias for the most uncertain queries: with $\tau=0.35$ and $\taub=0.40$, queries with $0.35<\Uomni\le0.40$ receive evidence without bias, and biasing every retrieved query lowers OK-VQA accuracy from 67.88 to 67.45 (Appendix~\ref{app:supp}).
- Gemini：\textit{Bias gate.} EAB is active on triggered queries with $\Uomni>\taub$. The second threshold $\taub\ge\tau$ reserves the bias for the most uncertain queries. With $\tau=0.35$ and $\taub=0.40$, queries with $0.35<\Uomni\le0.40$ receive evidence without bias, and biasing every triggered query lowers OK-VQA accuracy from 67.88 to 67.45 (Appendix~\ref{app:supp}).

## G07-02（Evidence Attention Bias，采纳）
- Gemini 的理由：Splits a run-on definition juggling three clauses to separate key definitions from gate scaling.
- 原文：Let $\mu_j=\sum_{i\in\mathrm{span}(d_j)}\alpha_i$ be the attention mass of chunk $j$, let the remaining keys form a group $j=0$ with $r_0=0$, and let $\lambda=\gamma\Uomni$ when the gate is active and $\lambda=0$ otherwise.
- Gemini：Let $\mu_j=\sum_{i\in\mathrm{span}(d_j)}\alpha_i$ be the attention mass of chunk $j$, and let the remaining keys form a group $j=0$ with $r_0=0$. Set $\lambda=\gamma\Uomni$ when the gate is active and $\lambda=0$ otherwise.

## G07-03（Evidence Attention Bias，采纳）
- Gemini 的理由：States the paragraph point in the lead sentence, eliminates a redundant summary sentence, and fixes awkward phrasing.
- 原文：Appendix~\ref{app:dca} gives the proof. EAB thus multiplies the odds between two chunks by at most $e^{1.2}\approx3.3$: it can overturn a positional preference of up to $3.3\times$ between a relevant and an irrelevant chunk and passes larger position effects through. The bias is therefore a bounded relevance prior.
- Gemini：As a bounded relevance prior, EAB multiplies the odds between two chunks by at most $e^{1.2}\approx3.3$ (proof in Appendix~\ref{app:dca}): it can overturn a positional preference of up to $3.3\times$ between a relevant and an irrelevant chunk, while larger position effects persist.

## G08-00（Experiments，采纳）
- Gemini 的理由：Capitalize the first word of each research question following its heading.
- 原文：\textbf{RQ1}: does \method{} improve answer quality on Web and other evidence formats? \textbf{RQ2}: how do the trigger, score normalization, modality routing, and EAB each contribute? \textbf{RQ3}: does EAB reduce order sensitivity, improve citation support, and make answers depend on the emphasized chunks? \textbf{RQ4}: how robust is the controller to retrieval errors, calibration, and the backbone, and what does it cost?
- Gemini：\textbf{RQ1}: Does \method{} improve answer quality on Web and other evidence formats? \textbf{RQ2}: How do the trigger, score normalization, modality routing, and EAB each contribute? \textbf{RQ3}: Does EAB reduce order sensitivity, improve citation support, and make answers depend on the emphasized chunks? \textbf{RQ4}: How robust is the controller to retrieval errors, calibration, and the backbone, and what does it cost?

## G09-00（Experimental Setup，采纳）
- Gemini 的理由：Cut redundant statement since a subset is inherently contained in the full set.
- 原文：(\tbd{[[selection rule]]}); the two WebQA sets overlap.
- Gemini：(\tbd{[[selection rule]]}).

## G09-01（Experimental Setup，采纳）
- Gemini 的理由：Use semicolons and gapping commas to clarify the parallel list of benchmarks and evidence types.
- 原文：OK-VQA~\cite{marino2019okvqa} tests image-based knowledge, the table subset of MultiModalQA~\cite{talmor2021mmqa} (MMQA-table) structured evidence, its full validation set (MMQA-full) routing among text, tables, and images, and Video-MME-Short~\cite{fu2025videomme} sampled short-video evidence.
- Gemini：OK-VQA~\cite{marino2019okvqa} tests image-based knowledge; the table subset of MultiModalQA~\cite{talmor2021mmqa} (MMQA-table), structured evidence; its full validation set (MMQA-full), routing among text, tables, and images; and Video-MME-Short~\cite{fu2025videomme}, sampled short-video evidence.

## G09-02（Experimental Setup，采纳）
- Gemini 的理由：Add comma after introductory adverbial phrase.
- 原文：On average a text chunk occupies
- Gemini：On average, a text chunk occupies

## G09-03（Experimental Setup，采纳）
- Gemini 的理由：Fix subject-verb agreement for the singular subject 'calibrated relevance'.
- 原文：the calibrated relevance $r_j$ of the first and the fifth chunk differ by \ph{} on average.
- Gemini：the calibrated relevance $r_j$ of the first and the fifth chunk differs by \ph{} on average.

## G09-04（Experimental Setup，采纳）
- Gemini 的理由：Eliminate repetitive use of 'apply' and 'applied' to tighten the sentence.
- 原文：apply 12 orderings of the five retrieval ranks, each applied identically to all questions, over three seeds.
- Gemini：apply 12 orderings of the five retrieval ranks identically to all questions, over three seeds.

## G09-05（Experimental Setup，采纳）
- Gemini 的理由：Add governing noun to clarify pronoun reference.
- 原文：; both are checked by alias matching
- Gemini：; both metrics are checked by alias matching

## G09-06（Experimental Setup，采纳）
- Gemini 的理由：Supply missing verb 'is' in coordinate clause.
- 原文：, and the coverage rate (Cov.) the share whose selected modalities contain all of it.
- Gemini：, and the coverage rate (Cov.) is the share whose selected modalities contain all of it.

## G09-07（Experimental Setup，采纳）
- Gemini 的理由：Insert auxiliary verb 'are' for grammatical parallelism.
- 原文：tables are serialized as text and videos given as sampled frames.
- Gemini：tables are serialized as text and videos are given as sampled frames.

## G09-08（Experimental Setup，改写后采纳）
- Gemini 的理由：Make the list item grammatical with 'a limit of' and split the independent clause with a semicolon.
- 原文：share the candidate pools, decoding limits, at most $k=5$ retrieved chunks, an input limit of \tbd{[[tokens; per-modality quota]]}, truncation rules, and answer-normalization scripts, and baselines receive \tbd{[[tuning budget]]}.
- Gemini：share the candidate pools, decoding limits, a limit of at most $k=5$ retrieved chunks, an input limit of \tbd{[[tokens; per-modality quota]]}, truncation rules, and answer-normalization scripts; baselines receive \tbd{[[tuning budget]]}.
- 最终：share the candidate pools, decoding limits, at most $k=5$ retrieved chunks, an input limit of \tbd{[[tokens; per-modality quota]]}, truncation rules, and answer-normalization scripts; baselines receive \tbd{[[tuning budget]]}.

## G09-09（Experimental Setup，采纳）
- Gemini 的理由：Replace redundant 'the same chunks' with 'those chunks'.
- 原文：whereas chunk replication duplicates the same chunks before truncation under the same input budget.
- Gemini：whereas chunk replication duplicates those chunks before truncation under the same input budget.

## G09-10（Experimental Setup，采纳）
- Gemini 的理由：Clarify preposition for equation inputs and replace pronoun 'they' with 'these queries'.
- 原文：estimate the median and IQR of Eq.~\eqref{eq:norm} and the ECDFs of the retriever scores; they come from
- Gemini：estimate the median and IQR for Eq.~\eqref{eq:norm} and the ECDFs of the retriever scores; these queries come from

## G10-00（Overall Performance (RQ1)，采纳）
- Gemini 的理由：Replaces 'operating threshold' with the locked term 'retrieval decision' to match terminology across the paper.
- 原文：the operating threshold accounts for \ph{} points and EAB for \ph{}
- Gemini：the retrieval decision accounts for \ph{} points and EAB for \ph{}

## G10-01（Overall Performance (RQ1)，改写后采纳）
- Gemini 的理由：Splits a compound sentence to separate accuracy gains from order-gap measurements and provides a noun for the bare pronoun 'this'.
- 原文：Over static concatenation in the same pipeline, the gain is 5.73\%, 7.98\%, 12.68\%, and 18.25\% on OK-VQA, \webimg{}, MMQA-table, and Video-MME-Short, and the sorted--reverse gap of static concatenation is 8.72, 9.35, 10.45, and 11.20 points on the same settings (Appendix~\ref{app:supp}). With four settings this is only an association.
- Gemini：Over static concatenation in the same pipeline, the gain is 5.73\%, 7.98\%, 12.68\%, and 18.25\% on OK-VQA, \webimg{}, MMQA-table, and Video-MME-Short. On the same settings, the sorted--reverse gap of static concatenation is 8.72, 9.35, 10.45, and 11.20 points (Appendix~\ref{app:supp}), though with four settings this correlation is only an association.
- 最终：Over static concatenation in the same pipeline, the gain is 5.73\%, 7.98\%, 12.68\%, and 18.25\% on OK-VQA, \webimg{}, MMQA-table, and Video-MME-Short. On the same settings, the sorted--reverse gap of static concatenation grows from 8.72 to 9.35, 10.45, and 11.20 points (Appendix~\ref{app:supp}), an association observed over only four settings.

## G11-00（Ablation Study (RQ2)，采纳）
- Gemini 的理由：Eliminate redundant repetition of "answer" and streamline the clause.
- 原文：the questions that NTE answers without retrieval are answered less accurately
- Gemini：questions answered without retrieval under NTE score lower

## G11-01（Ablation Study (RQ2)，采纳）
- Gemini 的理由：Clarify that NTE skips retrieval rather than skipping the questions themselves.
- 原文：Without the grounding term, NTE skips questions whose evidence attention is still diffuse.
- Gemini：Without the grounding term, NTE skips retrieval on questions whose evidence attention is still diffuse.

## G11-02（Ablation Study (RQ2)，采纳）
- Gemini 的理由：Use the locked term "chunk replication" and remove the redundant preposition.
- 原文：Replication emphasizes the same chunks as EAB, but only with integer copy counts and with duplicated tokens that can displace other evidence after truncation.
- Gemini：Chunk replication emphasizes the same chunks as EAB, but only with integer copy counts and duplicated tokens that can displace other evidence after truncation.

## G11-03（Ablation Study (RQ2)，采纳）
- Gemini 的理由：Supply the missing verb and split the relevance and gating ablations into distinct sentences.
- 原文：Uniform weights give 65.85 on OK-VQA and random relevance 65.15, so the gain requires chunk-specific relevance; a fixed bias $\gamma r_j$ that ignores $\Uomni$, close to a frozen OpenDecoder-style prior, gives \ph.
- Gemini：Uniform weights give 65.85 on OK-VQA and random relevance gives 65.15, so the gain requires chunk-specific relevance. A fixed bias $\gamma r_j$ that ignores $\Uomni$, close to a frozen OpenDecoder-style prior, gives \ph.

## G12-00（Evidence Order and Citation Support，采纳）
- Gemini 的理由：States the paragraph's core finding in the topic sentence and corrects the phrasing attributing permutation to the table.
- 原文：Table~\ref{tab:order} permutes a fixed set of retrieved chunks.
- Gemini：When a fixed set of retrieved chunks is permuted, EAB reduces order sensitivity (Table~\ref{tab:order}).

## G12-01（Evidence Order and Citation Support，采纳）
- Gemini 的理由：Replaces 'ordering' to avoid ambiguity with evidence orderings.
- 原文：The other three settings show the same ordering.
- Gemini：The other three settings show the same pattern.

## G12-02（Evidence Order and Citation Support，采纳）
- Gemini 的理由：Clarifies that orderings, rather than EAB, induce changes in correctness across permutations.
- 原文：At the question level, EAB changes the correctness of \ph\% of OK-VQA questions across orderings, against \ph\% for static concatenation.
- Gemini：At the question level, \ph\% of OK-VQA questions change correctness across orderings under EAB, against \ph\% for static concatenation.

## G12-03（Evidence Order and Citation Support，改写后采纳）
- Gemini 的理由：Replaces unidiomatic phrasing 'with the same direction on' with 'with the same trend on'.
- 原文：On OK-VQA, attribution precision rises from 38.53 to 72.45 and invalid citations fall from 12.4\% to 3.1\%, with the same direction on tables and WebQA.
- Gemini：On OK-VQA, attribution precision rises from 38.53 to 72.45 and invalid citations fall from 12.4\% to 3.1\%, with the same trend on tables and WebQA.
- 最终：On OK-VQA, attribution precision rises from 38.53 to 72.45 and invalid citations fall from 12.4\% to 3.1\%, and the same holds on tables and WebQA.

## G12-04（Evidence Order and Citation Support，改写后采纳）
- Gemini 的理由：Cuts repeated occurrences of 'chunks' and tightens the causal explanation.
- 原文：EAB raises the attention mass of high-relevance chunks, and the generator tends to cite the chunks it attends to, so more citations point to chunks that support the answer.
- Gemini：EAB raises the attention mass of high-relevance chunks, and because the generator tends to cite attended chunks, more citations support the answer.
- 最终：EAB raises the attention mass of high-relevance chunks, and because the generator tends to cite the chunks it attends to, more citations point to chunks that support the answer.

## G12-05（Evidence Order and Citation Support，改写后采纳）
- Gemini 的理由：Replaces 'preserve this ordering' with 'confirm this advantage' to avoid confusing attribution gains with chunk orderings.
- 原文：A 150-query human audit (41.20 against 75.10) and two other NLI checkers preserve this ordering (Appendix~\ref{app:supp}).
- Gemini：A 150-query human audit (41.20 against 75.10) and two other NLI checkers confirm this advantage (Appendix~\ref{app:supp}).
- 最终：A 150-query human audit (41.20 against 75.10) and two other NLI checkers show the same advantage (Appendix~\ref{app:supp}).

## G13-00（Dependence on the Emphasized Chunks，采纳）
- Gemini 的理由：Remove the redundant appendix citation that is repeated in the following sentence.
- 原文：(Table~\ref{tab:causal}; construction in Appendix~\ref{app:protocol})
- Gemini：(Table~\ref{tab:causal})

## G13-01（Dependence on the Emphasized Chunks，改写后采纳）
- Gemini 的理由：Replace the ambiguous pronoun with its noun referent and format the reference parenthetically.
- 原文：while keeping it; Appendix~\ref{app:protocol} gives the controls.
- Gemini：while keeping the target chunk (controls in Appendix~\ref{app:protocol}).
- 最终：while keeping the target chunk (construction and controls in Appendix~\ref{app:protocol}).

## G14-00（Robustness to Retrieval Errors，采纳）
- Gemini 的理由：Replace the ambiguous pronoun 'it' with the explicit subject 'EAB'.
- 原文：With Gaussian noise ($\sigma=0.1$, $0.2$) on the relevance, it scores \ph{} and \ph.
- Gemini：With Gaussian noise ($\sigma=0.1$, $0.2$) on the relevance, EAB scores \ph{} and \ph.

## G14-01（Robustness to Retrieval Errors，改写后采纳）
- Gemini 的理由：Split a four-clause sentence to separate the theoretical odds bound from its empirical limit.
- 原文：Low-$\Uomni$ queries receive no bias, and Proposition~\ref{thm:tilt} bounds the change in attention odds at one row by $e^{1.2}$, but this bound does not limit how often answers are misled, which the Misled column measures.
- Gemini：Low-$\Uomni$ queries receive no bias, and Proposition~\ref{thm:tilt} bounds the change in attention odds at one row by $e^{1.2}$. That bound does not limit how often answers are misled, as reported in the Misled column.
- 最终：Low-$\Uomni$ queries receive no bias, and Proposition~\ref{thm:tilt} bounds the change in attention odds at one row by $e^{1.2}$. This bound does not limit how often answers are misled, which the Misled column measures.

## G15-00（Calibration, Transfer, and Cost，采纳）
- Gemini 的理由：Add a comma after the introductory prepositional phrase to prevent a garden-path reading.
- 原文：With 100 unlabeled calibration queries OK-VQA reaches 66.85, and with 500, 67.88.
- Gemini：With 100 unlabeled calibration queries, OK-VQA reaches 66.85, and with 500, 67.88.

## G15-01（Calibration, Transfer, and Cost，采纳）
- Gemini 的理由：Split the four-clause sentence to state the rationale before the latency metric and eliminate repeated need.
- 原文：Entropy and routing take $O(N)$ operations for $N$ evidence tokens, but triggered queries also need a second prefill and EAB needs eager or SDPA attention, so we report end-to-end latency.
- Gemini：Entropy and routing take $O(N)$ operations for $N$ evidence tokens. Because triggered queries need a second prefill and EAB requires eager or SDPA attention, we report end-to-end latency.

## G15-02（Calibration, Transfer, and Cost，采纳）
- Gemini 的理由：Split the compound sentence at the semicolon into two focused statements and cut an unnecessary article.
- 原文：On OK-VQA, skipping retrieval for 21.5\% of queries lowers the average latency from 430\,ms for always-retrieve static concatenation to 382\,ms; uncertainty computation adds 15\,ms, against 250 of 639\,ms for the corresponding step of SAM-RAG.
- Gemini：On OK-VQA, skipping retrieval for 21.5\% of queries lowers average latency from 430\,ms for always-retrieve static concatenation to 382\,ms. Uncertainty computation adds 15\,ms, against 250 of 639\,ms for the corresponding step of SAM-RAG.

## G16-00（Error and Case Analysis，采纳）
- Gemini 的理由：Ensure parallel phrasing for categorized errors across OK-VQA and WebQA.
- 原文：Missing top-$k$ evidence causes 42.5\% of categorized OK-VQA errors and 38.2\% on WebQA (Appendix~\ref{app:error}).
- Gemini：Missing top-$k$ evidence causes 42.5\% of categorized errors on OK-VQA and 38.2\% on WebQA (Appendix~\ref{app:error}).

## G16-01（Error and Case Analysis，采纳）
- Gemini 的理由：Remove a wordy relative clause to streamline the sentence.
- 原文：A reranker can recover gold evidence that lies in the candidate pool but outside the top $k$ (recall \ph\% against \ph\% at $k=5$), \ai{consistent with the gain of adding it to the controller (Table~\ref{tab:main}).}
- Gemini：A reranker can recover gold evidence in the candidate pool but outside the top $k$ (recall \ph\% against \ph\% at $k=5$), \ai{consistent with the gain of adding it to the controller (Table~\ref{tab:main}).}

## G17-00（Conclusion，采纳）
- Gemini 的理由：Use the present tense to describe what the paper presents.
- 原文：We presented \method{}, a training-free latent-space controller for multimodal Web answering: an uncertainty score on evidence spans decides whether to retrieve, modality-level attention decides where, and EAB reuses the score as a bounded relevance prior.
- Gemini：We present \method{}, a training-free latent-space controller for multimodal Web answering: an uncertainty score on evidence spans decides whether to retrieve, modality-level attention decides where, and EAB reuses the score as a bounded relevance prior.

## G19-00（Evaluation Details，采纳）
- Gemini 的理由：Remove ambiguous pronouns and state the purpose of the retrieval setup directly.
- 原文：\webimg{} contains the text-only validation questions whose supporting facts include an image; it is retrieved from the full text-and-image pool, so it tests routing from text queries to visual evidence.
- Gemini：\webimg{} contains the text-only validation questions whose supporting facts include an image; retrieving evidence from the full text-and-image pool tests routing from text queries to visual evidence.

## G19-01（Evaluation Details，采纳）
- Gemini 的理由：Split unrelated implementation details into separate sentences.
- 原文：EAB is an additive mask of shape $[\mathrm{batch},1,q_{\mathrm{len}},kv_{\mathrm{len}}]$ that broadcasts across heads and grows along the key dimension as positions enter the key--value (KV) cache, and $\epsilon=10^{-9}$ in all attention and entropy computations.
- Gemini：EAB is an additive mask of shape $[\mathrm{batch},1,q_{\mathrm{len}},kv_{\mathrm{len}}]$ that broadcasts across heads and grows along the key dimension as positions enter the key--value (KV) cache. Attention and entropy computations use $\epsilon=10^{-9}$.

## G19-02（Evaluation Details，采纳）
- Gemini 的理由：Use locked terminology 'question' instead of 'example' for a dataset item.
- 原文：For answer $y$ to example $x$, cited chunks $C(x)$, and a binary support checker $\mathcal{S}$, let $\mathrm{Supp}(x)=\sum_{e\in C(x)}\mathcal{S}(y,e)$.
- Gemini：For answer $y$ to question $x$, cited chunks $C(x)$, and a binary support checker $\mathcal{S}$, let $\mathrm{Supp}(x)=\sum_{e\in C(x)}\mathcal{S}(y,e)$.

## G19-03（Evaluation Details，采纳）
- Gemini 的理由：Replace the pronoun 'it' with 'this indicator' and tighten phrasing.
- 原文：The flip indicator of question $x$ is $1$ if the correctness of its answer differs between any two orderings, and the flip rate averages it over questions.
- Gemini：The flip indicator of question $x$ is $1$ if answer correctness differs between any two orderings, and the flip rate averages this indicator over questions.

## G19-04（Evaluation Details，采纳）
- Gemini 的理由：Use locked terminology 'question' instead of 'query' for dataset items.
- 原文：Gold@2--5 contains \webimg{} queries whose top-ranked chunk is not a gold fact while a gold fact appears at ranks 2--5.
- Gemini：Gold@2--5 contains \webimg{} questions whose top-ranked chunk is not a gold fact while a gold fact appears at ranks 2--5.

## G20-00（Proofs for Evidence Attention Bias，改写后采纳）
- Gemini 的理由：Use the locked term evidence span and eliminate the repetitive phrasing 'span span'.
- 原文：Chunk $j\in\{1,\dots,k\}$ occupies span $\mathrm{span}(d_j)$;
- Gemini：Chunk $j\in\{1,\dots,k\}$ occupies evidence span $\mathrm{span}(d_j);
- 最终：Chunk $j\in\{1,\dots,k\}$ occupies evidence span $\mathrm{span}(d_j)$;

## G20-01（Proofs for Evidence Attention Bias，改写后采纳）
- Gemini 的理由：State why before what, cut filler words, and split a three-clause run-on sentence.
- 原文：The factor $e^{\lambda r_j}$ is shared by all keys of the span, so proportions within a span are preserved, and summing over the span gives Eq.~\eqref{eq:local} with $Z(\lambda)=\sum_l\mu_le^{\lambda r_l}$.
- Gemini：Because all keys in the span share $e^{\lambda r_j}$, proportions within the span are preserved; summing over the span gives Eq.~\eqref{eq:local} with $Z(\lambda)=\sum_l\mu_le^{\lambda r_l}.
- 最终：Because all keys in the span share $e^{\lambda r_j}$, proportions within the span are preserved; summing over the span gives Eq.~\eqref{eq:local} with $Z(\lambda)=\sum_l\mu_le^{\lambda r_l}$.

## G20-02（Proofs for Evidence Attention Bias，采纳）
- Gemini 的理由：Use the locked term chunk replication.
- 原文：Replication (superscript $\mathrm{rep}$)
- Gemini：Chunk replication (superscript $\mathrm{rep}$)

## G20-03（Proofs for Evidence Attention Bias，改写后采纳）
- Gemini 的理由：Add a comma after the introductory phrase to prevent reading 'copies' as a verb.
- 原文：with $c_j(a)>0$ visible copies the effective bias is $\log c_j(a)$.
- Gemini：with $c_j(a)>0$ visible copies, the effective bias is $\log c_j(a).
- 最终：with $c_j(a)>0$ visible copies, the effective bias is $\log c_j(a)$.

## G20-04（Proofs for Evidence Attention Bias，采纳）
- Gemini 的理由：Use the locked term chunk replication.
- 原文：between replication and EAB
- Gemini：between chunk replication and EAB

## G21-00（Supplementary Results，采纳）
- Gemini 的理由：Eliminates repetitive coordination and contrasts the drop in accuracy with the rise in retrieval rates.
- 原文：, and raises their retrieval rates from 78.5\%, 77.8\%, and 78.5\% to 83.2\%, 81.6\%, and 80.4\%.
- Gemini：, while raising retrieval rates from 78.5\%, 77.8\%, and 78.5\% to 83.2\%, 81.6\%, and 80.4\%.

## G21-01（Supplementary Results，采纳）
- Gemini 的理由：Splits a three-clause run-on sentence into two balanced sentences.
- 原文：on \webimg{}; attribution precision rises from 31.40 to 63.85 on MMQA-table (OK-VQA and \webimg{} in Table~\ref{tab:uomni_ablation}); and invalid citations fall from
- Gemini：on \webimg{}. Attribution precision rises from 31.40 to 63.85 on MMQA-table (OK-VQA and \webimg{} in Table~\ref{tab:uomni_ablation}), and invalid citations fall from

## G21-02（Supplementary Results，采纳）
- Gemini 的理由：Cuts redundant phrasing and splits a long compound sentence.
- 原文：Replacing DeBERTa-v3-large with FLAN-T5-xxl or Llama-3-8B-NLI as the support checker gives attribution precision of 37.85 and 39.20 for static concatenation against 71.60 and 73.15 for EAB on OK-VQA, and 30.15 and 32.55 against 62.45 and 64.70 on MMQA-table; invalid citations stay lower for EAB under every checker
- Gemini：Replacing DeBERTa-v3-large with FLAN-T5-xxl or Llama-3-8B-NLI gives attribution precision of 37.85 and 39.20 for static concatenation against 71.60 and 73.15 for EAB on OK-VQA, and 30.15 and 32.55 against 62.45 and 64.70 on MMQA-table. Invalid citations stay lower for EAB under every checker

## G21-03（Supplementary Results，改写后采纳）
- Gemini 的理由：Resolves a dangling participial modifier and replaces loose coordination with a semicolon.
- 原文：Varying $\tau$ on OK-VQA with chunk replication as the evidence-use operation, lowering retrieval from 100\% to 78.53\% raises accuracy to 67.14 and attribution precision to 69.20, and accuracy stays near 67 down to about 50\% retrieval.
- Gemini：Varying $\tau$ on OK-VQA with chunk replication as the evidence-use operation to lower retrieval from 100\% to 78.53\% raises accuracy to 67.14 and attribution precision to 69.20; accuracy stays near 67 down to about 50\% retrieval.
- 最终：With chunk replication as the evidence-use operation on OK-VQA, varying $\tau$ to lower retrieval from 100\% to 78.53\% raises accuracy to 67.14 and attribution precision to 69.20; accuracy stays near 67 down to about 50\% retrieval.

## G21-04（Supplementary Results，改写后采纳）
- Gemini 的理由：Clarifies that the controller improves accuracy over the baseline rather than raising the baseline itself.
- 原文：the controller raises static concatenation on OK-VQA, MMQA-table, and Video-MME-Short from 66.50, 50.80, and 53.45 to 71.20, 58.15, and 63.40 with InternVL2-40B, and from 62.45, 46.20, and 48.90 to 66.10, 52.35, and 56.45 with Llama-3.2-Vision-11B.
- Gemini：the controller raises accuracy over static concatenation on OK-VQA, MMQA-table, and Video-MME-Short from 66.50, 50.80, and 53.45 to 71.20, 58.15, and 63.40 with InternVL2-40B, and from 62.45, 46.20, and 48.90 to 66.10, 52.35, and 56.45 with Llama-3.2-Vision-11B.
- 最终：the controller raises OK-VQA, MMQA-table, and Video-MME-Short accuracy from 66.50, 50.80, and 53.45 under static concatenation to 71.20, 58.15, and 63.40 with InternVL2-40B, and from 62.45, 46.20, and 48.90 to 66.10, 52.35, and 56.45 with Llama-3.2-Vision-11B.

## G21-05（Supplementary Results，采纳）
- Gemini 的理由：Replaces loose coordination with a contrastive conjunction to distinguish unretrieved and retrieved subsets.
- 原文：$\Uomni$, and retrieved questions reach
- Gemini：$\Uomni$, while retrieved questions reach

## G21-06（Supplementary Results，拒绝）
- Gemini 的理由：Removes colloquial phrasing ("of a run").
- 原文：reports all arms of a run on the 5{,}046 OK-VQA validation questions
- Gemini：reports all arms on the 5{,}046 OK-VQA validation questions

## G21-07（Supplementary Results，采纳）
- Gemini 的理由：Contrasts the increasing accuracy with the decreasing share of unsupported claims.
- 原文：at $\gamma=1.2$, and the share of answers with unsupported claims is 18.5\%, 14.2\%, and 10.5\%.
- Gemini：at $\gamma=1.2$, while the share of answers with unsupported claims is 18.5\%, 14.2\%, and 10.5\%.

## G21-08（Supplementary Results，采纳）
- Gemini 的理由：Resolves a category error and aligns terminology with the question-as-dataset-item convention.
- 原文：single-token outputs are 5 of 32 development questions (15.6\%) under no bias, decode-only EAB, and EAB-P, and 51 and 50 of 256 held-out answers in the two text arms of Table~\ref{tab:headroom}.
- Gemini：single-token outputs occur on 5 of 32 development questions (15.6\%) under no bias, decode-only EAB, and EAB-P, and on 51 and 50 of 256 held-out questions in the two text arms of Table~\ref{tab:headroom}.

## G22-00（Error and Case Analysis，采纳）
- Gemini 的理由：Replaces the informal operator phrasing with standard mathematical wording and aligns the singular subject with the following sentence while eliminating redundant words.
- 原文：Scores use lexical accuracy times normalized BARTScore, a public approximation of the WebQA metric that is not on the scale of Table~\ref{tab:main}.
- Gemini：The score is lexical accuracy multiplied by normalized BARTScore, a public approximation of the WebQA metric not on the scale of Table~\ref{tab:main}.

## G22-01（Error and Case Analysis，采纳）
- Gemini 的理由：Uses standard machine learning terminology by adding 'size'.
- 原文：These pipeline runs use one Apple M4 Pro GPU (48\,GB, MPS) with generation batch 1.
- Gemini：These pipeline runs use one Apple M4 Pro GPU (48\,GB, MPS) with generation batch size 1.

## G22-02（Error and Case Analysis，采纳）
- Gemini 的理由：Corrects faulty agency where a question set was described as comparing arms, matching the active construction of P080.
- 原文：A second set of 256 WebQA validation questions with image evidence, drawn in advance from the official non-text pool and disjoint from \webimg{} and from all development and calibration data, compares three arms on the same cross-encoder candidates: raw scores, ECDF percentiles of the scores, and percentile selection with EAB (\tbd{[[TBD: decode-only EAB or EAB-P]]}).
- Gemini：On a second set of 256 WebQA validation questions with image evidence, drawn in advance from the official non-text pool and disjoint from \webimg{} and all development and calibration data, we compare three arms on the same cross-encoder candidates: raw scores, ECDF percentiles of the scores, and percentile selection with EAB (\tbd{[[TBD: decode-only EAB or EAB-P]]}).

## G23-00（Limitations and Ethical Considerations，改写后采纳）
- Gemini 的理由：Splits a three-clause comma splice into two sentences, fixes the awkward head phrasing, and resolves the ambiguous pronoun.
- 原文：Grounding entropy ignores how much attention goes to evidence overall, averaging heads can mix heads that attend to different chunks, and it depends on tokenization and patch density.
- Gemini：Grounding entropy ignores total evidence attention, and averaging across heads can mix heads that attend to different chunks. This score also depends on tokenization and patch density.
- 最终：Grounding entropy ignores total evidence attention, and averaging across heads can mix heads that attend to different chunks. Grounding entropy also depends on tokenization and patch density.

## G23-01（Limitations and Ethical Considerations，采纳）
- Gemini 的理由：Separates two unrelated limitations conjoined by a coordinating conjunction into distinct sentences.
- 原文：Normalization statistics are estimated per benchmark and backbone, and the videos are short clips, so long videos with temporal reasoning remain outside the evaluation.
- Gemini：Normalization statistics are estimated per benchmark and backbone. The evaluated videos are short clips, leaving long videos with temporal reasoning outside the evaluation.
