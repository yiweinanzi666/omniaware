import sys, json, traceback
sys.path.insert(0, r"C:\Users\22688\Desktop\www_code\omniaware\src")
from omniaware.baselines import universalrag as ur

# 1. UniversalRAG route fallback: label available only in another corpus
r = ur.to_route(["table"], {"text", "image"})
print("1 to_route(['table'], {text,image}) ->", r)
r2 = ur.to_route(["video"], {"text", "image"})
print("1b to_route(['video'], {text,image}) ->", r2)

# 2. select_chunks with that route: does it filter or fall back to joint?
try:
    import omniaware.run as run
    e = {"qid": "q1", "fixed_evidence": False, "candidates": [
        {"cid": "t1", "modality": "text", "title": "", "text": "a"},
        {"cid": "t2", "modality": "text", "title": "", "text": "b"},
        {"cid": "i1", "modality": "image", "title": "", "text": "c"}]}
    rel = {"t1": .9, "t2": .8, "i1": .3}
    print("2a route=None top-2 ->", [c["cid"] for c in run.select_chunks(e, rel, 2, "rank", 0, 0, None)])
    print("2b route modalities=['image'] top-2 ->", [c["cid"] for c in run.select_chunks(e, rel, 2, "rank", 0, 0, {"modalities": ["image"]})])
    print("2c UniversalRAG route for ['video'] (modalities empty) top-2 ->",
          [c["cid"] for c in run.select_chunks(e, rel, 2, "rank", 0, 0, r2)])
except Exception:
    traceback.print_exc()

# 3. Official M2RAG summarizer: what reaches the model for an OK-VQA style query with a question image
try:
    from omniaware.baselines.m2rag_official import build_official_summarizer
    from omniaware.baselines.m2rag_webqa import official_query, prepare_sources
    class TokW:
        def encode(self, t, add_special_tokens=False): return t.split()
        def decode(self, toks): return " ".join(map(str, toks))
    class Rec:
        name = "vlm-rec"
        def __init__(self): self.calls = []
        def chat(self, messages):
            self.calls.append(messages)
            return {"response": "summary text", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        def num_tokens_from_text(self, text): return len(text.split())
        def trim_text(self, text, max_tokens, side="right"): return " ".join(text.split()[:max_tokens])
    src = [dict(source_id="k1", fact_id="k1", modality="text", text="Passage about dogs.", r_j=.9)]
    prepared = prepare_sources(TokW(), src, {}, {"k1": .9})
    q = official_query("qid1", "What breed is the dog in the image?", prepared)
    print("3 official_query aux_images:", q["aux_images"])
    b = Rec()
    s = build_official_summarizer(b, multi_stage=False)
    out = s.summarize(query_list=[q], dry_run=False)
    print("3 summarizer result keys ok:", "placeholder_response" in out, "output_images" in out)
    msgs = b.calls[0]
    kinds = [(m.get("type"), (m.get("image_url") or {}).get("url") if m.get("type") == "image_url" else None) for m in msgs[0]["content"]] if isinstance(msgs[0], dict) and "content" in msgs[0] else [(m.get("type")) for m in msgs]
    print("3 message part types sent to backend:", kinds)
except Exception:
    traceback.print_exc()
