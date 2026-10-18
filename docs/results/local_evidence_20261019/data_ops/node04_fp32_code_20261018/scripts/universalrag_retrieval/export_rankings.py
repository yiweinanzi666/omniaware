"""Export UniversalRAG's own per-modality retrieval as candidate rankings (shared pools for every arm).

Run from the UniversalRAG checkout, inside its environment, after its script/0_dataset.sh and
script/1_preprocess.sh (features under eval/features/):

    cd external/UniversalRAG
    python ../../scripts/universalrag_retrieval/export_rankings.py --target hybridqa --top 10 \
        --output $OMNIAWARE_ASSETS/universalrag/rankings

Pass --modalities explicitly: HybridQA "paragraph table", VideoRAG-Synth "video clip" (the modalities that
data.load_universalrag reads by default). Writes <output>/<target>/<modality>.jsonl with
{"qid", "candidates": [{"source_id", "score"}]}, using the
retriever classes and feature files exactly as eval/eval.py init_inference_resources configures them
(alpha 0.2 for video/clip script features). Omni-Aware recomputes its own BGE/CLIP relevance on these
candidates; UniversalRAG's router decides which modality lists its baseline arm reads.
"""
import argparse
import json
from pathlib import Path
import sys

MODALITIES = ("paragraph", "document", "table", "image", "clip", "video")


# Subtitle ("script") features blended in with weight alpha.  script/1_preprocess.sh writes them as
# <name>_clipscript.pkl and <name>_vidscript.pkl (preprocess/extract_{clip,video}_feats.py), while eval/eval.py
# reads clip_script.pkl and video_script.pkl; either spelling is accepted.
SCRIPT_FEATURES = {"clip": ("clip_script.pkl", "clip_clipscript.pkl"),
                   "video": ("video_script.pkl", "video_vidscript.pkl")}


def script_features(kind: str, alpha: float) -> list[str]:
    """Path list for the script features of ``kind``; empty when alpha is 0 (the retrievers then ignore them)."""
    if alpha == 0:
        return []
    names = SCRIPT_FEATURES[kind]
    for name in names:
        if Path("eval/features", name).exists():
            return [f"eval/features/{name}"]
    raise SystemExit(f"--alpha {alpha} blends the {kind} subtitle features, but none of "
                     f"{', '.join('eval/features/' + n for n in names)} exists. Run script/1_preprocess.sh (it writes "
                     f"{names[1]} from the subtitles in dataset/), or pass --alpha 0 to export without them.")


def retrievers(target: str, alpha: float, wanted):
    from universalrag.retriever import ClipRetriever, ImgRetriever, TableRetriever, TextRetriever, VidRetriever
    q3, v2v = f"eval/features/query/qwen3/{target}.pkl", f"eval/features/query/vlm2vec/{target}.pkl"
    build = {
        "paragraph": lambda: TextRetriever(queryfeats_path=q3, textfeats_path=["eval/features/paragraph.pkl"]),
        "document": lambda: TextRetriever(queryfeats_path=q3, textfeats_path=["eval/features/document.pkl"]),
        "table": lambda: TableRetriever(queryfeats_path=q3, tablefeats_path=["eval/features/table.pkl"]),
        "image": lambda: ImgRetriever(queryfeats_path=v2v, imgfeats_path=["eval/features/image.pkl"]),
        "clip": lambda: ClipRetriever(queryfeats_path=v2v, clipfeats_path=["eval/features/clip.pkl"],
                                      clipscriptfeats_path=script_features("clip", alpha), alpha=alpha),
        "video": lambda: VidRetriever(queryfeats_path=v2v, vidfeats_path=["eval/features/video.pkl"],
                                      vidscriptfeats_path=script_features("video", alpha), alpha=alpha),
    }
    return {m: build[m]() for m in wanted}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--modalities", nargs="+", default=list(MODALITIES), choices=MODALITIES,
                    help="HybridQA: paragraph table; VideoRAG-Synth: video clip (the defaults of data.py)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path("src").resolve()))
    queries = json.loads(Path(f"dataset/query/{args.target}.json").read_text())
    out_dir = args.output / args.target
    out_dir.mkdir(parents=True, exist_ok=True)
    for modality, retriever in retrievers(args.target, args.alpha, args.modalities).items():
        kw = {"long_text": True} if modality == "document" else {}
        with (out_dir / f"{modality}.jsonl").open("w") as fh:
            for q in queries:
                ids, scores = retriever.retrieve(q["index"], top_k=args.top, **kw)
                fh.write(json.dumps({"qid": q["index"], "candidates": [
                    {"source_id": str(i), "score": float(s)} for i, s in zip(ids, scores)]}) + "\n")
        print(f"[rankings] {args.target}/{modality}: {len(queries)} queries", flush=True)
    (out_dir / "export.json").write_text(json.dumps({"top": args.top, "alpha": args.alpha,
                                                      "modalities": args.modalities}, indent=1))


if __name__ == "__main__":
    main()
