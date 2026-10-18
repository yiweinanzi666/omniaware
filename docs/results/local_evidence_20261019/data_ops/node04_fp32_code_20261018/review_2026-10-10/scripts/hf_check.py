import fnmatch, json, sys
from huggingface_hub import HfApi
api = HfApi()
CHECKS = [
    ("Qwen/Qwen3-VL-8B-Instruct", None, None),
    ("Qwen/Qwen3-VL-2B-Instruct", None, None),
    ("OpenGVLab/InternVL3_5-8B-HF", None, None),
    ("BAAI/bge-large-en-v1.5", "d4aa6901d3a41ba39fb536a557fa166f842b0e09", ["*.json", "*.txt", "model.safetensors", "README.md"]),
    ("openai/clip-vit-large-patch14", None, ["*.json", "*.txt", "model.safetensors"]),
    ("BAAI/bge-reranker-v2-m3", None, ["*.json", "*.model", "model.safetensors"]),
    ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7", None),
    ("facebook/bart-large-cnn", "main", ["config.json", "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"]),
]
for repo, rev, allow in CHECKS:
    try:
        info = api.model_info(repo, revision=rev or "main")
        files = [s.rfilename for s in info.siblings]
        print(f"OK {repo} @{rev or 'main'} sha={info.sha} n_files={len(files)}")
        if allow:
            matched = sorted({f for f in files for p in allow if fnmatch.fnmatch(f, p)})
            unmatched_allow = [p for p in allow if not any(fnmatch.fnmatch(f, p) for f in files)]
            print("   allow matched:", matched)
            if unmatched_allow:
                print("   !! allow patterns with NO match:", unmatched_allow)
        big = [f for f in files if f.endswith((".bin", ".safetensors", ".pth"))]
        if allow is None:
            print("   weight files (no allow filter -> all downloaded):", big)
        if repo.startswith("MoritzLaurer"):
            print("   all files:", files)
    except Exception as e:
        print(f"FAIL {repo} @{rev or 'main'}: {type(e).__name__}: {str(e)[:200]}")
