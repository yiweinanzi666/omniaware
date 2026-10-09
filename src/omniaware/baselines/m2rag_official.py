"""Load the pinned official M2RAG summarizer with an already-created backend.

Only project-local imports are redirected into a private namespace. The four
upstream source files are executed unchanged; eager API/training backend imports
are not part of this adapter. No third-party dependency is substituted.
"""

import builtins
from functools import lru_cache, partial
import re
import sys
from types import ModuleType

from .. import paths


OFFICIAL_COMMIT = "ea6b9977f5a153215b38cd9a6aa3e4702ffbda22"
OFFICIAL_ROOT = paths.CACHE / "official_m2rag"
_NAMESPACE = "_eab_official_m2rag"


def _official_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and (name == "src" or name.startswith("src.")):
        name = _NAMESPACE + name[3:]
    return builtins.__import__(name, globals, locals, fromlist, level)


def _package(suffix):
    name = _NAMESPACE + suffix
    module = ModuleType(name)
    module.__path__ = [str(OFFICIAL_ROOT / "src" / suffix.lstrip(".").replace(".", "/"))]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _source(suffix):
    name = _NAMESPACE + "." + suffix
    path = OFFICIAL_ROOT / "src" / (suffix.replace(".", "/") + ".py")
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rsplit(".", 1)[0]
    module.__builtins__ = dict(vars(builtins), __import__=_official_import)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


class _InjectedBackend:
    def __new__(cls, backend):
        return backend


@lru_cache(maxsize=1)
def _load_official():
    for suffix in ("", ".utils", ".model", ".summarizer", ".summarizer.multi_modal"):
        _package(suffix)
    registry = _source("utils.registry")
    model = sys.modules[_NAMESPACE + ".model"]
    model.MODEL = registry.Registry("model")
    model.MODEL.register_module(_InjectedBackend)
    model.build_model = partial(registry.build_from_config, registry=model.MODEL)
    summarizer = sys.modules[_NAMESPACE + ".summarizer"]
    summarizer.SUMMARIZER = registry.Registry("summarizer")
    summarizer.build_summarizer = partial(registry.build_from_config, registry=summarizer.SUMMARIZER)
    _source("summarizer.base")
    _source("summarizer.multi_modal.base")
    official = _source("summarizer.multi_modal.end_to_end")

    @summarizer.SUMMARIZER.register_module
    class LocalM2RAGVLMSummarizer(official.M2RAGVLMSummarizer):
        # Upstream drops the minus sign from its documented -1/no-image answer.
        def parse_image_selection_result(self, text, max_index):
            lines = text.split("\n")
            integers = re.findall(r"[+-]?\d+", lines[-1])
            index = int(integers[-1]) if integers else -1
            return {"summary_line_id": index if 0 <= index <= max_index else -1,
                    "image_caption": " ".join(lines[:-1]).strip()}

    return summarizer


def build_official_summarizer(backend, *, multi_stage=False, use_orig_image=False):
    """Return the native summarizer with the local signed-index parser repair.

    The backend owns decoding and output budgets: upstream calls only
    ``backend.chat(messages=...)``. Its name must begin with ``vlm``; its token
    methods follow the official model contract. Upstream may mutate image dicts
    and rewrites cached URLs in its returned message logs after backend calls.
    """
    return _load_official().build_summarizer({
        "type": "LocalM2RAGVLMSummarizer",
        "params": {
            "model_config": {"type": "_InjectedBackend", "params": {"backend": backend}},
            "multi_stage": multi_stage,
            "use_orig_image": use_orig_image,
        },
    })


def query_from_t15(record):
    """Map saved, already-truncated T15 blocks to the official offline schema.

    Webpage index i in the native prompt maps to evidence_blocks[i].fact_id.
    Source IDs stay in metadata; the native prompt's numbering is unchanged.
    No references, predictions, or unsliced candidate text enter the query.
    """
    return {
        "query_id": record["qid"],
        "content": record["question"],
        "webpages": [{
            "webpage_id": block["fact_id"],
            "title": block["fact_id"],
            "url": "",
            "cleaned_webpage_splits": [{"split_id": 0, "score": 0, "text": block["text"]}],
            "images": [],
        } for block in record["evidence_blocks"]],
        "aux_images": [],
    }
