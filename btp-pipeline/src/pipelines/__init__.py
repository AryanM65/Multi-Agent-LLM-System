"""
pipelines/ — Three separately-runnable agent pipeline modules.

Each module exposes:
  - A typed Input / Output dataclass pair
  - A run_*_pipeline() function that calls sample_node() internally
  - A __main__ CLI entry-point

The underlying MLX model singleton (nodes._get_model) is loaded once
regardless of how many of the three pipelines are imported in the same
process.

Quick imports:
    from src.pipelines import run_retriever_pipeline
    from src.pipelines import run_reasoner_pipeline
    from src.pipelines import run_writer_pipeline
"""

from src.pipelines.retriever_pipeline import RetrieverInput, RetrieverOutput, run_retriever_pipeline
from src.pipelines.reasoner_pipeline  import ReasonerInput,  ReasonerOutput,  run_reasoner_pipeline
from src.pipelines.writer_pipeline    import WriterInput,    WriterOutput,    run_writer_pipeline

__all__ = [
    "RetrieverInput", "RetrieverOutput", "run_retriever_pipeline",
    "ReasonerInput",  "ReasonerOutput",  "run_reasoner_pipeline",
    "WriterInput",    "WriterOutput",    "run_writer_pipeline",
]
