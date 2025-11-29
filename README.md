# The Compiler Paper: Neuro-Symbolic Code Analysis

This repository contains the implementation of the research project. It is an enterprise-grade system for analyzing source code using a hybrid Neuro-Symbolic approach, combining **LLVM IR** graph extraction with **Graph Neural Networks (GNN)** and **CodeBERT**.

## Project Achievements

We have successfully built an end-to-end pipeline that surpasses traditional static analysis tools by understanding both the *structure* (Control/Data Flow) and *semantics* (Instruction meanings) of code.

### Key Technical Milestones
1.  **Deep Tech Core (C++ & LLVM 17):**
    *   Implemented a custom `GraphExtractor` pass using the **LLVM New Pass Manager**.
    *   Developed **TOON (Token-Oriented Object Notation)** serialization for efficient, LLM-ready graph representation.
    *   Integrated **Memory Dependence Analysis (Alias Analysis)** to detect implicit data flows through memory.
2.  **Modern ML Architecture (Python):**
    *   **Hybrid GNN:** A novel architecture fusing semantic embeddings from **CodeBERT** with structural features processed by **Relational GCNs**.
    *   **Production Stack:** Built on **PyTorch Lightning** + **Hydra** for scalable configuration and training.
    *   **Dockerized:** Fully reproducible environment with pre-configured LLVM 17 and CUDA support.
3.  **Automated Pipeline:**
    *   Scripts for massive dataset ingestion (POJ-104, CodeNet).
    *   Parallel graph extraction using `clang` and `opt`.
    *   Evaluation suite with t-SNE visualization and CLI inference tools.


## Future Work & Optimization

While the system is operational, several avenues exist to push it to SOTA performance:

1.  **Scale Training:**
    *   Train on the full POJ-104 (train set) and CodeNet for 50+ epochs.
    *   Use A100 GPUs to handle larger batch sizes with CodeBERT.
2.  **Advanced Model Architectures:**
    *   Replace RGCN with **Graph Transformer** (e.g., Graphormer) to capture long-range dependencies.
    *   Fine-tune CodeBERT end-to-end (currently used as a frozen feature extractor).
3.  **TOON Optimization:**
    *   Implement a custom tokenizer for TOON to reduce sequence length for LLMs.
    *   Experiment with providing the raw TOON string directly to GPT-4/Llama 3 for reasoning tasks.
4.  **Cross-Language Validation:**
    *   Verify if Rust/Go code compiles to isomorphic graphs, enabling zero-shot cross-language clone detection.

## Repository Structure
*   `llvm_pass/`: C++ Source for LLVM Plugin.
*   `ml_core/`: PyTorch Lightning models, dataloaders, and training scripts.
*   `data_pipeline/`: Scripts for dataset management and compilation.
*   `docker/`: Container configuration.
