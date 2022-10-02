# Paper Outline

## Narrative

1. State of the art performance (proof pending ...) with multiple sequence alignment on 'omic sequences.
2. Broadly applicable to sequence data from multiple different sources.
3. Novel algorithm for building a guide tree, and for recursively aligning sets of sequences.
4. Speed and stability of Rust. Bindings in Python. Executable CLI tool.
5. New metrics for Alignment Quality.

## Figures

Nature Methods limits us to "up to 6 figures and/or tables" in the main manuscript.
We can add more in the supplement.

### 1. Overview

* An overview of the algorithm.
* Build the tree using Levenstein edit distance.
* Recursively align sibling clusters of sequences into the parent cluster. using Smith-Waterman or Needleman-Wunch between central sequences.
* Compressed storage of the full set of sequences.
* Individual sequences are retrievable as if the whole set were in a full MSA.
* Search for sequences in the compressed space.

### 2. State of the art performance

* Speed of the aligner.
* Compression and low memory utilization

### 3. Methods: Tree building, sequence alignment and alignment metrics

...

### 4. Methods: Sequence retrieval, search and compression

...

## Tables

### 1. Results and Benchmarks

...

### 2. Other ??

...

## Target Journals/Conferences

1. Nature Methods
2. Cell Genomics
3. IEEE

## References

...
