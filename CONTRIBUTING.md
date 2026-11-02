# Contributing

## Clean-room rule

Do not copy, translate, paraphrase, or structurally imitate implementation code, tests,
comments, native binaries, or file organization from `satijalab/sctransform`,
`Bowen-Man/sctransform-cpu`, or another GPL implementation. They may be run outside this
repository as black-box numerical oracles. Only independently derived mathematics,
public API documentation, papers, permissively licensed dependencies, and provenance-
documented fixtures or numeric outputs may enter the repository.

When behavior is not determined by a paper or official public documentation, record it as
`unknown` and test a proposed contract against an external oracle. Do not use source
inspection to resolve it.

## Pull requests

Development starts from `main`, uses a focused feature branch, and is squash-merged only
after required CI checks pass. Do not commit environments, caches, downloaded datasets,
reference binaries, or large benchmark output.

Keep the implementation small: remove unused abstractions instead of preserving speculative
APIs, normalize sparse inputs at one boundary, and densify only where the documented
algorithm requires a gene row/block or dense residual output. Before opening a PR, run:

```bash
ruff check .
ruff format --check .
python -m pytest
```
