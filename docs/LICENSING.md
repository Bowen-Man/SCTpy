# Licensing and independent provenance

Snapshot date: **2026-11-02**.

## Project license

This repository uses **BSD-3-Clause**. It is the common permissive license used by central
scverse projects including AnnData (observed at `bcdcf2cfb1d47ecba5471f4a3a89ed91439b6797`)
and Scanpy. It is compatible with the observed licenses of likely integration dependencies:

| Project | Observed version/commit | License |
|---|---|---|
| AnnData | tag 0.13.4 / `bcdcf2cfb1d47ecba5471f4a3a89ed91439b6797` | BSD-3-Clause |
| rapids-singlecell | 0.18.1 / `acdb4b01aea38d3a4459fd25f1feca9e2f354e67` | MIT |
| CuPy | 14.2.0 / `f4f7f7e461b647743b13b37775c2121885c844c2` | MIT |
| RMM | main `006f950a634908de9688cfe1881947ad4ca62b59` | Apache-2.0 |
| ScaleSC | `a7362f5cc88b3a7d3b246e85bd2a4ad53d9bdc35` | MIT |

Compatibility here means those permissive packages may be optional dependencies; it does
not grant permission to copy their source or remove their notices. **confirmed from official
documentation** for licenses; compatibility is a standard license interpretation, not
legal advice.

## GPL isolation

`satijalab/sctransform`, `glmGamPoi` 1.22.0, the fixed
`Bowen-Man/sctransform-cpu` v0.4.0 oracle, current `Zikun-Yang/SCTransform`, and the
`saketkc/pySCTransform` license file are GPL-covered.
They are not dependencies, build inputs, or vendored artifacts. Public R source may be
consulted to identify algorithmic behavior, but the Python design, naming, vectorization,
tests, and documentation are independently written. No reference R runtime, Fortran
library, DLL, shared object, wheel, native binary, source text, test, comment, or file
layout is copied or shipped here.

The official papers are open-access CC BY 4.0 and may be used to derive mathematical
behavior with attribution. Public API manuals and implementation source may establish
observable behavior; this does not permit copying protected expression. Oracle programs
run in an external environment and may produce numeric golden outputs; every
admitted fixture/output must carry a provenance record that identifies its source counts,
license, oracle version, manifest, command contract, and generation date.

## Excluded implementation sources

Even when a repository advertises MIT, it is excluded if its own documentation describes
direct translation of GPL implementation code. In particular, `GenomicAI/truecell` is not
an implementation source. Repositories with no explicit license, such as the observed
`atarashansky/SCTransformPy` snapshot, are also unusable as sources.

## Dependency policy

The CPU install depends only on published Python packages with permissive licenses. CUDA,
CuPy, RMM, and RAPIDS packages must remain optional. The project itself contains no
compiled extension. Dependency and license snapshots are reviewed before each release.
