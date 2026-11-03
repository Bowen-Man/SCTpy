# Implementation survey

Snapshot date: **2026-11-02** (Asia/Shanghai). Git commits are the observed default-branch
heads on that date. This survey used papers, official documentation, repository metadata,
license files, and project READMEs. It did **not** inspect the implementation source of the
GPL R package or `Bowen-Man/sctransform-cpu`.

Evidence labels used below:

- **confirmed from paper** — stated in a peer-reviewed methods section;
- **confirmed from official documentation** — stated in package/API documentation,
  repository metadata, a license, or a project README;
- **measured** — produced by a recorded execution in this repository;
- **inferred** — an engineering interpretation of confirmed facts;
- **unknown** — not fixed by the sources inspected.

## Primary method and estimator sources

| Item | Snapshot | License | Implementation-relevant finding |
|---|---|---|---|
| `satijalab/sctransform` | v0.4.3, `49e35b5aeb76a602910207cbfde1561093340be3` | GPL-3.0 | Public documentation says v2 selects `glmGamPoi_offset`, 2,000 cells, fixed slope/offset fitting, and Poisson-gene exclusion. It is an oracle only. **confirmed from official documentation** |
| Hafemeister & Satija (2019), DOI `10.1186/s13059-019-1874-1` | version of record 2019-12-23 | CC BY 4.0 article | Defines NB regression, regularization across gene abundance, Pearson residuals, and residual-variance ranking. **confirmed from paper** |
| Choudhary & Satija (2022), DOI `10.1186/s13059-021-02584-9` | Genome Biology 23:27 | CC BY 4.0 article | Defines v2 fixed slope, offset interpretation, Poisson-gene rule, per-gene theta, and the residual standard-deviation floor. **confirmed from paper** |
| `glmGamPoi` | fixed oracle 1.22.0 / `RELEASE_3_22` `908cde9d595259b4a317343ddfbcf20036ef0895`; current release 1.24.0; devel 1.25.1 | 1.22.0 and release 1.24.0: GPL-3; devel metadata: MIT + license file | It defines the v2 estimator's observable Cox--Reid and local-trend semantics. It is a semantic reference only, not a dependency; no source expression is copied. The license transition is recorded rather than collapsed into one claim. **confirmed from official documentation and source** |

The public method contract supports the following v2 facts:

- `x_gc ~ NB(mu_gc, theta_g)`, `log(mu_gc) = beta_g0 + log(n_c)`, and
  `Var(x_gc) = mu_gc + mu_gc^2 / theta_g`. **confirmed from paper**
- With `log10(n_c)` as predictor, the fixed slope is `ln(10)`. **confirmed from paper**
- Genes with arithmetic mean below `0.001` or sample variance no greater than the mean
  are excluded from regularization and use `theta = infinity`. **confirmed from paper**
- V2 retains a learned/regularized intercept; it does not fix the intercept for all genes.
  **confirmed from paper**
- The residual denominator has a gene-level floor `nonzero_count_median / 5`.
  **confirmed from paper**
- The low-level documented residual clipping default is `[-sqrt(cells), sqrt(cells)]`;
  Seurat's wrapper documents `[-sqrt(cells/30), sqrt(cells/30)]`. The API layer must name
  which convention it applies. **confirmed from official documentation**
- Public docs name `od_factor` regularization, a kernel bandwidth of `3 * bw.SJ`, and a
  geometric-mean pseudocount of 1, but the paper does not completely specify all
  numerical edge behavior. **confirmed from official documentation; unknown** for
  optimizer and boundary details.

## Python and GPU projects

| Project | Snapshot and license | Coverage relevant to this project |
|---|---|---|
| `saketkc/pySCTransform` | package metadata 0.1.1, `ec1216eb59c31cd9f4d70f9876a6990e92bbb0e8`; GPL-3.0 license file (despite stale BSD setup metadata) | README exposes a v2-like fixed-slope/Poisson-exclusion path and says `batch_var` is unsupported. Completeness of native v2 theta fitting is **unknown** without prohibited source inspection. Oracle only. |
| `GenomicAI/truecell` | v2.0.0, `f1ba3a8e5288e7397b422e4430f3033955969056`; MIT | README claims v1/v2 behavior but also calls portions direct translations/ports of R/Seurat. It is excluded as an implementation source regardless of the repository license; black-box comparison is possible. **confirmed from official documentation** |
| `Zikun-Yang/SCTransform` | v0.2.0, `8ade6eded351689ff53e37958a8190d70279fe31`; GPL-3.0-only | README claims native v1/v2, explicitly describes ports of R routines, and lists corrected counts as unfinished. GPL oracle only; not an implementation source. **confirmed from official documentation** |
| `atarashansky/SCTransformPy` | no release, `30dc3f0a41528efe50b880987b47a11b5397b94b`; no license detected | 2021 README describes a Python port with one log-UMI latent variable and translated theta MLE; no documented v2 Poisson handling or v2 residual floor. Treat as partial v1-era work and unusable source. **confirmed from official documentation; inferred** scope classification. |
| `scverse/rapids-singlecell` | v0.18.1, `acdb4b01aea38d3a4459fd25f1feca9e2f354e67`; MIT | `normalize_pearson_residuals` uses analytic residuals with one user-supplied theta shared by all genes (default 100). It does not fit gene-specific intercept/theta, exclude Poisson genes, or regularize parameters, so it is **not complete SCT v2**. **confirmed from official documentation** |
| `interactivereport/ScaleSC` | documented 0.1.0, `a7362f5cc88b3a7d3b246e85bd2a4ad53d9bdc35`; MIT | Its documented normalization is library-size scaling plus `log1p`, not SCT. Algorithm work includes GPU Harmony/marker operations; its broadly reusable scaling contribution is cell chunking, explicit CPU/GPU movement, and sparse-index-limit avoidance. **confirmed from official documentation; inferred** separation. |

## Analytic Pearson residuals are not SCT v2

Both use an NB2 variance and a library-size offset. Analytic residual normalization fixes
one theta across genes and obtains the offset-model mean analytically. SCT v2 instead
learns gene-specific intercepts and theta, identifies Poisson-like genes, regularizes
gene parameters as a function of abundance, and applies the v2 residual denominator
floor. Similar formulas at the final residual step do not make the fitting procedures
equivalent. **confirmed from paper and official documentation**

## Engineering decisions

- Use papers and public documentation to define the contract; use all GPL projects only
  as external processes producing numeric observations.
- Start with float64 and a genes-by-cells numerical core. AnnData conversion is a separate
  cells-by-genes adapter.
- Add corrected UMI only after an independent specification and numerical gate; continue
  to defer batch effects and extra covariates.
- Record selection indices and names in a manifest; never rely on an implicit RNG stream.
- Follow the official v2 selection order: sample cells, recheck `min_cells`, exclude
  non-overdispersed genes, then inverse-density sample by log geometric mean. Use a named
  PCG64 stream rather than adding an R RNG compatibility layer. **confirmed from official
  source; project decision** for the RNG.
- Do not add a GPU implementation until CPU stage timings demonstrate a target.
