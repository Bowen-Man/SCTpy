# Licensed real-data validation plan

Status: provenance, deterministic derivation, and the first Level-A/B/C differential were
completed on 2026-11-03.

## Source

The selected source is **10k Human PBMCs, 3' v3.1, Chromium X**, published by
10x Genomics on 2021-09-14 and analyzed with Cell Ranger 6.1.0. The official dataset page
states that it is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This is compatible with
redistributing derived factual metrics with attribution, but the 39.5 MB source matrix is
downloaded on demand and is not committed to this repository.

- Official page:
  <https://www.10xgenomics.com/datasets/10k-human-pbmcs-3-ht-v3-1-chromium-x-3-1-high>
- Fixed input file:
  <https://cf.10xgenomics.com/samples/cell-exp/6.1.0/10k_PBMC_3p_nextgem_Chromium_X/10k_PBMC_3p_nextgem_Chromium_X_filtered_feature_bc_matrix.h5>
- Expected byte length: `39,503,734`
- Expected SHA-256:
  `42a789ff3f92f0e117d1ff4ffcb2fa8edee87dd2a03d83a81502709344a38bf0`

**confirmed from official documentation**: CC BY 4.0, donor/sample description, Cell
Ranger version, and publication date. **measured on 2026-11-03** after hash verification:
the HDF5 matrix is 36,601 features by 11,996 barcodes with 27,164,608 nonzero entries;
all features are `Gene Expression` on `GRCh38`. Ensembl feature IDs and barcodes are
unique, while ten gene-symbol collisions make symbols unsuitable as canonical names.

The deterministic 3,000-cell derivation has 6,788,544 nonzero entries and occupies about
51.9 MiB as canonical CSR before compression (12.7 MB in the generated fixture archive).
It retains 17,155 genes at `min_cells=5`, of which 14,004 satisfy the step-1
overdispersion rule. Its canonical sparse-count SHA-256 is
`d9c23eda8654b17fab1a3672df4961b53f2d9e63356dd631de567c76db570532`.
**measured on 2026-11-03**

## Deterministic validation fixture

The validation fixture will be derived without biological QC or expression-dependent cell
filtering:

1. Verify the source byte length and SHA-256 before parsing.
2. Read the 10x CSC arrays and make the genes-by-cells orientation explicit.
3. Keep `Gene Expression` features in source order. Use unique Ensembl IDs as gene names;
   retain gene symbols only as metadata.
4. Select 3,000 barcodes without replacement using NumPy PCG64 seed `20261103`, then sort
   the selected source indices. Preserve barcode strings exactly.
5. Retain raw integer UMI counts. The SCT `min_cells=5` rule, not an upstream custom
   filter, determines eligible genes.
6. Run Level A with `set.seed(20261103)`, `n_genes=2000`, and `n_cells=2000`, then freeze
   its exported ordered gene/cell selection as the manifest replayed by Levels B and C.
   Order is part of the contract because it defines 500-gene fit bins; the manifest, not
   an assumption about a cross-language random stream, defines the comparison.
7. Record the derived sparse matrix hash, ordered-name hashes, shapes, `nnz`, selection
   manifest hash, package versions, and every numerical threshold before inspecting the
   differential result.

The full residual matrix and downloaded HDF5 file stay under ignored
`benchmark-results/real-data/`. A small committed result summary may contain only factual
metrics, hashes, named branch differences, and HVG overlaps. It must carry this source
attribution and may not contain an external implementation or runtime.

## Acceptance and resource budget

The first run uses `SCTConfig.r_v2(n_genes=2000, n_cells=2000)` with the persisted
manifest. It compares raw/final parameters, finite/Inf masks, residual summaries, and HVG
Top 1000/2000 exactly as the synthetic gate does. The initial local budget is 30 minutes
and 8 GiB peak RSS per implementation. Exceeding either budget stops the run and records
the failure; it does not trigger an unreviewed increase.

Real-data validation is an additional generalization check. It does not replace the
synthetic frozen CI gate and will not be used to weaken its tolerances.

`benchmarks.prepare_pbmc_fixture` verifies the source hash before reading it and stores
the derived counts as canonical CSR arrays. The manual `Real-data validation` workflow is
run either from the Actions UI or by applying the `real-data-validation` label to a PR.
It installs both external oracles only inside that isolated job and uploads compact
diagnostics, never the source HDF5, sparse fixture, full residual matrix, or native binary.

## First differential result

The isolated GitHub Actions run
[`41492647448`](https://github.com/Bowen-Man/sctransform-native/actions/runs/41492647448)
used R 4.5.2, `sctransform` 0.4.3, `glmGamPoi` 1.22.0, `matrixStats` 1.5.0,
and Level B at commit `b79041796e3f7ca3b33029adde5d7132eedd2430`. The R fit selected
2,000 genes and 2,000 cells; 142 raw theta values were infinite and 159 parameter outliers
were reported. Level C matched both named masks exactly. **measured**

| Level C versus Level A | Result |
|---|---:|
| raw intercept RMSE | `4.83e-9` |
| raw log10(theta) RMSE | `1.03e-5` |
| regularized intercept RMSE | `2.02e-4` |
| regularized log10(theta) RMSE | `2.08e-8` |
| Pearson residual RMSE | `1.77e-5` |
| residual-variance RMSE | `2.20e-5` |
| residual-variance Spearman | `0.9999999952` |
| HVG Top 1000/2000/3000 Jaccard | `1.0 / 1.0 / 1.0` |

The complete HVG order was not identical; its Spearman correlation was `0.9999999952`.
Level C's recorded fit and total stage times were 5.88 s and 7.32 s on the hosted runner.
These are one-fixture measurements, not general accuracy or performance claims. The
pre-registered broad real-data/full-differential thresholds passed against both Level A
and Level B; no threshold was changed after observing this run.

An initial controlled run canonicalized the same named selection into source order. It
preserved the selected sets but changed 500-gene bin membership, producing raw-intercept
RMSE `6.06e-3`. Replaying the oracle's ordered manifest reduced that value to `4.83e-9`.
This identifies selection order as part of the numerical contract when gene-bin fitting
is enabled. **measured**
