# Licensed real-data validation plan

Status: provenance and deterministic derivation fixed on 2026-11-03; differential results
are not yet claimed.

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
6. Create and persist one explicit v2 selection manifest for all Level-A/B/C runs. No
   cross-language random stream is compared.
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
