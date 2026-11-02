args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: export_r_oracle.R FIXTURE_PREFIX OUTPUT_DIR")
}

fixture_prefix <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(sctransform))

umi <- readMM(paste0(fixture_prefix, ".mtx"))
umi <- as(umi, "dgCMatrix")
gene_names <- readLines(paste0(fixture_prefix, ".genes.txt"), warn = FALSE)
cell_names <- readLines(paste0(fixture_prefix, ".cells.txt"), warn = FALSE)
rownames(umi) <- gene_names
colnames(umi) <- cell_names

started <- proc.time()[["elapsed"]]
result <- vst(
  umi,
  vst.flavor = "v2",
  n_genes = NULL,
  n_cells = NULL,
  return_cell_attr = TRUE,
  verbosity = 0
)
elapsed <- proc.time()[["elapsed"]] - started

write_f64 <- function(value, name) {
  connection <- file(file.path(output_dir, name), open = "wb")
  on.exit(close(connection))
  writeBin(as.double(value), connection, size = 8, endian = "little")
}

write_matrix_f64 <- function(value, name) {
  write_f64(t(as.matrix(value)), name)
}

model_fit <- result[["model_pars_fit"]]
model_raw <- result[["model_pars"]]
gene_attr <- result[["gene_attr"]]
residuals <- result[["y"]]
genes <- rownames(model_fit)
cells <- colnames(residuals)
raw_genes <- rownames(model_raw)
fit_cells <- result[["cells_step1"]]
outlier_mask <- result[["model_pars_outliers"]]
if (is.null(names(outlier_mask)) || anyNA(outlier_mask)) {
  stop("oracle returned an invalid model_pars_outliers vector")
}
hvg_order <- order(
  gene_attr[genes, "residual_variance"],
  decreasing = TRUE,
  method = "radix"
)

writeLines(genes, file.path(output_dir, "genes.txt"), useBytes = TRUE)
writeLines(cells, file.path(output_dir, "cells.txt"), useBytes = TRUE)
writeLines(raw_genes, file.path(output_dir, "fit_genes.txt"), useBytes = TRUE)
writeLines(fit_cells, file.path(output_dir, "fit_cells.txt"), useBytes = TRUE)
writeLines(genes[hvg_order], file.path(output_dir, "hvg_genes.txt"), useBytes = TRUE)
writeLines(
  names(outlier_mask)[outlier_mask],
  file.path(output_dir, "outlier_genes.txt"),
  useBytes = TRUE
)
writeLines(
  genes[is.infinite(model_fit[genes, "theta"])],
  file.path(output_dir, "poisson_genes.txt"),
  useBytes = TRUE
)

write_f64(model_fit[genes, "(Intercept)"], "regularized_intercept.f64")
write_f64(model_fit[genes, "log_umi"], "regularized_slope.f64")
write_f64(model_fit[genes, "theta"], "regularized_theta.f64")
write_f64(model_raw[raw_genes, "(Intercept)"], "raw_intercept.f64")
write_f64(model_raw[raw_genes, "theta"], "raw_theta.f64")
write_matrix_f64(residuals[genes, cells, drop = FALSE], "residuals.f64")
write_f64(gene_attr[genes, "residual_mean"], "residual_mean.f64")
write_f64(gene_attr[genes, "residual_variance"], "residual_variance.f64")

metadata <- c(
  paste("schema_version", "1", sep = "\t"),
  paste("oracle", "satijalab/sctransform", sep = "\t"),
  paste("sctransform_version", as.character(packageVersion("sctransform")), sep = "\t"),
  paste("glmGamPoi_version", as.character(packageVersion("glmGamPoi")), sep = "\t"),
  paste("matrixStats_version", as.character(packageVersion("matrixStats")), sep = "\t"),
  paste("R_version", R.version.string, sep = "\t"),
  paste("genes", length(genes), sep = "\t"),
  paste("cells", length(cells), sep = "\t"),
  paste("fit_genes", length(raw_genes), sep = "\t"),
  paste("fit_cells", length(fit_cells), sep = "\t"),
  paste("parameter_outliers", sum(outlier_mask), sep = "\t"),
  paste("elapsed_seconds", format(elapsed, digits = 17), sep = "\t"),
  paste("generated_on", format(Sys.Date()), sep = "\t")
)
writeLines(metadata, file.path(output_dir, "metadata.tsv"), useBytes = TRUE)
