# Ahlmann-Eltze, Huber & Anders, Nat. Methods (2025), Eq. 1.
# Both embeddings use RPE1 training expression. Rows are genes, columns are conditions.
suppressPackageStartupMessages(library(Matrix))
work <- commandArgs(trailingOnly = TRUE)[[1]]
labels <- function(name) readLines(file.path(work, name), warn = FALSE)
genes <- labels("genes.txt")
conditions <- labels("conditions.txt")
queries <- labels("queries.txt")
readouts <- labels("readouts.txt")
x <- matrix(readBin(file.path(work, "expression.f64"), "double",
                    n = length(genes) * length(conditions), size = 8, endian = "little"),
            nrow = length(genes), dimnames = list(genes, conditions))
stopifnot(!any(queries %in% conditions), all(queries %in% genes), "ctrl" %in% conditions)
set.seed(1)
gene_embedding <- irlba::prcomp_irlba(x, n = 10)$x
rownames(gene_embedding) <- genes
target_embedding <- t(irlba::prcomp_irlba(x, n = 10)$x)
colnames(target_embedding) <- genes
target_embedding <- cbind(target_embedding, ctrl = 0)
training <- intersect(colnames(target_embedding), conditions)
stopifnot(length(training) == length(conditions))
p <- target_embedding[, training, drop = FALSE]
y <- x[, training] - x[, "ctrl"]
intercept <- rowMeans(y)
g <- gene_embedding
penalty <- Diagonal(10, x = 0.1)
w <- solve(crossprod(g) + penalty) %*% t(g) %*% (y - intercept) %*%
     t(p) %*% solve(tcrossprod(p) + penalty)
prediction <- as.matrix(g %*% w %*% target_embedding[, queries, drop = FALSE]) + intercept
writeBin(as.double(prediction[readouts, , drop = FALSE]),
         file.path(work, "response.f64"), size = 8, endian = "little")
