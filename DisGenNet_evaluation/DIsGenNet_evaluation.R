option_list <- list(
  make_option(c("-g", "--genes_df"), type="character", help="Path to genes CSV file"),
  make_option(c("-o", "--output"), type="character", help="Path for output CSV file")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)
genes_df_path <- args[1]
output_path <- args[2]

if (is.null(opt$genes_df) || is.null(opt$output)) {
  stop("You must provide both --genes and --output options.")
}

required_packages <- c("devtools", "dplyr", "disgenet2r")
missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

if (length(missing_packages) > 0) {
  install.packages(missing_packages, dependencies = TRUE)
}

library(devtools)
library(dplyr)
library(disgenet2r)

authorization <- readLines("authorization.txt")
email <- authorization[1]
password <- authorization[2]

gene_names_list <- read.csv(opt$genes_df)$symbol

results <- gene2disease(api_key = get_disgenet_api_key(email, password), gene = gene_names_list, verbose = TRUE)

df <- as.data.frame(results@qresult)

write.csv(df, file = opt$output, row.names = FALSE)

cat("Data exported to", output_path, "\n")


