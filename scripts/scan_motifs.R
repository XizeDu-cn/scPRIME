#!/usr/bin/env Rscript
# Count JASPAR 2020 CORE motif matches in peak sequences with motifmatchr.
suppressPackageStartupMessages({
  library(optparse); library(Biostrings); library(TFBSTools)
  library(JASPAR2020); library(motifmatchr); library(SummarizedExperiment)
  library(HDF5Array); library(rhdf5)
})
options <- list(
  make_option('--fasta',type='character',help='Peak sequences from bedtools getfasta; record IDs match h5ad peaks'),
  make_option('--species',type='character',default='Homo sapiens'),
  make_option('--output',type='character',help='Output HDF5 filename'))
opt <- parse_args(OptionParser(option_list=options))
if (is.null(opt$fasta)||is.null(opt$output)) stop('--fasta and --output are required')
if (file.exists(opt$output)) stop('Output already exists')
set.seed(2020)
sequences <- readDNAStringSet(opt$fasta)
pwms <- getMatrixSet(JASPAR2020,list(species=opt$species,collection='CORE'))
counts <- motifCounts(matchMotifs(pwms,sequences,out='scores'))
motif_names <- vapply(pwms,function(x) x@name,character(1))
motif_names <- gsub('::','_',motif_names,fixed=TRUE)
writeHDF5Array(t(counts),opt$output,name='mat',with.dimnames=FALSE,verbose=FALSE)
h5write(names(sequences),opt$output,'row_names')
h5write(motif_names,opt$output,'col_names')
h5write(names(pwms),opt$output,'pwm_accessions')
