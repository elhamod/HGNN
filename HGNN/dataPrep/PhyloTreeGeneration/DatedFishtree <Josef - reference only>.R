library(datelife)
library(treeplyr)
library(rotl)

tree <- read.tree("./ottfishtree.tre")
fishtree <- read.tree("./timetree.tre.xz")

## OpenTree + Datelife
synth_tree <- get_otol_synthetic_tree(ott_ids = gsub("ott", "", tree$tip.label))
synth_tree_dated <- get_dated_otol_induced_subtree(ott_ids=gsub("ott", "", tree$tip.label)) 

## Get species names from ottids to match to fishtree of life
species <- rotl::taxonomy_taxon_info(as.integer(gsub("ott", "", tree$tip.label))) # Strip out the "ott" prefix and get taxonomy info
species <- sapply(species, function(x) x$name) #Grab name
species <- gsub(" ", "_", species) #Replace space with underscore

fishtree_td <- make.treedata(fishtree, data.frame(genspec=species, ottid=tree$tip.label)) #Match tree and data
ftol <- fishtree_td$phy #Pull out tree
ftol$tip.label <- as.character(fishtree_td[["ottid"]]) #Replace tip labels with original tip labels

# Save the trees
write.tree(ftol, file="~/Downloads/fishTOL.phy") 
write.tree(synth_tree_dated, file="~/Downloads/datelife.phy")

plot(synth_tree)
plot(synth_tree_dated)
plot(ftol)
