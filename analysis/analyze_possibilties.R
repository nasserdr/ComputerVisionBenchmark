library(readr)
library(stringr)
library(tidyr)
file_counts <- read_csv("file_counts.csv")
names(file_counts) <- c('Name', 'FilesN', 'SubDirN')
file_counts$Model <- str_split(file_counts$Name, "chunks", simplify = TRUE)[,1]
file_counts$Config <- str_split(file_counts$Name, "chunks", simplify = TRUE)[,2]
file_counts$Model <- str_remove(file_counts$Model, "^_|_$")
file_counts$Config <- str_replace(file_counts$Config, "^_|_$", "")
file_counts <- separate(file_counts, 'Config', into = c('col1', "IPB", 'col2', "MI"), sep = "_")
file_counts$col1 <- NULL
file_counts$col2 <- NULL
file_counts$Name <- NULL
file_counts$Possible <- ifelse(file_counts$FilesN >= 5, "Yes", "No")
file_counts$FilesN <- NULL
file_counts$SubDirN <- NULL
file_counts$MI <- NULL
write.csv(file_counts, file = "Model_Vs_BatchSize_Possibility.csv", row.names = FALSE)
