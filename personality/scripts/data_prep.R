score_single <-function(dataset) {
  data <- list(speaker=dataset[,1], features=dataset[,2:length(dataset)])
  above_mean <- apply(data$features, 2, function(x) x - mean(x))
  is_above_mean <- apply(above_mean, 1:2, function(x) 
    ifelse(x > 0, 1, 0))
  is_above_mean
}

#Returns a dataframe of labels
score <- function(datalist) {
  new_scores <- lapply(datalist, score_single)
  sum_scores <- add(new_scores)
  classes <- apply(sum_scores, 1:2, function(x) ifelse(x >= 6, 1, 0))
  data.frame(datalist[[1]][,1, drop=FALSE], classes)
}

#Helper function, generic addition
add <- function(x) Reduce("+", x)

#Used to generate the labels csv file 
generate.full.data <- function() {
  #Import data
  temp <- list.files(pattern= "*.csv", path="Personality_Scores/")
  for (i in 1:length(temp)) assign(sub(".csv", "", temp[i]), read.csv(paste("Personality_Scores",temp[i], sep="/")))
  
  #Data as list (TODO: unneccessary)
  datasets <- list(Score_001, Score_002, Score_003, Score_004, Score_005, Score_006, Score_007,
                   Score_008, Score_009, Score_010, Score_011)
  classes.full <- score(datasets)
  write.csv(classes.full, "full_classes.csv", row.names=FALSE)
  classes.full
}

merge_classes <- function() {
  meta <- read.csv("../data/csvs/Metadata.csv")
  classes <- read.csv("../data/csvs/personality_scores.csv")
  both <- merge(meta, classes)
  write.csv2(both, "../data/csvs/labels_all.csv", row.names=FALSE)
  both
}