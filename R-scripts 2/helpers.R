####### Custom theme used for all plots in the project
library(ggthemes)
library(ggplot2)

# Define the basics 
theme_pv <- function(base_size = 12,
                      base_family = 'mono',
                      base_rect_size = base_size / 170){
  
  
  theme(
    # Define the text in general
    text = element_text(family = base_family),
    
    # Define the plot title
    plot.title = element_text(color = 'black', face = "bold", hjust = 0, size = 17),
    
    # Define titles for the axes
    axis.title = element_text(
      colour = rgb(105, 105, 105, maxColorValue = 255),
      size = rel(1), margin = margin(t = 20, r = 20)),
    axis.title.x = element_text(margin = margin(t = 10, b = 10)),
    axis.title.y = element_text(margin = margin(r = 10)),
    
    # Define the style of the caption
    plot.caption = element_text(hjust = 1, 
                                colour = rgb(150, 150, 150, maxColorValue = 255)),
    
    # Define the style of the axes text and ticks
    axis.text = element_text(
      color = rgb(105, 105, 105, maxColorValue = 255), size = rel(0.7)),
    axis.ticks = element_line(colour = rgb(105, 105, 105, maxColorValue = 255)),
    axis.ticks.length.y = unit(.25, "cm"),
    axis.ticks.length.x = unit(.25, "cm"),
    
    # Style the grid lines
    panel.grid.major.y = element_line(
      rgb(105, 105, 105, maxColorValue = 255), linetype = "dotted", size = rel(1.3)),
    panel.ontop = FALSE,
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.x = element_blank(),
    
    # Specify the background of the plot panel
    panel.background = element_rect(fill = rgb(245, 245, 245, maxColorValue = 255)),
    
    # Define the style of the legend
    legend.key = element_rect(fill = "white", colour = NA),
    legend.title = element_text(size = rel(1.2)),
    
    # Define the aspect ratio of the panel 
    aspect.ratio = 0.7
  )
}

######### extract metrics from tfruns output
read_metrics <- function(path, files = NULL)
  # 'path' is where the runs are --> e.g. "path/to/runs"
{
  path <- paste0(path, "/")
  if ( is.null(files) ) files <- list.files(path) 
  n <- length(files)
  out <- vector("list", n)
  for ( i in n:1 ) {
    dir <- paste0(path, files[i], "/tfruns.d/")
    out[[i]] <- jsonlite::fromJSON(paste0(dir, "metrics.json")) 
    out[[i]]$flags <- jsonlite::fromJSON(paste0(dir, "flags.json")) 
    out[[i]]$evaluation <- jsonlite::fromJSON(paste0(dir, "evaluation.json"))
  }
  return(out) 
}


# function to plot the learning curves for preprocessing and architecture. It takes as input the list of 
# validation and training data  (fits) and names for the legend (coln) and returns a data frame which can be 
# passed to ggplot
data_pt <- function(fits, coln) {
  n <- length(fits)
  
  # create empty dataframe to store the performance metrics
  learn <- data.frame(matrix(ncol = length(coln), nrow = 100))
  
  for (i in 1:n) {
    # bind the accuracy metrics together
    learn[, c((i * 2) - 1, i * 2)] <- cbind(fits[[i]]$metrics$accuracy, fits[[i]]$metrics$val_accuracy)
  }
  
  # add the column names
  colnames(learn) <- coln
  
  #id variable for position in matrix 
  learn$id <- 1:nrow(learn) 
  
  #reshape to long format
  plot_learn <- melt(learn, id.var = 'id')
  
  plot_learn
}

