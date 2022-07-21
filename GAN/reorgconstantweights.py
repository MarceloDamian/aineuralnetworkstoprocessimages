
import pandas as pandasForSortingCSV
import csv
import pandas as pd
  
# assign dataset
csvData = pandasForSortingCSV.read_csv("./constantweights.csv")
                                         
# displaying unsorted data frame
print("\nBefore sorting:")
print(csvData)
  
# sort data frame
csvData.sort_values(["index"], 
                    axis=0,
                    inplace=False)
  
csvData.to_csv("./enumeratedconstantweights.csv")

# displaying sorted data frame
print("\nAfter sorting:")
print(csvData)
