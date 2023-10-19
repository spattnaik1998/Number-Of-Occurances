# Number-Of-Occurances
The code creates a large array that its columns are the words of the dictionary with number of occurances of each word as columns and the document as rows.

Term Frequency is the number of times a term occurs in a document. Numerator is the number of occurances of a word, and the denominator is the sum of all the words in the document. Next, we calculate "Inverse Document Frequency" for all the documents and finally calculate TF-IDF(w) and create a TF-IDF matrix.

Dataset: gs://metcs777-buck1/wiki-categorylinks.csv.bz2, gs://metcs777-buck1/WikipediaPagesOneDocPerLine1m.txt
