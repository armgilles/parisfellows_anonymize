# parisfellows_anonyme
Automatically pseudo-anonymise name of people in Cour des Comptes's jurisprudence

- We explore 138 documents.
- We have more than 12 k different words.
- We have more 420 k words (with 3147 positive / others are negative)

## How to :

Donwload data from this [link](https://www.dropbox.com/s/ab2durc3nvslh7s/data.zip?dl=0) then dezip it. You should see a directory ```data```on root.

**Run script**
python reading_doc_files.py --> Create data.csv file with all features and structure
python trainning.py         --> Train the model and give some metrics

