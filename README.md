# parisfellows_anonyme
Automatically pseudo-anonymise name of people in Cour des Comptes's jurisprudence

- We explore 138 documents.
- We have more than 12 k different words.
- We have more 420 k words (with 3147 positive / others are negative)

## How to :

Donwload data from this [link](https://www.dropbox.com/s/qxbgx3t7oooz6xw/data.zip?dl=0) then dezip it. You should see a directory ```data```on root.

### **Run script : **

- python reading_doc_files.py --> Create data.csv file with all features and structure
- python trainning.py         --> Train the model and give some metrics
- get_prediction.py           --> Read & processs a .docx (line 220) to anonymise it in ouput directory.

Create ouput files :
- ```[name_of_file]_log.csv``` : Log of this file (warning is a bool)
- ```[name_of_file].txt``` : Return the text with anonymise result.
- ```[name_of_file].html``` : Return the text in html balise with color (green seems OK, Red mean warning this could be a error).


*result of html file* :

![image](https://cloud.githubusercontent.com/assets/8374843/18081661/8d46ae36-6e9b-11e6-82b6-8ec96a1d5889.png)

