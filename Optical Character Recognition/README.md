PART 1: PART OF SPEECH TAGGING
----------------------


RUNNING THE CODE
----------------
python3 label.py training_file testing_file

CODE DESCRIPTION
----------------
1. Model training done using the train data set
2. Implemented 3 models for predicted POS tags:
    -   Simple
    -   HMM
    -   Complex
3. Model testing done using the test data set
4. Printed the results

RESULTS
-------
==> So far scored 2000 sentences with 29442 words.

  Words correct:  Sentences correct:
                    
0. Ground truth:    100.00%         100.00%

1. Simple:          93.92%          47.50%
      
2. HMM:             90.74%          35.65%
         
3. Complex:         92.70%          39.20%
                  

********************************************************************

PART 2: OPTICAL CHARACTER RECOGNITION (OCR)
-------------------------------------------

To Run:
--------
python3 ocr.py train-image-file.png train-text.txt test-image-file.png

Code Description:
------------------
1. Simple Model
2. Viterbi Model
3. Initial probability and Emission probability calculation
4. Final answer and Design considerations

Results: (test-19-0.png)
--------
Simple:    GINSBURG- BREYER- SOTOMAYOR- and KAGAN- JJ!- joined!

Viterbi:   GINSBURG- BREYER- SOTOMAYOR- and KAGAN- JJ!- joined!

Final answer:    

GINSBURG- BREYER- SOTOMAYOR- and KAGAN- JJ!- joined!

