HEAD v1.0 (Spanish version)
===========================

References
-----------

Vilares, David and Gómez-Rodríguez, Carlos. "HEAD-QA: A Healthcare Dataset for Complex Reasoning", ACL (short papers), 2019. 

Content
-------



|_ *.gold -> A tsv gold file that maps question IDs to the ground truth answer ID to such question. One file per exam.
|_ HEAD.json -> It contains the whole data for HEAD-QA (used in the so-called 'unsupervised' setting).
|_ train_HEAD.json -> It contains the training set of HEAD-QA (used as the training set in the so-called 'supervised' setting) 
|_ dev_HEAD.json -> A json file containing the development set of HEAD-QA (used in the 'supervised' setting).
|_ test_HEAD.json -> A json file ctaining the test set of HEAD-QA (used in the 'supervised' setting).

Structure of the JSON files
---------------------------
|_version
|_language
|_exams
  |_Cuaderno_YEAR_1_CATEGORY-ACRONYM (many, each of them is an exam)
    |_name
    |_category (medicine, biology, chemistry, etc)
    |_year
    |_data (list): It contains the questions/answers for a given exam
      |_element (many)
        |_qid: ID of the question in the **PDF exam**
        |_qtext: question text
        |_image: Path to the image the question refers (if any, just a few questions from the medicine exams use them)
        |_ra: ID of the right answer (it's one of the aids, see bellow)
        |_answers (many)
          |_ aid: answer ID (within the question)
          |_ atext: answer text


Missing questions
-----------------
Note that some question IDs might be missing for some exams. This usually corresponds to questions that were considered invalid after the exam (because its formulation was wrong, confusing or not accurate, for example). We removed these IDs from HEAD-QA (note that the question ids refers to the original ID in the exam in PDF format and not after the questions were removed)

Legal requirements
-------------------

The questions have been designed by the Ministerio de Sanidad, Consumo y Bienestar Social (https://www.mscbs.gob.es/).

The Ministerio de Sanidad, Consumo y Biniestar Social allows the redistribution of the exams and their content under [certain conditions](https://www.mscbs.gob.es/avisoLegal/home.htm): 

- The denaturalization of the content of the information is prohibited in any circumstance.
- The user is obliged to cite the source of the documents subject to reuse.
- The user is obliged to indicate the date of the last update of the documents object of the reuse.

Date of the last update of the documents object of the reuse: January, 14th, 2019.
