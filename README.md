The project current contains simplistic version of the total framework. 

Installation
============
Before doing anything, install the required packages. Note that, `requirements.txt` contains packages borrowed the developer's environment. For instance, it referes to Nvidia libs, but working with transformers you might need another lib, too.

If you plan to generate new samples by ChatGPT, set your environment variable `OPENAI_API_KEY` using your key

Software structure
==================

    |- workflow.py  -- main file to run. Setup which module you want to run...
    |- chatgpt.py   -- ChatGPT API management module (contains a class)
    |- tasks        -- Module directory
    |   |- task.py                    -- Base Task class
    |   |- counterfactual.py          -- Counterfactual task class
    |   |- counterfactual_semantic.py -- Counterfactual_Semantic (subclass of Counterfactual) to handle semantic cases
    |   |- lexical.py                 -- Lexical task class
    |   |- semantic.py                -- Semantic task class
    |   |- syntactic.py               -- Syntactic task class
    |   |- terms.py                   -- Term task class
    |   |- testing.py                 -- Testing task class


Database
========

Sample SQLite3 output database `original.db` is provided. It contains the following data:

Tables
------

# Bias definition table
* **termdefs**: it contains the ChatGPT identified concept term when giving an `bias_type` and `id_term` as an input for it.
  * **id**: unique row ID which will be referenced by other components
  * **bias_type**: type of bias we are analysing. It is user defined.
  * **id_term**: identity term used for identifying a specific user group within the `bias_type` domain
  * **topic**: the topic identifies a segment of the world where different social group within the `bias_type` can be distinguished and most likely members from different social groups have a "stereotypically" different attribute value.
  * **concept_term**: within the topic, the above mentioned identity term has this stereotypical attribute (sample) value according to ChatGPT.

# Module tables
Each module has basically the same structure of table:
* initial stereotype table (`<basetable>`):
  * **id**: unique row ID
  * **refid**: reference to a previous model row ID from where the data is generated/derived. In the case of `baseline` it points to the `termdefs` table's row ID. Otherwise, it points to the `baseline`'s row ID
  * **bias_type**: same as in termdefs
  * **id_term**: same as in termdefs
  * **concept_term**: same as in termdefs (except the `semantic` level where there is no such attribute)
  * **sentence**: sentence (test case) we are working with
  * **flagged**: default value 0 (meaning: there seems to be an OK sample), 1 (meaning: it will not work for testing because it does not meet counterfactual fairness specification), 2 (meanining: it is rather because of user error)
* counterfactual fairness table (usually `counterfact_<basetable>`)
  * **id**: unique row ID
  * **refid**: reference to the `<basetable>` row id
  * **bias_type**: same as in termdefs
  * **id_term**: same as in termdefs
  * **concept_term**: same as in termdefs (except the `semantic` level where there is no such attribute)
  * **sentence**: (counterfactual) sentence (other side of the test case) we are working with
  * **flagged**: default value 0 (meaning: there seems to be an OK sample), 1 (meaning: it will not work for testing because it does not meet counterfactual fairness specification), 2 (meanining: it is rather because of user error)
* unified data view (usually `<basetable>_data`): union of the `<basetable>` and the `counterfact_<basetable>` where the `flagged` attribute is ommitted. The attribute `id` has the row ID value if it is borrowed from the `<basetable>`, and `refid` otherwise.
* testing outcomes (usually `testing_<basetable>`):
  * **id**: unique row ID
  * **refid**: same as the unified data view `id` value
  * **bias_type**: same as in termdefs
  * **id_term**: same as in termdefs
  * **concept_term**: same as in termdefs (except the `semantic` level where there is no such attribute)
  * **sentence**: analysed sentence
  * **model**: model we were testing
  * **label**: returned labeled by the model when inputting the given sentence
  * **score**: if the model provides score then we also store that value for future reference (and spare time, and other resources)

We have different modules, that is we have the following values for `<basetable>`:
* **baseline**: the naive generation of test sentences based on `termdefs` data
* **lexical**: lexical augmentation of the `baseline` sentences
* **syntactic**: syntactic augmentation of the `baseline` sentences
* **semantic**: semantic augmentation of the `baseline` sentences

