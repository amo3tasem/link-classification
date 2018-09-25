# Project Description
This is a text classification model designed to take a link as an input, and return if this webpage topic is related to an Arabic series called ‘Ra7eem’ or not.

## Dataset
Dataset for this project is not mine, and I have no right to upload it here, but I'm sure this approach should work with other arabic annotated link-based datasets.

## Exports
I'v exported ```Ra7eem.pkl```, ```count_vect.pkl```, and ```tfidf.pkl``` if anyone wants to experiment with the model you can import them using sklear ```joblib``` feature.

## Dataset Description
You are given a small dataset in a csv format of around 900 links. This dataset is divided in approximately
50%-50% portions in which the first portion contains links related to the TV show “Ra7em - رحیم ” series and the
other portion of links are not related to this series. (The CSV file has two columns, the first corresponds to links
column and the second corresponds to class of the link where 1 means related and 0 means not related).


```python
import pandas as pd
import numpy as np
```


```python
df = pd.read_csv('Dataset.csv')
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>link</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.4helal.tv/video/series-Rahim-01.html</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>http://krmalk.tv/video/watch.php?vid=8e54b1d51</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://www.mzarita.tv/video/watch.php?vid=dfa...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.mzarita.tv/video/watch.php?vid=157...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.mzarita.tv/video/watch.php?vid=c78...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Feature Selection (Web Scraping)
### For this task I decided to load all text from each webpage, as there's no regular structure
You are required to extract some features from each link and its web page source that can be used to classify
the link either being related to this series or not.


```python
result = []
where_to_start_again = -1
indexes_dropped = []
```


```python
import re
import requests
from bs4 import BeautifulSoup
should_restart = True
while should_restart:    
    should_restart = False
    for index, row in df[where_to_start_again+1:].iterrows():
        where_to_start_again = index
        try:
            html = requests.get(row['link'])
            headers = {'User-Agent':'Mozilla/5.0'}
            soup = BeautifulSoup(html.text, "html.parser")
            data = soup.findAll(text=True)

            def visible(element):
                if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
                    return False
                elif re.match('<!--.*-->', str(element.encode('utf-8'))):
                    return False
                return True
            print(index, row['link'])
            result.append([list(filter(visible, data)),row['class']])
        except:
            indexes_dropped.append(where_to_start_again)
            should_restart = True
            break
        
```


```python
scraped_df = pd.DataFrame(result, columns=['text', 'class'])
```

## Feature Engineering
### Text cleaning and stemming

```python
scraped_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[
, 
, 
, 
, 
, 
, 
, 
, 
, 
, 
, 
, 
, 
, 
, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[
, &lt;![endif], 
, 
, 
, 
, 
, 
, 
, [if lt IE ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[
, &lt;![endif], 
, 
, 
, 
, 
, 
, 
, 
, 
, 
, 
...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[
, &lt;![endif], 
, 
, 
, 
, 
, 
, 
, 
, 
, 
, 
...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[
, &lt;![endif], 
, 
, 
, 
, 
, 
, 
, 
, 
, 
, 
...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Single text for each cell join


```python
scraped_df.text = [' '.join(x) for x in scraped_df.text]
```


```python
scraped_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>\n \n \n \n \n \n \n \n \n \n \n \n \n \n \n \...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>\n &lt;![endif] \n \n \n \n \n \n \n [if lt IE 9]...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>\n &lt;![endif] \n \n \n \n \n \n \n \n \n \n \n ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>\n &lt;![endif] \n \n \n \n \n \n \n \n \n \n \n ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>\n &lt;![endif] \n \n \n \n \n \n \n \n \n \n \n ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Remove non-arabic characters


```python
import re
scraped_df.text = [' '.join(re.sub(r'[^\u0600-\u06FF]', ' ', x).split()) for x in scraped_df.text]
```


```python
scraped_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>تسجيل الدخول افلام اجنبية انواع الافلام سلاسل ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>تسجيل دخول تسجيل دخول كلمة المرور تسجيل دخول ن...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>الصفحة الرئيسية الجديد افلام المزاريطة افلام ع...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>الصفحة الرئيسية الجديد افلام المزاريطة افلام ع...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>الصفحة الرئيسية الجديد افلام المزاريطة افلام ع...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Removing special characters


```python
scraped_df.text = [x.replace('؟','').replace('،','').replace('؛','').replace(',','').replace('ـ','') for x in scraped_df.text]
```


```python
scraped_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>تسجيل الدخول افلام اجنبية انواع الافلام سلاسل ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>تسجيل دخول تسجيل دخول كلمة المرور تسجيل دخول ن...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>الصفحة الرئيسية الجديد افلام المزاريطة افلام ع...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>الصفحة الرئيسية الجديد افلام المزاريطة افلام ع...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>الصفحة الرئيسية الجديد افلام المزاريطة افلام ع...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Removing Arabic stop-words
* Using nltk arabic stopwords corpus
* Filtering stop-words helps preventing redundant features



```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\amoat\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True




```python
arabic_stopwords = list(set(nltk.corpus.stopwords.words("arabic")))

```


```python
def remove_stop_words(text):
    filtered_word_list = text #make a copy of the word_list
    for word in text: # iterate over word_list
        if word in arabic_stopwords: 
            filtered_word_list.remove(word) # remove word from filtered_word_list if it is a stopword
    return filtered_word_list
```


```python
scraped_df.text = [remove_stop_words(x) for x in scraped_df.text]
```


```python
scraped_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>تسجيل الدخول افلام اجنبية انواع الافلام سلاسل ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>تسجيل دخول تسجيل دخول كلمة المرور تسجيل دخول ن...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>الصفحة الرئيسية الجديد افلام المزاريطة افلام ع...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>الصفحة الرئيسية الجديد افلام المزاريطة افلام ع...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>الصفحة الرئيسية الجديد افلام المزاريطة افلام ع...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
temp = scraped_df.copy()
```


```python
scraped_df = temp
```

# Modeling Starts here

Very good term frequency inverse document frequency [tutorial](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

### Count Vectorizer


```python
X_train = scraped_df['text']
y_train = scraped_df['class']
```


```python
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape
```




    (885, 11825)




```python
count_vect.vocabulary_.get(u'رحيم')
```




    6148



### Term Frequencies


```python
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
```




    (885, 11825)



### Model Selection


```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

models = [('LR', LogisticRegression()),
         ('KNN', KNeighborsClassifier()),
         ('CART', DecisionTreeClassifier()),
         ('NB', GaussianNB()),
         ('SVM', SVC()),
         ('RF', RandomForestClassifier())]
seed = 1073
results = []
names = []
scoring = 'accuracy'
X = X_train_tfidf.toarray()
Y = y_train
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
```

    LR: 0.870110 (0.045558)
    KNN: 0.742377 (0.201805)
    CART: 0.893948 (0.048822)
    NB: 0.805094 (0.117445)
    SVM: 0.034065 (0.091057)
    RF: 0.867901 (0.060408)
    

## Random Forest had good accuracy
* So let's setup a grid for its hyperparameters and see if we can acheive even better accuracy


```python
import pprint
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 250, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 7)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
random_grid
```




    {'bootstrap': [True, False],
     'max_depth': [10, 26, 43, 60, 76, 93, 110, None],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10],
     'n_estimators': [50, 100, 150, 200, 250]}




```python
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X, Y)

print( rf_random.best_estimator_ )
print( rf_random.best_score_ )
print( rf_random.best_params_ )
```

    Fitting 5 folds for each of 100 candidates, totalling 500 fits
    

    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:   22.4s
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed:  1.7min
    [Parallel(n_jobs=-1)]: Done 349 tasks      | elapsed:  4.3min
    [Parallel(n_jobs=-1)]: Done 500 out of 500 | elapsed:  6.1min finished
    

    RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=2, min_samples_split=10,
                min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
    0.900564971751
    {'n_estimators': 250, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': None, 'bootstrap': False}
    

## Final Training with best parameters

* The grid pumbed the cross-validation accuracy to 90%
* This grid could  go deeper and probably better accuracy but due to computional limitiations.


```python
clf = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=76, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
clf.fit(X,Y)
```




    RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                max_depth=76, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=2, min_samples_split=10,
                min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)



# Testing Pipeline


```python
import random
urls_to_predict = df.sample(10)
```


```python
def scrape_links_to_predict(urls_to_predict):    
    result = []
    for link in urls_to_predict:
        try:
            html = requests.get(link)
            headers = {'User-Agent':'Mozilla/5.0'}
            soup = BeautifulSoup(html.text, "html.parser")
            data = soup.findAll(text=True)

            def visible(element):
                if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
                    return False
                elif re.match('<!--.*-->', str(element.encode('utf-8'))):
                    return False
                return True
            result.append(list(filter(visible, data)))
        except:
            print('Site won\'t allow scraping this link: ', link)
            
            continue
    return result
```


```python
def stemming_text(text_to_predict):
    text_to_predict = [' '.join(x) for x in text_to_predict]
    text_to_predict = [' '.join(re.sub(r'[^\u0600-\u06FF]', ' ', x).split()) for x in text_to_predict]
    text_to_predict = [x.replace('؟','').replace('،','').replace('؛','').replace(',','').replace('ـ','') for x in text_to_predict]
    text_to_predict = [remove_stop_words(x) for x in text_to_predict]
    return text_to_predict
```


```python
def to_tfidf(text_to_predict):
    cv_to_predict = count_vect.transform(text_to_predict)
    tfidf_to_predict = tfidf_transformer.transform(cv_to_predict)
    return tfidf_to_predict
```


```python
def prepare_to_predict(urls_to_predict):
    text_to_predict = stemming_text(scrape_links_to_predict(urls_to_predict))
    return to_tfidf(text_to_predict)
```


```python
clf.predict(prepare_to_predict(['https://www.youtube.com/watch?v=vnqz19l2N8M']))
```




    array([1], dtype=int64)




```python
clf.predict(prepare_to_predict(['https://www.elcinema.com/work/1010439']))

```




    array([1], dtype=int64)



elsayyad on YouTube (test)


```python
clf.predict(prepare_to_predict(['https://www.youtube.com/watch?v=6D8o3dGgQCE']))
```




    array([0], dtype=int64)



Exporting transformations for reproducibility
```python
from sklearn.externals import joblib
joblib.dump(clf, 'Ra7eem.pkl')
joblib.dump(count_vect, "count_vect.pkl")
joblib.dump(tfidf_transformer, "tfidf.pkl")
```
