__author__ = 'janos'

"""
In this example a classifier is built that can distinguish
names of beer from brand names of drugs.

I wanted a somewhat realistic not canned example to learn
basic classification and testing in scikit-learns.

The beer names act as corpus of non-medical words. The branded
drugs are derived from a top 100 list of branded drugs.
A more complete list of branded drugs could be derived from
RxNorm current prescribe content.

To run install pandas and sklearn (scikit) on your machine or use
the simple to install Anaconda python distribution.

I found that the ngram_range=(1,1) worked better as indicated by
higher precision than either nagram_range(1,2) and ngram_range(2,2)

Branded drug names which are basically made up words for marketing
purposes should have a different statistical pattern of
character frequency. In the sample run of the program is given below the
letter 'x' occurs in 0.1% of the characters and while in the
beer names occurs in 3% of the characters.
"""

"""
Frequency of characters in drug names:
     0.221014
e    0.083333
a    0.074275
i    0.063406
t    0.063406
n    0.063406
s    0.056159
l    0.052536
o    0.052536
r    0.043478
c    0.034420
v    0.032609
p    0.021739
b    0.019928
x    0.018116
u    0.018116
g    0.014493
y    0.014493
m    0.012681
h    0.012681
d    0.010870
w    0.005435
z    0.005435
q    0.003623
j    0.001812
dtype: float64
Frequency of characters in beer names:
     0.275100
e    0.088353
a    0.083333
r    0.064257
l    0.053213
o    0.047189
t    0.046185
i    0.035141
b    0.034137
s    0.031124
c    0.029116
n    0.027108
h    0.025100
p    0.023092
u    0.020080
d    0.020080
k    0.019076
w    0.015060
g    0.014056
y    0.012048
m    0.011044
z    0.005020
f    0.004016
.    0.003012
v    0.003012
j    0.002008
-    0.002008
'    0.002008
x    0.001004
7    0.001004
4    0.001004
/    0.001004
!    0.001004
dtype: float64

Frequency of brand name / Frequency of beer name
/Users/janos/anaconda/lib/python2.7/site-packages/pandas/core/config.py:570: DeprecationWarning: height has been deprecated.

  warnings.warn(d.msg, DeprecationWarning)
   frequency_b  frequency_d  ratio of frequency
i     0.035141     0.063406            1.804348
s     0.031124     0.056159            1.804348
n     0.027108     0.063406            2.338969
v     0.003012     0.032609           10.826087
x     0.001004     0.018116           18.043478

Frequency of different string lengths for brand drug names:
7     0.42
8     0.18
15    0.10
6     0.10
18    0.06
9     0.06
10    0.04
11    0.02
5     0.02
dtype: float64

Frequency of different string length for beer names:
21    0.12
11    0.08
13    0.08
25    0.06
19    0.06
18    0.06
17    0.06
14    0.06
16    0.04
7     0.04
20    0.04
8     0.04
15    0.04
10    0.04
5     0.02
6     0.02
34    0.02
12    0.02
27    0.02
22    0.02
23    0.02
24    0.02
3     0.02
dtype: float64
             precision    recall  f1-score   support

       beer       1.00      0.88      0.94        17
 brand drug       0.89      1.00      0.94        16

avg / total       0.95      0.94      0.94        33

Confusion matrix:
[[15  2]
 [ 0 16]]

"""

import csv
import random
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix


def read_top_100_drugs(filename="./data/top_100_drugs.csv"):
    """List of 2013 top drugs from: http://www.medscape.com/viewarticle/820011
    Data is originally from IMS Health
    """

    with open(filename) as f:
        dict_reader = csv.DictReader(f, delimiter="\t")
        drug_list = []
        for row in dict_reader:
            drug_list += [row]
    return drug_list


def drug_names_on_drug_list(drug_list):
    """Create a list of drug names"""
    return [dl["Drug (brand name)"] for dl in drug_list]


def build_analyzer(ngram_range=(1, 1)):
    """Build an analyzer"""
    cv = CountVectorizer(analyzer='char_wb', ngram_range=ngram_range)
    return cv.build_analyzer()


def build_frequency_list(name_list):
    """Generate a list of each char"""
    analyzer = build_analyzer()
    char_list = []
    for name in name_list:
        char_list += analyzer(name)
    return char_list


def read_beer_as_frame():
    """The CSV file "beers.csv" is from http://openbeerdb.com/
    which is licensed under the Open Database License.
    """
    beers = pd.read_csv("./data/beers.csv")
    return beers


def get_random_name(name_list):
    """Beer list is rather large so we sample from the beer names. Also it contains blanks, nan, etc"""
    length_of_list = len(name_list)
    while True:
        random_index = random.randrange(0, length_of_list-1)
        selected_name = name_list[random_index]
        if selected_name.__class__ == "".__class__:
            max_ord_value = max([ord(c) for c in selected_name])
            if max_ord_value < 128 and len(selected_name.strip()) > 0:
                break

    return selected_name


def get_random_n_cleaned_names(name_list, n=100):
    """Select a random subset"""
    random_name_list = []
    for i in range(n):
        random_name_list += [get_random_name(name_list)]

    return random_name_list


def get_characters_as_data_frame(name_list):
    character_list = build_frequency_list(name_list)
    return pd.DataFrame(character_list, columns=["character"])


def main():
    drug_name_list = drug_names_on_drug_list(read_top_100_drugs())
    random_drug_name_list = get_random_n_cleaned_names(drug_name_list, n=50)
    drug_character_df = get_characters_as_data_frame(random_drug_name_list)

    print("Frequency of characters in drug names:")
    drug_character_counts = drug_character_df["character"].value_counts(normalize=True)
    print(drug_character_counts)

    beers = read_beer_as_frame()

    random_beer_name_list = get_random_n_cleaned_names(beers["name"], n=50)
    beer_character_df = get_characters_as_data_frame(random_beer_name_list)
    print("Frequency of characters in beer names:")
    beer_character_counts = beer_character_df["character"].value_counts(normalize=True)
    print(beer_character_counts)
    beer_character_counts_df = pd.DataFrame(beer_character_counts, columns=["frequency_b"])
    drug_character_counts_df = pd.DataFrame(drug_character_counts, columns=["frequency_d"])

    bccd = beer_character_counts_df.join(drug_character_counts_df)

    bccd["ratio of frequency"] = bccd["frequency_d"] / bccd["frequency_b"]

    print("")
    print("Frequency of brand name / Frequency of beer name")
    print(bccd[bccd["ratio of frequency"] >= 1.5])

    drug_name_df = pd.DataFrame(random_drug_name_list, columns=["name"])
    beer_name_df = pd.DataFrame(random_beer_name_list, columns=["name"])

    drug_name_df["name_length"] = drug_name_df["name"].apply(lambda x: len(x))
    beer_name_df["name_length"] = beer_name_df["name"].apply(lambda x: len(x))
    print("")
    print("Frequency of different string lengths for brand drug names:")
    drug_name_letter_frequency = drug_name_df["name_length"].value_counts(normalize=True)

    print(drug_name_letter_frequency)
    print("")

    print("Frequency of different string length for beer names:")
    beer_name_letter_frequency = beer_name_df["name_length"].value_counts(normalize=True)
    print(beer_name_letter_frequency)

    drug_name_df["class"] = "brand drug"
    beer_name_df["class"] = "beer"

    names_df = pd.concat([drug_name_df, beer_name_df])

    names = names_df["name"].values
    class_of_names = names_df["class"].values

    x_train, x_test, y_train, y_test = train_test_split(names, class_of_names, test_size=0.33)

    cv = CountVectorizer(analyzer="char_wb", ngram_range=(1, 1))

    x_train_ngram = cv.fit_transform(x_train)

    svmi = svm.SVC()
    svmi.fit(x_train_ngram, y_train)

    y_test_predict = svmi.predict(cv.transform(x_test))

    print(classification_report(list(y_test), list(y_test_predict)))

    print("Confusion matrix:")
    print(confusion_matrix(list(y_test), list(y_test_predict)))


if __name__ == "__main__":
    main()
