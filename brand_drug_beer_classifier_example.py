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
letter 'x' occurs in 2.3% of the characters and while in the
beer names occurs in 0.3% of the characters.
"""

"""
Frequency of characters in drug names:
     0.235714
a    0.103571
e    0.066071
n    0.060714
i    0.058929
t    0.050000
o    0.046429
v    0.044643
r    0.042857
s    0.033929
l    0.032143
p    0.023214
x    0.023214
u    0.019643
f    0.019643
c    0.019643
h    0.017857
g    0.016071
b    0.014286
y    0.014286
m    0.014286
k    0.010714
z    0.010714
3    0.005357
d    0.003571
w    0.003571
-    0.003571
1    0.001786
j    0.001786
q    0.001786
dtype: float64
Frequency of character in beer names:
     0.267191
e    0.101179
a    0.068762
r    0.054028
l    0.053045
i    0.052063
o    0.049116
t    0.047151
b    0.034381
n    0.034381
s    0.031434
c    0.025540
u    0.024558
d    0.024558
p    0.023576
h    0.021611
f    0.012770
k    0.012770
m    0.012770
w    0.010806
g    0.009823
y    0.007859
x    0.002947
9    0.002947
.    0.002947
z    0.002947
j    0.001965
q    0.001965
'    0.001965
v    0.000982
1    0.000982
-    0.000982
dtype: float64
             precision    recall  f1-score   support

       beer       0.91      0.77      0.83        13
 brand drug       0.86      0.95      0.90        20

avg / total       0.88      0.88      0.88        33

Confusion matrix:
[[10  3]
 [ 1 19]]

"""

import csv
import random
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix


def read_top_100_drugs(filename="top_100_drugs.csv"):
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
    which is licensed under the Open Drug Database.
    """
    beers = pd.read_csv("beers.csv")
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
    print(drug_character_df["character"].value_counts(normalize=True))

    beers = read_beer_as_frame()

    random_beer_name_list = get_random_n_cleaned_names(beers["name"], n=50)
    beer_character_df = get_characters_as_data_frame(random_beer_name_list)
    print("Frequency of characters in beer names:")
    print(beer_character_df["character"].value_counts(normalize=True))

    drug_name_df = pd.DataFrame(random_drug_name_list, columns=["name"])
    beer_name_df = pd.DataFrame(random_beer_name_list, columns=["name"])

    drug_name_df["name_length"] = drug_name_df["name"].apply(lambda x: len(x))
    beer_name_df["name_length"] = beer_name_df["name"].apply(lambda x: len(x))

    print("Frequency of different string lengths for brand drug names:")
    print(drug_name_df["name_length"].value_counts(normalize=True))
    print("Frequence of different string length for beer names:")
    print(beer_name_df["name_length"].value_counts(normalize=True))

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
