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
letter 'x' occurs in 2.4% of the characters and while in the
beer names occurs in 0.1% of the characters.
"""

"""
Frequency of characters in drug names:
     0.222222
e    0.098148
a    0.085185
i    0.072222
n    0.061111
r    0.057407
t    0.050000
s    0.038889
o    0.035185
l    0.035185
v    0.033333
c    0.027778
x    0.024074
p    0.022222
h    0.016667
m    0.016667
d    0.016667
u    0.016667
b    0.014815
g    0.012963
f    0.009259
z    0.007407
k    0.007407
3    0.005556
y    0.003704
-    0.003704
q    0.001852
1    0.001852
w    0.001852
dtype: float64
Frequency of characters in beer names:
     0.272541
e    0.094262
a    0.073770
l    0.055328
o    0.051230
t    0.046107
i    0.045082
r    0.044057
s    0.040984
n    0.034836
b    0.027664
p    0.026639
c    0.022541
h    0.022541
u    0.018443
m    0.017418
y    0.017418
d    0.016393
g    0.013320
w    0.011270
k    0.009221
f    0.009221
z    0.005123
v    0.005123
"    0.002049
\    0.002049
-    0.002049
0    0.002049
1    0.002049
.    0.001025
)    0.001025
(    0.001025
x    0.001025
2    0.001025
7    0.001025
6    0.001025
j    0.001025
5    0.001025
dtype: float64
Frequency of different string lengths for brand drug names:
7     0.30
8     0.18
9     0.14
6     0.14
15    0.06
18    0.04
13    0.04
10    0.04
5     0.04
11    0.02
dtype: float64
Frequency of different string length for beer names:
12    0.12
16    0.08
8     0.08
22    0.08
17    0.08
14    0.08
10    0.06
20    0.06
15    0.06
21    0.04
19    0.04
11    0.04
6     0.02
7     0.02
9     0.02
37    0.02
33    0.02
24    0.02
25    0.02
32    0.02
3     0.02
dtype: float64
             precision    recall  f1-score   support

       beer       0.93      0.82      0.87        17
 brand drug       0.83      0.94      0.88        16

avg / total       0.88      0.88      0.88        33

Confusion matrix:
[[14  3]
 [ 1 15]]

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
    print("Frequency of different string length for beer names:")
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
