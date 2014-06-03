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
     0.230303
a    0.092929
i    0.072727
e    0.070707
n    0.066667
s    0.050505
r    0.044444
o    0.044444
t    0.042424
l    0.036364
x    0.030303
c    0.028283
v    0.026263
p    0.026263
y    0.020202
g    0.018182
b    0.018182
z    0.014141
u    0.014141
d    0.012121
3    0.008081
f    0.006061
h    0.006061
m    0.006061
1    0.004040
-    0.004040
k    0.004040
q    0.002020
dtype: float64
Frequency of characters in beer names:
()
Frequency of brand name / Frequency of beer name
/Users/janos/anaconda/lib/python2.7/site-packages/pandas/core/config.py:570: DeprecationWarning: height has been deprecated.

  warnings.warn(d.msg, DeprecationWarning)
   frequency_b  frequency_d  ratio of frequency
i     0.043436     0.072727            1.674343
n     0.029923     0.066667            2.227957
g     0.011583     0.018182            1.569697
y     0.007722     0.020202            2.616162
3     0.001931     0.008081            4.185859
z     0.001931     0.014141            7.325253
v     0.001931     0.026263           13.604040
x     0.001931     0.030303           15.696970
1     0.000965     0.004040            4.185859
-     0.000965     0.004040            4.185859
q     0.000965     0.002020            2.092929

Frequency of different string lengths for brand drug names:
7     0.40
6     0.20
9     0.10
8     0.08
5     0.08
15    0.06
13    0.04
10    0.04
dtype: float64

Frequency of different string length for beer names:
10    0.12
16    0.10
21    0.08
19    0.08
28    0.08
6     0.06
12    0.06
14    0.06
9     0.04
11    0.04
13    0.04
31    0.04
18    0.04
17    0.04
15    0.02
22    0.02
24    0.02
25    0.02
26    0.02
27    0.02
dtype: float64
             precision    recall  f1-score   support

       beer       0.94      0.88      0.91        17
 brand drug       0.88      0.94      0.91        16

avg / total       0.91      0.91      0.91        33

Confusion matrix:
[[15  2]
 [ 1 15]]

Process finished with exit code 0


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
    drug_character_counts = drug_character_df["character"].value_counts(normalize=True)
    print(drug_character_counts)

    beers = read_beer_as_frame()

    random_beer_name_list = get_random_n_cleaned_names(beers["name"], n=50)
    beer_character_df = get_characters_as_data_frame(random_beer_name_list)
    print("Frequency of characters in beer names:")
    beer_character_counts = beer_character_df["character"].value_counts(normalize=True)

    beer_character_counts_df = pd.DataFrame(beer_character_counts, columns=["frequency_b"])
    drug_character_counts_df = pd.DataFrame(drug_character_counts, columns=["frequency_d"])

    bccd = beer_character_counts_df.join(drug_character_counts_df)

    bccd["ratio of frequency"] = bccd["frequency_d"] / bccd["frequency_b"]

    print()
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
