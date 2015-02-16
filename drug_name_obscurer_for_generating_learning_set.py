__author__ = 'janos'


"""
The goal is to create an obsuficated list of drugs that appear in data integration issues in
Health Information Exchanges. As we are feeding from several different EHRs they will have different
structured and unstructured ways of passing on a drug list.

In generating this list the aim is to create a challenge set with a
gold standard for malformed medication lists so machine learning algorithms can be applied and
rigorously tested.

As an example of the drug names:
Lisinopril 10 mg
LISINOPRIL 10MG Tablet
10 mg Lisinopril
10mg Lisinopril
20  milligram capsule of lisinopril on a daily basis

ABILIFY (ARIPRAZOLE)
Lisinopril 10 mg daily

The list should include misspelling of common drug names and some lists should not include all tokens
but may include other garbage.

["BRAND_NAME", "GENERIC_NAME", "STRENGTH", "UNIT", "STRENGTH_UNIT", "FREQUENCY", "ROUTE", "FORM"]

The token ["UNCLASSIFIED"] should be applied to other tokens.

First example of a parse above would be:

["10mg", "Lisinopril"], ["QUANTITY_UNIT", "BRAND_NAME"] ->
{"STRENGTH_UNIT" -> "10mg", "BRAND_NAME": "lisinopril"} ->
{"STRENGTH": "10", "UNIT": "mg", "BRAND_NAME": "lisinopril"}

A second example:
"10  milligram capsule of hydrochlorothiazide HCL on a daily basis"
{"STRENGTH": "10", "UNIT": "milligram", "GENERIC_NAME": " hydrochlorothiazide HCL", "FREQUENCY": "on a daily basis"}

We may also get to put garbage in the list as an example:

"100 mg"
"5 STROKE"
"5555787987987"

"""

import random

FREQUENCY_VOCABULARY = ["taken daily", "TAKEN DAILY" "takes twice daily", "as needed"]

TOKENS_TO_CAPTURE = ["BRAND_NAME", "GENERIC_NAME", "STRENGTH", "UNIT", "STRENGTH_UNIT", "FREQUENCY", "ROUTE", "FORM"]

TOKEN_TEMPLATES = [["BRAND_NAME"], ["GENERIC_NAME"], ["STRENGTH_UNIT", "BRAND_NAME"], ["STRENGTH_UNIT", "GENERIC_NAME"],
                   ["STRENGTH", "UNIT", "GENERIC_NAME"], ["STRENGTH", "UNIT", "BRAND_NAME"],
                   ["GENERIC_NAME", "STRENGTH", "UNIT"], ["BRAND_NAME", "STRENGTH", "UNIT"],
                   ["GENERIC_NAME", "STRENGTH", "UNIT", "FORM"], ["BRAND_NAME", "STRENGTH", "UNIT", "FORM"],
                   ["GENERIC_NAME", "FREQUENCY"], ["BRAND_NAME", "FREQUENCY"],
                   ["ROUTE", "GENERIC_NAME", "FREQUENCY"], ["ROUTE", "BRAND_NAME", "FREQUENCY"],
                   ["ROUTE", "GENERIC_NAME", "STRENGTH", "UNIT", "FREQUENCY"],
                   ["ROUTE", "BRAND_NAME", "STRENGTH", "UNIT", "FREQUENCY"],
                   ["ROUTE", "GENERIC_NAME", "STRENGTH_UNIT", "FREQUENCY"],
                   ["ROUTE", "GENERIC_NAME", "STRENGTH_UNIT", "FREQUENCY"],
                   ["BRAND_NAME", "(", "GENERIC_NAME", ")"],
                   ["BRAND_NAME", "(", "GENERIC_NAME", ")", "STRENGTH_UNIT", "FORM", "-"],
                   ["BRAND_NAME", "-", "GENERIC_NAME"],
                   ["ROUTE", "GENERIC_NAME", "STRENGTH_UNIT", "FREQUENCY"],
                   ]

SPELLING_ERROR_RATE = [0.99, 0.995, 1.00]
SPACING_PROBABILITY = [0.99, 0.995, 1.00]
SPACING_BEFORE_AFTER = {"(": (1, 0), ")": (0, 1), "-": (1, 1), ",": (0, 1)}


def generate_collapsed_drug_name(drug_detail_dict, token_pattern):
    generated_drug_string = ""
    i = 0

    for token in token_pattern:
        if token in drug_detail_dict:
            extracted_token = drug_detail_dict[token]
        else:
            extracted_token = token

        if token in SPACING_BEFORE_AFTER:
            spacing = SPACING_BEFORE_AFTER[token]
            if spacing[0]:
                generated_drug_string += " "
            else:
                generated_drug_string = generated_drug_string.rstrip()

        generated_drug_string += extracted_token

        if token in SPACING_BEFORE_AFTER:
            spacing = SPACING_BEFORE_AFTER[token]
            if not spacing[1]:
                generated_drug_string = generated_drug_string.rstrip()
            else:
                generated_drug_string += " "
        else:
            generated_drug_string += " "

        i += 1

    #TODO: Add random spacing
    #TODO: Add random spelling mistakes

    return " ".join(generated_drug_string.rstrip().split())


if __name__ == "__main__":
    drug_dict = {"GENERIC_NAME": "Atorvastatin", "BRAND_NAME": "LIPITOR", "STRENGTH": "10", "UNIT": "mg", "FREQUENCY": "once daily",
                 "STRENGTH_UNIT": "5MG", "ROUTE": "ORAL", "FORM": "TABLET"}

    print(drug_dict)
    print(generate_collapsed_drug_name(drug_dict,  ["GENERIC_NAME", "STRENGTH", "UNIT"]))
    print(generate_collapsed_drug_name(drug_dict,  ["BRAND_NAME", "(", "GENERIC_NAME", ")"]))
    print(generate_collapsed_drug_name(drug_dict,  ["STRENGTH_UNIT", "BRAND_NAME", "(", "GENERIC_NAME", ")", "FREQUENCY"]))