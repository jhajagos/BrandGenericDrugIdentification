__author__ = 'janos'

import sys
sys.path.append("..")

import drug_name_obscurer_for_generating_learning_set as dno


import unittest


class TestDrugNameObscurer(unittest.TestCase):
    def setUp(self):
        self.example_drug_name = {"GENERIC_NAME": "Atorvastatin", "BRAND_NAME": "LIPITOR",
                                              "STRENGTH": "5", "UNIT": "mg", "FREQUENCY": "once daily",
                                              "STRENGTH_UNIT": "5MG", "FORM": "TABLET", "ROUTE": "ORAL"}

    def test_drug_name_obscurer(self):
        compressed_name1 = dno.generate_collapsed_drug_name(self.example_drug_name, ["BRAND_NAME", "(", "GENERIC_NAME",
                                                                                     ")", "STRENGTH_UNIT", "FORM", "-",
                                                                                     "FREQUENCY"])

        self.assertEquals("LIPITOR (Atorvastatin) 5MG TABLET - once daily", compressed_name1)

        compressed_name2 = dno.generate_collapsed_drug_name(self.example_drug_name, ["BRAND_NAME", " ( ", "GENERIC_NAME",
                                                                                     " ) ", "STRENGTH_UNIT", "FORM", "-",
                                                                                     "FREQUENCY"])

        self.assertEquals("LIPITOR ( Atorvastatin ) 5MG TABLET - once daily", compressed_name2)

        compressed_name3 = dno.generate_collapsed_drug_name(self.example_drug_name, ["BRAND_NAME", ",", "FREQUENCY"])
        self.assertEquals("LIPITOR, once daily", compressed_name3)

        compressed_name4 = dno.generate_collapsed_drug_name(self.example_drug_name, ["BRAND_NAME", ",", "FREQUENCY"])
        self.assertEquals("LIPITOR, once daily", compressed_name4)

        compressed_name5 = dno.generate_collapsed_drug_name(self.example_drug_name, ["ROUTE", "BRAND_NAME", "FREQUENCY"])
        self.assertEquals("ORAL LIPITOR once daily", compressed_name5)

        compressed_name6 = dno.generate_collapsed_drug_name(self.example_drug_name, ["GENERIC_NAME", "STRENGTH", "UNIT"])
        self.assertEquals("Atorvastatin 5 mg", compressed_name6)