
from io import BytesIO
from zipfile import ZipFile
import urllib.request
import pandas as pd

def prep_data(binned=False):
    url = urllib.request.urlopen(
        "https://archive.ics.uci.edu/static/public/373/drug+consumption+quantified.zip")
    colnames = ["att" + str(i) for i in range(32)]

    my_zip_file = ZipFile(BytesIO(url.read()))
    f = my_zip_file.namelist()[0]

    dta = pd.read_csv(my_zip_file.open(f),
                      header=None,
                      names=colnames,
                      skipinitialspace=True)

    dta = dta.drop(["att0"], axis=1)

    # dta.loc[dta['Class'] != "A", 'Class'] = "B"
    dta.att28 = dta.att28.replace({"CL0": 1,
                                   "CL1": 2,
                                   "CL2": 2,
                                   "CL3": 3,
                                   "CL4": 3,
                                   "CL5": 3,
                                   "CL6": 3})

    dta = pd.get_dummies(dta)

    if binned:
        for col in list(dta)[:12]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")

