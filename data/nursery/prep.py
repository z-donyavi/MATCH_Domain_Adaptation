from io import BytesIO
from zipfile import ZipFile
import urllib.request
import pandas as pd


def prep_data(binned=False):
    url = urllib.request.urlopen(
        "https://archive.ics.uci.edu/static/public/76/nursery.zip")
    
    colnames = ["att" + str(i + 1) for i in range(9)]

    my_zip_file = ZipFile(BytesIO(url.read()))
    f = my_zip_file.namelist()[1]

    dta = pd.read_csv(my_zip_file.open(f),
                      header=None,
                      names=colnames,
                      skipinitialspace=True)

    dta.att9 = dta.att9.replace({"not_recom": 0, "recommend": 1, "very_recom": 1, "priority": 1, "spec_prior": 2})
    dta = pd.get_dummies(dta)

    return dta

# dta.to_pickle("dta.pkl")


