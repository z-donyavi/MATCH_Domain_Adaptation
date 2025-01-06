
from io import BytesIO
from zipfile import ZipFile
import urllib.request
import pandas as pd


def prep_data(binned=False):
    url = urllib.request.urlopen(
        "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip")
    
    my_zip_file = ZipFile(BytesIO(url.read()))
    f = my_zip_file.namelist()[0]

    dta = pd.read_csv(my_zip_file.open(f),
                      header=0,
                      skipinitialspace=True)

    dta = dta.drop(["date", "rv1", "rv2"], axis=1)

    bins = [0, 50, 100, 2000]
    labels = [1, 2, 3]
    dta['Appliances'] = pd.cut(dta['Appliances'], bins=bins, labels=labels)
    dta['Appliances'] = dta['Appliances'].astype("int64")

    if binned:
        for col in list(dta)[1:]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")

