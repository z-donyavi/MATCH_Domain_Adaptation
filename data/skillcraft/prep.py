from io import BytesIO
from zipfile import ZipFile
import urllib.request
import pandas as pd

def prep_data(binned=False):
    url = urllib.request.urlopen("https://archive.ics.uci.edu/static/public/272/skillcraft1+master+table+dataset.zip")

    my_zip_file = ZipFile(BytesIO(url.read()))
    f = my_zip_file.namelist()[-1]

    dta = pd.read_csv(my_zip_file.open(f),
                      header=0,
                      skipinitialspace=True,
                      na_values="?")

    dta = dta.dropna()
    dta = dta.drop("GameID", axis=1)

    bins = [0, 3, 5, 8]
    labels = [1, 2, 3]
    dta['LeagueIndex'] = pd.cut(dta['LeagueIndex'], bins=bins, labels=labels)
    dta['LeagueIndex'] = dta['LeagueIndex'].astype("int64")

    if binned:
        for col in list(dta)[1:]:
            dta[col] = pd.qcut(dta[col], q=4, labels=False, duplicates='drop')
            dta[col] = dta[col].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")



