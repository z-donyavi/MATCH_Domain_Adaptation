from io import BytesIO
from zipfile import ZipFile
import urllib.request
import pandas as pd


def prep_data(binned=False):
    url = urllib.request.urlopen(
        "https://archive.ics.uci.edu/static/public/262/turkiye+student+evaluation.zip")

    my_zip_file = ZipFile(BytesIO(url.read()))
    f = my_zip_file.namelist()[0]

    dta = pd.read_csv(my_zip_file.open(f),
                      sep=',',
                      skipinitialspace=True)

    dta = dta.drop(columns=['class', 'nb.repeat'])

    # dta.to_pickle("dta.pkl")

    return dta



