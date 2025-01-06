
from io import BytesIO
from zipfile import ZipFile
import urllib.request
import pandas as pd

def prep_data(binned=False):
   
    url = urllib.request.urlopen("https://archive.ics.uci.edu/static/public/30/contraceptive+method+choice.zip")

    my_zip_file = ZipFile(BytesIO(url.read()))
    f = my_zip_file.namelist()[0]
    
    colnames = ["Age", "Education", "HusbandEducation", "NumberChildren", "Islamic", "Work", "HusbandJob",
                "LivingStandard", "MediaExposure", "Contraceptive"]

    dta = pd.read_csv(my_zip_file.open(f),
                      header=None,
                      names=colnames,
                      skipinitialspace=True)

    dta = pd.get_dummies(dta, columns=["HusbandJob"])

    if binned:
        bins = [15, 25, 30, 35, 40, 50]
        labels = [1, 2, 3, 4, 5]
        dta['Age'] = pd.cut(dta['Age'], bins=bins, labels=labels)
        dta['Age'] = dta['Age'].astype("int64")

        bins = [-1, 0, 1, 2, 3, 5, 20]
        labels = [0, 1, 2, 3, 4, 5]
        dta['NumberChildren'] = pd.cut(dta['NumberChildren'], bins=bins, labels=labels)
        dta['NumberChildren'] = dta['NumberChildren'].astype("int64")

        # dta.to_pickle("dta_binned.pkl")

    return dta

# dta.to_pickle("dta.pkl")


