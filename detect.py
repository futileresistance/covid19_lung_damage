from utils import download_items
download_items()

from covid_segm_model.covid_model import model as covid_model
from lungmask.lung_model import model as lung_model
from utils import count_injury_percentage

patient_scans_path = 'test_cases/ct/coronacases_org_001.nii'
_, results = count_injury_percentage(patient_scans_path, lung_model, covid_model) 
print(results)


