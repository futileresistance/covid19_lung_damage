from utils import download_items
from covid_segm_model.covid_model import model as covid_model
from lungmask.lung_model import model as lung_model
from utils import count_injury_percentage_nii
from utils import count_injury_percentage_dcm
from utils import resultStrMasksImgBytesArrayToJson
from flask import Flask

patient_scans_path_nii = 'test_cases/ct/coronacases_org_001.nii'
patient_scans_path_dcm = 'test_cases/ct/coronacases_org_001.DCM'

#download_items()
#_, results,masks = count_injury_percentage_dcm(patient_scans_path_dcm, lung_model, covid_model) 
#resultStrMasksImgBytesArrayToJson(results,masks)
#print(resultStrMasksImgBytesArrayToJson(results,masks))

app = Flask(__name__)
@app.route("/")
def test_local_files():
    #download_items() nii files procedure
    #_, results,masks = count_injury_percentage_nii(patient_scans_path_nii, lung_model, covid_model) 
    #resultStrMasksImgBytesArrayToJson(results,masks)
    return '200 -  OK'

from flask import request
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(patient_scans_path_dcm)
        #download_items()
        _, results,masks = count_injury_percentage_dcm(patient_scans_path_dcm, lung_model, covid_model) 
        jso_string = resultStrMasksImgBytesArrayToJson(results,masks)
        print('return ',jso_string)
        return jso_string 



