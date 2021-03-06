import nibabel as nib
import numpy as np
import cv2
import torch
import gdown
import os.path
import pydicom
import json
import base64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#test cv 
png_path = 'test_cases/ct/'
png_name = 'coronacases_org.png'

def read_dcm(filepath):
    dcm = pydicom.dcmread(filepath)
    array_uint8 = dcm.pixel_array
    array = array_uint8.astype(np.float64)

    if array.ndim == 2:
        array = array[..., None]
 
    return array
    
def read_nii(filepath):
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    
    if array.ndim == 2:
        array = array[..., None]
        
    return read_ct(array)
    
def read_ct(array):
    array = np.rot90(np.array(array))
    print('patients data loaded...')

    return array

def make_covid_pred(image, covid_model):
    if torch.cuda.is_available():
        covid_model.cuda()
    image = (image*255).astype(np.uint8)
    image = torch.as_tensor(image / np.max(image), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    pred = covid_model(image.to(device))
    return pred[0].cpu().detach().numpy()

def diceScore(inputs, targets):
    inputs[inputs>0.2] = 1
    inputs[inputs<0.2] = 0
    im1 = np.asarray(inputs).astype(np.bool)
    im2 = np.asarray(targets).astype(np.bool)
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def prepare_slice(image_slice):
    img = image_slice
     
    max_value = np.max(img)

    if max_value <= 255:
        img = np.divide((img - 0), 365)
    else:
        img[img < -1024] = -1024
        img[img > 600] = 600
        img = np.divide((img + 1024), 1624)
    
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
    img_tens = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img, img_tens

def get_lung_mask(model, image_slice, device=device):
    if torch.cuda.is_available():
        model.cuda()
        pred = model(image_slice.cuda().to(device))
        pls = torch.max(pred, 1)[1].detach().cpu().numpy().astype(np.uint8)[0]
        pls[pls>0] = 1
        return pls
    pred = model(image_slice.to(device))
    pls = torch.max(pred, 1)[1].detach().cpu().numpy().astype(np.uint8)[0]
    pls[pls>0] = 1
    return pls

def get_max_coords(tup1, tup2):
    x1,y1,w1,h1 = tup1
    x2,y2,w2,h2 = tup2
    min_x, min_y  = min(x1,x2)-5, min(y1,y2)
    max_w, max_h = w1+w2+30, max(h1,h2)
    return min_x, min_y, max_w, max_h

def make_decision(ldist1, rdist1, ldist2, rdist2, area1, area2):
    if area1 < 100 or area2 < 100:
        return False
    else:
        if not (ldist1 < 50 or rdist1 < 50):
            return False
        elif not (ldist2 < 50 or rdist2 < 50):
            return False
        else:
            return True

def compute_dist(x,y,w,h):
    c_x = x+w//2
    c_y = y+h//2
    dot = np.array([c_x,c_y])
    left_center = np.array([100,130])
    right_center = np.array([160,130])
    left_dist = np.linalg.norm(dot-left_center)
    right_dist = np.linalg.norm(dot-right_center)
    return left_dist, right_dist

def _cropper(ct_img, orig_ct):
    ct_img = (ct_img * 255).astype(np.uint8)
    _, mask = cv2.threshold(ct_img * 255, 1, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) < 2:
        print('return', 'areas < 2',len(areas),len(contours))
        return None, False
    x = np.argsort(areas)

    # first lung
    max_index = x[-1]
    cnt1 = contours[max_index]
    area1 = cv2.contourArea(cnt1)
    coords1 = cv2.boundingRect(cnt1)
    x1, y1, w1, h1 = coords1
    left_dist1, right_dist1 = compute_dist(x1, y1, w1, h1)

    # second lung
    max_index = x[-2]
    cnt2 = contours[max_index]
    area2 = cv2.contourArea(cnt2)
    coords2 = cv2.boundingRect(cnt2)
    x2, y2, w2, h2 = coords2
    left_dist2, right_dist2 = compute_dist(x2, y2, w2, h2)

    decision = make_decision(left_dist1, right_dist1, left_dist2, right_dist2, area1, area2)
    
    if decision:
        x, y, w, h = get_max_coords(coords1, coords2)
        if x < 0 or y < 0:
            return ct_img, False                  
        cropped_ct = orig_ct[y:y + h, x:x + w]
        cropped_ct = cv2.resize(cropped_ct, dsize=(256, 256), interpolation=cv2.INTER_AREA)  
        return cropped_ct, True
    else:                   
        return ct_img, False


def get_lung_crop(image_slice, model):
    image_slice, prepared_slice = prepare_slice(image_slice)
    lung_mask = get_lung_mask(model, prepared_slice)
    cv2.imwrite(png_path+str(0)+png_name,lung_mask) 
    masked_slice = cv2.bitwise_and(image_slice, image_slice, mask=lung_mask)
    cr_ct, res = _cropper(masked_slice, image_slice) 
    
    return cr_ct, res, lung_mask

def count_relative_square(image1, image2):
    square1 = np.sum(image1)
    square2 = np.sum(image2)
    return square1/square2

def count_injury_percentage_nii(path_to_patient_file, lung_model, covid_model):
    patient = read_nii(path_to_patient_file)
    return count_injury_percentage(patient, patient.shape[2], lung_model,covid_model, False)

def count_injury_percentage_dcm(path_to_patient_file, lung_model, covid_model):
    patient = read_dcm(path_to_patient_file)
    return count_injury_percentage(patient, patient.shape[2], lung_model, covid_model, True)

def count_injury_percentage(patient, num_slices, lung_model, covid_model,dcm):
    print('count_injury_percentage1 ')
    inj_squares_list = []
    lung_masks = []
    pretty_result = "nan"
    print('count_injury_percentage2 ',num_slices)
    for i in range(num_slices):#
        ct_slice = patient[:,:,i]
        cropped_ct, result, lung_mask = get_lung_crop(ct_slice, lung_model)
        if result:
            predict = make_covid_pred(cropped_ct, covid_model)
            rel_sq = count_relative_square(predict[0], lung_mask)
            inj_squares_list.append(rel_sq)
            pretty_result = f"{np.mean(inj_squares_list)*100:0.2f}"
            print(pretty_result)
        else:
            print('nan')
        lung_mask = cv2.resize(lung_mask, dsize=(ct_slice.shape[0], ct_slice.shape[1]), interpolation=cv2.INTER_AREA)
        
        if(dcm):
            lung_mask = np.rot90(np.array(lung_mask),1)
            
        lung_masks.append(lung_mask)              
    return inj_squares_list, pretty_result, lung_masks

def resultStrMasksImgBytesArrayToJson(results, masks,size,idx):
    results_json_string = json.dumps({'results' : str(results)})
    my_dict = json.loads(results_json_string)
    for i in range(len(masks)):
        data = masks[i]  
        print(data.shape)
        data = data.astype(np.float64)
        retval, t = cv2.imencode('.png', data)
        s = base64.b64encode(t)
        y = {"picture":str(s)} 
        w = {"size":str(size)} 
        z = {"idx":str(idx)} 
        my_dict.update(y)
        my_dict.update(w)
        my_dict.update(z)
    return json.dumps(my_dict)

def download_items():
    ct_path_nii = 'test_cases/ct/coronacases_org_001.nii'
    ct_path_dcm = 'test_cases/ct/coronacases_org_001.DCM'
    mask_path = 'test_cases/mask/coronacases_001.nii'
    covid_model_path = 'covid_segm_model/lungNET_take3'
    lung_model_path = 'lungmask/lung_seg_model.pth'


    if not os.path.exists(ct_path_nii):
        ct_url_nii = 'https://drive.google.com/uc?id=1mxxV1IM18ES1-8IUocBqSEsFhoTl1pUY'
        print('downloading test ct nii ...')
        gdown.download(ct_url_nii, ct_path_nii, quiet=False)
        
    if not os.path.exists(ct_path_dcm):
        ct_url_dcm = 'https://drive.google.com/file/d/1pgn9f8Spm70EvDktBlZZiQ6sqSCvXE8o/view?usp=sharing'
        print('downloading test ct dcm...')
        gdown.download(ct_url_dcm, ct_path_dcm, quiet=False)

    if not os.path.exists(mask_path):
        mask_url = 'https://drive.google.com/uc?id=1JxX-4w7CkMxifq5w2L7BQJBiT0ZtYJOS'
        print('downloading test mask ...')
        gdown.download(mask_url, mask_path, quiet=False)

    if not os.path.exists(lung_model_path):
        lung_model_url = 'https://drive.google.com/uc?id=1CHatlZDtYbpFoUOE7XhILVIzBJXkp-nS'
        print('downloading lung model  ...')
        gdown.download(lung_model_url, lung_model_path, quiet=False)

    if not os.path.exists(covid_model_path):
        covid_model_url = 'https://drive.google.com/uc?id=1tXtQxLM4FYPuoFcqLOLw9A6Q2TiwpF4c'
        print('downloading covid model  ...')
        gdown.download(covid_model_url, covid_model_path, quiet=False)
