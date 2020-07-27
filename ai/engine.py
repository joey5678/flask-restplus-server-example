import requests
import json
import sys
import numpy as np
import cv2
from cls_id import CLS_NAMES, SEG_CLS_NAMES

from web_client_mrcnn import InferenceConfig, ForwardModel, detect_mask_single_image_using_restapi, calculate_areas


# from mock.object import IMAGE_BASE64

IA_SERVER_IP = '10.240.108.54'
IA_SEG_REST_PORT = 8501
IA_CLS_REST_PORT = 8508
SEG_ENDPOINT = 'tree_model'
CLS_ENDPOINT = 'tree_cls_model'
SEG_REST_API_URL = "http://%s:%s/v1/models/%s:predict" % (IA_SERVER_IP, IA_SEG_REST_PORT, SEG_ENDPOINT)
CLS_REST_API_URL = "http://%s:%s/v1/models/%s:predict" % (IA_SERVER_IP, IA_CLS_REST_PORT, CLS_ENDPOINT)

def get_areas(img_data):
    """
    # body = {"instances": [1.0, 2.0, 5.0]}
    # data = json.dumps(body)
    # url = 'http://10.240.108.54:8501/v1/models/half_plus_two:predict'
    """
    preprocess_obj = ForwardModel(InferenceConfig())
    r = detect_mask_single_image_using_restapi(img_data, preprocess_obj, SEG_REST_API_URL)
    crown_parts, trunk_part, root_part, tree_locs, tree_rois = calculate_areas(img_data, r)

    print(crown_parts)
    print(trunk_part)
    print(root_part)
    print(tree_locs)
    print(tree_rois)

    roi_imgs = []
    for roi_cls, roi_loc in tree_rois.items():
        x0, y0, x1, y1 = roi_loc
        roi_imgs.append(img_data[y0:y1, x0:x1])

    get_cls_results(roi_imgs)
    

def get_cls_results(roi_images):
    normal_roi_list = []
    for roi_image in roi_images:
        roi_image = cv2.resize(roi_image, (256, 256))/255.
        normal_roi_list.append(roi_image.tolist())

    headers = {"content-type": "application/json"}
    data = json.dumps({"instances": normal_roi_list})
    json_response = requests.post(CLS_REST_API_URL, data=data, headers=headers)
    
    try:
        predictions = json.loads(json_response.text)['predictions']
        return cls_postprocess(predictions)
    except:
        print("[DEBUG] RESTFUL CLS failed: {}".format(json_response.text))
        return None

def test_restful():
    body = {'source_id': 'test',
        'request_type': 10,
        'sub_type': 'test',
        'content_id':'testttttttest',
        'content_type': 20,
        'content_text': 'aaaaaaa',
        'image': IMAGE_BASE64}

    headers = {"content-type": "application/json"}
    auth_body = {"identifier":"111111111", "req_source":"222222222"}
    data = json.dumps(body)
    url = 'http://127.0.0.1:5000/api/v1/analysis/'
    url = 'http://127.0.0.1:5000/auth/'
    json_response = requests.post(url, data=data, headers=headers)
    print(json_response.text)


def cls_postprocess(predictions, im_width=None, im_height=None):
    results = []
    for prediction in predictions:
        top1_id = np.array(prediction).argmax()
        print(CLS_NAMES[top1_id])
        print(prediction[top1_id])
        results.append((top1_id, prediction[top1_id], CLS_NAMES[top1_id]))
    return results

def det_postprocess(prediction, im_width=None, im_height=None):
    num_detections = int(prediction.pop('num_detections'))
    output_dict = {}
    for key, value in prediction.items():
        output_dict[key] = value[0:num_detections]

    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = [int(v) for v in output_dict['detection_classes']]
    max_boxes = len(output_dict['detection_boxes'])
    result = []
    min_score_thresh = 0.1
    for i in range(max_boxes):
        if output_dict['detection_scores'][i] < min_score_thresh:
            continue
        print(output_dict['detection_scores'][i])
        print()
        print(output_dict['detection_boxes'][i])
        ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
        if im_width and im_height:
            bbox = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
        else:
            bbox = (-1, -1, -1, -1)

        result.append({
            'class_name': Label_Map.get(output_dict['detection_classes'][i], "Unknown"),
            'class_id': output_dict['detection_classes'][i],
            'norm_bbox': output_dict['detection_boxes'][i],
            'bbox': bbox
        })

    return result

if __name__ == "__main__":
    if True:
        image_path="./test.jpg"
        img = cv2.imread(image_path)
        img = cv2.resize(img, (256, 256))
        get_areas(img)
        # get_cls_results([img])
    else:
        test_restful()