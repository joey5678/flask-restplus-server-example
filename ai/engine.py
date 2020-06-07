import requests
import json
import sys

import cv2

from mock.object import IMAGE_BASE64


Label_Map = {
    1: 'tree_head',
    2: 'tree_body',
    3: 'tree_root',
    4: 'house',
    5: 'person',
    6: 'other',
}


def infer(img_data):
    headers = {"content-type": "application/json"}
    # body = {"instances": [1.0, 2.0, 5.0]}
    # data = json.dumps(body)
    # url = 'http://10.240.108.54:8501/v1/models/half_plus_two:predict'

    data = json.dumps({"instances": [img_data]})
    url = 'http://10.240.108.54:8501/v1/models/tree_model:predict'
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']

    return predictions[0]

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

def postprocess(prediction, im_width=None, im_height=None):
    num_detections = int(prediction.pop('num_detections'))
    output_dict = {}
    for key, value in prediction.items():
        output_dict[key] = value[0:num_detections]

    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = [int(v) for v in output_dict['detection_classes']]
    max_boxes = len(output_dict['detection_boxes'])
    result = []
    min_score_thresh = 0.5
    for i in range(max_boxes):
        if output_dict['detection_scores'][i] < min_score_thresh:
            continue
        # print(output_dict['detection_scores'][i])
        # print()
        # print(output_dict['detection_boxes'][i])
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
    if False:
        image_path="./test.jpg"
        img = cv2.imread(image_path)
        img = cv2.resize(img, (446, 446))
        print(postprocess(infer(img.tolist()), 446, 446))
    else:
        test_restful()