import base64
import time

import cv2
import numpy as np

import config
from copy import copy



# return preprocessed image ready for mask detector
def image_preprocessing(input_image):
    out_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    out_image = cv2.resize(src=out_image, dsize=config.RESIZED_DIM, interpolation=cv2.INTER_NEAREST)
    out_image = cv2.normalize(out_image, None, 0, 255, cv2.NORM_MINMAX)
    out_image = out_image / 255

    return out_image


def __draw_bbs(image, mask, pt1, pt2, confidence):
    if mask:
        color = config.GREEN
        text = f"mask: {confidence:2.2f}"
    else:
        color = config.RED
        text = f"no_mask: {confidence:2.2f}"

    cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=color, thickness=1)
    cv2.putText(img=image, text=text, org=(pt1[0] - 10, pt2[1] + 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4,
                color=color)


def detect_masks(input_image, bounding_boxes, mask_detector):
    cropped_faces = []

    # using a copy of the original image to extract the cropped faces
    # this prevents artifacts caused by __draw_bbs e.g. "mask" or "no_mask" text overlapping on the cropped faces
    clean_image = copy(input_image)

    for bb in bounding_boxes:
        # extracting coordinates
        coords = bb["box"]
        x1, y1 = coords[0], coords[1]
        x2, y2 = coords[0] + coords[2], coords[1] + coords[3]

        cropped_face_raw = clean_image[y1: y2, x1: x2]
        cropped_face = image_preprocessing(input_image=cropped_face_raw)

        # adding channel and batch size
        cropped_face = np.expand_dims(cropped_face, axis=-1)
        cropped_face = np.expand_dims(cropped_face, axis=0)

        with graph.as_default():
            out = mask_detector.predict(cropped_face)[0]

        if out[0] > out[1]:
            mask = True
            probability = out[0]
        else:
            mask = False
            probability = out[1]

        base64_face = cv_image_to_base64(
            cv2.cvtColor(
                cv2.resize(
                    src=cropped_face_raw,
                    dsize=config.RESIZED_DIM,
                    interpolation=cv2.INTER_NEAREST
                ),
                cv2.COLOR_BGR2RGB
            )
        )
        cropped_faces.append({
            "face": base64_face,
            "mask": mask,
            "probability": str(probability)})

        __draw_bbs(image=input_image, mask=mask, pt1=(x1, y1), pt2=(x2, y2), confidence=probability)

    output_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    return output_image, cropped_faces


def cv_image_to_base64(input_image):
    _, img_arr = cv2.imencode(config.RESPONSE_IMAGE_FORMAT, input_image)
    img_bytes = img_arr.tobytes()
    img_b64_bytes = base64.b64encode(img_bytes)
    img_b64_str = img_b64_bytes.decode("utf-8")

    base64_header = "data:image/png;base64, "
    img_b64_str = base64_header + img_b64_str
    return img_b64_str
