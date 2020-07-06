import cv2
from tensorflow import keras
import numpy as np
import time

start = time.time()
mask_detector = keras.models.load_model("models/model-best.h5")
end = time.time() - start

print(f"it took {end} to load the model")

# return preprocessed image ready for mask detector
def image_preprocessing(input_image):
    out_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    out_image = cv2.resize(src=out_image, dsize=(150, 150), interpolation=cv2.INTER_NEAREST)
    out_image = cv2.normalize(out_image, None, 0, 255, cv2.NORM_MINMAX)
    out_image = out_image / 255

    print(out_image.shape)

    return out_image


def __draw_bbs(image, mask, pt1, pt2, confidence):
    if mask:
        color = (0, 255, 0, 255)
        text = f"mask: {confidence:2.2f}"
    else:
        color = (255, 0, 0, 255)
        text = f"no_mask: {confidence:2.2f}"

    cv2.rectangle(img=image, pt1=pt1, pt2=pt2, color=color, thickness=1)
    cv2.putText(img=image, text=text, org=(pt1[0], pt2[1] + 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=color, thickness=2)


def detect_masks(input_image, bounding_boxes):

    for bb in bounding_boxes:
        # extracting coordinates
        coords = bb["box"]
        x1, y1 = coords[0], coords[1]
        x2, y2 = coords[0] + coords[2], coords[1] + coords[3]

        cropped_face = input_image[y1: y2, x1: x2]
        cropped_face = image_preprocessing(input_image=cropped_face)

        # adding channel and batch size
        cropped_face = np.expand_dims(cropped_face, axis=-1)
        cropped_face = np.expand_dims(cropped_face, axis=0)

        out = mask_detector.predict(cropped_face)[0]

        if out[0] > out[1]:
            mask = True
            confidence = out[0]
        else:
            mask = False
            confidence = out[1]

        __draw_bbs(image=input_image, mask=mask, pt1=(x1, y1), pt2=(x2, y2), confidence=confidence)

    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    cv2.waitKey(0)


