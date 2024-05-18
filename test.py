import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

#load trained model
loaded_model = load_model("Project_Saved_Models/trained_model.h5")

#perform denoising
def denoise(image):

    #denoising using Non-local mean algorithm
    out = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    return out

def main1(path):

    image = cv2.imread(path)
    image_resize=cv2.resize(image,(128,128))
    output=denoise(image_resize)
    # convert it to grayscale
    img_yuv = cv2.cvtColor(output,cv2.COLOR_BGR2YUV)
    # apply histogram equalization 
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    print(hist_eq.shape)
    out_image=np.expand_dims(hist_eq,axis=0)/255
    print(out_image.shape)


    my_pred = loaded_model.predict(out_image)
    print(my_pred)
    my_pred=np.argmax(my_pred,axis=1)
    print(my_pred)

    if my_pred==0:
        print("MildDemented")
    elif my_pred==1:
        print("ModerateDemented")
    elif my_pred==2:
        print("NonDemented")
    elif my_pred==3:
        print("VeryMildDemented")



if __name__=="__main__":
    from tkinter.filedialog import askopenfilename
    path=askopenfilename()
    main1(path)
