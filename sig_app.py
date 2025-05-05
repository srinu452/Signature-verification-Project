import os
import numpy as np
import pandas as pd
import streamlit as st 
import cv2
import matplotlib.pyplot as plt
import pickle
import joblib

# Load the model from the Pickle file
#with open("sig_model.pkl", "rb") as f:
load_model = joblib.load(r"C:\Users\gadam\PycharmProjects\NewSignatureVerification\.venv\UpdatedSignatureVerifcation\ModelFiles/Signature_Forgery_Detector.joblib")

def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    ret, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized_img = cv2.resize(img, (200, 200))
    cv2.imwrite(image_path, resized_img)

def img_to_array(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img = img/255.0
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img

def verification(img1,img2):
    preprocess(img1)
    preprocess(img2)
    
    image1=cv2.imread(img1)
    image2=cv2.imread(img2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image1)
    axes[0].set_title("Image 1")
    axes[1].imshow(image2)
    axes[1].set_title("Image 2")

    st.pyplot(fig)
    test_img1 = img_to_array(img1)
    test_img2 = img_to_array(img2)
    
    prediction = load_model.predict([test_img1, test_img2])
    prediction = (prediction > 0.75).astype(int)
    if(prediction[0] ==0):
        return "Signatre is genuine"
    else:
        return "Signature is forged"
    
def main():
    st.title("Signature Verifier")
    
    img1 = st.file_uploader("Upload the image of the original signature: ", type=["png", "PNG", "jpg", "jpeg"], key="image1")
    img2 = st.file_uploader("Upload the image of the signature to be tested: ", type=["png", "PNG", "jpg", "jpeg"], key="image2")

    if img1 is not None and img2 is not None:
        if not os.path.exists("uploaded_images"):
            os.mkdir("uploaded_images")

        image1_path = os.path.join("uploaded_images", "image1.png")
        with open(image1_path, "wb") as f:
            f.write(img1.read())

        image2_path = os.path.join("uploaded_images", "image2.png")
        with open(image2_path, "wb") as f:
            f.write(img2.read())
            
        ver=''
        
        if st.button("Get verification results"):
            ver = verification(image1_path, image2_path)
        st.success(ver) 

if __name__ == "__main__":
    main()
