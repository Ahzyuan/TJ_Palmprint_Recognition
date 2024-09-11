import cv2,os
import numpy as np

'''
img: gray img
'''

def laplacian(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, cv2.CV_8UC3, kernel)

def hist(image):
    return cv2.equalizeHist(image)

def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def log(img,c=100):
    result = np.uint8(c * np.log(1.0 + img))
    return result

if __name__=='__main__':
    img_name='s1_01478.bmp'
    img=cv2.imread(os.path.join(r'C:\Users\Administrater\Desktop\pre',img_name),flags=cv2.IMREAD_GRAYSCALE)
    
    la_res=laplacian(img)
    he_res=hist(img)
    clahe_res=clahe(img)
    log_res=log(img)

    for name,res_img in {'LA':la_res,'HE':he_res,'CLAHE':clahe_res,'LOG':log_res}.items():
        cv2.imwrite(r'C:\Users\Administrater\Desktop\pre'+f'\\{name}_{img_name}.png',res_img)
    
    print('Done!')