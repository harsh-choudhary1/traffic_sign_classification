import numpy as np
import cv2 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import os 

data_dir = "/Users/harsh/Coding/Projects/GTSRB/Training"
categories = os.listdir(data_dir)

data = []
labels = []

valid_extensions = (".jpg", ".jpeg", ".png", ".ppm", ".bmp")
for category in categories:
    category_path = os.path.join(data_dir, category)
    if not os.path.isdir(category_path):
        print(f"Skipping non-directory item: {category_path}")
        continue
    for img_name in os.listdir(category_path):
        if img_name.endswith(valid_extensions):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            # if img is None:
            #     print(f"Could not read image: {img_path}
            #     continue
            img = cv2.resize(img, (64, 64))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hog_features = hog(
                gray_img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=False,
            )
            data.append(hog_features)
            labels.append(category)


# data = np.array[:14]
# labels = np.array[:14]
data = np.array(data)
labels = np.array(labels)


x_train , x_test , y_train , y_test = train_test_split(data , labels, test_size=0.2, train_size= 0.2 , random_state=42)

svm_model = SVC(kernel='linear',random_state=42)

svm_model.fit(x_train,y_train)

y_pred = svm_model.predict(x_test)

accuracy = accuracy_score(y_test , y_pred )
print(f"Model Accuracy:- {accuracy * 100:.2f}%")

def predict_sign(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(64,64))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_features = hog(gray_img , orientations = 9 , pixels_per_cell = ( 8,8 ) , cells_per_block = (2,2 ) , block_norm = 'L2-Hys', visualize = False)
    hog_features = hog_features.reshape(1 , -1 )

    prediction = model.predict(hog_features)
    return prediction[0]
new_image = "/Users/harsh/Coding/Projects/mag-11WMT-t_CA0-jumbo.jpg"
predicted_class = predict_sign(new_image, svm_model)
print(f"Presicted sign :- {predicted_class}")
