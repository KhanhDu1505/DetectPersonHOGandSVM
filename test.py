import cv2
import numpy as np

# Hàm để trích xuất đặc trưng HOG từ hình ảnh
def get_hog_features(image):
    hog = cv2.HOGDescriptor()
    return hog.compute(image)

# Chuẩn bị dữ liệu huấn luyện
# Load các positive samples (các hình ảnh chứa đối tượng cần theo dõi)
positive_samples = "archive/1"
for i in range(1, 6):
    img = cv2.imread(f'positive_sample_{i}.jpg')
    positive_samples.append(img)

# Load các negative samples (các hình ảnh không chứa đối tượng)
negative_samples = "archive/0"
for i in range(1, 6):
    img = cv2.imread(f'negative_sample_{i}.jpg')
    negative_samples.append(img)

# Tạo các nhãn tương ứng cho dữ liệu huấn luyện
labels = np.hstack((np.ones(len(positive_samples)), np.zeros(len(negative_samples))))

# Trích xuất đặc trưng HOG từ các hình ảnh huấn luyện
features = []
for img in positive_samples + negative_samples:
    hog_features = get_hog_features(img)
    features.append(hog_features)

features = np.array(features).squeeze()

# Huấn luyện SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(features, cv2.ml.ROW_SAMPLE, labels)

# Xác định vị trí của đối tượng trong khung hình
def detect_object(frame):
    hog_features = get_hog_features(frame)
    prediction = svm.predict(hog_features.reshape(1, -1))
    if prediction[1] == 1:
        # Đối tượng được dự đoán có mặt trong khung hình
        # Tìm kiếm vùng quan tâm xung quanh vị trí dự đoán
        x, y, w, h = prediction[0][0]
        roi = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame, roi
    else:
        # Không có đối tượng được dự đoán có mặt trong khung hình
        return frame, None

# Đọc video
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Xác định vị trí của đối tượng trong khung hình
    frame, roi = detect_object(frame)

    cv2.imshow('Object Tracking', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()