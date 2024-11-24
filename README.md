# 🫁 Lung Nodule Detection on CT Images

**🚀 Authors:**  
- **Karim Sleiman**
- A. Gimesi  
- S. Dash  
- J.S. Ibarra  
- S. Chattopadhyay  

**📍 Affiliation:**  
University of Cassino, Cassino, Italy  

---

## 🌟 Overview  
This project aims to detect lung nodules from CT scans to aid in early lung cancer diagnosis. Using the LUNA16 dataset, the system integrates traditional machine learning techniques for false positive reduction and deep learning models like RetinaNet, YOLOv8, and V-Net for object detection and segmentation. The primary goal is to develop a comprehensive pipeline for lung nodule detection while addressing key challenges such as class imbalance, image variability, and computational limitations.

---

## 📂 Dataset and Preprocessing  
The LUNA16 dataset, derived from the LIDC/IDRI database, was used for this study. It includes 888 CT scans with annotations for nodules ≥3 mm. Preprocessing involved lung segmentation using a custom pipeline that included:
1. **Clipping Intensity Values**: Standardizing Hounsfield Unit (HU) ranges.
2. **Binary Thresholding**: Differentiating air-filled lung tissues from surrounding structures.
3. **Noise Removal and Morphological Filtering**: Enhancing the quality of lung masks.

The segmentation results were used for candidate extraction, which involved generating initial nodule candidates and refining them through post-processing to reduce false positives.

---

## 🔍 Methods Used  

### **Machine Learning Pipeline**  
A set of hand-engineered features (e.g., texture, intensity, and morphological properties) were extracted from nodule candidates. These features were used in models such as:
- **Random Forest (RF)**: Ensemble-based classifier for handling high-dimensional data.
- **Support Vector Machines (SVM)**: Effective for imbalanced datasets.
- **LightGBM**: Gradient-boosted decision trees optimized for speed and memory.

Feature selection methods like PCA, mRMR, and RFE were applied to reduce dimensionality, and random undersampling was used to address class imbalance.

---

### **Deep Learning Approaches**  
1. **RetinaNet**: A state-of-the-art object detection model with Focal Loss to address severe class imbalance. The model predicts bounding boxes around nodules in 2D slices.
2. **YOLOv8**: A high-speed object detection model optimized for small datasets, used with custom lung-segmented images.
3. **Faster R-CNN**: A two-stage detector, combining region proposal networks with CNN-based classification.
4. **V-Net**: A 3D convolutional neural network for volumetric segmentation, which identifies nodules directly in 3D CT scans. The architecture utilized Dice coefficient-based loss for improved accuracy in imbalanced datasets.

---

## 📌 Future Work  
Future efforts aim to address computational constraints for handling larger datasets, explore transformer-based architectures for more robust detection, and enhance the models' generalizability across varying CT imaging conditions.

---

📫 **Contact:**  
Karim Sleiman - karim.s2000@icloud.com  
