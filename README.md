# CV Annotation Toolkit

A lightweight Python toolkit for **visualizing bounding boxes**, **validating annotations**, and **auditing datasets** across common computer vision formats.  
Designed to work smoothly on **servers and remote machines (no GUI required)**.

---

## 🚀 Supported Annotation Formats

- YOLO (You Only Look Once)
- Pascal VOC (Visual Object Classes)
- COCO (Common Objects in Context)

---

## ✨ Features

- Draw bounding boxes on images from YOLO / VOC annotations
- Batch visualization for quick dataset sanity checks
- Dataset statistics and class distribution analysis
- Detect missing labels and invalid bounding boxes
- Server-friendly (outputs are saved as images, no display needed)

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/SUBINOY551/agri-vision-toolkit.git
cd agri-vision-toolkit
pip3 install -r requirements.txt

###Currently working on data-centric weed detection for Indian agriculture using MH-Weed16 and a self-collected jute-field weed dataset from West Bengal.
