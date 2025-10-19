
---

# Machine Learning and NLP Projects Portfolio

👥 Group Members
This project was developed by the following group members:

* Anthonia Othetheaso
* Obuye Emmanuel Chukwuemeke 
* Eunice Fagbemide
* Daizy Jepchumba Kiplagat 
* Mark Ireri

This repository contains a collection of projects demonstrating classical machine learning, deep learning (CNNs), and natural language processing (NLP) techniques.
Each project addresses a distinct problem, from image classification to sentiment analysis.

---

## 🚀 Projects

### 1. Iris Species Classification

* **File:** `iris_classification.py`
* **Purpose:** This script implements a classical machine learning model to classify the species of Iris flowers based on their features.
* **Model/Algorithm:** **Decision Tree Classifier**.
* **Key Libraries:** **Scikit-learn** (for the model and metrics), **Pandas**, **Matplotlib**, and **Streamlit** (for the interactive web application interface).
* **Functionality:** It trains the model, displays a sample of the **Iris Species** dataset, calculates and presents performance metrics (**Accuracy**, **Precision**, **Recall**, **Confusion Matrix**), and visualizes the structure of the Decision Tree.

### 2. Natural Language Processing with spaCy

* **File:** `nlp-with-spacy (1).ipynb`
* **Purpose:** A Jupyter Notebook exploring fundamental Natural Language Processing tasks on text data.
* **Dataset:** **Amazon reviews**.
* **Key Libraries:** **spaCy** (for core NLP tasks), **Pandas**.
* **Focus:** The notebook demonstrates how to process text and is used to analyze the reviews and determine their **sentiment** (Positive, Negative, or Neutral).

### 3. CNN for MNIST Digit Recognition

* **File:** `TF_cnn_MNIST.ipynb`
* **Purpose:** A deep learning project that implements a Convolutional Neural Network (CNN) for image classification.
* **Model/Algorithm:** **Convolutional Neural Network (CNN)**.
* **Dataset:** **MNIST** (handwritten digits).
* **Key Libraries:** **TensorFlow** / **Keras**.
* **Performance:** The model is trained over 5 epochs, achieving a high test accuracy (e.g., **99.02%**). The notebook also generates a plot of training accuracy and loss over the epochs.
* **Live Demo:** You can interact with the deployed Streamlit web application here: https://mnist-digit-classifier-5.streamlit.app/

---

## ✍️ Project Deliverables & Submission Requirements

In addition to the code files, we submitted the following supplementary materials as Required:

### 1. Project Report (PDF)

A comprehensive PDF report, covering the project's theoretical and practical aspects. This report has been:
* **Shared as an article** in the Community, Peer Group Review.
* **Included** in this GitHub Repository.

The report contain:
* **Answers to theoretical questions** related to the models and concepts used.
* **Screenshots of model outputs** (e.g., accuracy graphs, NER results, confusion matrices).
* A section dedicated to **Ethical Reflection** on the dataset, model usage, and potential societal impact.

### 2. Presentation (Video)

A short video presentation explaining our approach toward the assignment.the video is a mergrd video where each of us explained the processes and codes.
The video is approximately **3 minutes** long. and will be shared on the Community platform, submitted there.

 ### 3. Ethics & Optimization
This section addresses advanced topics related to model deployment and responsibility. 

 i.### **Ethical Considerations**
 
* **Model:** MNIST Digit Classification
* **Potential Biases:** Data representation bias (limited handwriting styles from American sources), accessibility concerns, and class imbalance in real-world use
* **Mitigation Strategies:** TensorFlow Fairness Indicators can use slice-based evaluation and fairness metrics to identify performance discrepancies.Other strategies include data augmentation and human-in-the-loop verification.

* **Model:** Amazon Reviews NLP
* **Potential Biases:** Language/Cultural bias (English-only), rule-based limitations (sarcasm, negation, context), demographic representation bias.
* **Mitigation Strategies:** spaCy's rule-based systems can be enhanced with customizable pipelines for negation handling and context-aware processing11. Other strategies include hybrid models and confidence scoring to flag ambiguous reviews

* ### Deployment Safeguards (General):
Maintain transparency about model limitations.
Establish user feedback mechanisms to dispute classifications.
Avoid automated high-stakes decisions without human verification.

ii. ###Troubleshooting Challenge
Task: Debug and fix a provided buggy TensorFlow script that contains errors such as dimension mismatches or incorrect loss functions.
Outcome: The debugged CNN successfully achieved a high test accuracy (approximately 99%) on the MNIST dataset after 5 optimized epochs. Detailed performance metrics, including accuracy and loss curves, and a per-digit confusion matrix, were generated

---

## ⚙️ Technologies Used

* **Python 3**
* **Scikit-learn:** Classical ML algorithms.
* **TensorFlow/Keras:** Deep learning (CNNs).
* **spaCy:** Natural Language Processing.
* **Pandas & NumPy:** Data handling and manipulation.
* **Matplotlib:** Data visualization.
* **Streamlit:** Web application deployment (for the MNIST project).
