# SMS-Spam-Detection
📩 SMS Spam Detection using Machine Learning  
This project focuses on detecting spam messages in SMS data using machine learning.  
By analyzing message content, the model classifies texts as either *spam* or *ham* (not spam), helping users avoid unwanted or malicious messages.

🚀 Features  
🧹 Data Preprocessing  
- Removed null or duplicate entries  
- Converted all text to lowercase  
- Removed punctuation, stopwords, and applied tokenization  
- Transformed text into numerical features using TF-IDF Vectorizer  

🌲 Model Training  
- Implemented Naive Bayes Classifier for binary classification  

📊 Model Evaluation  
- Evaluated using Accuracy, Precision, Recall, and F1-score  
- Displayed results using a Confusion Matrix  

🔍 Visualization  
- Plotted heatmap of confusion matrix  
- Visualized spam vs ham distribution  

📈 Full Dataset Predictions  
- Predicted spam or ham labels for all messages in the dataset  

🛠 Tech Stack  
- Python  
- Pandas  
- NumPy  
- scikit-learn  
- NLTK  
- Matplotlib  
- Seaborn  

📂 Dataset  
Dataset Source: [SMS Spam Collection Dataset] (https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
📊 Model Workflow  
- Load and inspect dataset  
- Preprocess text (cleaning, tokenization, vectorization)  
- Split into train and test sets  
- Train Naive Bayes Classifier  
- Evaluate and visualize model performance  
- Generate predictions on unseen messages  


📈 Results  
Achieved ~98% accuracy with SVM Classifier  
Successfully filtered out spam messages with high precision and recall
