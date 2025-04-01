# Customer Query Classification Using NLP

## Course: NLP (Semester 6) - Pillai College of Engineering

## Project Overview
This project is part of the Natural Language Processing (NLP) course for Semester 6 students at Pillai College of Engineering. The focus is on Customer Query Classification, where we apply Machine Learning (ML) and Deep Learning (DL) techniques to categorize customer queries into predefined categories. The project involves text preprocessing, feature extraction, model training, and evaluation to build an efficient classification system.

You can learn more about the college by visiting the official website of Pillai College of Engineering.

## Acknowledgements
We would like to express our sincere gratitude to the following individuals for their guidance and support throughout this project:

**Theory Faculty:**
- Dhiraj Amin
- Sharvari Govilkar

**Lab Faculty:**
- Dhiraj Amin
- Neha Ashok
- Shubhangi Chavan

## Project Title
Customer Query Classification Using NLP and Ensemble Learning

## Project Abstract
Customer support queries are a crucial part of business operations, providing insight into customer concerns and service performance. However, manually categorizing these queries is time-consuming and inefficient. This project aims to develop an automated classification system that categorizes customer queries into predefined categories such as Account, Cancellation Fee, Contact, Delivery, Feedback, Invoice, Newsletter, Order, Payment, Refund, and Shipping Address.

The system is built using Natural Language Processing (NLP) techniques and an ensemble model combining XGBoost and LSTM. The Customer Support Intent Dataset from Kaggle is used for training and evaluation. The Streamlit framework is employed to develop an interactive web application where users can input queries and receive real-time classifications.

The project evaluates the effectiveness of both models independently and integrates their predictions through a voting mechanism, ensuring robust and accurate results.

## Dataset Details
**Dataset Name:** Customer Support Intent Dataset  
**Source:** Kaggle  
**Description:**
- A large-scale dataset containing customer support queries and their corresponding intent labels.
- Includes thousands of examples categorized into areas such as billing, technical support, and feedback.
- Useful for training NLP models to accurately classify customer support queries.

## Algorithms Used
**Machine Learning & Deep Learning Models:**
- **XGBoost** – A gradient boosting algorithm used for high-performance classification.
- **LSTM (Long Short-Term Memory)** – A deep learning model capable of capturing sequential dependencies in text data.
- **Ensemble Approach** – Combines XGBoost and LSTM results through a voting mechanism to improve accuracy.
- 
## Model Performance Summary

| Model       | Accuracy  |
|------------|----------|
| CNN        | 0.958264 |
| LSTM       | 0.190317 |
| CNN-BiLSTM | 0.971619 |

---

## CNN Model Performance

**Accuracy:** 0.9582637729549248

### Classification Report
```
                   precision    recall  f1-score   support

         ACCOUNT       0.84      0.99      0.91       124
        DELIVERY       1.00      0.96      0.98       114
           ORDER       0.98      0.90      0.94       118
          REFUND       1.00      0.94      0.97       118
SHIPPING_ADDRESS       1.00      1.00      1.00       125

        accuracy                           0.96       599
       macro avg       0.96      0.96      0.96       599
    weighted avg       0.96      0.96      0.96       599
```

---

## LSTM Model Performance

**Accuracy:** 0.19031719532554256

### Classification Report
```
                   precision    recall  f1-score   support

         ACCOUNT       0.19      0.27      0.22       124
        DELIVERY       0.00      0.00      0.00       114
           ORDER       0.14      0.08      0.11       118
          REFUND       0.18      0.22      0.20       118
SHIPPING_ADDRESS       0.21      0.36      0.27       125

        accuracy                           0.19       599
       macro avg       0.15      0.19      0.16       599
    weighted avg       0.15      0.19      0.16       599
```
**Note:** Warnings indicate that some labels have no predicted samples, leading to ill-defined precision values.

---

## CNN-BiLSTM Model Performance

**Accuracy:** 0.9716193656093489

### Classification Report
```
                   precision    recall  f1-score   support

         ACCOUNT       0.96      1.00      0.98       124
        DELIVERY       1.00      0.97      0.99       114
           ORDER       0.91      0.97      0.94       118
          REFUND       1.00      0.91      0.95       118
SHIPPING_ADDRESS       1.00      1.00      1.00       125

        accuracy                           0.97       599
       macro avg       0.97      0.97      0.97       599
    weighted avg       0.97      0.97      0.97       599
```
## Conclusion
From the results, the CNN-BiLSTM model performs the best with **97.16% accuracy**, followed closely by the CNN model with **95.83% accuracy**. The LSTM model performed poorly, achieving only **19.03% accuracy**.

**Key takeaways from the project:**
- LSTM effectively captures the sequential patterns in text, improving classification accuracy.
- XGBoost provides interpretability and robustness in structured text classification.
- The ensemble model outperforms standalone models by leveraging the strengths of both approaches.
- The Streamlit web application enables real-time classification, making it easy to use in practical customer support systems.

This project highlights the potential of NLP in automating customer support workflows, optimizing response times, and improving customer service operations.

### Recommendations
- **Use CNN-BiLSTM** for the best performance.
- **Avoid LSTM standalone** as it fails to generalize well in this task.
- Consider further hyperparameter tuning or additional data augmentation techniques to improve model performance.

---

**Author:** Aniket Kumar Saini, Sarthak Suryakant Satam, Sahil Shivaji Salunkhe, Pratik Pandurang Pawar


