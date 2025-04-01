# Customer Query Classification Using NLP and Ensemble Learning

## Course: NLP (Semester 6) - Pillai College of Engineering

## Acknowledgements
We would like to express our sincere gratitude to the following individuals for their guidance and support throughout this project:

**Theory Faculty:**
- Dhiraj Amin
- Sharvari Govilkar

**Lab Faculty:**
- Dhiraj Amin
- Neha Ashok
- Shubhangi Chavan

## Project Overview
This project is part of the Natural Language Processing (NLP) course for Semester 6 students at Pillai College of Engineering. We focus on Customer Query Classification, applying Machine Learning (ML) and Deep Learning (DL) techniques to categorize customer queries into predefined categories. The project involves text preprocessing, feature extraction, model training, and evaluation to build an efficient classification system.

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
- **CNN-BiLSTM** - A hybrid model combining Convolutional Neural Networks and Bidirectional LSTM.
- **BERT & RoBERTa** - Transformer-based models for advanced language understanding.
- **Ensemble Approach** – Combines multiple models through a voting mechanism to improve accuracy.

## Model Performance Results

### Machine Learning Models Performance

#### Random Forest (Accuracy: 70.45%)
| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| 0     | 0.78      | 0.78   | 0.78     |
| 1     | 0.61      | 0.57   | 0.59     |
| 2     | 0.72      | 0.74   | 0.73     |
| 3     | 0.74      | 0.72   | 0.73     |
| 4     | 0.67      | 0.71   | 0.69     |
| **Macro Avg** | **0.70** | **0.70** | **0.70** |
| **Weighted Avg** | **0.70** | **0.70** | **0.70** |

#### XGBoost (Accuracy: 70.12%)
| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| 0     | 0.80      | 0.77   | 0.78     |
| 1     | 0.60      | 0.57   | 0.58     |
| 2     | 0.73      | 0.76   | 0.75     |
| 3     | 0.71      | 0.72   | 0.71     |
| 4     | 0.66      | 0.69   | 0.68     |
| **Macro Avg** | **0.70** | **0.70** | **0.70** |
| **Weighted Avg** | **0.70** | **0.70** | **0.70** |

#### Logistic Regression (Accuracy: 62.77%)
| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| 0     | 0.78      | 0.78   | 0.78     |
| 1     | 0.57      | 0.58   | 0.57     |
| 2     | 0.74      | 0.70   | 0.72     |
| 3     | 0.59      | 0.59   | 0.59     |
| 4     | 0.47      | 0.48   | 0.48     |
| **Macro Avg** | **0.63** | **0.63** | **0.63** |
| **Weighted Avg** | **0.63** | **0.63** | **0.63** |

#### SVM (Accuracy: 58.26%)
| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| 0     | 0.62      | 0.87   | 0.72     |
| 1     | 0.53      | 0.29   | 0.38     |
| 2     | 0.72      | 0.68   | 0.70     |
| 3     | 0.47      | 0.68   | 0.56     |
| 4     | 0.60      | 0.39   | 0.47     |
| **Macro Avg** | **0.59** | **0.58** | **0.57** |
| **Weighted Avg** | **0.59** | **0.58** | **0.57** |

### Deep Learning Models Performance

#### CNN (Accuracy: 95.83%)
| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| ACCOUNT | 0.84 | 0.99 | 0.91 |
| DELIVERY | 1.00 | 0.96 | 0.98 |
| ORDER | 0.98 | 0.90 | 0.94 |
| REFUND | 1.00 | 0.94 | 0.97 |
| SHIPPING_ADDRESS | 1.00 | 1.00 | 1.00 |
| **Macro Avg** | **0.96** | **0.96** | **0.96** |
| **Weighted Avg** | **0.96** | **0.96** | **0.96** |

#### LSTM (Accuracy: 19.03%)
| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| ACCOUNT | 0.19 | 0.27 | 0.22 |
| DELIVERY | 0.00 | 0.00 | 0.00 |
| ORDER | 0.14 | 0.08 | 0.11 |
| REFUND | 0.18 | 0.22 | 0.20 |
| SHIPPING_ADDRESS | 0.21 | 0.36 | 0.27 |
| **Macro Avg** | **0.15** | **0.19** | **0.16** |
| **Weighted Avg** | **0.15** | **0.19** | **0.16** |

#### CNN-BiLSTM (Accuracy: 97.16%)
| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| ACCOUNT | 0.96 | 1.00 | 0.98 |
| DELIVERY | 1.00 | 0.97 | 0.99 |
| ORDER | 0.91 | 0.97 | 0.94 |
| REFUND | 1.00 | 0.91 | 0.95 |
| SHIPPING_ADDRESS | 1.00 | 1.00 | 1.00 |
| **Macro Avg** | **0.97** | **0.97** | **0.97** |
| **Weighted Avg** | **0.97** | **0.97** | **0.97** |

### Transformer Models Performance

#### BERT (Accuracy: 99.83%, MCC: 0.9979, Eval Loss: 0.0156)
| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| ACCOUNT | 1.00 | 1.00 | 1.00 |
| DELIVERY | 1.00 | 0.99 | 1.00 |
| ORDER | 0.99 | 1.00 | 1.00 |
| REFUND | 1.00 | 1.00 | 1.00 |
| SHIPPING_ADDRESS | 1.00 | 1.00 | 1.00 |
| **Macro Avg** | **1.00** | **1.00** | **1.00** |
| **Weighted Avg** | **1.00** | **1.00** | **1.00** |

#### RoBERTa (Accuracy: 99.83%, MCC: 0.9979, Eval Loss: 0.0148)
| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| ACCOUNT | 1.00 | 1.00 | 1.00 |
| DELIVERY | 1.00 | 0.99 | 1.00 |
| ORDER | 0.99 | 1.00 | 1.00 |
| REFUND | 1.00 | 1.00 | 1.00 |
| SHIPPING_ADDRESS | 1.00 | 1.00 | 1.00 |
| **Macro Avg** | **1.00** | **1.00** | **1.00** |
| **Weighted Avg** | **1.00** | **1.00** | **1.00** |

## Key Findings

1. **Transformer models (BERT & RoBERTa)** achieved the highest performance with near-perfect accuracy (99.83%) and F1-scores (1.00) across all classes.
2. **CNN-BiLSTM** was the best performing deep learning architecture (97.16% accuracy), with excellent precision and recall balance.
3. **Standard LSTM** performed poorly on its own (19.03% accuracy), with very low precision and recall values.
4. Among traditional ML models, **Random Forest** and **XGBoost** performed similarly well (~70% accuracy).
5. **Class imbalance issues** were observed in some models (particularly SVM), as evidenced by the disparity between precision and recall values.
6. The **ensemble approach** combining multiple models improved robustness and accuracy.

## Implementation

- An interactive web application was developed using **Streamlit** for real-time customer query classification.
- The system accepts user input and provides instant classification results.
- The ensemble model combines predictions from multiple models for improved accuracy.

## Recommendations

1. **Use transformer-based models** (BERT/RoBERTa) for highest accuracy when computational resources allow.
2. **CNN-BiLSTM** offers an excellent balance of performance and computational efficiency.
3. **Avoid standalone LSTM** as it fails to generalize well for this specific task.
4. For resource-constrained environments, **Random Forest** or **XGBoost** provide reasonable performance.
5. Consider further hyperparameter tuning or additional data augmentation techniques for continued improvement.
6. Address class imbalance issues to improve model performance, particularly for SVM.

## Practical Applications

This project demonstrates how NLP can automate customer support workflows by:
- Automatically routing customer queries to appropriate departments
- Optimizing response times by prioritizing urgent issues
- Providing insights into common customer concerns
- Improving overall customer service operations

## Authors
- Aniket Kumar Saini
- Sarthak Suryakant Satam
- Sahil Shivaji Salunkhe
- Pratik Pandurang Pawar
