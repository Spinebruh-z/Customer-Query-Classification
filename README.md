Customer Query Classification Using NLP
Course: NLP (Semester 6) - Pillai College of Engineering
Project Overview
This project is part of the Natural Language Processing (NLP) course for Semester 6 students at Pillai College of Engineering. The focus is on Customer Query Classification, where we apply Machine Learning (ML) and Deep Learning (DL) techniques to categorize customer queries into predefined categories. The project involves text preprocessing, feature extraction, model training, and evaluation to build an efficient classification system.

You can learn more about the college by visiting the official website of Pillai College of Engineering.
Acknowledgements
We would like to express our sincere gratitude to the following individuals for their guidance and support throughout this project:
Theory Faculty:
- Dhiraj Amin
- Sharvari Govilkar
Lab Faculty:
- Dhiraj Amin
- Neha Ashok
- Shubhangi Chavan
Project Title
Customer Query Classification Using NLP and Ensemble Learning
Project Abstract
Customer support queries are a crucial part of business operations, providing insight into customer concerns and service performance. However, manually categorizing these queries is time-consuming and inefficient. This project aims to develop an automated classification system that categorizes customer queries into predefined categories such as Account, Cancellation Fee, Contact, Delivery, Feedback, Invoice, Newsletter, Order, Payment, Refund, and Shipping Address.

The system is built using Natural Language Processing (NLP) techniques and an ensemble model combining XGBoost and LSTM. The Customer Support Intent Dataset from Kaggle is used for training and evaluation. The Streamlit framework is employed to develop an interactive web application where users can input queries and receive real-time classifications.

The project evaluates the effectiveness of both models independently and integrates their predictions through a voting mechanism, ensuring robust and accurate results.
Dataset Details
Dataset Name: Customer Support Intent Dataset
Source: Kaggle
Description:
- A large-scale dataset containing customer support queries and their corresponding intent labels.
- Includes thousands of examples categorized into areas such as billing, technical support, and feedback.
- Useful for training NLP models to accurately classify customer support queries.
Algorithms Used
Machine Learning & Deep Learning Models:
XGBoost – A gradient boosting algorithm used for high-performance classification.
LSTM (Long Short-Term Memory) – A deep learning model capable of capturing sequential    dependencies in text data.
Ensemble Approach – Combines XGBoost and LSTM results through a voting mechanism to improve accuracy.
Comparative Analysis
Category        	Precision (%)	Recall (%)	F1-Score (%)
ACCOUNT	          98	          100	        99
CANCELLATION_FEE	100	          96	        98
CONTACT	          100	          100	        100
DELIVERY	        100	          100	        100
FEEDBACK	        100	          100	        100
INVOICE	          100	          100	        100
NEWSLETTER	      100	          91	        95
ORDER	            100	          100	        100
PAYMENT	          100	          100	        100
REFUND	          100	          99	        99
SHIPPING_ADDRESS	100	          100        	100
Overall Accuracy	100	          99	        99
Conclusion
This Customer Query Classification project demonstrates how Natural Language Processing (NLP), Machine Learning (ML), and Deep Learning (DL) can be leveraged to automate text classification. The ensemble approach combining XGBoost and LSTM significantly enhances the accuracy of classification compared to using individual models.

Key takeaways from the project:
- LSTM effectively captures the sequential patterns in text, improving classification accuracy.
- XGBoost provides interpretability and robustness in structured text classification.
- The ensemble model outperforms standalone models by leveraging the strengths of both approaches.
- The Streamlit web application enables real-time classification, making it easy to use in practical customer support systems.

This project highlights the potential of NLP in automating customer support workflows, optimizing response times, and improving customer service operations.
