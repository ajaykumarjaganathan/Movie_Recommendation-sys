# Streamlit_Movie_Recommendation
# **Movie Recommendation System**  

A **collaborative filtering-based** movie recommendation system built using:  
- **K-Nearest Neighbors (KNN)** for user-based recommendations  
- **Singular Value Decomposition (SVD)** for matrix factorization  
- **MovieLens 100K Dataset**  

## **Features**  
✅ Personalized movie recommendations  
✅ User-based & model-based collaborative filtering  
✅ Evaluation using RMSE (Root Mean Squared Error)  
✅ Flask API for deployment  

## **Installation**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/movie-recommender.git
   cd movie-recommender
   ```  

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

## **Dataset**  
Download the **MovieLens 100K dataset** from:  
- [GroupLens](https://grouplens.org/datasets/movielens/)  
- [Kaggle](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)  

Place `ratings.csv` and `movies.csv` in the `data/` folder.  

## **Usage**  
### **1. Training & Evaluation**  
Run the Jupyter notebook:  
```bash
jupyter notebook Movie_Recommendation_System.ipynb
```  
- **Exploratory Data Analysis (EDA)**  
- **KNN (User-Based CF) Training**  
- **SVD (Matrix Factorization) Training**  
- **Generating Recommendations**  

### **2. Running the Flask API**  
```bash
python app.py
```  
Send a POST request to `http://127.0.0.1:5000/recommend` with JSON:  
```json
{
    "user_id": 1,
    "n": 5
}
```  
**Response:**  
```json
[
    {"Movie": "Shawshank Redemption, The (1994)", "Predicted Rating": 4.5},
    {"Movie": "Godfather, The (1972)", "Predicted Rating": 4.4},
    {"Movie": "Pulp Fiction (1994)", "Predicted Rating": 4.3},
    {"Movie": "Schindler's List (1993)", "Predicted Rating": 4.2},
    {"Movie": "Fight Club (1999)", "Predicted Rating": 4.1}
]
```  

## **Technologies Used**  
- **Python**  
- **Pandas, NumPy, Matplotlib, Seaborn** (Data Processing & Visualization)  
- **Surprise** (Scikit-learn for recommender systems)  
- **Flask** (API Deployment)  

## **Future Improvements**  
- Hybrid (Content + Collaborative) Recommendations  
- Real-time recommendations using **Apache Kafka + Spark**  
- **Neural Collaborative Filtering (NCF)** with TensorFlow  

## **License**  
MIT License  

---
**Contributions welcome!** 🎥🍿  

🔗 **(https://github.com/ajaykumarjaganathan/Movie_Recommendation-sys.git)**  
📧 **Contact**: ajaykumarimpex833.com  
x
