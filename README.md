# 🏗️ BuildCast - Building Energy Prediction

This project is a **machine learning pipeline** for predicting building energy usage using **synthetic datasets**.  
It includes **data preprocessing, model training (LightGBM + PyTorch), evaluation, and visualization** of results.

---

## 📌 Features
- Preprocessing with **StandardScaler** and **OneHotEncoder**  
- **LightGBM baseline model** for regression  
- **PyTorch neural network model** implementation  
- **Evaluation metrics**: MAE, RMSE, R²  
- **Graphical visualization** of predicted vs actual values  
- **Synthetic dataset generation** with 20,000 data points for better visualization  

---

## 📂 Project Structure
📦 Building-Energy-Prediction
- 📜 main.py # Main training + evaluation pipeline
- 📜 requirements.txt # Dependencies
- 📜 buildings.csv # Example dataset
- 📜 README.md # Project documentation

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/QuantumCoderrr/BuildCast.git
cd BuildCast
```

2. Create a virtual environment & install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

pip install -r requirements.txt
```
---

## ▶️ Usage
1. Run with existing dataset
```bash
python main.py
```
2. Generate & use synthetic dataset (20,000 points)
```bash
python generate_dataset.py
python main.py
```
---

## 📊 Visualization
The model outputs a scatter plot comparing actual vs predicted energy usage.
This helps visualize how well the model generalizes on test data.

## 📈 Example Output
Metrics (MAE, RMSE, R²) printed on console
Scatter plot of predictions vs actual values

## 🛠️ Tech Stack
Python 3.11+
Pandas, NumPy
Scikit-learn
LightGBM
PyTorch
Matplotlib

---

## 👩‍💻 Author
Sandip Ghosh, Aishika Majumdar and Sandhita Poddar

---

## 📜 License
This project is licensed under the MIT License.
Feel free to use, modify, and share with proper attribution.
