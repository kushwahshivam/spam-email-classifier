Setup Instructions
1️⃣ Clone the repository

git clone https://github.com/kushwahshivam/spam-email-classifier.git
cd spam-email-classifier

2️⃣ Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

3️⃣ Install required packages
pip install -r requirements.txt

Prepare the Dataset

Place a CSV file at data/dataset.csv.

Required columns:

text – email content

label – either spam or ham

Example:

label,text
ham,"Hey! Are we still meeting for lunch today?"
spam,"Win a brand new car! Click here to claim your prize now."

Train the Model

From the backend folder, run:

python train.py --data ../data/dataset.csv --model_dir models


Splits data into train/test

Trains a TF-IDF + Multinomial Naive Bayes pipeline

Saves the trained pipeline to backend/models/spam_model.pkl

Run the Backend API
cd backend
python app.py


Flask API will run at http://127.0.0.1:5000/

Endpoints:

/ → Frontend page (index.html)

/api/predict → POST email text to get spam/ham prediction
