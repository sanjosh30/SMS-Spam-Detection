📩 SMS Spam Detection

A machine learning project that classifies SMS messages as spam or ham (not spam) using Natural Language Processing and a TF-IDF + Multinomial Naive Bayes pipeline. The project covers the full workflow: data cleaning, exploratory data analysis, text preprocessing, comparison of 11 classifiers plus ensemble methods, and model export for deployment.

📌 Highlights
Cleaned and de-duplicated the SMS Spam Collection dataset (5,572 → 5,169 messages)
Built a custom NLP preprocessing pipeline (tokenization, stop-word removal, stemming)
Compared 11 classifiers, plus Voting and Stacking ensembles
Final model: TF-IDF + Multinomial Naive Bayes with 97.1% accuracy and 100% precision on the test set (zero legitimate messages flagged as spam)
Exported the trained vectorizer and model as .pkl files, ready to plug into a web app
📂 Dataset
File: spam.csv (SMS Spam Collection dataset)
Original size: 5,572 messages
After removing 403 duplicates: 5,169 messages
Class distribution:
Class	Count	Share
Ham (0)	4,516	87.37%
Spam (1)	653	12.63%

The dataset is imbalanced, which is why precision is tracked alongside accuracy throughout the project.

🔄 Project Workflow
1. Data Cleaning
Dropped the three empty unnamed columns
Renamed v1 → target and v2 → text
Label-encoded the target (ham = 0, spam = 1)
Checked for null values and removed duplicate rows
2. Exploratory Data Analysis
Class distribution pie chart
Added features: number of characters, words, and sentences per message
Compared distributions for ham vs. spam using histograms, a pair plot, and a correlation heatmap
Finding: spam messages are noticeably longer than ham (about 138 characters / 28 words on average, versus about 70 characters / 17 words for ham)
3. Text Preprocessing

Each message goes through a transform_text() function:

Convert to lowercase
Tokenize with NLTK
Keep only alphanumeric tokens (removes punctuation and symbols)
Remove English stop-words
Apply Porter stemming

Word clouds and top-30 word bar charts were generated for both spam and ham to visualize the most frequent terms in each class.

4. Model Building
Vectorization: TfidfVectorizer(max_features=3000)
Train/test split: 80/20, random_state=2
Compared three Naive Bayes variants (Gaussian, Multinomial, Bernoulli)
Compared 11 classifiers on accuracy and precision
Tried Voting (soft) and Stacking ensembles
5. Model Export

The fitted TF-IDF vectorizer and the Multinomial Naive Bayes model are saved with pickle as vectorizer.pkl and model.pkl.

📊 Results
Naive Bayes variants
Model	Accuracy	Precision
GaussianNB	87.43%	51.82%
MultinomialNB	97.10%	100.00%
BernoulliNB	98.36%	99.19%
Classifier comparison (sorted by precision)
Algorithm	Accuracy	Precision
K-Nearest Neighbors	90.52%	100.00%
Multinomial Naive Bayes	97.10%	100.00%
Random Forest	97.39%	98.26%
SVC (sigmoid kernel)	97.58%	97.48%
Extra Trees	97.49%	97.46%
Logistic Regression	95.55%	96.00%
XGBoost	97.00%	95.73%
Gradient Boosting	95.07%	93.07%
Bagging	95.84%	86.82%
Decision Tree	93.33%	84.16%
AdaBoost	92.17%	82.02%
Ensemble methods
Model	Accuracy	Precision
Voting (SVC + NB + Extra Trees, soft)	97.97%	98.35%
Stacking (final estimator: Random Forest)	97.97%	94.66%
Why Multinomial Naive Bayes?

In spam filtering, wrongly marking a genuine message as spam is costlier than letting a spam message through, so precision was the priority metric. Multinomial Naive Bayes reached 100% precision with strong accuracy (97.1%) and, on the test set, produced no false positives (confusion matrix: [[896, 0], [30, 108]]). It is also lightweight and fast, which makes it well suited for deployment, whereas the ensembles add model size and complexity without a clear precision gain.

🛠️ Tech Stack
Language: Python
Data handling: NumPy, Pandas
Visualization: Matplotlib, Seaborn, WordCloud
NLP: NLTK (tokenization, stop-words, Porter stemmer)
Machine learning: Scikit-learn, XGBoost
Model persistence: Pickle
📁 Project Structure
├── sms_spam_detection.ipynb   # Full notebook: cleaning, EDA, preprocessing, modeling
├── spam.csv                   # Dataset
├── vectorizer.pkl             # Fitted TF-IDF vectorizer
├── model.pkl                  # Trained Multinomial Naive Bayes model
└── README.md
▶️ How to Run
Clone or download this repository.
Install the dependencies:
bash
   pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud xgboost
Download the NLTK data (one time):
python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   nltk.download('stopwords')
Open and run the notebook:
bash
   jupyter notebook sms_spam_detection.ipynb
🔮 Using the Saved Model

Predict on a new message using the exported files. The transform_text function must be the same one defined in the notebook.

python
import pickle

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

message = "Congratulations! You have won a free prize. Call now to claim."

transformed = transform_text(message)       # same preprocessing as in the notebook
vector = tfidf.transform([transformed])
prediction = model.predict(vector)[0]

print("Spam" if prediction == 1 else "Not Spam")

Note: Load the pickle files with the same scikit-learn version used to train them, otherwise you may see version warnings or errors. Only load pickle files from sources you trust.

🚀 Future Improvements
Deploy as a Streamlit web app for real-time message classification
Improve recall (the model currently misses some spam messages) using threshold tuning or class weighting
Experiment with n-grams and larger vocabulary sizes
Try deep learning approaches (LSTM or transformer-based models)
