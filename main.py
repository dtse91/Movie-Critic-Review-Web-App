from flask import Flask,render_template,request,url_for

#EDA Packages
import pandas as pd
import numpy as np

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/",methods=['POST'])
def predict():
#	 Link to dataset from github
    df= pd.read_csv('critics.csv')
    df_data = df[["quote","fresh"]]
	# Features and Labels
    df_x = df_data.quote.values.astype('S')
    df_y = df_data.fresh
    print(df_y)
    
    # Extract Feature With CountVectorizer
    corpus = df_x
    tv = TfidfVectorizer(min_df=0.001, stop_words='english', ngram_range=(1,3))
    X = tv.fit_transform(corpus) # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB(alpha=1)
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	# ytb_model = open("naivebayes_spam_model.pkl","rb")
	# clf = joblib.load(ytb_model)

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = tv.transform(data).toarray()
        my_prediction = int(clf.predict(vect))
        
    return render_template('results.html',prediction = my_prediction,comment = comment)
	


if __name__ == '__main__':
	app.run(host="127.0.0.1",port=8080,debug=True)
    
    
#https://stackoverflow.com/questions/19071512/socket-error-errno-48-address-already-in-use