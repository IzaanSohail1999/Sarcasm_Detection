from flask import Flask, request
from flask_cors import CORS
# import json
from nltk.stem import PorterStemmer
import pandas as pd
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
# import sklearn.externals.joblib as extjoblib
import joblib
Size = 0

app = Flask(__name__)
CORS(app)



@app.route('/query')
def get_sarcasm():
    query = request.args.get('query')
    print(query)

    query = pd.Series(query)
    
    query = query.apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))

    ps = PorterStemmer()
    query = query.apply(lambda x: x.split())
    query = query.apply(lambda x : ' '.join([ps.stem(word) for word in x]))

    tv1 = joblib.load("tfidfVectorizer.pkl")
    query = tv1.transform(query).toarray()

    lsvc = joblib.load("linearsvc_84acc.pkl")

    result_array = lsvc.predict(query)

    if str(result_array[0]) == '0':
        # print("if")
        result = "not sarcastic"
    else:
        # print("else")
        result = "sarcastic"
    
    print("predicted result:",result,"headline:",query)
    
    return {
        "result": result
    }

app.run(host="localhost")