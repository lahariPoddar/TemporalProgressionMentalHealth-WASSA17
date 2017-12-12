import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor

train_fraction = 0.9
text_fields = set([5, 11, 17, 23, 29])
label_field = 30
numericColumns = []

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
scaler = MinMaxScaler(feature_range=(0,1))

def getTexts(data):
	texts = []
	'''
	for i in range(0, data.shape[1]):
		print('field:',i, data[i][0])
	'''
	for i in range(0, data.shape[0]):
		text = ''
		for text_field in text_fields:
			#print(data[i][text_field])
			text += data[i][text_field]
		texts.append(text)
	return texts

def getFeatures(data, forTrain=False):
	texts = getTexts(data)
	bowFeatures = getBOWFeatures(texts, forTrain)
	numericFeatures = data[:,numericColumns]
	if forTrain:
		numericFeatures = scaler.fit_transform(numericFeatures)
	else:
		numericFeatures = scaler.transform(numericFeatures)
	features = np.concatenate((numericFeatures, bowFeatures), axis=1)
	#features = bowFeatures
	return features

def getBOWFeatures(texts, forTrain=False):
	clean_data = []
	for text in texts:
		clean_data.append(review_to_words(text))
	if forTrain:
		data_features = vectorizer.fit_transform(clean_data)
	else:
		data_features = vectorizer.transform(clean_data)
	data_features = data_features.toarray()
	return data_features
	
def review_to_words( raw_review ):   
	letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
	words = letters_only.lower().split()                             
	stops = set(stopwords.words("english"))                  
	meaningful_words = [w for w in words if not w in stops]   
	return( " ".join( meaningful_words ))

data = pd.read_csv("feature_file_5.tsv", header=None, delimiter="\t")
print(data.shape)
#print(data[17][0])

for column in range(0, data.shape[1]):
	if column not in text_fields and column != label_field:
		numericColumns.append(column)

train_size = int(data.shape[0]*train_fraction)
train = data.iloc[0:train_size,:]
test = data.iloc[train_size:data.shape[0],:]

train_X = getFeatures(train.values, True)
test_X = getFeatures(test.values)

train_Y = train.values[:,label_field]
test_Y = test.values[:,label_field]

#clf = linear_model.LinearRegression()
#clf = SVR(kernel='rbf', C=1e3, gamma=0.1)
clf = ensemble.RandomForestRegressor(n_estimators=100)
#clf =  DecisionTreeRegressor(max_depth=5)
clf.fit(train_X, train_Y)


predicted_Y = clf.predict(test_X)

mae = mean_absolute_error(test_Y, predicted_Y)

print('mae is:', mae)


