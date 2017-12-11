from os import listdir
from os.path import join
import numpy
from pandas import read_csv
import math
import keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,merge,average,Dense,Dropout,concatenate
from keras.layers import GRU,Lambda,LSTM,Reshape
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers.wrappers import Bidirectional,TimeDistributed
from keras.layers.embeddings import Embedding
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

#feature file for baseline models
def write_feature_file(features,labels):
        f = open('feature_file.tsv','w')
        for i in range(0,len(labels)):
                f.write('\t'.join(features[i].flatten()))
                f.write('\t'+str(labels[i])+'\n')
        f.close()

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),0:]
		dataX.append(a)
		dataY.append(dataset[i + look_back,0])
	return numpy.array(dataX), numpy.array(dataY)

# encode an array of text to an array of numbers
def encode_text(tokenizer,text_matrix,max_length):
	X_train = []
	for sample in text_matrix:
		text = tokenizer.texts_to_sequences(sample)
		seq = sequence.pad_sequences(text, maxlen=max_length)
		X_train.append(seq)
	return numpy.array(X_train)


# fix random seed for reproducibility
numpy.random.seed(7)
dirPath = 'data/authorFiles_normalized_score'
files = listdir(dirPath)
# load the dataset
all_features,all_labels = [],[]
max_look_back = 5
look_back = 5
text_dim = 64
max_length = 100
top_words = 50000
num_other_features =5
num_features = num_other_features+text_dim

input_sequence = Input((look_back,max_length,))
other_features = Input((look_back,num_other_features,))
def backend_reshape(x):
    return K.reshape(x, (-1,max_length))
def backend_reshape_2(x):
    return K.reshape(x, (-1,look_back,text_dim))

for fileName in files:
	dataframe = read_csv(join(dirPath,fileName), sep = '\t')
	post_type0 = dataframe['type0'].values.astype('float32')
	post_type1 = dataframe['type1'].values.astype('float32')
	post_type2 = dataframe['type2'].values.astype('float32')
	tslp = dataframe['tslp'].values.astype('float32')
	text = dataframe['text'].values.astype('str')
	score = dataframe['mpqa_score_np'].values.astype('float32')

	
	# normalize the dataset
	post_features = numpy.column_stack((tslp,post_type0,post_type1,post_type2))
	scaler = MinMaxScaler(feature_range=(0, 1))
	post_features = scaler.fit_transform(post_features)

	features = numpy.column_stack((score,post_features,text))
	features, labels = create_dataset(features, max_look_back)
	#print('feature shape : ',features.shape)
	all_features.extend(features)
	all_labels.extend(labels)
	

all_features = numpy.array(all_features)
all_labels = numpy.array(all_labels)

#write_feature_file(all_features,all_labels)

# split into train and test sets
train_size = int(len(all_features) * 0.9)
test_size = len(all_features) - train_size
start_idx = max_look_back-look_back
end_idx = max_look_back

trainX, testX = all_features[0:train_size,start_idx:end_idx,0:-1], all_features[train_size:len(all_features),start_idx:end_idx,0:-1]
trainY, testY = all_labels[0:train_size], all_labels[train_size:len(all_features)]

text_trainX, text_testX = all_features[0:train_size,start_idx:end_idx,-1], all_features[train_size:len(all_features),start_idx:end_idx,-1]


print('Shape of trainX: ',trainX.shape)
print('Shape of testX: ',testX.shape)
print('Shape of text_trainX: ',text_trainX.shape)
print('Shape of text_testX: ',text_testX.shape)

all_text=numpy.append(text_trainX.flatten(),text_testX.flatten())
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(all_text)

word_index = tokenizer.word_index
embeddings_index = {}
f = open('glove.6B.50d.txt',encoding='utf-8')
print('Starting to read embedding file')
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = numpy.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


embedding_matrix = numpy.zeros((len(word_index) + 1, 50))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


text_trainX = encode_text(tokenizer,text_trainX,max_length)
text_testX = encode_text(tokenizer,text_testX,max_length)

text_model = Sequential()
text_model.add(Lambda(backend_reshape,input_shape=(look_back,max_length),output_shape=(max_length,)))
text_model.add(Embedding(len(word_index)+1, 50, weights=[embedding_matrix],trainable = True, input_length=max_length))
text_model.add(LSTM(text_dim,return_sequences=True))
text_model.add(Dropout(0.7))
text_model.add(LSTM(text_dim))
text_model.add(Dropout(0.7))
text_model.add(Lambda(backend_reshape_2,output_shape=(look_back,text_dim,)))


text_feature = text_model(input_sequence)


# create and fit the LSTM networks
x1=(LSTM(256, input_shape=(look_back, num_other_features)))(other_features)
x1=Dropout(0.6)(x1)
x1=(Dense(16))(x1)

x2=(LSTM(256, input_shape=(look_back, text_dim)))(text_feature)
x2=Dropout(0.6)(x2)
x2=(Dense(16))(x2)

x = concatenate([x1,x2])
x = (Dense(1))(x)


model = Model(inputs=[other_features,input_sequence], outputs=[x])
model.compile(loss='mean_absolute_error', optimizer='adam')
print(model.summary())
model.fit([trainX,text_trainX], trainY, validation_data=[[testX,text_testX],testY], epochs=50, batch_size=32)
