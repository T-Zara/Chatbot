import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Download Only Once
nltk.download('punkt')
nltk.download('wordnet')


# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the dataset
# data = [
#    {
#        "question": "What makes Jessup Cellars unique compared to other tasting rooms in Yountville?",
#        "answer": "Jessup Cellars has a casual and inviting atmosphere and was the first tasting room opened in Yountville in 2003. You have the option of sitting inside our stunning art gallery or you may choose to enjoy the patio with giant umbrellas. We also have space available for private groups and special accomodations and snacks for your children. Our fine art is meticulously curated by our lead artist Jermaine Dante who exhibits his colorful creations in large formats in our spacious gallery where you can take in, or take home the inspiring art while imbibing your favorite Jessup wines."
#    }
# ]

with open('corpus.json') as file:
    data = json.load(file)

# Prepare data
words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

for item in data:
    word_list = nltk.word_tokenize(item['question'])
    words.extend(word_list)
    documents.append((word_list, item['answer']))
    if item['answer'] not in classes:
        classes.append(item['answer'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build neural network
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
mdl = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', mdl)
