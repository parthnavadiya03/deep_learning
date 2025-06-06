# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(filepath):
    texts = []
    labels = []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                text, label = line.rsplit(';', 1)
                texts.append(text)
                labels.append(label)
    return pd.DataFrame({'text': texts, 'label': labels})

# Load each file
train_data = load_data("/Users/parthnavadiya/Desktop/Exams/Deep Learning/archive/train.txt")
val_data = load_data("/Users/parthnavadiya/Desktop/Exams/Deep Learning/archive/val.txt")
test_data = load_data("/Users/parthnavadiya/Desktop/Exams/Deep Learning/archive/test.txt")

# Encode labels consistently across datasets
label_encoder = LabelEncoder()

# Fit on train labels only, then transform all
train_data['label'] = label_encoder.fit_transform(train_data['label'])
val_data['label'] = label_encoder.transform(val_data['label'])
test_data['label'] = label_encoder.transform(test_data['label'])

# Check shapes and classes distribution (optional)
print(f"Train shape: {train_data.shape}")
print(f"Validation shape: {val_data.shape}")
print(f"Test shape: {test_data.shape}")
print("Label classes:", label_encoder.classes_)


# %%
# Encode emotion labels to integers
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['label'])
val_labels = label_encoder.transform(val_df['label'])
test_labels = label_encoder.transform(test_df['label'])

# %%
# Save the classes for later use
emotion_classes = label_encoder.classes_
print("Emotion classes:", list(emotion_classes))

# %%
# Set parameters
vocab_size = 10000  # max number of words to keep
max_len = 50        # max length of a sentence (can be adjusted)
oov_token = "<OOV>" # for out-of-vocabulary words

# Create tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_df['text'])

# %%
# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_df['text'])
val_sequences = tokenizer.texts_to_sequences(val_df['text'])
test_sequences = tokenizer.texts_to_sequences(test_df['text'])

# Pad sequences
train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
val_padded = pad_sequences(val_sequences, maxlen=max_len, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

# Check shape
print("Train padded shape:", train_padded.shape)


# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Define model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=50),
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(len(emotion_classes), activation='softmax')  # output layer
])

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Model summary
model.summary()


# %%
# Build the model manually by giving input shape
model.build(input_shape=(None, 50))


# %%
# Print summary again
model.summary()


# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenizer setup
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['text'])

# Tokenize and pad sequences
max_length = 50
x_train = pad_sequences(tokenizer.texts_to_sequences(train_data['text']), maxlen=max_length, padding='post')
x_val = pad_sequences(tokenizer.texts_to_sequences(val_data['text']), maxlen=max_length, padding='post')
x_test = pad_sequences(tokenizer.texts_to_sequences(test_data['text']), maxlen=max_length, padding='post')

# One-hot encode labels
y_train = to_categorical(train_data['label'])
y_val = to_categorical(val_data['label'])
y_test = to_categorical(test_data['label'])


# %%
print(x_train.shape)  # Should be (num_samples, sequence_length)
print(y_train.shape)  # Should be (num_samples,)
print(y_train[:10])   # Should print integers like [0, 1, 2, 3]


# %%
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=50),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# %%
import numpy as np
print(np.sum(y_train, axis=0))  # Counts of each class in train labels

# %%
from sklearn.utils import class_weight
import numpy as np

# Convert one-hot labels back to class indices
y_train_labels = np.argmax(y_train, axis=1)

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weight_dict = dict(enumerate(class_weights))

print(class_weight_dict)


# %%
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=32,
    class_weight=class_weight_dict
)



