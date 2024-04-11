import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract data and labels
data = data_dict['data']
labels = data_dict['labels']

# Flatten the data
data_flat = []
for sample in data:
    # Concatenate left and right hand keypoints
    flattened_sample = []
    for hand_keypoints in sample:
        # Check if hand_keypoints is not None and is iterable
        if hand_keypoints is not None and hasattr(hand_keypoints, '__iter__'):
            flattened_sample.extend(hand_keypoints)
    data_flat.append(flattened_sample)

# Convert to NumPy arrays
data_flat = np.array(data_flat)
labels = np.array(labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_flat, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
pickle.dump(model, open('model2.pkl', 'wb'))
