# evaluate.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

(class_names) = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# load model
model = tf.keras.models.load_model("saved_model/cifar10_cnn")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype("float32")/255.0
y_test = y_test.flatten()

y_pred = np.argmax(model.predict(x_test), axis=1)

print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.ylabel("True")
plt.xlabel("Predicted")
plt.show()
