# Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import re  # import regular expressions to process emails
import numpy
from scipy.io import loadmat

import svm_funcs

# ==================== Part 1: Email Preprocessing ====================
print("\nPreprocessing sample email (emailSample1.txt)\n")

# Extract Features
with open('./emailSample1.txt') as fid:
    file_contents = fid.read()

word_indices = svm_funcs.process_email(file_contents, False)

# Print Stats
print('-------------')
print('Word Indices:')
print('-------------')
print(word_indices)

# ==================== Part 2: Feature Extraction ====================
print("\nExtracting features from sample email (emailSample1.txt)\n")

# Extract Features
features = svm_funcs.email_features(word_indices)

# Print Stats
print("Length of feature vector: %d" % len(features))
print("Number of non-zero entries: %d" % sum(features > 0))

# =========== Part 3: Train Linear SVM for Spam Classification ========
# Load the Spam Email dataset
# You will have X, y in your environment
data = loadmat("./spamTrain.mat")
x_train = data['X'].astype(float)
y_train = data['y']
y_train = y_train.reshape(-1)
num_examples, num_features = x_train.shape
print("Spam example Ex.6. training #examples:", num_examples, "#features:", num_features)


print("\nTraining Linear SVM (Spam Classification)")
print("This may take 1 to 2 minutes...\n")

reg_C = 0.1
model = svm_funcs.svm_train(svm_funcs.linear_kernel, x_train, y_train, reg_C, tol=1e-3, max_passes=20)

train_pred = svm_funcs.svm_predict(model, x_train)  # Compute the training accuracy
train_acc = numpy.mean(train_pred == y_train)
print("Training Accuracy: %.2f" % (train_acc*100))

# =================== Part 4: Test Spam Classification ================
# Load the test dataset
# You will have Xtest, ytest in your environment
data = loadmat("./spamTest.mat")
x_test = data['Xtest'].astype(float)
y_test = data['ytest']
y_test = y_test.reshape(-1)

print("\nEvaluating the trained Linear SVM on a test set...")
test_pred = svm_funcs.svm_predict(model, x_test)
test_acc = numpy.mean(test_pred == y_test)
print("\nTest Accuracy: %.2f" % (test_acc*100))

# ================= Part 5: Top Predictors of Spam ====================
# Sort the weights and obtin the vocabulary list
# NOTE some words have the same weights, so their order might be different than in the text above
idx = numpy.argsort(model['w'])
top_idx = idx[-15:][::-1]
vocab_list = svm_funcs.get_vocab_list()

print("\nTop predictors of spam:")
print("%-15s %-15s" % ('word', 'weight'))
print("----" + " "*12 + "------")
for word, w in zip(numpy.array(vocab_list)[top_idx], model['w'][top_idx]):
    print("%-15s %0.2f" % (word, w))


# # =================== Part 6: Try Your Own Emails =====================
filename = './emailSample1.txt'

with open(filename) as fid:
    file_contents = fid.read()

word_indices = svm_funcs.process_email(file_contents, verbose=False)
x = svm_funcs.email_features(word_indices)
p = svm_funcs.svm_predict(model, x)

print("\nProcessed %s\nSpam Classification: %s" % (filename, 'spam' if p else 'not spam'))
