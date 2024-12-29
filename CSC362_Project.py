'''
Sebastian J Courdy
'''
import numpy as np # Importing numpy
import pandas as pd # Importing pandas
import matplotlib.pyplot as plt # Importing matplot to plot graphs

# Part 1
df = pd.read_csv("pd_hoa_activities_cleaned.csv") # Load the file

unique_tasks = df['task'].unique() # Gets unique values in task column
task_dict = {task: idx for idx, task in enumerate(unique_tasks)} # Creates a dictionary that matches each task to an index

df['task_encoded'] = df['task'].map(task_dict) # Maps task column to the corresponding index
df['class_encoded'] = df['class'].apply(lambda x: 0 if x == 'HOA' else 1) # Assigns 0 for HOA and 1 to others

X = df[['task_encoded', 'duration', 'age']].values # Converts relevant features into an array
y = df['class_encoded'].values # Converts target feature variable into an array

split = 0.8 # 80/20 split
training_size = int(len(df) * split) # Calculates number of training examples 


X_train = X[:training_size] # Training features
y_train = y[:training_size] # Training levels


X_test = X[training_size:]  # Test features
y_test = y[training_size:]  # Test labels

# Part 2
print("x_train first five values:\n", X_train[:5]) # Prints first 5 values of x_train
print("\ny_train first five values:\n", y_train[:5]) # Prints first 5 values of y_train

print("\nx_train type:", type(X_train)) # Prints the type of x_train (array in numpy)
print("\ny_train type:", type(y_train)) # Prints the type of y_train (array in numpy)

print("\nx_train shape:", X_train.shape) # Prints the shape of x_train
print("\ny_train shape:", y_train.shape) # Prints the shape of y_train

# Part 3
# Histogram showing age
plt.figure(figsize = (10, 6)) 
plt.hist(df['age'], bins = 30, color = 'skyblue', edgecolor = 'black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Histogram')
plt.show()

# Bar chart for task
task_frequency = df["task"].value_counts()
plt.figure(figsize=(10, 6))
task_frequency.plot(kind = "bar", color = "skyblue", edgecolor = "black")
plt.title("Task Frequency Distribution")
plt.xlabel("Task")
plt.ylabel("Frequency")
plt.grid(axis = "y", linestyle = "--", alpha = 0.7)
plt.show()

# Histogram showing duration of tasks
plt.figure(figsize=(10, 6))
plt.hist(df['duration'], bins = 30, color = 'skyblue', edgecolor = 'black')
plt.xlabel('Duration of Tasks')
plt.ylabel('Frequency')
plt.title('Duration of Tasks Histogram')
plt.show()

# Part 4
# Implements the sigmoid function
def sigmoid(x):
    max_value = 500 # Max value to avoid an error
    min_value = -500 # Min value to avoid an error
    
    x = np.minimum(np.maximum(x, min_value), max_value) # Ensures values outisde the range dont cause overflow
    return 1 / (1 + np.exp(-x))

#Part 5
# Implement the cost function for the logistic Regression model
def cost_function_log_refression(X, y, theta):
    m = len(y) # Gets the number of training examples

    x = np.dot(X, theta)
    h = 1 / (1 + np.exp(-x)) # Applies sigmoid function to get the predicted probabilities

    cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) # Calculates the chance of a negative log
    
    return cost

np.random.seed(0) # Set to 0
X = np.random.rand(100, 3) # 100 examples and three features
X[:, 0] = 1 # The bias term
y = np.random.randint(0, 2, 100) # Either 0 or 1 binary labels
theta = np.zeros(X.shape[1]) # Initializes theta as 0
alpha = 0.1 # The learning rate 
num_of_iterations = 100 # Number of iterations 

costs = []

for i in range(num_of_iterations):
    x = np.dot(X, theta) # Computes the linear combination
    h = 1 / (1 + np.exp(-x)) # Applies the sigmoid function
    gradient = (1 / len(y)) * np.dot(X.T, (h - y)) # Computes the gradient
    theta -= alpha * gradient # Updates theta parameter
    cost = cost_function_log_refression(X, y, theta) # Calculates the cost after updating the parameters
    costs.append(cost) # Append the current cost to the list

plt.figure(figsize = (10, 6))
plt.plot(range(num_of_iterations), costs, color = "blue", label = "Cost")
plt.title("Cost Function over Iterations", fontsize = 14)
plt.xlabel("Iterations", fontsize = 12)
plt.ylabel("Cost", fontsize = 12)
plt.grid(True, linestyle = "--", alpha = 0.7)
plt.legend()
plt.show()

# Part 6
def logistic_regression_gradient(X, y, theta, alpha, num_of_iterations):
    m = len(y) # Gets the number of training examples
    costs = [] # Initializes an empty list
    
    for i in range(num_of_iterations):
        x = np.dot(X, theta) # Computes linear combination
        h = sigmoid(x) # Applies the sigmoid function

        gradient = (1 / m) * np.dot(X.T, (h - y)) # Computes the gradient
        theta -= alpha * gradient # Updates the the theta parameter

        final_cost = cost_function_log_refression(X, y, theta) # Calculates the cost

    return theta, final_cost

initializations = [
    (np.zeros(X.shape[1]), 0.01), # Initializes with 0's and a learning rate of 0.01
    (np.ones(X.shape[1]), 0.1), # Initializes with 1's and a learning rate of 0.01
    (np.random.rand(X.shape[1]), 0.5) # Has a random initialization and a learning rate of 0.5
]

num_of_iterations = 100

print()
for idx, (theta_init, alpha) in enumerate(initializations, 1):
    theta, final_cost = logistic_regression_gradient(X, y, theta_init.copy(), alpha, num_of_iterations)
    print("Test", idx, ": Final cost =", final_cost, ", Learning rate =", alpha)

# Part 7
optimal_theta = None # Stores the lowest cost
lowest_cost = float('inf') # Initial infinity so everything will be lower

if final_cost < lowest_cost:
        lowest_cost = final_cost
        optimal_theta = theta # Save the parameter with the lowest cost

print("\nOptimal Parameters (theta):", optimal_theta)
print("\nLowest Cost:", lowest_cost)

# Part 8
def predict(X, theta, y = None, verbose = True):
    probabilities = sigmoid(np.dot(X, theta))  # Computes the probabilities
    predictions = (probabilities >= 0.5).astype(int)  # Converts probability to prediction

    if y is not None:
        accuracy = np.mean(predictions == y) * 100  # Calculate accuracy as percentage of correct predictions
        
        if verbose:
            print("\nModel Accuracy:", accuracy, "%")
        return predictions, accuracy
    
    return predictions

predictions, accuracy = predict(X, optimal_theta, y)

# Part 9
def predict_test(X, y, theta):
    predictions = predict(X, theta, y, verbose = False)[0] # Calls predict function to get predictions
    accuracy = np.mean(predictions == y) * 100 # Converts to a percentage 

    return accuracy

training_accuracy = predict_test(X_train, y_train, optimal_theta) # Calculates accuracy of training model
print("\nTraining Set Accuracy:", training_accuracy, "%") # Prints training model accuracy
test_accuracy = predict_test(X_test, y_test, optimal_theta) # Calculates the test model accuracy
print("\nTest Set Accuracy:", test_accuracy, "%") # Prints test model accuracy


from sklearn.svm import SVC # Importing SVC from scikit

# Part 1 
svm_model = SVC(kernel = 'linear') # Use a linear kernel
svm_model.fit(X_train, y_train) # Training the model

# Part 2
svm_training_predictions = svm_model.predict(X_train) # Using the trained model to make predictions

# Part 3
svm_training_accuracy = svm_model.score(X_train, y_train) * 100 # Calculates accuracy as a percentage on training set
svm_test_accuracy = svm_model.score(X_test, y_test) * 100 # Calculates accuracy as a percentage on test set

print("\nSVM Training Set Accuracy:", svm_training_accuracy, "%") # Outputs training set accuracy
print("\nSVM Test Set Accuracy:", svm_test_accuracy, "%") # Outputs test set accuracy
