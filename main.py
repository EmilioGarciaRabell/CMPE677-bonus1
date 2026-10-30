from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def main():
    iris = datasets.load_iris()
    U = iris.data
    Y = np.zeros((len(iris.target), 3))
    Y[np.arange(iris.target.size), iris.target] = 1

    # A) Divide the data into two sets
    rng = np.random.default_rng(seed=42)
    indices = np.arange(len(U))
    rng.shuffle(indices)

    split_idx80 = int(len(U) * 0.8)

    train_indices = indices[:split_idx80]
    test_indices = indices[split_idx80:]

    U_train = U[train_indices]
    Y_train = Y[train_indices]

    U_test = U[test_indices]
    Y_test = Y[test_indices]

    # B) Perform classification using k nearest neighbors 
    
    Y_test_pred = K_classifier(U_test, U_train, Y_train, 5)
    Y_train_pred = K_classifier(U_train, U_train, Y_train, 5)
    print_results(Y_test_pred, Y_train_pred, Y_test, Y_train, "K nearest neighbors")
    
    # C) Perform classification on the data using a linear regression model with the pseudoinverse method. Report the training and test accuracy
    Y_train_pred = linear_regression(U_train, U_train, Y_train)
    Y_test_pred = linear_regression(U_test, U_train, Y_train)
    print_results(Y_test_pred, Y_train_pred, Y_test, Y_train, "Linear Regression")

    # D) Logistic regression
    Y_train_pred_logistic, mse_history = logistic_regression(
        U_train, U_train, Y_train,
        iterations=5000,
        learning_rate=0.01,
        lambda_val=0
    )

    Y_test_pred_logistic, _ = logistic_regression(
        U_test, U_train, Y_train,
        iterations=5000,
        learning_rate=0.01,
        lambda_val=0
    )

    print_results(
        Y_test_pred_logistic,
        Y_train_pred_logistic,
        Y_test,
        Y_train,
        "Logistic Regression"
    )

    plt.plot(range(len(mse_history)), mse_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Logistic Regression Training MSE")
    plt.savefig("mse_history.png")
    plt.close()

    # E) Ridge regularization
    lambdas = [0.01, 0.1, 1, 10]

    train_accuracies = []
    test_accuracies = []

    for lambda_val in lambdas:
        Y_train_pred, _ = logistic_regression(
            U_train, U_train, Y_train,
            iterations=5000,
            learning_rate=0.01,
            lambda_val=lambda_val
        )

        Y_test_pred, _ = logistic_regression(
            U_test, U_train, Y_train,
            iterations=5000,
            learning_rate=0.01,
            lambda_val=lambda_val
        )

        train_accuracy = np.sum(Y_train_pred == np.argmax(Y_train, axis=1)) / len(Y_train)

        test_accuracy = np.sum(Y_test_pred == np.argmax(Y_test, axis=1)) / len(Y_test)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print_results(
            Y_test_pred,
            Y_train_pred,
            Y_test,
            Y_train,
            f"Logistic Regression with λ={lambda_val}"
        )

    plt.figure(figsize=(10, 6))

    plt.plot(
        lambdas,
        train_accuracies,
        marker='o',
        label="Training Accuracy"
    )

    plt.plot(
        lambdas,
        test_accuracies,
        marker='o',
        label="Test Accuracy"
    )

    plt.xscale("log")
    plt.xlabel("λ")
    plt.ylabel("Accuracy")
    plt.title("Impact of Ridge Regularization on Logistic Regression")
    plt.legend()
    plt.savefig("ridge_regularization_results.png")
    plt.close()

    # F) Confusion matrix for logistic regression test set
    confusion_matrix = np.zeros((3, 3), dtype=int)

    for i in range(len(Y_test_pred_logistic)):
        actual_class = np.argmax(Y_test[i])
        predicted_class = Y_test_pred_logistic[i]

        confusion_matrix[actual_class, predicted_class] += 1

    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=[0, 1, 2]
    )

    cm_display.plot()
    plt.title("Logistic Regression Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("Confusion Matrix:")
    print(confusion_matrix)


def print_results(Y_test_pred, Y_train_pred, Y_test, Y_train, model_type):
    test_accuracy = np.sum(Y_test_pred == np.argmax(Y_test, axis=1)) / len(Y_test)
    train_accuracy = np.sum( Y_train_pred == np.argmax(Y_train, axis=1)) / len(Y_train)
    print("\n")
    print(f"Training accuracy ({model_type}):", round(train_accuracy, 3))
    print("Test accuracy:", round(test_accuracy, 3))
    print("\n---------------------------------------------------------")


def K_classifier(U, U_trainK, Y_trainK, K):
    # Go through Ui in U_test
    Y_pred = []
    for ui in U:
        # calculate eucledian distance using U_trainK
        distances = []
        for j, x in enumerate(U_trainK):
            difference = ui - x
            current_distance = np.sqrt(np.sum(difference**2))
            distances.append((current_distance, np.argmax(Y_trainK[j])))
        distances.sort(key=lambda x: x[0])

        classes_count = dict()
        for neighbor in range(K):

            current_class = distances[neighbor][1]
            if current_class in classes_count:
                classes_count[current_class] += 1
            else:
                classes_count[current_class] = 1

        Y_pred.append(max(classes_count, key=classes_count.get))

    Y_pred = np.array(Y_pred)
    return Y_pred

def linear_regression(U, U_trainL, Y_trainL):
    ones = np.ones((U_trainL.shape[0], 1))



    # Augment the matrix U
    augmented_Utrain = np.column_stack((ones, U_trainL))

    W = np.linalg.pinv(augmented_Utrain) @ Y_trainL

    ones = np.ones((U.shape[0], 1))
    augmented_U = np.column_stack((ones, U))

    Y_pred = augmented_U@W

    predicted_classes = np.argmax(Y_pred, axis=1)
    return predicted_classes

def logistic_regression(U, U_train, Y_train, iterations, learning_rate, lambda_val):
    mse_history = []

    ones = np.ones((U_train.shape[0], 1))
    augmented_Utrain = np.column_stack((ones, U_train))
    
    W = np.zeros((augmented_Utrain.shape[1], Y_train.shape[1]))
    for _ in range(iterations):
        Z = augmented_Utrain @ W
        Y_pred = sigmoid(Z)

        mse = np.mean((Y_train - Y_pred) ** 2)
        mse_history.append(mse)
        gradient = (augmented_Utrain.T @ (Y_pred - Y_train)) / len(Y_train)
        regularization = lambda_val * W.copy()
        regularization[0, :] = 0
        W = W - learning_rate * (gradient + regularization)

    ones = np.ones((U.shape[0], 1))
    augmented_U = np.column_stack((ones, U))

    Z = augmented_U @ W
    Y_pred = sigmoid(Z)
    predicted_classes = np.argmax(Y_pred, axis=1)
    return predicted_classes, mse_history


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

if __name__ == "__main__":
    main()