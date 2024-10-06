import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from functools import partial
from neural_network import MLP, Value
import pickle
def to_value_list(arr):
    return [Value(float(x)) for x in arr]

def one_hot(y, num_classes=10):
    return [1.0 if i == y else 0.0 for i in range(num_classes)]

def softmax(x):
    exp_x = [v.exp() for v in x]
    sum_exp_x = sum(exp_x)
    return [v / sum_exp_x for v in exp_x]

def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-7
    return -sum(t * (p + epsilon).log() for t, p in zip(y_true, y_pred))


def train_batch(model_params, batch_data):
    # print("Training batch with process ID:", mp.current_process().pid)
    X_batch, y_batch = batch_data
    
    # Reconstruct the model from parameters
    input_size = 784
    hidden_size = 10
    output_size = 10
    model = MLP(input_size, [hidden_size, output_size])
    for p, p_data in zip(model.parameters(), model_params):
        p.data = p_data
    
    batch_loss = Value(0)
    for x, y in zip(X_batch, y_batch):
        x_value = to_value_list(x)
        y_pred = softmax(model(x_value))
        y_true = one_hot(y)
        loss = cross_entropy_loss(y_pred, y_true)
        batch_loss = batch_loss + loss
    batch_loss.backward()
    return batch_loss.data, [p.grad for p in model.parameters()]

def parallel_train(model, X_train, y_train, batch_size, learning_rate, num_processes):
    num_samples = len(X_train)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    batches = [(X_train[i:i+batch_size], y_train[i:i+batch_size]) 
               for i in range(0, num_samples, batch_size)]

    model_params = [p.data for p in model.parameters()]
    
    with mp.Pool(processes=num_processes) as pool:
        train_batch_partial = partial(train_batch, model_params)
        results = pool.map(train_batch_partial, batches)

    total_loss = sum(loss for loss, _ in results)
    avg_loss = total_loss / num_samples

    # Accumulate gradients
    for _, grads in results:
        for p, grad in zip(model.parameters(), grads):
            p.grad += grad

    # Update parameters with gradient clipping
    for p in model.parameters():
        grad_clip = 1.0  # Set an appropriate clipping value
        p.grad = max(min(p.grad, grad_clip), -grad_clip)  # Clip gradients
        p.data += -learning_rate * p.grad
        p.grad = 0

    return avg_loss

def main():
    # Load and preprocess MNIST data
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, cache=True)
    X = mnist.data[:1280] / 255.0
    y = mnist.target[:1280].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    input_size = 784
    hidden_size = 10
    output_size = 10
    model = MLP(input_size, [hidden_size, output_size])
    
    # Training loop
    print("Starting training...")
    epochs = 100
    batch_size = 32
    learning_rate = 1
    num_processes = 8 # mp.cpu_count()

    prev_loss = 100
    for epoch in range(epochs):
        if prev_loss > 5:
            learning_rate = 0.1
        elif prev_loss > 2.8 and prev_loss <= 5: 
            learning_rate = 0.08
        elif prev_loss > 2.2 and prev_loss <= 2.8:
            learning_rate = 0.05
        elif prev_loss > 2.0 and prev_loss <= 2.5:
            learning_rate = 0.01
        elif prev_loss > 1.7 and prev_loss <= 2.0:
            learning_rate = 0.001
        else:
            learning_rate = 0.0005
        print(f"Epoch {epoch+1}/{epochs}, Learning rate: {learning_rate}")
        avg_loss = parallel_train(model, X_train, y_train, batch_size, learning_rate, num_processes)
        prev_loss = avg_loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Evaluation
    print("Evaluating model...")
    correct = sum(1 for x, y in zip(X_test, y_test)
                  if np.argmax([o.data for o in softmax(model(to_value_list(x)))]) == y)
    accuracy = correct / len(y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    

if __name__ == "__main__":
    model = main()