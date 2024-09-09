import torch
import torch.nn.functional as F
import argparse


def create_vocabulary(file_path):
    """
    Creates a character-to-index and index-to-character mapping from a file containing names.

    Args:
        file_path (str): Path to the file containing names, one per line.

    Returns:
        tuple:
            - char_to_idx (dict): A dictionary mapping characters to their indices.
            - idx_to_char (dict): A dictionary mapping indices back to characters.
            - vocab_size (int): The size of the vocabulary.
    """
    names = open(file_path, 'r').read().splitlines()

    char = set(''.join(names))

    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(char))}
    char_to_idx['.'] = 0

    idx_to_char = {v: k for k, v in char_to_idx.items()}

    return char_to_idx, idx_to_char, len(idx_to_char)


def create_dataset(file_path):
    """
    Creates a dataset from a file containing names, converting characters to indices.

    Args:
        file_path (str): Path to the file containing names, one per line.

    Returns:
        tuple:
            - X (torch.Tensor): Input tensor containing character indices for the model.
            - y (torch.Tensor): Output tensor containing the next character indices.
    """
    names = open(file_path, 'r').read().splitlines()

    X = []
    y = []

    for name in names:
        x_ = CONTEXT_SIZE * [0]
        for ch in name + '.':
            X.append(x_)
            y.append(char_to_idx[ch])
            x_ = x_[1:] + [char_to_idx[ch]]

    X = torch.tensor(X)
    y = torch.tensor(y)

    return X, y


class LinearLayer:
    """
    A simple Multilayer Perceptron layer (fully connected layer).

    Args:
        in_features (int): Number of input features.
        hidden_units (int): Number of units in the hidden layer.
    """

    def __init__(self, in_features, hidden_units):
        self.W = torch.randn((in_features, hidden_units)) * 0.01
        self.b = torch.randn(hidden_units) * 0

    def __call__(self, X):
        """
        Forward pass of the mlp layer.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the linear transformation.
        """
        self.out = X @ self.W + self.b
        return self.out

    def params(self):
        """
        Returns the parameters of the linear layer.

        Returns:
            list: List containing weights and bias.
        """
        return [self.W] + [self.b]


class BatchNormalization:
    """
    Batch Normalization layer.

    Args:
        dim (int): Dimension of the input features.
        training (bool): Whether the layer is in training mode or not.
    """

    def __init__(self, dim, training=True):
        self.training = training

        self.batch_gain = torch.ones(dim)
        self.batch_bias = torch.zeros(dim)

        self.all_batch_mean = torch.zeros(dim)
        self.all_batch_std = torch.ones(dim)

    def __call__(self, X):
        """
        Forward pass for the batch normalization layer.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying batch normalization.
        """
        if self.training:
            batch_mean = X.mean(0, keepdim=True)
            batch_std = X.std(0, keepdim=True)
            with torch.no_grad():
                self.all_batch_mean = 0.99 * self.all_batch_mean + 0.01 * batch_mean
                self.all_batch_std = 0.99 * self.all_batch_std + 0.01 * batch_std
        else:
            batch_mean = self.all_batch_mean
            batch_std = self.all_batch_std

        self.out = self.batch_gain * (X - batch_mean) / batch_std + self.batch_bias
        return self.out

    def params(self):
        """
        Returns the parameters of the batch normalization layer.

        Returns:
            list: List containing gain and bias parameters.
        """
        return [self.batch_gain, self.batch_bias]


class TanH:
    """
    TanH activation layer.
    """

    def __call__(self, X):
        """
        Forward pass for the TanH activation layer.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the TanH activation function.
        """
        self.out = torch.tanh(X)
        return self.out

    def params(self):
        """
        Returns an empty list as Tanh has no parameters.

        Returns:
            list: An empty list.
        """
        return []


class Embeddings:
    """
    Embedding layer to convert input indices to dense vectors.

    Args:
        vocab_size (int): Size of the vocabulary.
        emb_size (int): Size of the embedding.
    """

    def __init__(self, vocab_size, emb_size):
        self.emb = torch.randn((vocab_size, emb_size))

    def __call__(self, X):
        """
        Forward pass for the embedding layer.

        Args:
            X (torch.Tensor): Input tensor with indices.

        Returns:
            torch.Tensor: Output tensor after embedding lookup.
        """
        self.out = self.emb[X]
        self.out = self.out.view(self.out.shape[0], -1)
        return self.out

    def params(self):
        """
        Returns the parameters of the embedding layer.

        Returns:
            list: List containing the embedding matrix.
        """
        return [self.emb]


class Sequential:
    """
    A module to run layers in sequence.

    Args:
        layers (list): List of layers to be executed sequentially.
    """

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, X):
        """
        Forward pass through all layers in the sequence.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after sequential processing.
        """
        for layer in self.layers:
            X = layer(X)
        self.out = X
        return self.out

    def params(self):
        """
        Returns all parameters from all layers in the sequence.

        Returns:
            list: List of parameters from all layers.
        """
        return [p for layer in self.layers for p in layer.params()]

# Script for training the character-level language model.
# This script reads input data, sets up the model, and trains it using the specified parameters.
# After training, it generates and outputs new sequences of characters (pet names in this case).
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="CharFlow")

    parser.add_argument('--file-path', '-f', type=str, default='names.txt', help='input file with words one per line')
    parser.add_argument('--steps', '-s', type=int, default=200000, help='number of training steps')
    parser.add_argument('--emb', '-e', type=int, default=20, help='embedding size')
    parser.add_argument('--hidden-units', '-hu', type=int, default=100, help='number of hidden units in a layer')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--context-size', '-c', type=int, default=3,
                        help='number of needed chars to predict the next one')
    args = parser.parse_args()

    CONTEXT_SIZE = args.context_size

    char_to_idx, idx_to_char, vocab_size = create_vocabulary(args.file_path)

    X, y = create_dataset(args.file_path)

    model = Sequential([
        Embeddings(vocab_size, args.emb),
        LinearLayer(args.emb * CONTEXT_SIZE, args.hidden_units), BatchNormalization(args.hidden_units), TanH(),
        LinearLayer(args.hidden_units, args.hidden_units), BatchNormalization(args.hidden_units), TanH(),
        LinearLayer(args.hidden_units, vocab_size),
    ])

    parameters = model.params()
    for p in parameters:
        p.requires_grad = True

    for i in range(args.steps):

        idx = torch.randint(0, X.shape[0], (args.batch_size,))
        X_batch, y_batch = X[idx], y[idx]

        logits = model(X_batch)
        loss = F.cross_entropy(logits, y_batch)

        for p in parameters:
            p.grad = None
        loss.backward()

        if i < 150000:
            lr = -0.01
        elif i < 200000:
            lr = -0.1
        else:
            lr = -0.001

        for p in parameters:
            p.data += lr * p.grad

    for layer in model.layers:
        layer.training = False

    for _ in range(10):

        out_list = []
        context = CONTEXT_SIZE * [0]
        while True:

            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            idx = torch.multinomial(probs, num_samples=1).item()

            context = context[1:] + [idx]
            out_list.append(idx)

            if idx == 0:
                break

        print(''.join(idx_to_char[idx] for idx in out_list[:-1]))
