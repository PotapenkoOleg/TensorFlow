import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict
import re

class TypeInferenceTransformer:
    """
    Transformer-based neural network for inferring data types from string representations.
    """
    
    def __init__(self, 
                 vocab_size: int = 128,  # ASCII characters
                 max_length: int = 50,
                 num_types: int = 10,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 ff_dim: int = 512,
                 dropout_rate: float = 0.1):
        """
        Initialize the type inference transformer.
        
        Args:
            vocab_size: Size of vocabulary (default: ASCII)
            max_length: Maximum string length
            num_types: Number of type classes to predict
            d_model: Dimension of the model
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            dropout_rate: Dropout rate
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_types = num_types
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Type mapping
        self.type_mapping = {
            0: 'integer',
            1: 'float',
            2: 'boolean',
            3: 'date',
            4: 'datetime',
            5: 'email',
            6: 'url',
            7: 'phone',
            8: 'json',
            9: 'string'
        }
        
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build the transformer model."""
        # Input layer
        inputs = tf.keras.layers.Input(shape=(self.max_length,))
        
        # Embedding layer
        embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.d_model
        )(inputs)
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.max_length, delta=1)
        position_embedding = tf.keras.layers.Embedding(
            input_dim=self.max_length,
            output_dim=self.d_model
        )(positions)
        
        # Combine embeddings
        x = embedding + position_embedding
        
        # Transformer encoder layers
        for _ in range(self.num_layers):
            x = self._transformer_encoder_layer(x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.num_types, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def _transformer_encoder_layer(self, x: tf.Tensor) -> tf.Tensor:
        """Single transformer encoder layer."""
        # Multi-head self-attention
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate
        )(x, x)
        
        # Dropout and residual connection
        attn_output = tf.keras.layers.Dropout(self.dropout_rate)(attn_output)
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward network
        ff_output = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.d_model)
        ])(x1)
        
        # Dropout and residual connection
        ff_output = tf.keras.layers.Dropout(self.dropout_rate)(ff_output)
        x2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x1 + ff_output)
        
        return x2
    
    def preprocess_string(self, s: str) -> np.ndarray:
        """Convert string to numerical representation."""
        # Convert to ASCII values (0-127)
        encoded = [ord(c) % self.vocab_size for c in s[:self.max_length]]
        
        # Pad to max_length
        if len(encoded) < self.max_length:
            encoded.extend([0] * (self.max_length - len(encoded)))
        
        return np.array(encoded)
    
    def generate_synthetic_data(self, samples_per_type: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data."""
        X = []
        y = []
        
        # Integer patterns
        for _ in range(samples_per_type):
            val = str(np.random.randint(-10000, 10000))
            X.append(self.preprocess_string(val))
            y.append(0)
        
        # Float patterns
        for _ in range(samples_per_type):
            val = f"{np.random.uniform(-1000, 1000):.{np.random.randint(1, 6)}f}"
            X.append(self.preprocess_string(val))
            y.append(1)
        
        # Boolean patterns
        boolean_values = ['true', 'false', 'True', 'False', 'TRUE', 'FALSE', '1', '0', 'yes', 'no']
        for _ in range(samples_per_type):
            val = np.random.choice(boolean_values)
            X.append(self.preprocess_string(val))
            y.append(2)
        
        # Date patterns
        for _ in range(samples_per_type):
            year = np.random.randint(1900, 2030)
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            formats = [f"{year}-{month:02d}-{day:02d}", 
                      f"{month:02d}/{day:02d}/{year}",
                      f"{day:02d}-{month:02d}-{year}"]
            val = np.random.choice(formats)
            X.append(self.preprocess_string(val))
            y.append(3)
        
        # DateTime patterns
        for _ in range(samples_per_type):
            year = np.random.randint(1900, 2030)
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 60)
            val = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"
            X.append(self.preprocess_string(val))
            y.append(4)
        
        # Email patterns
        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'example.com']
        for _ in range(samples_per_type):
            name = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 
                                           size=np.random.randint(3, 10)))
            domain = np.random.choice(domains)
            val = f"{name}@{domain}"
            X.append(self.preprocess_string(val))
            y.append(5)
        
        # URL patterns
        protocols = ['http://', 'https://']
        domains = ['example.com', 'test.org', 'website.net']
        for _ in range(samples_per_type):
            protocol = np.random.choice(protocols)
            domain = np.random.choice(domains)
            path = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz/'), 
                                          size=np.random.randint(0, 20)))
            val = f"{protocol}{domain}/{path}"
            X.append(self.preprocess_string(val))
            y.append(6)
        
        # Phone patterns
        for _ in range(samples_per_type):
            formats = [
                f"+1-{np.random.randint(100, 999)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}",
                f"({np.random.randint(100, 999)}) {np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}",
                f"{np.random.randint(100, 999)}.{np.random.randint(100, 999)}.{np.random.randint(1000, 9999)}"
            ]
            val = np.random.choice(formats)
            X.append(self.preprocess_string(val))
            y.append(7)
        
        # JSON patterns
        for _ in range(samples_per_type):
            json_patterns = [
                '{"key": "value"}',
                '[1, 2, 3]',
                '{"name": "John", "age": 30}',
                '["a", "b", "c"]'
            ]
            val = np.random.choice(json_patterns)
            X.append(self.preprocess_string(val))
            y.append(8)
        
        # General string patterns
        for _ in range(samples_per_type):
            length = np.random.randint(5, 30)
            val = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 
                                          size=length))
            X.append(self.preprocess_string(val))
            y.append(9)
        
        return np.array(X), np.array(y)
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, 
              X_train: np.ndarray = None, 
              y_train: np.ndarray = None,
              validation_split: float = 0.2,
              epochs: int = 20,
              batch_size: int = 32):
        """Train the model."""
        if X_train is None or y_train is None:
            print("Generating synthetic training data...")
            X_train, y_train = self.generate_synthetic_data()
        
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
            ]
        )
        
        return history
    
    def predict(self, string: str) -> Dict[str, float]:
        """Predict the type of a string."""
        # Preprocess
        x = self.preprocess_string(string)
        x = np.expand_dims(x, axis=0)
        
        # Predict
        predictions = self.model.predict(x, verbose=0)[0]
        
        # Return as dictionary
        results = {}
        for i, prob in enumerate(predictions):
            results[self.type_mapping[i]] = float(prob)
        
        return results
    
    def predict_type(self, string: str) -> str:
        """Get the most likely type for a string."""
        predictions = self.predict(string)
        return max(predictions, key=predictions.get)


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = TypeInferenceTransformer()
    
    # Compile model
    model.compile_model(learning_rate=0.001)
    
    # Train on synthetic data
    print("Training model on synthetic data...")
    history = model.train(epochs=20, batch_size=64)
    
    # Test predictions
    test_strings = [
        "42",
        "3.14159",
        "true",
        "2024-01-15",
        "2024-01-15 14:30:00",
        "user@example.com",
        "https://www.example.com",
        "+1-555-123-4567",
        '{"name": "John"}',
        "Hello, World!"
    ]
    
    print("\nPredictions:")
    print("-" * 50)
    for test_str in test_strings:
        pred_type = model.predict_type(test_str)
        probs = model.predict(test_str)
        print(f"String: '{test_str}'")
        print(f"Predicted type: {pred_type}")
        print(f"Confidence: {probs[pred_type]:.2%}")
        print("-" * 50)