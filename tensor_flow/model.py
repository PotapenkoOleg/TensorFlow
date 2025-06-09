import tensorflow as tf
import numpy as np
from typing import Tuple, Dict
import uuid
import datetime


# Model Config

VOCAB_SIZE: int = 128
MAX_LENGTH: int = 100
MODEL_DIMENSION: int = 128
NUM_HEADS: int = 8
NUM_LAYERS: int = 4
FF_DIM: int = 512
DROPOUT_RATE: float = 0.1

def preprocess_string(input_str: str) -> np.ndarray:
    input_str = input_str[:MAX_LENGTH-1].strip().upper()
    encoded = [ord(c) % VOCAB_SIZE for c in input_str]
    if len(encoded) < MAX_LENGTH:
        encoded.extend([0] * (MAX_LENGTH - len(encoded)))
    return np.array(encoded)

def generate_synthetic_data(samples_per_type: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data."""
    X = []
    y = []
    
    # Integer patterns
    for _ in range(samples_per_type):
        val = str(np.random.randint(-10000, 10000))
        X.append(val)
        y.append(0)
    
    # Float patterns
    for _ in range(samples_per_type):
        val = f"{np.random.uniform(-1000, 1000):.{np.random.randint(1, 6)}f}"
        X.append(val)
        y.append(1)
    
    # Boolean patterns
    boolean_values = ['TRUE', 'FALSE', '1', '0', 'YES', 'NO']
    for _ in range(samples_per_type):
        val = np.random.choice(boolean_values)
        X.append(val)
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
        X.append(val)
        y.append(3)
    
    # DateTime patterns
    for _ in range(samples_per_type):
        year = np.random.randint(1900, 2030)
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)
        hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        val = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"
        X.append(val)
        y.append(4)
    
    # UUID patterns
    for _ in range(samples_per_type):
        val = str(uuid.uuid4())
        X.append(val)
        y.append(4)
    
    # General string patterns
    for _ in range(samples_per_type):
        length = np.random.randint(5, 30)
        val = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ '), size=length))
        X.append(val)
        y.append(6)
    
    return np.array(X), np.array(y)

class TypeInferenceTransformer:
    """
    Transformer-based neural network for inferring data types from string representations
    """
    def __init__(
        self, 
        vocab_size: int = VOCAB_SIZE,
        max_length: int = MAX_LENGTH,
        model_dimension: int = MODEL_DIMENSION,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        ff_dim: int = FF_DIM,
        dropout_rate: float = DROPOUT_RATE
        ):
        """
        Initialize the type inference transformer.
        
        Args:
            vocab_size: Size of vocabulary (default: ASCII)
            max_length: Maximum string length
            d_model: Dimension of the model
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            dropout_rate: Dropout rate
        """
        self.__vocab_size = vocab_size
        self.__max_length = max_length
        self.__d_model = model_dimension
        self.__num_heads = num_heads
        self.__num_layers = num_layers
        self.__ff_dim = ff_dim
        self.__dropout_rate = dropout_rate
        
        # Type mapping
        self.__type_mapping = {
            0: 'int',
            1: 'float',
            2: 'boolean',
            3: 'date',
            4: 'datetime',
            5: 'uuid',
            6: 'string'
        }
        # num_classes: Number of type classes to predict
        self.__num_classes = len(self.__type_mapping)
        
        self.__model = self.__build_model()
        
    def __build_model(self) -> tf.keras.Model:
        """Build the transformer model."""
        # Input layer
        inputs = tf.keras.layers.Input(shape=(self.__max_length,))
        
        # Embedding layer
        embedding = tf.keras.layers.Embedding(
            input_dim=self.__vocab_size,
            output_dim=self.__d_model
        )(inputs)
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.__max_length, delta=1)
        position_embedding = tf.keras.layers.Embedding(
            input_dim=self.__max_length,
            output_dim=self.__d_model
        )(positions)
        
        # Combine embeddings
        x = embedding + position_embedding
        
        # Transformer encoder layers
        for _ in range(self.__num_layers):
            x = self.__transformer_encoder_layer(x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.__dropout_rate)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(self.__dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.__num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def __transformer_encoder_layer(self, x: tf.Tensor) -> tf.Tensor:
        """Single transformer encoder layer."""
        # Multi-head self-attention
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=self.__num_heads,
            key_dim=self.__d_model // self.__num_heads,
            dropout=self.__dropout_rate
        )(x, x)
        
        # Dropout and residual connection
        attn_output = tf.keras.layers.Dropout(self.__dropout_rate)(attn_output)
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward network
        ff_output = tf.keras.Sequential([
            tf.keras.layers.Dense(self.__ff_dim, activation='relu'),
            tf.keras.layers.Dropout(self.__dropout_rate),
            tf.keras.layers.Dense(self.__d_model)
        ])(x1)
        
        # Dropout and residual connection
        ff_output = tf.keras.layers.Dropout(self.__dropout_rate)(ff_output)
        x2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x1 + ff_output)
        
        return x2
    
    def summary(self):
        return self.__model.summary()
    
    def __compile_model(self, learning_rate: float):
        self.__model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def fit(
            self, 
            training_batches,
            validation_batches,
            batch_size: int,
            epochs: int,
            learning_rate: float,
        ):
        
        model.__compile_model(learning_rate=learning_rate)
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_log_dir = f"./logs/fit/{current_time}"
        
        history = self.__model.fit(
            training_batches,
            validation_data=validation_batches,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
                tf.keras.callbacks.ModelCheckpoint('./best_model/best_model_{}.keras'.format(current_time), monitor='val_loss', save_best_only=True),
                tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
            ]
        )
        
        return history
    
    def evaluate(self, test_batches):
        return self.__model.evaluate(test_batches)
    
    def predict(self, string: str) -> Dict[str, float]:
        x = preprocess_string(string)
        x = np.expand_dims(x, axis=0)
        
        predictions = self.__model.predict(x, verbose=0)[0]
        
        results = {}
        for i, prob in enumerate(predictions):
            results[self.__type_mapping[i]] = float(prob)
        
        return results
    
    def predict_type(self, string: str) -> str:
        predictions = self.predict(string)
        return max(predictions, key=predictions.get)
    

if __name__ == "__main__":
    
    model = TypeInferenceTransformer()
    
    model.summary()
   
    BATCH_SIZE: int = 32
    EPOCHS: int = 20
    LEARNING_RATE: float = 0.001
    
    NUM_TRAINING_EXAMPLES: int = 640
    NUM_VALIDATION_EXAMPLES: int = 320
    NUM_TEST_EXAMPLES: int = 320
            
    X_train, y_train = generate_synthetic_data(samples_per_type=NUM_TRAINING_EXAMPLES)
    X_train = np.array([preprocess_string(s) for s in X_train])
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    training_batches = train_dataset.cache().shuffle(NUM_TRAINING_EXAMPLES//4).batch(BATCH_SIZE).prefetch(1)  
    
    X_validation, y_validation = generate_synthetic_data(samples_per_type=NUM_VALIDATION_EXAMPLES)  
    X_validation = np.array([preprocess_string(s) for s in X_validation])
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_validation, y_validation))
    validation_batches = validation_dataset.cache().batch(BATCH_SIZE).prefetch(1)

    X_test, y_test = generate_synthetic_data(samples_per_type=NUM_TEST_EXAMPLES)  
    X_test = np.array([preprocess_string(s) for s in X_test])
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_batches = test_dataset.cache().batch(BATCH_SIZE).prefetch(1)

    history = model.fit(training_batches, validation_batches, batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    
    ### -----------------------------------------------------------------------------------------------------------------------------------------------
    
    loss, accuracy = model.evaluate(test_batches)
    print('\nLoss on the TEST Set: {:,.3f}'.format(loss))
    print('Accuracy on the TEST Set: {:.3%}'.format(accuracy))
    
    ### -----------------------------------------------------------------------------------------------------------------------------------------------
    
    # saved_keras_model_filepath = './best_model/best_model_20250605-235156.keras'
    # model = tf.keras.models.load_model(saved_keras_model_filepath)
    # model.summary()
    
    ### -----------------------------------------------------------------------------------------------------------------------------------------------
    
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