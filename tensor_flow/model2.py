import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict, Optional
import re
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class BaseTypeInferenceModel:
    """Base class for type inference models."""
    
    def __init__(self, 
                 vocab_size: int = 128,
                 max_length: int = 50,
                 num_types: int = 10):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_types = num_types
        
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
        
    def preprocess_string(self, s: str) -> np.ndarray:
        """Convert string to numerical representation."""
        encoded = [ord(c) % self.vocab_size for c in s[:self.max_length]]
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


class TransformerModel(BaseTypeInferenceModel):
    """Transformer-based model."""
    
    def __init__(self, d_model: int = 128, num_heads: int = 8, num_layers: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=(self.max_length,))
        
        # Embedding
        x = tf.keras.layers.Embedding(self.vocab_size, self.d_model)(inputs)
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.max_length, delta=1)
        pos_embedding = tf.keras.layers.Embedding(self.max_length, self.d_model)(positions)
        x = x + pos_embedding
        
        # Transformer layers
        for _ in range(self.num_layers):
            # Multi-head attention
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads
            )(x, x)
            x = tf.keras.layers.LayerNormalization()(x + attn_output)
            
            # Feed-forward
            ff_output = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(self.d_model)
            ])(x)
            x = tf.keras.layers.LayerNormalization()(x + ff_output)
        
        # Classification
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(self.num_types, activation='softmax')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)


class CNNModel(BaseTypeInferenceModel):
    """CNN-based model for character-level classification."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=(self.max_length,))
        
        # Embedding
        x = tf.keras.layers.Embedding(self.vocab_size, 128)(inputs)
        
        # Conv layers
        x = tf.keras.layers.Conv1D(128, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(256, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(512, 3, activation='relu')(x)
        
        # Classification
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_types, activation='softmax')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)


class LSTMModel(BaseTypeInferenceModel):
    """LSTM-based model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=(self.max_length,))
        
        # Embedding
        x = tf.keras.layers.Embedding(self.vocab_size, 128)(inputs)
        
        # LSTM layers
        x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(64)(x)
        
        # Classification
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_types, activation='softmax')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)


class EnsembleTypeInference:
    """Ensemble methods for type inference."""
    
    def __init__(self, base_models: Optional[List[BaseTypeInferenceModel]] = None):
        """
        Initialize ensemble with base models.
        
        Args:
            base_models: List of base models. If None, creates default models.
        """
        if base_models is None:
            self.base_models = [
                TransformerModel(d_model=128, num_heads=8, num_layers=4),
                TransformerModel(d_model=64, num_heads=4, num_layers=6),
                CNNModel(),
                LSTMModel()
            ]
        else:
            self.base_models = base_models
            
        self.meta_learner = None
        self.weights = None
        self.type_mapping = self.base_models[0].type_mapping
        
    def compile_models(self, learning_rate: float = 0.001):
        """Compile all base models."""
        for model in self.base_models:
            model.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
    
    def train_base_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                         epochs: int = 15, batch_size: int = 32):
        """Train all base models."""
        histories = []
        
        for i, model in enumerate(self.base_models):
            print(f"\nTraining model {i+1}/{len(self.base_models)}: {model.__class__.__name__}")
            
            history = model.model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
                ]
            )
            histories.append(history)
            
        return histories
    
    def voting_ensemble(self, X: np.ndarray, voting: str = 'soft') -> np.ndarray:
        """
        Voting ensemble prediction.
        
        Args:
            X: Input data
            voting: 'hard' for majority voting, 'soft' for probability averaging
            
        Returns:
            Predictions
        """
        if voting == 'hard':
            # Hard voting: majority vote
            predictions = []
            for model in self.base_models:
                pred = model.model.predict(X, verbose=0)
                predictions.append(np.argmax(pred, axis=1))
            
            predictions = np.array(predictions)
            # Majority vote
            final_pred = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), 0, predictions
            )
            return final_pred
        
        else:  # soft voting
            # Soft voting: average probabilities
            predictions = []
            for model in self.base_models:
                pred = model.model.predict(X, verbose=0)
                predictions.append(pred)
            
            # Average probabilities
            avg_pred = np.mean(predictions, axis=0)
            return np.argmax(avg_pred, axis=1)
    
    def weighted_voting_ensemble(self, X: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Weighted voting ensemble.
        
        Args:
            X: Input data
            weights: Model weights. If None, uses equal weights.
            
        Returns:
            Predictions
        """
        if weights is None:
            weights = np.ones(len(self.base_models)) / len(self.base_models)
        
        predictions = []
        for model in self.base_models:
            pred = model.model.predict(X, verbose=0)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += weights[i] * pred
        
        return np.argmax(weighted_pred, axis=1)
    
    def train_stacking_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, 
                               n_folds: int = 5):
        """
        Train stacking ensemble with cross-validation.
        
        Args:
            X_train: Training data
            y_train: Training labels
            n_folds: Number of folds for cross-validation
        """
        # Generate meta-features using cross-validation
        meta_features = np.zeros((len(X_train), len(self.base_models) * 10))  # 10 classes
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for i, model in enumerate(self.base_models):
            print(f"\nGenerating meta-features for model {i+1}/{len(self.base_models)}")
            
            fold_predictions = np.zeros((len(X_train), 10))
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                # Train on fold
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                
                # Clone model for this fold
                fold_model = model.__class__(**model.__dict__)
                fold_model.model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                fold_model.model.fit(
                    X_fold_train, y_fold_train,
                    epochs=10,
                    batch_size=32,
                    verbose=0
                )
                
                # Predict on validation fold
                fold_pred = fold_model.model.predict(X_fold_val, verbose=0)
                fold_predictions[val_idx] = fold_pred
            
            # Store meta-features
            meta_features[:, i*10:(i+1)*10] = fold_predictions
        
        # Train meta-learner
        print("\nTraining meta-learner...")
        self.meta_learner = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(len(self.base_models) * 10,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        self.meta_learner.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.meta_learner.fit(
            meta_features, y_train,
            validation_split=0.2,
            epochs=20,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
    
    def stacking_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using stacking ensemble.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained. Call train_stacking_ensemble first.")
        
        # Generate meta-features
        meta_features = []
        for model in self.base_models:
            pred = model.model.predict(X, verbose=0)
            meta_features.append(pred)
        
        meta_features = np.concatenate(meta_features, axis=1)
        
        # Meta-learner prediction
        final_pred = self.meta_learner.predict(meta_features, verbose=0)
        return np.argmax(final_pred, axis=1)
    
    def boosting_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, 
                         n_rounds: int = 5, learning_rate: float = 0.1):
        """
        Adaptive boosting ensemble.
        
        Args:
            X_train: Training data
            y_train: Training labels
            n_rounds: Number of boosting rounds
            learning_rate: Learning rate for weight updates
        """
        n_samples = len(X_train)
        sample_weights = np.ones(n_samples) / n_samples
        self.weights = []
        
        for round in range(n_rounds):
            print(f"\nBoosting round {round+1}/{n_rounds}")
            
            # Select model for this round
            model_idx = round % len(self.base_models)
            model = self.base_models[model_idx]
            
            # Train with weighted samples
            # Note: TensorFlow doesn't directly support sample weights in fit,
            # so we simulate by resampling
            indices = np.random.choice(n_samples, size=n_samples, p=sample_weights)
            X_resampled = X_train[indices]
            y_resampled = y_train[indices]
            
            model.model.fit(
                X_resampled, y_resampled,
                epochs=5,
                batch_size=32,
                verbose=0
            )
            
            # Calculate error
            predictions = model.model.predict(X_train, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            incorrect = pred_classes != y_train
            
            # Calculate weighted error
            error = np.sum(sample_weights[incorrect]) / np.sum(sample_weights)
            
            # Calculate model weight
            if error > 0.5:
                continue
                
            alpha = learning_rate * np.log((1 - error) / error)
            self.weights.append(alpha)
            
            # Update sample weights
            sample_weights[incorrect] *= np.exp(alpha)
            sample_weights /= np.sum(sample_weights)
    
    def predict_type(self, string: str, method: str = 'voting') -> str:
        """
        Predict the type of a string using specified ensemble method.
        
        Args:
            string: Input string
            method: Ensemble method ('voting', 'weighted', 'stacking')
            
        Returns:
            Predicted type
        """
        # Preprocess
        x = self.base_models[0].preprocess_string(string)
        x = np.expand_dims(x, axis=0)
        
        if method == 'voting':
            pred = self.voting_ensemble(x, voting='soft')
        elif method == 'weighted':
            pred = self.weighted_voting_ensemble(x, self.weights)
        elif method == 'stacking':
            pred = self.stacking_predict(x)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self.type_mapping[pred[0]]
    
    def predict_probabilities(self, string: str) -> Dict[str, float]:
        """Get probability distribution over all types."""
        x = self.base_models[0].preprocess_string(string)
        x = np.expand_dims(x, axis=0)
        
        predictions = []
        for model in self.base_models:
            pred = model.model.predict(x, verbose=0)
            predictions.append(pred)
        
        # Average probabilities
        avg_pred = np.mean(predictions, axis=0)[0]
        
        results = {}
        for i, prob in enumerate(avg_pred):
            results[self.type_mapping[i]] = float(prob)
        
        return results


# Example usage and evaluation
if __name__ == "__main__":
    # Create ensemble
    ensemble = EnsembleTypeInference()
    
    # Generate training data
    print("Generating synthetic training data...")
    X_train, y_train = ensemble.base_models[0].generate_synthetic_data(samples_per_type=500)
    
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    # Split data
    split_idx = int(0.8 * len(X_train))
    X_test = X_train[split_idx:]
    y_test = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    # Compile models
    ensemble.compile_models()
    
    # Train base models
    print("\nTraining base models...")
    ensemble.train_base_models(X_train, y_train, epochs=10)
    
    # Evaluate individual models
    print("\n" + "="*60)
    print("Individual Model Performance:")
    print("="*60)
    for i, model in enumerate(ensemble.base_models):
        pred = model.model.predict(X_test, verbose=0)
        pred_classes = np.argmax(pred, axis=1)
        acc = accuracy_score(y_test, pred_classes)
        print(f"{model.__class__.__name__}: {acc:.4f}")
    
    # Evaluate ensemble methods
    print("\n" + "="*60)
    print("Ensemble Methods Performance:")
    print("="*60)
    
    # Voting ensemble
    pred_voting = ensemble.voting_ensemble(X_test, voting='soft')
    acc_voting = accuracy_score(y_test, pred_voting)
    print(f"Soft Voting Ensemble: {acc_voting:.4f}")
    
    # Hard voting
    pred_hard = ensemble.voting_ensemble(X_test, voting='hard')
    acc_hard = accuracy_score(y_test, pred_hard)
    print(f"Hard Voting Ensemble: {acc_hard:.4f}")
    
    # Train stacking ensemble
    print("\nTraining stacking ensemble...")
    ensemble.train_stacking_ensemble(X_train, y_train, n_folds=3)
    
    # Evaluate stacking
    pred_stacking = ensemble.stacking_predict(X_test)
    acc_stacking = accuracy_score(y_test, pred_stacking)
    print(f"Stacking Ensemble: {acc_stacking:.4f}")
    
    # Test predictions
    print("\n" + "="*60)
    print("Example Predictions:")
    print("="*60)
    
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
    
    for test_str in test_strings:
        voting_pred = ensemble.predict_type(test_str, method='voting')
        stacking_pred = ensemble.predict_type(test_str, method='stacking')
        probs = ensemble.predict_probabilities(test_str)
        
        print(f"\nString: '{test_str}'")
        print(f"Voting prediction: {voting_pred}")
        print(f"Stacking prediction: {stacking_pred}")
        print(f"Top 3 probabilities:")
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for type_name, prob in sorted_probs:
            print(f"  - {type_name}: {prob:.2%}")