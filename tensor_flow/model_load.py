import tensorflow as tf
import logging
import numpy as np

if __name__ == "__main__":
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)
    
    CLASS_NAMES = {
        0: 'int',
        1: 'float',
        2: 'boolean',
        3: 'time',
        4: 'date',
        5: 'datetime',
        6: 'uuid',
        7: 'string'
    }
    VOCAB_SIZE: int = 128
    MAX_LENGTH: int = 100
    
    def preprocess_string(input: str) -> np.ndarray:
        input = input.strip().upper()
        encoded = [ord(c) % VOCAB_SIZE for c in input[:MAX_LENGTH]]
        if len(encoded) < MAX_LENGTH:
            encoded.extend([0] * (MAX_LENGTH - len(encoded)))
        return np.array(encoded)

    saved_keras_model_filepath = '/home/oleg/Developer/GitHub/TensorFlow/TensorFlow/tensor_flow/best_model_20250606-220704.keras'
    model = tf.keras.models.load_model(saved_keras_model_filepath)
    
    
    # test_strings = [
    #     "42",
    #     "3.14159",
    #     "true",
    #     "2024-01-15",
    #     "2024-01-15 14:30:00",
    #     "user@example.com",
    #     "a20d5384-3110-4567-b5a9-9b581a40e1f8",
    #     "+1-555-123-4567",
    #     '{"name": "John"}',
    #     "Hello, World!"
    # ]

    x = preprocess_string("2024-01-15 14:30:00")
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    predictions = model.predict(x)[0]
    print(predictions.shape)
    max_index = int(np.argmax(predictions)) 
    print("Predicted class: ", CLASS_NAMES[max_index])
    print("Probability: ", predictions[max_index])
    # results = {}
    # for i, prob in enumerate(predictions):
    #     results[CLASS_NAMES[i]] = float(prob)
    # print(results)