# -*- coding: utf-8 -*-
"""
Implementation of the Seq2Seq LSTM model with Attention for CEEMDAN/ICEEMDAN decomposed data.
"""
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, AdditiveAttention, Layer, Embedding
from tensorflow.keras.models import Model
import numpy as np

try:
    from .. import config # Relative import from parent directory (src)
except ImportError:
    import config # Fallback for direct execution

class Encoder(Model):
    """
    Encoder part of the Seq2Seq model.
    Consists of two LSTM layers.
    """
    def __init__(self, lstm_units_list, standard_dropout_rate, recurrent_dropout_rate, name="encoder"):
        super(Encoder, self).__init__(name=name)
        if len(lstm_units_list) != 2:
            raise ValueError("lstm_units_list must contain units for exactly two LSTM layers.")
        
        self.lstm_units_list = lstm_units_list
        self.standard_dropout_rate = standard_dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        
        self.lstm1 = LSTM(self.lstm_units_list[0],
                          return_sequences=True, # Returns the full sequence for Attention
                          return_state=True,     # Returns the last hidden and cell state
                          dropout=self.standard_dropout_rate, # Corrected to use instance attribute
                          recurrent_dropout=self.recurrent_dropout_rate,
                          name="encoder_lstm_1")
        self.lstm2 = LSTM(self.lstm_units_list[1], # Corrected indentation
                          return_sequences=True, # Returns the full sequence for Attention
                          return_state=True,     # Returns the last hidden and cell state
                          dropout=self.standard_dropout_rate, # Corrected to use instance attribute
                          recurrent_dropout=self.recurrent_dropout_rate,
                          name="encoder_lstm_2")
        # Potentially Bidirectional, but keeping it simple first as per initial plan.
        # If bidirectional:
        # self.bilstm1 = Bidirectional(LSTM(...), name="encoder_bilstm_1")
        # self.bilstm2 = Bidirectional(LSTM(...), name="encoder_bilstm_2")

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'lstm_units_list': self.lstm_units_list,
            'standard_dropout_rate': self.standard_dropout_rate,
            'recurrent_dropout_rate': self.recurrent_dropout_rate
        })
        return config

    def call(self, x_encoder_input):
        # x_encoder_input shape: (batch_size, encoder_timesteps, K+1 features)
        # For a single LSTM layer:
        # entire_output_seq, final_state_h, final_state_c = self.lstm1(x_encoder_input)
        
        # For two LSTM layers:
        # The states from the first LSTM layer are typically not directly passed to the decoder
        # if the second LSTM layer also returns states. The second layer's states are used.
        x, state_h1, state_c1 = self.lstm1(x_encoder_input) # state_h1, state_c1 are from the first LSTM layer
        encoder_outputs, state_h2, state_c2 = self.lstm2(x) # state_h2, state_c2 are from the second LSTM layer
        
        # Return all four states to be used by the decoder
        return encoder_outputs, [state_h1, state_c1, state_h2, state_c2]


class BahdanauAttention(Layer):
    """
    Bahdanau Attention Layer.
    """
    def __init__(self, units, name="bahdanau_attention"):
        super(BahdanauAttention, self).__init__(name=name)
        self.units = units # Store units
        self.W1 = Dense(units, name="W1_dense") # For encoder_outputs
        self.W2 = Dense(units, name="W2_dense") # For decoder_hidden_state
        self.V = Dense(1, name="V_dense")       # To compute the score

    def call(self, query, values):
        # query shape: (batch_size, decoder_hidden_dim) -> decoder's previous hidden state
        # values shape: (batch_size, encoder_timesteps, encoder_hidden_dim) -> encoder_outputs

        # Expand query to be (batch_size, 1, decoder_hidden_dim) to broadcast addition
        query_with_time_axis = tf.expand_dims(query, 1)

        # Score calculation
        # self.W1(values) shape: (batch_size, encoder_timesteps, units)
        # self.W2(query_with_time_axis) shape: (batch_size, 1, units)
        # Broadcasting addition gives shape: (batch_size, encoder_timesteps, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_with_time_axis))) # Shape: (batch_size, encoder_timesteps, 1)

        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1) # Shape: (batch_size, encoder_timesteps, 1)

        # Context vector
        # context_vector shape after sum: (batch_size, encoder_hidden_dim)
        context_vector = attention_weights * values # Element-wise multiplication
        context_vector = tf.reduce_sum(context_vector, axis=1) # Sum over encoder_timesteps

        return context_vector, attention_weights

    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({'units': self.units})
        return config

class Decoder(Model):
    """
    Decoder part of the Seq2Seq model with Attention.
    Consists of two LSTM layers, an Attention layer, and a Dense output layer.
    """
    def __init__(self, lstm_units_list, output_feature_dim, standard_dropout_rate, recurrent_dropout_rate, attention_units, name="decoder"):
        super(Decoder, self).__init__(name=name)
        if len(lstm_units_list) != 2:
            raise ValueError("lstm_units_list must contain units for exactly two LSTM layers.")

        self.lstm_units_list = lstm_units_list
        self.output_feature_dim = output_feature_dim
        self.standard_dropout_rate = standard_dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.attention_units = attention_units
        
        # Embedding for the single price input to match LSTM dimensions if needed,
        # or if we want a learnable representation.
        # Assuming LSTM units are, e.g., 64. If input is 1D price, embedding helps.
        # Let's make embedding_dim a parameter or derive it. For now, assume it matches first LSTM unit.
        self.embedding_dim = self.lstm_units_list[0] # Or a separate config
        self.embedding = Dense(self.embedding_dim, activation='relu', name="decoder_input_embedding") # Simple Dense as embedding

        self.attention = BahdanauAttention(self.attention_units, name="decoder_attention") # Use stored attribute
        
        self.lstm1 = LSTM(self.lstm_units_list[0],
                          return_sequences=True, # Must return sequences for the next LSTM layer
                          return_state=True,
                          dropout=self.standard_dropout_rate, # Use instance attribute
                          recurrent_dropout=self.recurrent_dropout_rate,
                          name="decoder_lstm_1")
        self.lstm2 = LSTM(self.lstm_units_list[1], # Corrected indentation
                          return_sequences=True, # Returns sequence for Dense layer at each step
                          return_state=True,
                          dropout=self.standard_dropout_rate, # Use instance attribute
                          recurrent_dropout=self.recurrent_dropout_rate,
                          name="decoder_lstm_2")
        
        self.dense_output = Dense(self.output_feature_dim, name="decoder_dense_output") # Corrected indentation, output_feature_dim is 1 for price

    def get_config(self): # Corrected indentation for method definition
        config = super(Decoder, self).get_config()
        config.update({
            'lstm_units_list': self.lstm_units_list,
            'output_feature_dim': self.output_feature_dim,
            'standard_dropout_rate': self.standard_dropout_rate,
            'recurrent_dropout_rate': self.recurrent_dropout_rate,
            'attention_units': self.attention_units
            # self.embedding_dim is derived, so not strictly needed if lstm_units_list is saved
        })
        return config

    def call(self, x_decoder_input, initial_states_list, encoder_outputs, training=None): # Corrected indentation for method definition
        # x_decoder_input shape for teacher forcing: (batch_size, H_forecast_period, 1) (scaled price y_t-1)
        # initial_states_list: list of [state_h, state_c] from the last encoder LSTM layer.
        # encoder_outputs shape: (batch_size, encoder_timesteps, encoder_lstm_units)
        
        target_seq_len = tf.shape(x_decoder_input)[1] # H_forecast_period
        batch_size = tf.shape(x_decoder_input)[0]

        # initial_states_list now contains [enc_state_h1, enc_state_c1, enc_state_h2, enc_state_c2]
        enc_state_h1, enc_state_c1, enc_state_h2, enc_state_c2 = initial_states_list

        # Initialize decoder LSTM layers with corresponding encoder LSTM states
        current_state_h_l1 = enc_state_h1
        current_state_c_l1 = enc_state_c1
        current_state_h_l2 = enc_state_h2
        current_state_c_l2 = enc_state_c2
        
        # Get sequence length as a Python integer for Python-style range
        # x_decoder_input shape for teacher forcing: (batch_size, H_forecast_period, 1)
        target_seq_len = x_decoder_input.shape[1] # Should be a Python int (e.g., H=5)
        if target_seq_len is None: # Should not happen if data is prepared correctly with fixed H
            raise ValueError("Decoder input sequence length (H) cannot be None in training/inference loop.")

        all_step_outputs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        # Initial input to the decoder (e.g., y_T_scaled for the first step)
        # x_decoder_input[:, 0, :] gives the first timestep for all batches
        current_step_input = x_decoder_input[:, 0:1, :] # Shape: (batch_size, 1, input_feature_dim=1)

        # Use Python range for the loop, as target_seq_len is now a Python int
        for t in range(target_seq_len):
            # 1. Embed the input for the current timestep
            embedded_input = self.embedding(current_step_input) # Shape: (batch_size, 1, embedding_dim)
            
            # 2. Calculate context vector using attention
            # Query for attention is the hidden state of the *previous* timestep of the *first* decoder LSTM layer.
            # For the first step, query can be encoder's final hidden state.
            # Let's use the hidden state of the second (outputting) LSTM layer from previous step.
            query_for_attention = current_state_h_l2 
            context_vector, attention_weights = self.attention(query_for_attention, encoder_outputs)
            # context_vector shape: (batch_size, encoder_lstm_units)
            # Expand context_vector to be (batch_size, 1, encoder_lstm_units) to concat with embedded_input
            context_vector_expanded = tf.expand_dims(context_vector, 1)

            # 3. Concatenate embedded input and context vector
            # embedded_input shape: (batch_size, 1, embedding_dim)
            # context_vector_expanded shape: (batch_size, 1, encoder_units_of_last_layer)
            # The dimensions must match or be compatible for concatenation if embedding_dim != encoder_units
            # For simplicity, let's assume embedding_dim is chosen appropriately, or we project context_vector.
            # A common approach: concat and then pass through a Dense layer before LSTM.
            # Or, ensure embedding_dim + attention_output_dim is what LSTM expects, or project.
            # For now, let's assume simple concatenation and that LSTM input dim is sum of these.
            # This implies the decoder LSTM input_dim needs to be embedding_dim + encoder_units.
            # This is getting complex. A simpler way:
            # Decoder LSTM input = embedded_input. Context vector is used to modulate LSTM state or output.
            # Alternative: input to LSTM is concat(embedded_input, context_vector_expanded)
            # This means decoder LSTM's input_dim must be embedding_dim + encoder_output_units.
            
            # Let's try: input to LSTM is concatenation of embedded input and context vector
            # This requires decoder LSTM input feature size to be embedding_dim + encoder_output_units
            # (e.g., 128 + 64 = 192 if embedding_dim=128 and encoder last layer=64 units).
            # Keras LSTM layers can often infer the input size from the first batch,
            # but it's crucial that this concatenated dimension is handled correctly.
            decoder_lstm_input = tf.concat([embedded_input, context_vector_expanded], axis=-1) # Shape: (batch_size, 1, embedding_dim + encoder_units)

            # 4. Pass through Decoder LSTMs
            # Layer 1
            # Initial state for the first step is from encoder. Subsequent steps use previous step's state.
            full_seq_output_l1, state_h_l1_next, state_c_l1_next = self.lstm1(
                decoder_lstm_input, initial_state=[current_state_h_l1, current_state_c_l1]
            )
            # Layer 2
            # The input to lstm2 is the output sequence from lstm1
            full_seq_output_l2, state_h_l2_next, state_c_l2_next = self.lstm2(
                full_seq_output_l1, initial_state=[current_state_h_l2, current_state_c_l2]
            )
            
            # 5. Get output for the current timestep from the final LSTM layer's output sequence
            # full_seq_output_l2 shape: (batch_size, 1, decoder_lstm2_units)
            # We take the output for the current single timestep
            current_step_lstm_output = full_seq_output_l2[:, 0, :] # Shape: (batch_size, decoder_lstm2_units)
            
            # 6. Pass through Dense layer to get prediction
            prediction = self.dense_output(current_step_lstm_output) # Shape: (batch_size, output_feature_dim=1)
            all_step_outputs = all_step_outputs.write(t, prediction)

            # 7. Prepare for next timestep (Teacher Forcing or Recursive)
            if training: # Teacher Forcing
                # Next input is the true value from the target sequence
                if t < target_seq_len - 1:
                    current_step_input = x_decoder_input[:, t+1:t+2, :]
                    # Ensure shape consistency for AutoGraph
                    current_step_input = tf.ensure_shape(current_step_input, [None, 1, 1])
            else: # Recursive prediction for inference
                # Next input is the current prediction
                current_step_input = prediction # Shape (batch_size, 1)
                current_step_input = tf.expand_dims(current_step_input, axis=1) # Reshape to (batch_size, 1, 1) for embedding
                # Ensure shape consistency for AutoGraph
                current_step_input = tf.ensure_shape(current_step_input, [None, 1, 1])

            # Update states for the next iteration
            current_state_h_l1, current_state_c_l1 = state_h_l1_next, state_c_l1_next
            current_state_h_l2, current_state_c_l2 = state_h_l2_next, state_c_l2_next

        # Stack the outputs from the TensorArray
        # The .stack() method will create a tensor by stacking the elements along axis 0.
        # If each element written was (batch_size, output_feature_dim),
        # then stacked will be (target_seq_len, batch_size, output_feature_dim).
        # We need to transpose it to (batch_size, target_seq_len, output_feature_dim).
        stacked_outputs = all_step_outputs.stack()
        return tf.transpose(stacked_outputs, perm=[1, 0, 2])


class Seq2SeqAttentionLSTM(Model):
    """
    Full Seq2Seq model with Attention, combining Encoder and Decoder.
    """
    def __init__(self, encoder_lstm_units, decoder_lstm_units, output_feature_dim,
                 encoder_standard_dropout, encoder_recurrent_dropout,
                 decoder_standard_dropout, decoder_recurrent_dropout,
                 attention_units, name="seq2seq_attention_lstm"):
        super(Seq2SeqAttentionLSTM, self).__init__(name=name)
        self.encoder_lstm_units = encoder_lstm_units
        self.decoder_lstm_units = decoder_lstm_units
        self.output_feature_dim = output_feature_dim
        self.encoder_standard_dropout = encoder_standard_dropout
        self.encoder_recurrent_dropout = encoder_recurrent_dropout
        self.decoder_standard_dropout = decoder_standard_dropout
        self.decoder_recurrent_dropout = decoder_recurrent_dropout
        self.attention_units = attention_units
        
        self.encoder = Encoder(self.encoder_lstm_units, self.encoder_standard_dropout, self.encoder_recurrent_dropout)
        self.decoder = Decoder(self.decoder_lstm_units, self.output_feature_dim,
                               self.decoder_standard_dropout, self.decoder_recurrent_dropout, self.attention_units)

    def get_config(self):
        config = super(Seq2SeqAttentionLSTM, self).get_config()
        config.update({
            'encoder_lstm_units': self.encoder_lstm_units,
            'decoder_lstm_units': self.decoder_lstm_units,
            'output_feature_dim': self.output_feature_dim,
            'encoder_standard_dropout': self.encoder_standard_dropout,
            'encoder_recurrent_dropout': self.encoder_recurrent_dropout,
            'decoder_standard_dropout': self.decoder_standard_dropout,
            'decoder_recurrent_dropout': self.decoder_recurrent_dropout,
            'attention_units': self.attention_units
        })
        return config

    def call(self, inputs, training=None):
        x_encoder_input, x_decoder_teacher_forcing_input = inputs
        
        encoder_outputs, encoder_states = self.encoder(x_encoder_input)
        
        # Pass encoder_states (list of [h,c] for the last layer of encoder)
        # and encoder_outputs to the decoder.
        # The decoder's call method handles the loop for teacher forcing.
        decoder_output = self.decoder(x_decoder_teacher_forcing_input, encoder_states, encoder_outputs, training=training)
        return decoder_output

    # Note: Removed the unused 'initial_decoder_input_embedding_dim' argument
    def predict_sequence(self, x_encoder_input_sample, true_y_T_scaled_sample, forecast_horizon_H):
        """
        Performs recursive sequence prediction for a single input sample.

        Args:
            x_encoder_input_sample (tf.Tensor): Single encoder input, shape (1, encoder_timesteps, K+1).
            true_y_T_scaled_sample (tf.Tensor): Scaled true price at time T (last day of encoder input), shape (1,1) or scalar.
                                                Used to form the first input to the decoder.
            forecast_horizon_H (int): Number of steps to predict.


        Returns:
            tf.Tensor: Predicted sequence, shape (1, H, output_feature_dim).
        """
        encoder_outputs, encoder_states = self.encoder(x_encoder_input_sample, training=False)
        
        # Initialize decoder states with corresponding encoder states
        # encoder_states is [enc_state_h1, enc_state_c1, enc_state_h2, enc_state_c2]
        current_state_h_l1, current_state_c_l1 = encoder_states[0], encoder_states[1]
        current_state_h_l2, current_state_c_l2 = encoder_states[2], encoder_states[3]

        # First input to decoder: true_y_T_scaled_sample
        # Ensure it's correctly shaped: (1, 1, 1) for embedding layer expecting (batch, timesteps, features)
        current_decoder_input_step = tf.reshape(true_y_T_scaled_sample, [1, 1, 1])

        predicted_sequence_list = []

        for _ in tf.range(forecast_horizon_H):
            embedded_input = self.decoder.embedding(current_decoder_input_step) # (1,1,embedding_dim)
            
            query_for_attention = current_state_h_l2 # Use hidden state of last decoder LSTM
            context_vector, _ = self.decoder.attention(query_for_attention, encoder_outputs)
            context_vector_expanded = tf.expand_dims(context_vector, 1)
            
            decoder_lstm_input = tf.concat([embedded_input, context_vector_expanded], axis=-1)

            # Decoder LSTM Layer 1
            full_seq_output_l1, state_h_l1_next, state_c_l1_next = self.decoder.lstm1(
                decoder_lstm_input, initial_state=[current_state_h_l1, current_state_c_l1], training=False
            )
            # Decoder LSTM Layer 2
            full_seq_output_l2, state_h_l2_next, state_c_l2_next = self.decoder.lstm2(
                full_seq_output_l1, initial_state=[current_state_h_l2, current_state_c_l2], training=False
            )
            
            current_step_lstm_output = full_seq_output_l2[:, 0, :]
            prediction_for_step = self.decoder.dense_output(current_step_lstm_output) # (1,1)
            
            predicted_sequence_list.append(prediction_for_step)
            
            # Next input is the current prediction
            current_decoder_input_step = tf.expand_dims(prediction_for_step, axis=1) # Reshape to (1,1,1)

            # Update states
            current_state_h_l1, current_state_c_l1 = state_h_l1_next, state_c_l1_next
            current_state_h_l2, current_state_c_l2 = state_h_l2_next, state_c_l2_next

        return tf.stack(predicted_sequence_list, axis=1) # Shape (1, H, 1)

if __name__ == '__main__':
    # Example of how to instantiate and test the model (very basic)
    print("Testing Seq2SeqAttentionLSTM model structure...")
    
    # Dummy parameters (replace with actual config values)
    encoder_timesteps = config.SEQ2SEQ_ENCODER_TIMESTEPS # 10
    k_plus_1_features = config.ICEEMDAN_TARGET_K_IMF + 1 # 9
    forecast_horizon = config.ICEEMDAN_H_FORECAST_PERIOD # 5
    output_dim = 1 # Predicting 1 value (price)
    
    encoder_units = config.SEQ2SEQ_ENCODER_LSTM_UNITS # e.g., [128, 64]
    decoder_units = config.SEQ2SEQ_DECODER_LSTM_UNITS # e.g., [128, 64]
    attention_units_val = 64 # Example

    std_dropout = config.LSTM_STANDARD_DROPOUT_RATE
    rec_dropout = config.LSTM_RECURRENT_DROPOUT_RATE

    # Create Encoder
    encoder_model = Encoder(lstm_units_list=encoder_units, 
                            standard_dropout_rate=std_dropout, 
                            recurrent_dropout_rate=rec_dropout)
    
    # Create Decoder
    # Decoder's LSTM input feature size will be embedding_dim + encoder_output_units (from attention context)
    # This needs careful dimension handling.
    # The current Decoder design's LSTM input is concat(embedded_price, context_vector)
    # embedded_price is self.decoder.embedding_dim (e.g., 128 if matches first LSTM unit)
    # context_vector is encoder's last layer units (e.g., 64)
    # So, decoder LSTM input_dim = 128 + 64 = 192. This is not explicitly set in LSTM layer.
    # Keras LSTM infers input_dim from first batch if not specified in Input layer.
    # This part is complex and needs to be robust.
    
    decoder_model = Decoder(lstm_units_list=decoder_units, 
                            output_feature_dim=output_dim,
                            standard_dropout_rate=std_dropout, 
                            recurrent_dropout_rate=rec_dropout,
                            attention_units=attention_units_val)

    # Create Seq2Seq Model
    seq2seq_model = Seq2SeqAttentionLSTM(
        encoder_lstm_units=encoder_units,
        decoder_lstm_units=decoder_units,
        output_feature_dim=output_dim,
        encoder_standard_dropout=std_dropout,
        encoder_recurrent_dropout=rec_dropout,
        decoder_standard_dropout=std_dropout,
        decoder_recurrent_dropout=rec_dropout,
        attention_units=attention_units_val
    )

    # Dummy input data
    batch_s = 4
    dummy_encoder_input = tf.random.normal((batch_s, encoder_timesteps, k_plus_1_features))
    dummy_decoder_teacher_input = tf.random.normal((batch_s, forecast_horizon, output_dim)) # Scaled prices

    print(f"Dummy Encoder Input shape: {dummy_encoder_input.shape}")
    print(f"Dummy Decoder Teacher Input shape: {dummy_decoder_teacher_input.shape}")

    # Test model call (training mode)
    try:
        predictions_train_mode = seq2seq_model([dummy_encoder_input, dummy_decoder_teacher_input], training=True)
        print(f"Predictions (train mode) shape: {predictions_train_mode.shape}") # Expected: (batch_s, forecast_horizon, output_dim)
    except Exception as e:
        print(f"Error during model call (train mode): {e}")
        import traceback
        traceback.print_exc()

    # Test predict_sequence (inference mode)
    try:
        single_enc_input = dummy_encoder_input[0:1, :, :] # Take one sample
        single_true_y_T = tf.random.normal((1,1)) # Dummy last known true price (scaled)
        
        # Need to define initial_decoder_input_embedding_dim for predict_sequence
        # This was from a previous plan. Let's use decoder.embedding.units
        # The predict_sequence method needs to align with how Decoder.call handles input.
        # The current predict_sequence is a bit diverged from the Decoder.call logic.
        # This test part needs refinement once the Decoder logic is fully set.
        
        # For now, let's just check if the model can be built.
        # seq2seq_model.build(input_shape=[dummy_encoder_input.shape, dummy_decoder_teacher_input.shape])
        # print("\nModel Summary:")
        # seq2seq_model.summary() # This will build the model if not already built by a call.
        
        # To properly test predict_sequence, we need to ensure its internal logic for input handling
        # and state management is consistent with the Decoder's call method.
        # The current predict_sequence has its own loop.
        
        print("\nAttempting to build and summarize the full Seq2Seq model...")
        # A call will build it:
        _ = seq2seq_model([dummy_encoder_input, dummy_decoder_teacher_input])
        seq2seq_model.summary()


        print("\nTesting predict_sequence (inference)...")
        # The predict_sequence method needs to be carefully aligned with the Decoder's call logic.
        # The current `predict_sequence` has its own internal loop.
        # A more integrated approach would modify Decoder.call to handle training=False.
        # For now, testing the existing predict_sequence structure.

        # The `initial_decoder_input_embedding_dim` argument in predict_sequence definition is unused and can be removed.
        # Let's assume it's removed from the definition for this call.
        # Corrected call without the dummy last argument:
        pred_inf = seq2seq_model.predict_sequence(single_enc_input, single_true_y_T, forecast_horizon)
        print(f"Inference prediction shape: {pred_inf.shape}") # Expected (1, H, 1)


    except Exception as e:
        print(f"Error during model build or predict_sequence test: {e}")
        import traceback
        traceback.print_exc()

    print("\nSeq2SeqAttentionLSTM structure test complete.")