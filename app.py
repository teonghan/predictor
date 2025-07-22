import streamlit as st
import pandas as pd
import pickle
import numpy as np # Needed for numerical operations if any

st.set_page_config(page_title="Model Prediction App", layout="centered")

st.title("🔮 Model Prediction App")
st.write("Upload your trained model (.pkl file) and enter new data or upload a list of data points to get predictions.")

# ---- Model Upload ----
uploaded_model_file = st.file_uploader("Upload your trained model (.pkl file)", type=["pkl"])

if uploaded_model_file is not None:
    try:
        # Load the model and metadata
        loaded_data = pickle.load(uploaded_model_file)

        model = loaded_data['model']
        feature_names = loaded_data['feature_names']
        target_column = loaded_data['target_column']
        is_regression = loaded_data['is_regression']
        label_encoder = loaded_data.get('label_encoder') # Will be None if regression
        original_predictor_cols = loaded_data['original_predictor_cols']
        categorical_unique_values = loaded_data['categorical_unique_values']
        one_hot_encoded_feature_map = loaded_data['one_hot_encoded_feature_map']


        st.success(f"Model for predicting '{target_column}' loaded successfully!")
        st.write(f"This is a {'Regression' if is_regression else 'Classification'} model.")
        st.write(f"The model expects the following features (after processing): {', '.join(feature_names)}")

        if not is_regression and label_encoder is not None:
            st.write("---")
            st.subheader("Target Class Mapping:")
            # Display the mapping for classification models
            class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
            st.write(class_mapping)
            st.info("This shows what each numerical prediction (0, 1, etc.) corresponds to in terms of actual categories.")


        st.markdown("---")
        st.header("How would you like to provide data for prediction?")
        prediction_mode = st.radio(
            "Select Prediction Mode:",
            ("Enter Single Data Point", "Upload List of Data Points (CSV/Excel)"),
            key="prediction_mode"
        )

        # --- Data Preprocessing Function ---
        def preprocess_data_for_prediction(raw_df, original_cols, cat_unique_vals, ohe_map, model_feature_names):
            processed_df = pd.DataFrame()

            for col in original_cols:
                if col in cat_unique_vals:
                    # Handle categorical features: one-hot encode
                    # Ensure the column exists in the raw_df
                    if col in raw_df.columns:
                        # Convert to category type with known categories to handle unseen values gracefully
                        # This is crucial if a category in new data wasn't in training data
                        raw_df[col] = pd.Categorical(raw_df[col], categories=cat_unique_vals[col])
                        encoded_temp = pd.get_dummies(raw_df[col], prefix=col, prefix_sep='_', dummy_na=False, drop_first=True)
                        processed_df = pd.concat([processed_df, encoded_temp], axis=1)
                    else:
                        # If the original categorical column is missing in the new data,
                        # create zero columns for all its OHE features
                        for ohe_col in ohe_map.get(col, []):
                            processed_df[ohe_col] = 0
                else:
                    # Handle numeric features
                    if col in raw_df.columns:
                        processed_df[col] = pd.to_numeric(raw_df[col], errors='coerce') # Coerce non-numeric to NaN
                    else:
                        # If a numeric column is missing, fill with a default (e.g., 0 or mean from training)
                        # For simplicity, we'll fill with 0, but in a real scenario, consider training data mean/median
                        processed_df[col] = 0

            # Ensure the order of columns matches the training data (model_feature_names)
            # and fill any missing columns (e.g., for categories not present in new_raw_data) with 0
            final_df = processed_df.reindex(columns=model_feature_names, fill_value=0)

            # Drop any columns that are in processed_df but not in model_feature_names (shouldn't happen with reindex)
            final_df = final_df[model_feature_names]

            return final_df


        if prediction_mode == "Enter Single Data Point":
            st.write("---")
            st.write("Please enter values for the original features below.")

            new_raw_data_input = {}
            st.subheader("Feature Values:")

            input_cols = st.columns(min(len(original_predictor_cols), 4))
            col_idx = 0
            for feature in original_predictor_cols:
                with input_cols[col_idx % 4]:
                    if feature in categorical_unique_values:
                        options = categorical_unique_values[feature]
                        new_raw_data_input[feature] = st.selectbox(f"{feature} (Category)", options, key=f"input_{feature}_cat")
                    else:
                        new_data_input_val = st.number_input(f"{feature} (Number)", value=None, key=f"input_{feature}_num", format="%.2f")
                        new_raw_data_input[feature] = new_data_input_val
                col_idx += 1

            if st.button("Make Prediction"):
                # Check if all numeric inputs are filled
                all_inputs_filled = True
                for col in original_predictor_cols:
                    raw_value = new_raw_data_input[col]
                    if col not in categorical_unique_values and raw_value is None:
                        st.warning(f"Please enter a value for the numeric feature: '{col}'")
                        all_inputs_filled = False

                if not all_inputs_filled:
                    st.stop()

                # Convert single input to a DataFrame for preprocessing
                single_input_df = pd.DataFrame([new_raw_data_input])
                final_input_df = preprocess_data_for_prediction(
                    single_input_df,
                    original_predictor_cols,
                    categorical_unique_values,
                    one_hot_encoded_feature_map,
                    feature_names
                )

                # Make prediction
                prediction = model.predict(final_input_df)

                st.markdown("---")
                st.subheader("Prediction Result:")
                if is_regression:
                    st.success(f"The predicted {target_column} is: **{prediction[0]:.2f}**")
                else:
                    predicted_class = label_encoder.inverse_transform(prediction)
                    st.success(f"The predicted {target_column} is: **{predicted_class[0]}**")

                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(final_input_df)
                        st.write("Prediction Probabilities:")
                        proba_df = pd.DataFrame(probabilities, columns=label_encoder.classes_)
                        st.dataframe(proba_df)

        elif prediction_mode == "Upload List of Data Points (CSV/Excel)":
            st.write("---")
            st.write("Upload a CSV or Excel file containing the features for prediction.")
            st.info(f"The file should contain columns matching your original predictor columns: {', '.join(original_predictor_cols)}")

            st.subheader("Expected Data Format:")
            st.markdown("Please ensure your uploaded file adheres to the following format for each column:")
            for feature in original_predictor_cols:
                if feature in categorical_unique_values:
                    st.markdown(f"- **'{feature}' (Categorical):** Must contain one of the following values: `{', '.join(map(str, categorical_unique_values[feature]))}`. Case-sensitive.")
                else:
                    # Heuristic to suggest integer or float based on common numeric column names
                    if any(x in feature.lower() for x in ['id', 'count', 'year', 'age', 'years', 'number']):
                        st.markdown(f"- **'{feature}' (Numeric/Integer):** Should contain whole numbers (e.g., `10`, `25`).")
                    else:
                        st.markdown(f"- **'{feature}' (Numeric/Decimal):** Can contain whole numbers or decimals (e.g., `10.5`, `25`).")
            st.markdown("---")


            uploaded_data_file = st.file_uploader("Choose a CSV or Excel file for prediction", type=["csv", "xlsx"])

            if uploaded_data_file is not None:
                try:
                    if uploaded_data_file.name.endswith(".csv"):
                        new_raw_data_df = pd.read_csv(uploaded_data_file)
                    else:
                        new_raw_data_df = pd.read_excel(uploaded_data_file)

                    st.success("Prediction data file loaded successfully!")
                    st.dataframe(new_raw_data_df.head())

                    # Check if all original predictor columns are present in the uploaded file
                    missing_cols = [col for col in original_predictor_cols if col not in new_raw_data_df.columns]
                    if missing_cols:
                        st.warning(f"The uploaded file is missing the following predictor columns: {', '.join(missing_cols)}. These will be treated as missing values.")
                        # Fill missing columns with appropriate defaults (e.g., 0 for numeric, first category for categorical)
                        for m_col in missing_cols:
                            if m_col in categorical_unique_values:
                                new_raw_data_df[m_col] = categorical_unique_values[m_col][0] # Default to first category
                            else:
                                new_raw_data_df[m_col] = 0.0 # Default to 0 for numeric

                    if st.button("Make Batch Predictions"):
                        # Preprocess the entire DataFrame
                        final_input_df = preprocess_data_for_prediction(
                            new_raw_data_df,
                            original_predictor_cols,
                            categorical_unique_values,
                            one_hot_encoded_feature_map,
                            feature_names
                        )

                        # Make predictions
                        batch_predictions = model.predict(final_input_df)

                        # Add predictions to the original DataFrame
                        new_raw_data_df[f'Predicted {target_column}'] = batch_predictions

                        if not is_regression and label_encoder is not None:
                            # Decode numerical predictions back to original class labels
                            new_raw_data_df[f'Predicted {target_column}'] = label_encoder.inverse_transform(batch_predictions)

                        st.markdown("---")
                        st.subheader("Batch Prediction Results:")
                        st.dataframe(new_raw_data_df)

                        # Provide download option for results
                        csv_output = new_raw_data_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv_output,
                            file_name=f"predictions_{target_column}.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"An error occurred while processing the uploaded data: {e}")
                    st.info("Please ensure your uploaded file is correctly formatted and contains the expected columns.")
                    st.exception(e) # Display full traceback for debugging

    except pickle.UnpicklingError:
        st.error("Invalid pickle file. Please ensure you uploaded a valid model .pkl file.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure the uploaded file is a valid model exported from the 'Basic Stats Explorer' app.")
        st.exception(e) # Display full traceback for debugging
