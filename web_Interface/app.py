import gradio as gr
import pickle
import pandas as pd

# -------------------------------
# Load model and scaler
# -------------------------------
with open("models/gradient_boosting_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/transformation_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# Feature names (MUST match training)
# -------------------------------
FEATURES = [
    "Relative Compactness",
    "Surface Area",
    "Wall Area",
    "Roof Area",
    "Overall Height",
    "Orientation",
    "Glazing Area",
    "Glazing Area Distribution"
]

# -------------------------------
# Prediction function
# -------------------------------
def predict_energy(
    relative_compactness,
    surface_area,
    wall_area,
    roof_area,
    overall_height,
    orientation,
    glazing_area,
    glazing_area_distribution
):
    # Create DataFrame instead of NumPy array
    input_df = pd.DataFrame([{
        "Relative Compactness": relative_compactness,
        "Surface Area": surface_area,
        "Wall Area": wall_area,
        "Roof Area": roof_area,
        "Overall Height": overall_height,
        "Orientation": orientation,
        "Glazing Area": glazing_area,
        "Glazing Area Distribution": glazing_area_distribution
    }])

    # Apply transformation
    input_transformed = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_transformed)

    return round(prediction[0][0], 2), round(prediction[0][1], 2)

# -------------------------------
# Gradio UI
# -------------------------------
with gr.Blocks(title="Energy Efficiency Prediction System") as demo:
    gr.Markdown("""
    # Building Energy Efficiency Predictor  
    Predict **Heating Load** and **Cooling Load** using a trained Gradient Boosting model.
    """)

    with gr.Row():
        with gr.Column():
            relative_compactness = gr.Number(label="Relative Compactness")
            surface_area = gr.Number(label="Surface Area")
            wall_area = gr.Number(label="Wall Area")
            roof_area = gr.Number(label="Roof Area")

        with gr.Column():
            overall_height = gr.Number(label="Overall Height")
            orientation = gr.Number(label="Orientation (1–4)")
            glazing_area = gr.Number(label="Glazing Area")
            glazing_area_distribution = gr.Number(label="Glazing Area Distribution (0–5)")

    predict_btn = gr.Button("Predict Energy Load")

    with gr.Row():
        heating_output = gr.Number(label="Predicted Heating Load")
        cooling_output = gr.Number(label="Predicted Cooling Load")

    predict_btn.click(
        fn=predict_energy,
        inputs=[
            relative_compactness,
            surface_area,
            wall_area,
            roof_area,
            overall_height,
            orientation,
            glazing_area,
            glazing_area_distribution
        ],
        outputs=[heating_output, cooling_output]
    )

# -------------------------------
# Launch
# -------------------------------
if __name__ == "__main__":
    demo.launch()
