import os
import joblib
from django.shortcuts import render

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'ml_models', 'multioutput_xgboost_model_r.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'ml_models', 'scaler_rr4.pkl'))

# List of (field_name, readable_label)
FIELDS = [
    ('relative_compactness', 'Relative Compactness'),
    ('wall_area', 'Wall Area'),
    ('orientation', 'Orientation'),
    ('glazing_area', 'Glazing Area'),
    ('glazing_area_distribution', 'Glazing Area Distribution')
]

# XGBoost model metrics
MODEL_METRICS = {
    'heating': {
        'rmse': 0.403,
        'mae': 0.284,
        'r2': f"{0.998 * 100:.1f}%",
        'adj_r2': f"{0.998 * 100:.1f}%"
    },
    'cooling': {
        'rmse': 1.136,
        'mae': 0.696,
        'r2': f"{0.986 * 100:.1f}%",
        'adj_r2': f"{0.985 * 100:.1f}%"
    }
}

placeholders = {
    'relative_compactness': {'value': 0.75, 'range': '0.62 – 0.98'},
    'wall_area': {'value': 318.5, 'range': '245 – 416.5'},
    'orientation': {'value': 3.5, 'range': '2 – 5'},
    'glazing_area': {'value': 0.25, 'range': '0 – 0.4'},
    'glazing_area_distribution': {'value': 3, 'range': '0 – 5'},
}

def predict_energy(request):
    result = None
    input_values = {}

    if request.method == 'POST':
        input_data = []
        for field, _ in FIELDS:
            value = request.POST.get(field)
            input_values[field] = value
            input_data.append(float(value))

        scaled = scaler.transform([input_data])
        prediction = model.predict(scaled)[0]

        result = {
            'heating': round(prediction[0], 2),
            'cooling': round(prediction[1], 2)
        }

    return render(request, 'predictor/form.html', {
        'fields': FIELDS,
        'result': result,
        'input_values': input_values,
        'model_metrics': MODEL_METRICS,
        'placeholders': placeholders
    })
