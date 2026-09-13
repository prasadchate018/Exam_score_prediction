from flask import Flask, render_template_string, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the SVM pickle model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'svm.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Embedded HTML & CSS Layout
HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Prediction Analytics</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.12);
            --accent-purple: #8b5cf6;
            --accent-pink: #ec4899;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 
                0 20px 50px rgba(0, 0, 0, 0.5),
                0 0 40px rgba(139, 92, 246, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            transition: box-shadow 0.4s ease;
        }

        .container:hover {
            box-shadow: 
                0 25px 60px rgba(0, 0, 0, 0.6),
                0 0 50px rgba(236, 72, 153, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(to right, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: capitalize;
            letter-spacing: 0.5px;
        }

        .input-group input, .input-group select {
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--accent-purple);
            box-shadow: 
                0 0 15px rgba(139, 92, 246, 0.3),
                inset 0 2px 4px rgba(0,0,0,0.3);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 1rem;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink));
            color: #fff;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 25px rgba(236, 72, 153, 0.3);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 35px rgba(236, 72, 153, 0.45);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.25rem;
            border-radius: 16px;
            background: rgba(139, 92, 246, 0.1);
            border: 1px solid rgba(139, 92, 246, 0.3);
            text-align: center;
            display: none;
            animation: fadeIn 0.4s ease forwards;
        }

        .result-box h3 {
            font-size: 1rem;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }

        .result-box .val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #fff;
            text-shadow: 0 0 12px rgba(167, 139, 250, 0.6);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>SVM Predictor</h1>
        <p>Enter parameters to compute performance score</p>
    </div>

    <form id="predictionForm" class="form-grid">
        <div class="input-group">
            <label>Age</label>
            <input type="number" step="any" name="age" required placeholder="e.g. 21">
        </div>
        <div class="input-group">
            <label>Gender</label>
            <select name="gender" required>
                <option value="0">Male</option>
                <option value="1">Female</option>
            </select>
        </div>
        <div class="input-group">
            <label>Course Code</label>
            <input type="number" step="any" name="course" required placeholder="e.g. 1">
        </div>
        <div class="input-group">
            <label>Study Hours</label>
            <input type="number" step="any" name="study_hours" required placeholder="e.g. 5.5">
        </div>
        <div class="input-group">
            <label>Class Attendance (%)</label>
            <input type="number" step="any" name="class_attendance" required placeholder="e.g. 85">
        </div>
        <div class="input-group">
            <label>Internet Access</label>
            <select name="internet_access" required>
                <option value="1">Yes</option>
                <option value="0">No</option>
            </select>
        </div>
        <div class="input-group">
            <label>Sleep Hours</label>
            <input type="number" step="any" name="sleep_hours" required placeholder="e.g. 7">
        </div>
        <div class="input-group">
            <label>Sleep Quality (1-5)</label>
            <input type="number" step="any" name="sleep_quality" required placeholder="e.g. 4">
        </div>
        <div class="input-group">
            <label>Study Method Code</label>
            <input type="number" step="any" name="study_method" required placeholder="e.g. 2">
        </div>
        <div class="input-group">
            <label>Facility Rating (1-5)</label>
            <input type="number" step="any" name="facility_rating" required placeholder="e.g. 3">
        </div>
        <div class="input-group">
            <label>Exam Difficulty (1-5)</label>
            <input type="number" step="any" name="exam_difficulty" required placeholder="e.g. 3">
        </div>

        <button type="submit" class="btn-submit">Generate Prediction</button>
    </form>

    <div id="result" class="result-box">
        <h3>Predicted Outcome</h3>
        <div id="resultValue" class="val">--</div>
    </div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        const resultBox = document.getElementById('result');
        const resultVal = document.getElementById('resultValue');

        if (data.status === 'success') {
            resultVal.textContent = data.prediction;
            resultBox.style.display = 'block';
        } else {
            resultVal.textContent = 'Error processing model';
            resultBox.style.display = 'block';
        }
    });
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_LAYOUT)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Features order matches model: age, gender, course, study_hours, class_attendance, 
        # internet_access, sleep_hours, sleep_quality, study_method, facility_rating, exam_difficulty
        features = [
            float(request.form.get('age', 0)),
            float(request.form.get('gender', 0)),
            float(request.form.get('course', 0)),
            float(request.form.get('study_hours', 0)),
            float(request.form.get('class_attendance', 0)),
            float(request.form.get('internet_access', 0)),
            float(request.form.get('sleep_hours', 0)),
            float(request.form.get('sleep_quality', 0)),
            float(request.form.get('study_method', 0)),
            float(request.form.get('facility_rating', 0)),
            float(request.form.get('exam_difficulty', 0))
        ]
        
        prediction = model.predict([features])[0]
        return jsonify({'status': 'success', 'prediction': round(float(prediction), 4)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
