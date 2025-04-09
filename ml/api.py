import os
import pickle
import io
import csv
from flask import Flask, request, jsonify, send_file, Response
from reportlab.pdfgen import canvas

# Define paths based on current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_category_path = os.path.join(script_dir, 'model_category.pkl')
model_type_path = os.path.join(script_dir, 'model_type.pkl')
vectorizer_path = os.path.join(script_dir, 'vectorizer.pkl')
le_category_path = os.path.join(script_dir, 'le_category.pkl')
le_type_path = os.path.join(script_dir, 'le_type.pkl')

# Load trained models, vectorizer, and label encoders
with open(model_category_path, 'rb') as f:
    model_category = pickle.load(f)
with open(model_type_path, 'rb') as f:
    model_type = pickle.load(f)
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)
with open(le_category_path, 'rb') as f:
    le_category = pickle.load(f)
with open(le_type_path, 'rb') as f:
    le_type = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'text' not in data:
        return jsonify({'error': "Missing 'text' field in request data"}), 400

    description = data['text']
    X = vectorizer.transform([description])

    # Predict with probabilities
    category_proba = model_category.predict_proba(X)[0]
    type_proba = model_type.predict_proba(X)[0]

    # Get numeric predictions from highest probability
    pred_category_num = model_category.classes_[category_proba.argmax()]
    pred_type_num = model_type.classes_[type_proba.argmax()]

    # Convert NumPy float32 values to native Python floats for JSON serialization
    confidence_category = float(max(category_proba))
    confidence_type = float(max(type_proba))

    # Convert numeric predictions back to string labels using the label encoders
    predicted_category = le_category.inverse_transform([pred_category_num])[0]
    predicted_type = le_type.inverse_transform([pred_type_num])[0]

    # Set uncertainty threshold
    threshold = 0.1
    if confidence_category < threshold:
        predicted_category = "Uncertain"
    if confidence_type < threshold:
        predicted_type = "Uncertain"

    return jsonify({
        'category': predicted_category,
        'category_confidence': confidence_category,
        'type': predicted_type,
        'type_confidence': confidence_type
    })

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json(force=True)
        # Validate required fields
        if not all(key in data for key in ['text', 'type', 'category']):
            return jsonify({'error': "Missing one or more required fields: 'text', 'type', 'category'"}), 400

        text = data['text']
        type_str = data['type']
        category_str = data['category']

        # Vectorize the input text using the pre-loaded vectorizer
        X = vectorizer.transform([text])

        # Convert the provided category and type labels to their numeric representation
        y_category = le_category.transform([category_str])
        y_type = le_type.transform([type_str])

        # Update the models using partial_fit if available
        if hasattr(model_category, 'partial_fit'):
            model_category.partial_fit(X, y_category, classes=le_category.classes_)
        else:
            return jsonify({'error': 'Model does not support incremental training for category.'}), 500

        if hasattr(model_type, 'partial_fit'):
            model_type.partial_fit(X, y_type, classes=le_type.classes_)
        else:
            return jsonify({'error': 'Model does not support incremental training for type.'}), 500

        # Save the updated models back to disk
        with open(model_category_path, 'wb') as f:
            pickle.dump(model_category, f)
        with open(model_type_path, 'wb') as f:
            pickle.dump(model_type, f)

        return jsonify({'message': 'Training successful'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/pdf')
def download_pdf():
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica", 16)
    p.drawString(100, 750, "Bank Statement / Prediction Details")
    p.setFont("Helvetica", 12)
    p.drawString(100, 730, "Account Holder: John Doe")
    p.drawString(100, 710, "Account Number: 123456789")
    p.drawString(100, 690, "Statement Date: 2025-03-28")
    p.drawString(100, 670, "Closing Balance: $1,000.00")
    p.drawString(100, 650, "Predicted Category: Sample Category")
    p.drawString(100, 630, "Predicted Type: Sample Type")
    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='statement.pdf', mimetype='application/pdf')

@app.route('/download/csv')
def download_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Field', 'Value'])
    writer.writerow(['Account Holder', 'John Doe'])
    writer.writerow(['Account Number', '123456789'])
    writer.writerow(['Statement Date', '2025-03-28'])
    writer.writerow(['Closing Balance', '1000.00'])
    writer.writerow([])
    writer.writerow(['Date', 'Description', 'Category', 'Debit', 'Credit', 'Balance', 'Type'])
    writer.writerow(['2025-03-28', 'Transaction 1', 'Sample Category', '100.00', '0.00', '900.00', 'Debit'])
    writer.writerow(['2025-03-28', 'Transaction 2', 'Sample Category', '0.00', '150.00', '1050.00', 'Credit'])
    output.seek(0)
    return Response(output.getvalue(), mimetype='text/csv', headers={"Content-Disposition": "attachment;filename=statement.csv"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
