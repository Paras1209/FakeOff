from flask import Flask, request, jsonify, render_template
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch


app = Flask(__name__)

model_dir = "" #trained model directory

model = ViTForImageClassification.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=True
)

processor = ViTImageProcessor.from_pretrained(model_dir)

print("Model and processor loaded succesfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error":"no image file found in the request"}),400
        
        file = request.files['image']
        image = Image.open(file.stream).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = output.logits
            predicted_class_id = logits.argmax(-1).item()

        predicted_label = model.config.id2label[predicted_class_id]

        return jsonify({
            "prediction": predicted_label,
            "confidence": torch.softmax(logits,dim=-1)[0][predicted_class_id].item()
        })
    
    except Exception as e:
        return jsonify({"error":str(e)}),500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)