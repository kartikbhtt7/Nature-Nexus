import torch
import numpy as np
from utils.audio_processing import preprocess_audio

class_names = [
    'fireworks', 'chainsaw', 'footsteps', 'car_horn', 'crackling_fire',
    'drinking_sipping', 'laughing', 'engine', 'breathing', 'hand_saw',
    'coughing', 'snoring', 'sneezing', 'siren'
]

class AudioClassifier(torch.nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.2),
            
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.2),
            
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def load_audio_model(model_path='models/audio_model.pth'):
    model = AudioClassifier(len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_audio(audio_path, model):
    # Preprocess audio
    spec = preprocess_audio(audio_path)
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(spec).unsqueeze(0)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get results
    pred_prob, pred_index = torch.max(probabilities, 1)
    return class_names[pred_index.item()], pred_prob.item()