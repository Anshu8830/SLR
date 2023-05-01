import numpy as np
from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
  if(request.method == 'POST'):
        audio_file = request.files['file']
        ##if audio_file.filename !='':
           # audio_file.save(audio_file.filename)
        actions = np.array(['hello', 'thanks', 'iloveyou']) 
        
        model = load_model('action.h5')
        
        mp_holistic = mp.solutions.holistic # Holistic model

        def mediapipe_detection(image, model):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
            image.flags.writeable = False                  # Image is no longer writeable
            results = model.process(image)                 # Make prediction
            image.flags.writeable = True                   # Image is now writeable 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
            return image, results

        def extract_keypoints(results):
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
            return np.concatenate([pose, face, lh, rh])


        # 1. New detection variables
        sequence = []
        cap = cv2.VideoCapture(audio_file.filename)

        # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            while cap.isOpened():

                ret, frame = cap.read()
                if not ret: 
                    cap.release()
                    break    
        
                image, results = mediapipe_detection(frame, holistic) 
                keypoints = extract_keypoints(results)
                sequence.insert(0,keypoints) 
                sequence = sequence[-30:]     
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    break
              
        resp = {"output": actions[np.argmax(res)]}
        return jsonify(resp)

      
if(__name__ == "__main__"):  app.run(debug=True, port=4000)
