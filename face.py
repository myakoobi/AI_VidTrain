import cv2
import os 
from ultralytics import YOLO 
import speech_recognition as sr

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Create dataset folders
os.makedirs("dataset/images/train", exist_ok=True)
os.makedirs("dataset/labels/train", exist_ok=True)

# To keep track of label IDs
class_name_to_id = {}
label_count = 0

recognizer = sr.Recognizer()

for ea in os.listdir('data'):
    if not ea.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        continue
    
    each = os.path.join('data', ea)
    cap = cv2.VideoCapture(each)

    ret, frame = cap.read()

    if not ret:
        print(f"[ERROR] Couldn't read first frame of {each}")
        cap.release()
        continue

    results = model(frame)
    ann_fr = results[0].plot()

    # Show the frame
    cv2.imshow("Annotated Frame", ann_fr)
    print("Speak the name of the object you're holding...")
    cv2.waitKey(500)  # Show window briefly before listening

    label = None

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating for background noise...")
        r.adjust_for_ambient_noise(source, duration=1)
        print("Listening now...")
        audio = r.listen(source)

    try:
        label = r.recognize_google(audio)
        label = label.strip().lower()
        print("You said:", label)
    except sr.UnknownValueError:
        print("Could not understand.")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    if label:
        if label not in class_name_to_id:
            class_name_to_id[label] = label_count
            label_count += 1

        class_id = class_name_to_id[label]

        image_filename = f"dataset/images/train/{label}_{ea.replace('.', '_')}.jpg"
        cv2.imwrite(image_filename, frame)

        h, w, _ = frame.shape
        label_filename = image_filename.replace("images", "labels").replace(".jpg", ".txt")

        x_center, y_center, box_w, box_h = 0.5, 0.5, 0.4, 0.4
        with open(label_filename, "w") as f:
            f.write(f"{class_id} {x_center} {y_center} {box_w} {box_h}\n")

        print(f"[SAVED] Frame: {image_filename}, Label: {label_filename}")
    else:
        print("[SKIPPED] No label recorded, skipping this frame.")


    cap.release()
    cv2.destroyAllWindows()

# Show final class map
print("\nClass map:", class_name_to_id)
# === Save custom_data.yaml for training ===
import yaml

yaml_data = {
    'path': os.path.abspath('dataset'),
    'train': 'images/train',
    'val': 'images/train',
    'nc': len(class_name_to_id),
    'names': [None] * len(class_name_to_id)
}

for name, idx in class_name_to_id.items():
    yaml_data['names'][idx] = name

with open('custom_data.yaml', 'w') as f:
    yaml.dump(yaml_data, f)

print("\n✅ Saved 'custom_data.yaml':")
print(yaml_data)

