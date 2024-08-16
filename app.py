from flask import Flask, request, send_file, render_template, url_for
import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the model
model_path = r'C:\Users\dharm\Downloads\parking\model\parking_tracking.pth'
num_classes = 2
image_shape = (64, 64)

# Initialize the model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define the transform
transform = transforms.Compose([
    transforms.Resize(image_shape),
    transforms.ToTensor(),
])

# Define constants
EMPTY = True
NOT_EMPTY = False

def empty_or_not(spot_bgr):
    if spot_bgr is None or spot_bgr.size == 0:
        return EMPTY

    try:
        # Convert BGR image to RGB and PIL Image
        spot_rgb = cv2.cvtColor(spot_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(spot_rgb)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()

        return EMPTY if prediction == 0 else NOT_EMPTY

    except Exception as e:
        print(f"Error processing image: {e}")
        return NOT_EMPTY

def get_parking_spots_bboxes(mask):
    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []
    coef = 1
    for i in range(1, totalLabels):
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)
        slots.append([x1, y1, w, h])
    return slots

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        input_video_path = 'uploaded_video.mp4'
        output_video_path = 'output_video.mp4'
        mask_path = r'C:\Users\dharm\Downloads\parking\mask_1920_1080.png'  # Update this path as needed
        
        file.save(input_video_path)
        
        # Load the mask
        mask = cv2.imread(mask_path, 0)

        # Process the video
        process_video(input_video_path, output_video_path, mask)
        
        # Redirect to download page
        return render_template('download.html', filename='output_video.mp4')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

def process_video(video_path, output_video_path, mask):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error reading video file")
        return

    # Get parking spot bounding boxes
    spots = get_parking_spots_bboxes(mask)

    spots_status = [None for _ in spots]
    previous_frame = None

    frame_nmr = 0
    step = 30
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1], frame.shape[0]))

    def calc_diff(im1, im2):
        return np.abs(np.mean(im1) - np.mean(im2))

    while ret:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_nmr % step == 0 and previous_frame is not None:
            # Process frame
            diffs = []
            for spot in spots:
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                diffs.append(calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :]))

            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4] if previous_frame is not None else range(len(spots))
            for spot_indx in arr_:
                spot = spots[spot_indx]
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                spot_status = empty_or_not(spot_crop)
                spots_status[spot_indx] = spot_status

        if frame_nmr % step == 0:
            previous_frame = frame.copy()

        for spot_indx, spot in enumerate(spots):
            spot_status = spots_status[spot_indx]
            x1, y1, w, h = spots[spot_indx]
            color = (0, 255, 0) if spot_status == EMPTY else (0, 0, 255)
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

        num_empty = sum(1 for status in spots_status if status == EMPTY)
        num_not_empty = len(spots) - num_empty
        total_slots = len(spots)
        status_text = f'Empty: {num_empty}, Not Empty: {num_not_empty}, Total Slots: {total_slots}'
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write the processed frame to the output video
        out.write(frame)

        frame_nmr += 1

    cap.release()
    out.release()

if __name__ == '__main__':
    app.run(debug=True)
