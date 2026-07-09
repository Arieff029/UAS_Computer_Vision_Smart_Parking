import cv2
from ultralytics import YOLO

# 1. Ambil model YOLO paling ringan
model = YOLO('yolov8n.pt')

# 2. Koordinat kotak parkir simulasi
parking_slots = [
    {"id": 1, "space": [150, 400, 260, 520]},  
    {"id": 2, "space": [290, 400, 400, 520]}, 
    {"id": 3, "space": [430, 400, 540, 520]}  
]

# 3. Baca file video
video_path = 'parking_lot.mp4' 
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video jika habis
        continue

    # Deteksi objek mobil (class 2) dan truk (class 7)
    results = model(frame, classes=[2, 7], verbose=False)
    detected_cars = results[0].boxes.xyxy.cpu().numpy()

    filled_slots = []
    for car in detected_cars:
        cx1, cy1, cx2, cy2 = car[:4]
        center_x = int((cx1 + cx2) / 2)
        center_y = int((cy1 + cy2) / 2)

        for slot in parking_slots:
            sx1, sy1, sx2, sy2 = slot["space"]
            if sx1 < center_x < sx2 and sy1 < center_y < sy2:
                filled_slots.append(slot["id"])

    # 4. Gambar kotak status warna-warni
    available_count = 0
    for slot in parking_slots:
        sx1, sy1, sx2, sy2 = slot["space"]
        if slot["id"] in filled_slots:
            color = (0, 0, 255) # Merah (Terisi)
            status_text = "TERISI"
        else:
            color = (0, 255, 0) # Hijau (Kosong)
            status_text = "KOSONG"
            available_count += 1
        
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 3)
        cv2.putText(frame, f"Slot {slot['id']}: {status_text}", (sx1, sy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"Slot Kosong: {available_count}/{len(parking_slots)}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow("UAS Computer Vision - Smart Parking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()