import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))
i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    i+=1
    if i == 1:
        print("Recording...")
    out.write(frame)
    if i == 450:
        break
print("Recording completed.")
cap.release()
out.release()
cv2.destroyAllWindows()