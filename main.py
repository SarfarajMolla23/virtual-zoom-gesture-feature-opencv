import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
cap.set(3, 1280)  # Set width
cap.set(4, 720)  # Set height

# Initialize hand detector
detector = HandDetector(detectionCon=0.7)
startDist = None
scale = 0
cx, cy = 500, 500

while True:
    success, img = cap.read()

    # Check if frame is captured
    if not success or img is None:
        print("Failed to capture frame. Please check your camera.")
        break

    # Detect hands
    hands, img = detector.findHands(img)

    # Load and verify image
    img1 = cv2.imread("test.jpg")
    if img1 is None:
        print(
            "Failed to load 'test.jpg'. Ensure the file exists in the working directory.")
        break

    if len(hands) == 2:
        # Check for specific finger gestures
        if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and \
                detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
            lmList1 = hands[0]["lmList"]
            lmList2 = hands[1]["lmList"]

            # Initialize start distance
            if startDist is None:
                length, info, img = detector.findDistance(
                    hands[0]["center"], hands[1]["center"], img)
                startDist = length

            # Calculate zoom scale
            length, info, img = detector.findDistance(
                hands[0]["center"], hands[1]["center"], img)
            scale = int((length - startDist) // 2)
            cx, cy = info[4:]
            print("Zoom Scale:", scale)
    else:
        startDist = None

    # Resize and overlay image
    try:
        h1, w1, _ = img1.shape
        newH, newW = max(1, ((h1 + scale) // 2) * 2), max(1, (
                    (w1 + scale) // 2) * 2)
        img1 = cv2.resize(img1, (newW, newH))

        # Ensure the dimensions fit within the main image
        y1, y2 = max(0, cy - newH // 2), min(img.shape[0], cy + newH // 2)
        x1, x2 = max(0, cx - newW // 2), min(img.shape[1], cx + newW // 2)

        # Avoid overlaying outside bounds
        img[y1:y2, x1:x2] = img1[:y2 - y1, :x2 - x1]
    except Exception as e:
        print("Error during image overlay:", e)

    # Display the result
    cv2.imshow("Image", img)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
