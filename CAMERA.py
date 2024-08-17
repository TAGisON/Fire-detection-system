import cv2
import imutils

# RTSP stream URL
rtsp_url = 'rtsp://admin:admin123@192.168.100.71:554/avstream/channel=1/stream=0.sdp'

# OpenCV VideoCapture for RTSP stream
cam = cv2.VideoCapture(rtsp_url)

def camera():
    # Read a frame from the RTSP stream
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        return None
    return img

def MobileCamera():
    # This function is now not needed, as RTSP is handled by VideoCapture
    pass

def main():
    while True:
        img = camera()
        if img is None:
            break
        
        # Resize image for display
        img = imutils.resize(img, width=640, height=480)
        
        # Display the image
        cv2.imshow('RTSP Stream', img)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close all OpenCV windows
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
