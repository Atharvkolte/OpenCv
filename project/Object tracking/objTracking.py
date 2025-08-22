import cv2
import sys

def select_tracker(tracker_type):
    if tracker_type == "BOOSTING":
        return cv2.legacy.TrackerBoosting_create()
    elif tracker_type == "KCF":
        return cv2.TrackerKCF_create()
    elif tracker_type == "MIL":
        return cv2.TrackerMIL_create()
    elif tracker_type == "TLD":
        return cv2.legacy.TrackerTLD_create()
    elif tracker_type == "MEDIANFLOW":
        return cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == "MOSSE":
        return cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        return cv2.TrackerCSRT_create()
    else:
        print("Invalid tracker type")
        sys.exit(1)

def main():
    tracker_types = ["BOOSTING", "KCF", "MIL", "TLD", "MEDIANFLOW", "MOSSE", "CSRT"]
    print("select tracker type:")
    for i,t_type in enumerate(tracker_types):
        print(f"{i+1}: {t_type}")
    tracker_choice = int(input("Enter the number corresponding to your choice: "))-1
    tracker_type=tracker_types[tracker_choice]
    
    tracker=select_tracker(tracker_type)
    video_path = input("Enter the path to the video file: ")
    if not video_path:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(video_path)
        
    if not video.isOpened():
        print("Error opening video file")
    
    ret, frame=video.read()
    if not ret:
        print("Error reading video frame")
        sys.exit(1)
    bbox = cv2.selectROI(frame, False)
    
    ret = tracker.init(frame, bbox)
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("End of video")
            break
        ret,bbox = tracker.update(frame)

        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            print("Tracking failure detected")
        cv2.putText(frame, tracker_type+" Tracker", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()