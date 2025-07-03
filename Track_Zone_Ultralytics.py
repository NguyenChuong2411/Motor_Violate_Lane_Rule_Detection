import cv2

from ultralytics import solutions

cap = cv2.VideoCapture(".\IMG_0403.MOV")
assert cap.isOpened(), "Error reading video file"

# Define region points
#IMG 0418 
#region_points = [(1480, 0), (392, 2159), (3839, 2159), (3836, 1674), (2620, 0), (1480, 0)]
# IMG 0410
# region_points = [
#     (710, 2159),
#     (796, 1917),
#     (900, 1582),
#     (1000, 1282),
#     (1080, 914),
#     (1092, 630),
#     (1072, 382),
#     (972, 103),
#     (904, 3),
#     (2548, 0),
#     (2868, 167),
#     (3288, 438),
#     (3720, 782),
#     (3839, 886),
#     (3839, 2159),
#     (710, 2159)  # Lặp lại điểm đầu tiên để khép kín vùng
# ]
# IMG 0403
region_points =[
    (435, 2146),
    (1313, 963),
    (2394, 952),
    (3026, 2159),
    (435, 2146)
]
# IMG 0406
# region_points = [
#     (1350, 1252),
#     (1864, 686),
#     (2816, 658),
#     (3193, 1179),
#     (1350, 1252)  # Lặp lại điểm đầu tiên để khép kín vùng
# ]

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("trackzone_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init trackzone (object tracking in zones, not complete frame)
trackzone = solutions.TrackZone(
    show=True,  # display the output
    region=region_points,  # pass region points
    model="WL.pt",  # use any model that Ultralytics support, i.e. YOLOv9, YOLOv10
    # line_width=2,  # adjust the line width for bounding boxes and text display
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break
    im0 = cv2.rotate(im0, cv2.ROTATE_180)
    results = trackzone(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the video file

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows