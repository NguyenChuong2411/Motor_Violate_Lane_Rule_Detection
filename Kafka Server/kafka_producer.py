import cv2
from confluent_kafka import Producer
import json
import time

def delivery_report(err, msg):
    """Callback to report the delivery status of a message."""
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def produce_video(video_path, bootstrap_servers='localhost:9092', topic='video-stream', resize_factor=0.5, quality=50):
    """Read video, optionally resize, and send frames to Kafka topic."""
    # Kafka producer configuration with increased message size
    conf = {
        'bootstrap.servers': bootstrap_servers,
        'client.id': 'video-producer',
        'message.max.bytes': 10000000  # 10MB, adjust as needed
    }
    producer = Producer(conf)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error reading video file")
        return

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video processing complete or frame is empty.")
            break

        # Optionally resize frame to reduce size
        if resize_factor < 1.0:
            frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

        # Encode frame as JPEG with specified quality
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        frame_bytes = buffer.tobytes()

        # Create message with metadata
        message = {
            'frame_id': frame_count,
            'timestamp': time.time(),
            'data': frame_bytes.hex(),  # Convert bytes to hex string
            'width': frame.shape[1],
            'height': frame.shape[0]
        }

        # Send to Kafka
        try:
            producer.produce(
                topic=topic,
                value=json.dumps(message).encode('utf-8'),
                callback=delivery_report
            )
            producer.poll(0)  # Trigger delivery callbacks
        except Exception as e:
            print(f"Error producing message: {e}")
            continue

        frame_count += 1
        time.sleep(0.01)  # Small delay to avoid overwhelming the broker

    # Send end-of-stream marker
    producer.produce(
        topic=topic,
        value=json.dumps({'frame_id': -1, 'timestamp': time.time(), 'data': None}).encode('utf-8'),
        callback=delivery_report
    )
    producer.flush()
    cap.release()
    print("Video transmission complete.")

if __name__ == "__main__":
    video_path = ".\IMG_0403.MOV"
    produce_video(video_path, resize_factor=0.5, quality=50)  # Adjust resize_factor and quality as needed