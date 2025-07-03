import cv2
import json
import numpy as np
from confluent_kafka import Consumer, KafkaError
from ultralytics import solutions

def consume_and_process(bootstrap_servers='localhost:9092', topic='video-stream', group_id='video-consumer-group'):
    """Consume video frames from Kafka and process with TrackZone."""
    # Kafka consumer configuration with increased message size
    conf = {
        'bootstrap.servers': bootstrap_servers,
        'group.id': group_id,
        'auto.offset.reset': 'earliest',
        'fetch.message.max.bytes': 10000000,
        'max.partition.fetch.bytes': 10000000
    }
    consumer = Consumer(conf)
    consumer.subscribe([topic])

    # Initialize video writer
    video_writer = None
    fps = 30  # Default FPS, adjust if known

    # Define region points
    region_points = [
        (435, 2146),
        (1313, 963),
        (2394, 952),
        (3026, 2159),
        (435, 2146)
    ]

    # Initialize TrackZone
    trackzone = solutions.TrackZone(
        show=True,
        region=region_points,
        model="WL.pt",
    )

    print("Starting consumer...")
    message_count = 0
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Consumer error: {msg.error()}")
                    break

            # Parse message
            message_count += 1
            print(f"--- Đã nhận được message #{message_count}. Đang xử lý... ---")
            try:
                message = json.loads(msg.value().decode('utf-8'))
            except json.JSONDecodeError as e:
                print(f"Lỗi: Không thể parse JSON của message #{message_count}: {e}. Bỏ qua message này.")
                continue

            if message['frame_id'] == -1:
                print("Received end-of-stream marker.")
                break

            # Decode frame
            try:
                frame_bytes = bytes.fromhex(message['data'])
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("cv2.imdecode trả về None")
            except Exception as e:
                print(f"Lỗi: Không thể giải mã frame cho message #{message_count}: {e}. Bỏ qua message này.")
                continue

            # Validate frame dimensions
            try:
                w, h = int(message['width']), int(message['height'])
                if w <= 0 or h <= 0:
                    raise ValueError(f"Invalid dimensions: width={w}, height={h}")
            except (KeyError, ValueError) as e:
                print(f"Lỗi: Metadata không hợp lệ cho message #{message_count}: {e}. Bỏ qua message này.")
                continue

            # Initialize video writer with first frame's dimensions
            if video_writer is None:
                video_writer = cv2.VideoWriter("trackzone_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            # Scale region points to match resized frame
            try:
                scale_x, scale_y = w / 3840, h / 2160
                scaled_points = [(int(x * scale_x), int(y * scale_y)) for x, y in region_points[:-1]] + [(int(region_points[-1][0] * scale_x), int(region_points[-1][1] * scale_y))]
                # Convert to NumPy array for cv2.polylines
                scaled_points_np = np.array(scaled_points, dtype=np.int32).reshape(-1, 1, 2)
                print(f"Scaled points for message #{message_count}: {scaled_points_np}")
            except Exception as e:
                print(f"Lỗi: Không thể scale region points cho message #{message_count}: {e}. Bỏ qua message này.")
                continue

            # Update TrackZone region
            trackzone.region = scaled_points_np

            # Process frame
            try:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                results = trackzone(frame)
                video_writer.write(results.plot_im)
            except Exception as e:
                print(f"Lỗi: Không thể xử lý frame cho message #{message_count}: {e}. Bỏ qua message này.")
                continue

    finally:
        consumer.close()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Consumer closed and video processing complete.")

if __name__ == "__main__":
    consume_and_process()