import cv2
import json
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, from_json
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType, BinaryType
from ultralytics import solutions

# --- Phần 1: Khởi tạo các biến và mô hình toàn cục ---

# Khai báo mô hình và các biến khác ở cấp độ toàn cục để chúng
# được khởi tạo một lần trên mỗi worker của Spark.
def init_model():
    global trackzone
    # Các điểm của vùng theo tọa độ gốc (3840x2160)
    region_points = [
        (435, 2146), (1313, 963), (2394, 952),
        (3026, 2159), (435, 2146)
    ]
    trackzone = solutions.TrackZone(
        show=False, # Đặt show=False khi chạy trên Spark
        region=region_points,
        model="WL.pt",
    )

# --- Phần 2: Định nghĩa Pandas UDF để xử lý khung hình ---

@udf(returnType=BinaryType())
def process_frame_udf(frame_data: pd.Series, width: pd.Series, height: pd.Series) -> pd.Series:
    """
    Pandas UDF để xử lý một loạt khung hình video với TrackZone.
    """
    init_model() # Khởi tạo mô hình trên worker
    results = []
    
    # Kích thước gốc của video để tính toán tỷ lệ scale
    original_w, original_h = 3840, 2160

    for i, data in enumerate(frame_data):
        try:
            # Giải mã khung hình
            frame_array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                results.append(None)
                continue

            # Lấy kích thước của khung hình đã được resize
            h, w, _ = frame.shape

            # Scale các điểm của vùng theo kích thước mới
            scale_x, scale_y = w / original_w, h / original_h
            scaled_points = [(int(p[0] * scale_x), int(p[1] * scale_y)) for p in trackzone.region]
            trackzone.region = np.array(scaled_points, dtype=np.int32).reshape(-1, 1, 2)
            
            # Xoay khung hình và xử lý
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            processed_frame = trackzone(frame)
            
            # Mã hóa lại khung hình đã xử lý để trả về
            _, buffer = cv2.imencode('.jpg', processed_frame)
            results.append(buffer.tobytes())
            
        except Exception as e:
            print(f"Lỗi khi xử lý khung hình: {e}")
            results.append(None)
            
    return pd.Series(results)

# --- Phần 3: Thiết lập Spark Session và Streaming DataFrame ---

def main():
    spark = SparkSession.builder \
        .appName("VideoProcessingSpark") \
        .master("local[*]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    # Đọc luồng dữ liệu từ Kafka
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "video-stream") \
        .option("startingOffsets", "earliest") \
        .option("kafka.fetch.message.max.bytes", "10000000") \
        .load()

    # Định nghĩa schema để parse JSON từ Kafka
    schema = StructType([
        StructField("frame_id", LongType(), True),
        StructField("timestamp", DoubleType(), True),
        StructField("data", StringType(), True),
        StructField("width", LongType(), True),
        StructField("height", LongType(), True)
    ])

    # Parse dữ liệu JSON và chuyển đổi chuỗi hex thành binary
    json_df = kafka_df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*") \
        .withColumn("frame_bytes", udf(lambda x: bytes.fromhex(x), BinaryType())(col("data")))

    # Lọc ra các thông điệp end-of-stream
    processed_df = json_df.filter(col("frame_id") != -1)

    # Áp dụng Pandas UDF để xử lý khung hình
    result_df = processed_df.withColumn(
        "processed_frame",
        process_frame_udf(col("frame_bytes"), col("width"), col("height"))
    )

    # --- Phần 4: Xử lý và hiển thị kết quả ---

    # Sử dụng foreachBatch để xử lý từng micro-batch
    query = result_df.writeStream \
        .foreachBatch(lambda batch_df, batch_id: {
            batch_df.persist(),
            [
                cv2.imshow('Processed Video', cv2.imdecode(np.frombuffer(row.processed_frame, np.uint8), cv2.IMREAD_COLOR))
                for row in batch_df.collect() if row.processed_frame is not None
            ],
            cv2.waitKey(1),
            batch_df.unpersist()
        }) \
        .start()
        
    query.awaitTermination()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
