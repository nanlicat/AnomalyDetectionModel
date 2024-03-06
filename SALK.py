import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# 假设你有一个名为time_series的numpy数组，包含你的时间序列数据

# 数据预处理
scaler = MinMaxScaler()
time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1))

# 划分数据集（这里简单起见，只使用最后一部分作为测试集）
train_size = int(len(time_series_scaled) * 0.8)
train, test = time_series_scaled[:train_size], time_series_scaled[train_size:]

# 构建LSTM模型
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(train.shape[1], 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 将数据重塑为[samples, time steps, features]格式
train_reshaped = np.reshape(train, (len(train) // 50, 50, 1))

# 训练模型
model.fit(train_reshaped, train[50:], epochs=100, verbose=0)

# 预测与计算重构误差
test_reshaped = np.reshape(test[:-1], (len(test) - 1) // 50, 50, 1)
predictions = model.predict(test_reshaped)
errors = test[50:] - predictions.flatten()

# 误差聚类
kmeans = KMeans(n_clusters=2, random_state=42).fit(errors.reshape(-1, 1))
labels = kmeans.labels_

# 假设较小的簇（或平均误差较大的簇）为异常点
anomalies = np.where(labels == 1)[0]  # 这里假设标签1代表异常簇

# 打印异常点的索引
print("Anomalies detected at indices:", anomalies)

# 评估（这里只是打印出异常点，实际评估可能需要其他指标）
print("Reconstruction errors:", errors[anomalies])

# 实际应用时，可能需要对测试集的每个点单独进行预测，而不是像上面那样分组进行
# 此外，K-means的簇数量n_clusters可能需要根据实际情况进行调整