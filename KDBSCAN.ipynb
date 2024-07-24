import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from qiskit import Aer, QuantumCircuit, transpile, assemble, execute

# Membaca data
data = pd.read_excel("data_supplier_100.xlsx")

# Normalisasi data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Harga', 'Kualitas', 'Waktu Pengiriman']])

# Fungsi untuk menerapkan gerbang kuantum (gunakan amplitudo encoding)
def terapkan_gerbang(qc, value, qubits):
    angle = 2 * np.arccos(np.sqrt(value))
    qc.ry(angle, qubits[0])

# Fungsi untuk menghitung jarak kuantum
def quantum_distance_circuit(point_a, point_b):
    n_qubits = len(point_a)  # Setiap fitur menggunakan 1 qubit
    qc = QuantumCircuit(n_qubits * 2)

    for i, (a, b) in enumerate(zip(point_a, point_b)):
        terapkan_gerbang(qc, a, [i])
        terapkan_gerbang(qc, b, [n_qubits + i])

    return qc

# Fungsi untuk menghitung kepadatan kuantum
def quantum_density_circuit(points):
    n_points = len(points)
    n_qubits = len(points[0])  # Setiap fitur menggunakan 1 qubit
    qc = QuantumCircuit(n_qubits * 2)

    for i in range(n_points):
        for j in range(i + 1, n_points):
            distance_circuit = quantum_distance_circuit(points[i], points[j])
            qc.compose(distance_circuit, range(n_qubits * 2), inplace=True)

    qc.measure_all()
    return qc

# Fungsi untuk menjalankan algoritma Quantum DBSCAN tanpa batch
def quantum_dbscan(data, epsilon, min_samples):
    n_points = len(data)
    clusters = [-1] * n_points
    cluster_id = 0

    simulator = Aer.get_backend('aer_simulator')

    qc = quantum_density_circuit(data)
    
    compiled_circuit = transpile(qc, simulator)
    qobj = assemble(compiled_circuit)
    result = execute(qc, backend=simulator, shots=1024).result()
    counts = result.get_counts(qc)

    # Ambil hasil pengukuran dan tentukan tetangga
    for i, point in enumerate(data):
        neighbors = [idx for idx, count in enumerate(counts.values()) if count > 0]

        if len(neighbors) < min_samples:
            clusters[i] = -1  # Noise
        else:
            if clusters[i] == -1:
                clusters[i] = cluster_id
                queue = [i]
                while queue:
                    current = queue.pop(0)
                    qc = quantum_density_circuit([data[current]])
                    
                    compiled_circuit = transpile(qc, simulator)
                    qobj = assemble(compiled_circuit)
                    result = execute(qc, backend=simulator, shots=1024).result()
                    counts = result.get_counts(qc)

                    neighbors = [idx for idx, count in enumerate(counts.values()) if count > 0]

                    for neighbor in neighbors:
                        if clusters[neighbor] == -1:
                            clusters[neighbor] = cluster_id
                        if clusters[neighbor] == 0:
                            queue.append(neighbor)
                cluster_id += 1

    return clusters, qc

# Jalankan Quantum DBSCAN
epsilon = 1
min_samples = 5
clusters, final_circuit = quantum_dbscan(scaled_data, epsilon, min_samples)

# Menambahkan hasil clustering ke data asli
data["clusters"] = clusters

data
