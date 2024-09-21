import numpy as np
import torch

# 첫 번째 파일 읽기
data_array1 = np.loadtxt('/data1/bubble3jh/ppg/data/six_ch/D500.dat', skiprows=1)
data_array1_modified = data_array1[:, 1:]

reshaped_data = np.expand_dims(data_array1_modified, axis=0)
reshaped_data = np.transpose(reshaped_data, (0, 2, 1))

# Numpy 배열을 PyTorch Tensor로 변환
tensor_data = torch.tensor(reshaped_data, dtype=torch.float32)
for i in range(tensor_data.shape[1]):
    # i번째 차원을 슬라이싱하여 새 텐서 생성
    sliced_tensor = tensor_data[:, i:i+1, :]
    
    # 새 텐서의 shape는 (1, 1, 33660)이 됩니다.
    print(f"Sliced tensor {i+1} shape: {sliced_tensor.shape}")
    
    # 새 텐서를 별도의 .pth 파일로 저장
    torch.save(sliced_tensor, f'/data1/bubble3jh/ppg/data/six_ch/d500_sliced_{i+1}.pth')
# Tensor를 .pth 파일로 저장
torch.save(tensor_data, '/data1/bubble3jh/ppg/data/six_ch/d500.pth')

# 저장된 Tensor의 shape 확인
print(tensor_data.shape)