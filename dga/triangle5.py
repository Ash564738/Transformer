import matplotlib.pyplot as plt
import numpy as np

def ternary_to_cartesian(ch4, c2h4, c2h6):
    """
    Chuyển đổi tọa độ tam giác đều chuẩn xác theo nhãn của hình vẽ:
    - Đỉnh trên (Top): 100% CH4 -> (50, 86.6025)
    - Đáy trái (Bottom-Left): 100% C2H6 -> (0, 0)
    - Đáy phải (Bottom-Right): 100% C2H4 -> (100, 0)
    """
    ch4, c2h4, c2h6 = np.array(ch4), np.array(c2h4), np.array(c2h6)
    total = ch4 + c2h4 + c2h6
    
    # Chuẩn hóa về thang 100%
    ch4_pct = (ch4 / total) * 100
    c2h4_pct = (c2h4 / total) * 100
    
    x = c2h4_pct + 0.5 * ch4_pct
    y = ch4_pct * (np.sqrt(3) / 2)
    return x, y

# Khởi tạo khung vẽ vuông tỉ lệ 1:1 bảo toàn tính đối xứng của tam giác đều
fig, ax = plt.subplots(figsize=(10, 9))
ax.set_aspect('equal')

# 1. Vẽ các đường lưới phụ (Grid lines) mảnh chạy từ 10% đến 90%
for i in range(10, 100, 10):
    # Đường song song với cạnh phải (C2H6 hằng số)
    x1, y1 = ternary_to_cartesian(0, 100 - i, i)
    x2, y2 = ternary_to_cartesian(100 - i, 0, i)
    ax.plot([x1, x2], [y1, y2], color='#e0e0e0', linestyle=':', linewidth=0.7, zorder=1)
    
    # Đường song song với cạnh đáy (CH4 hằng số)
    x1, y1 = ternary_to_cartesian(i, 0, 100 - i)
    x2, y2 = ternary_to_cartesian(i, 100 - i, 0)
    ax.plot([x1, x2], [y1, y2], color='#e0e0e0', linestyle=':', linewidth=0.7, zorder=1)
    
    # Đường song song với cạnh trái (C2H4 hằng số)
    x1, y1 = ternary_to_cartesian(0, i, 100 - i)
    x2, y2 = ternary_to_cartesian(100 - i, i, 0)
    ax.plot([x1, x2], [y1, y2], color='#e0e0e0', linestyle=':', linewidth=0.7, zorder=1)

# 2. Định nghĩa đa giác các phân vùng theo tọa độ (CH4, C2H4, C2H6) từ các giao điểm ranh giới
# Phối màu chuẩn theo hình ảnh mẫu đầu tiên bạn gửi
zones = {
    'PD': {
        'points': [(100, 0, 0), (99, 1, 0), (84, 1, 15), (85, 0, 15)],
        'color': '#ffebad', 'text_pos': (92, 0.5, 7.5)
    },
    'O': {
        'points': [(85, 0, 15), (84, 1, 15), (46, 1, 53), (47, 0, 53)],
        'color': '#fce3a1', 'text_pos': (65, 0.5, 34.5)
    },
    'ND': {
        'points': [(47, 0, 53), (46, 1, 53), (37, 10, 53), (0, 10, 90), (0, 0, 100)],
        'color': '#f4f6f9', 'text_pos': (20, 4, 76)
    },
    'S': {
        'points': [(84, 1, 15), (75, 10, 15), (37, 10, 53), (46, 1, 53)],
        'color': '#b2ebb5', 'text_pos': (60, 5.5, 34.5)
    },
    'T2': {
        'points': [(99, 1, 0), (84, 1, 15), (49, 36, 15), (64, 36, 0)],
        'color': '#faedd9', 'text_pos': (74, 18, 8)
    },
    'C': {
        'points': [(75, 10, 15), (49, 36, 15), (0, 36, 64), (0, 10, 90)],
        'color': '#adb5bd', 'text_pos': (31, 23, 46)
    },
    'T3': {
        'points': [(64, 36, 0), (49, 36, 15), (0, 36, 64), (0, 100, 0)],
        'color': '#fce8eb', 'text_pos': (25, 55, 20)
    }
}

# Vẽ đổ màu đa giác và gán nhãn chữ vào vùng trung tâm tương ứng
for zone_name, info in zones.items():
    # Chuyển đổi toàn bộ mảng điểm góc sang tọa độ Cartesian phẳng (X, Y)
    cart_pts = [ternary_to_cartesian(p[0], p[1], p[2]) for p in info['points']]
    
    # Tạo đối tượng vẽ đa giác ranh giới đen mảnh
    polygon = plt.Polygon(cart_pts, closed=True, facecolor=info['color'], 
                          edgecolor='#343a40', linewidth=1.2, zorder=2)
    ax.add_patch(polygon)
    
    # Lấy tọa độ nhãn tối ưu theo cấu trúc hình học thực tế
    tx, ty = ternary_to_cartesian(*info['text_pos'])
    ax.text(tx, ty, zone_name, color='black', weight='bold', fontsize=11, 
            ha='center', va='center', zorder=3)

# 3. Ghi nhãn số phần trăm và nhãn Trục chuẩn theo đồ thị gốc
# Cạnh trái: CH4 (%) chạy từ đáy trái lên đỉnh
ax.text(17, 50, 'CH$_4$ (%)', fontsize=12, weight='semibold', rotation=60, ha='center')
# Cạnh phải: C2H4 (%) chạy từ đỉnh xuống đáy phải
ax.text(83, 50, 'C$_2$H$_4$ (%)', fontsize=12, weight='semibold', rotation=-60, ha='center')
# Cạnh đáy: C2H6 (%) chạy từ đáy phải sang đáy trái
ax.text(50, -6, 'C$_2$H$_6$ (%)', fontsize=12, weight='semibold', ha='center')

# Ghi nhãn 3 đỉnh chính 100% cố định
ax.text(50, 89, '100% CH$_4$', fontsize=11, weight='bold', ha='center')
ax.text(-2, -3, '100% C$_2$H$_6$', fontsize=11, weight='bold', ha='right')
ax.text(102, -3, '100% C$_2$H$_4$', fontsize=11, weight='bold', ha='left')

# Vẽ viền đen dày bọc ngoài cùng của tam giác tổng thể
triangle_border = [ternary_to_cartesian(100,0,0), ternary_to_cartesian(0,0,100), ternary_to_cartesian(0,100,0)]
outer_poly = plt.Polygon(triangle_border, closed=True, facecolor='none', edgecolor='black', linewidth=2, zorder=4)
ax.add_patch(outer_poly)

# Cấu hình giới hạn trục hiển thị xung quanh đồ thị sạch sẽ
ax.set_xlim(-10, 110)
ax.set_ylim(-10, 95)
ax.axis('off')

# Hiển thị đồ thị
plt.show()
