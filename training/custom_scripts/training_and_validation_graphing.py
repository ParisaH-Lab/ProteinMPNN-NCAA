#!/usr/bin/env python

import matplotlib.pyplot as plt

# Sample data from the provided text
data = """epoch: 1, step: 2, time: 24.3, train: 14204.103516, valid: 8516.636719, train_acc: 1.000000, valid_acc: 1.000000
epoch: 2, step: 4, time: 23.5, train: 14208.680664, valid: 8516.868164, train_acc: 1.000000, valid_acc: 1.000000
epoch: 3, step: 7, time: 48.1, train: 15670.544922, valid: 8503.640625, train_acc: 1.000000, valid_acc: 1.000000
epoch: 4, step: 10, time: 42.9, train: 15656.347656, valid: 8489.713867, train_acc: 1.000000, valid_acc: 1.000000
epoch: 5, step: 13, time: 36.4, train: 11814.107422, valid: 8467.605469, train_acc: 1.000000, valid_acc: 1.000000
epoch: 6, step: 16, time: 32.8, train: 11798.794922, valid: 8450.231445, train_acc: 1.000000, valid_acc: 1.000000
epoch: 7, step: 19, time: 38.0, train: 12475.080078, valid: 8433.985352, train_acc: 1.000000, valid_acc: 1.000000
epoch: 8, step: 22, time: 35.3, train: 12422.578125, valid: 8407.260742, train_acc: 1.000000, valid_acc: 1.000000
epoch: 9, step: 25, time: 42.9, train: 13950.422852, valid: 8394.762695, train_acc: 1.000000, valid_acc: 1.000000
epoch: 10, step: 28, time: 39.1, train: 13898.954102, valid: 8365.418945, train_acc: 1.000000, valid_acc: 1.000000
epoch: 11, step: 31, time: 38.6, train: 11522.575195, valid: 10665.867188, train_acc: 1.000000, valid_acc: 1.000000
epoch: 12, step: 34, time: 35.1, train: 11480.632812, valid: 10617.922852, train_acc: 1.000000, valid_acc: 1.000000
epoch: 13, step: 37, time: 37.7, train: 12850.627930, valid: 8253.327148, train_acc: 1.000000, valid_acc: 1.000000
epoch: 14, step: 40, time: 35.4, train: 12777.086914, valid: 8211.171875, train_acc: 1.000000, valid_acc: 1.000000
epoch: 15, step: 43, time: 45.3, train: 14059.559570, valid: 8158.466309, train_acc: 1.000000, valid_acc: 1.000000
epoch: 16, step: 46, time: 41.6, train: 14012.525391, valid: 8104.697266, train_acc: 1.000000, valid_acc: 1.000000
epoch: 17, step: 48, time: 38.4, train: 15680.113281, valid: 10273.961914, train_acc: 1.000000, valid_acc: 1.000000
epoch: 18, step: 50, time: 35.0, train: 15602.474609, valid: 10218.579102, train_acc: 1.000000, valid_acc: 1.000000
epoch: 19, step: 53, time: 37.7, train: 10536.393555, valid: 7953.452148, train_acc: 1.000000, valid_acc: 1.000000
epoch: 20, step: 56, time: 33.3, train: 10446.086914, valid: 7894.704590, train_acc: 1.000000, valid_acc: 1.000000
epoch: 21, step: 60, time: 56.8, train: 12783.100586, valid: 7809.741699, train_acc: 1.000000, valid_acc: 1.000000
epoch: 22, step: 64, time: 52.8, train: 12635.652344, valid: 7718.582520, train_acc: 1.000000, valid_acc: 1.000000
epoch: 23, step: 67, time: 49.6, train: 14366.685547, valid: 11869.157227, train_acc: 1.000000, valid_acc: 1.000000
epoch: 24, step: 70, time: 47.4, train: 14215.148438, valid: 11738.751953, train_acc: 1.000000, valid_acc: 1.000000
epoch: 25, step: 74, time: 47.7, train: 11185.025391, valid: 7454.859375, train_acc: 1.000000, valid_acc: 1.000000
epoch: 26, step: 78, time: 44.1, train: 11024.855469, valid: 7341.851562, train_acc: 1.000000, valid_acc: 1.000000
epoch: 27, step: 81, time: 44.8, train: 12370.322266, valid: 9205.397461, train_acc: 1.000000, valid_acc: 1.000000
epoch: 28, step: 84, time: 41.6, train: 12223.275391, valid: 9090.231445, train_acc: 1.000000, valid_acc: 1.000000
epoch: 29, step: 87, time: 41.3, train: 11410.679688, valid: 7076.575195, train_acc: 1.000000, valid_acc: 1.000000
epoch: 30, step: 90, time: 37.8, train: 11249.563477, valid: 6986.332520, train_acc: 1.000000, valid_acc: 1.000000
epoch: 31, step: 93, time: 39.5, train: 11118.946289, valid: 6870.162598, train_acc: 1.000000, valid_acc: 1.000000
epoch: 32, step: 96, time: 35.9, train: 10987.796875, valid: 6774.292480, train_acc: 1.000000, valid_acc: 1.000000
epoch: 33, step: 99, time: 44.6, train: 11972.585938, valid: 6670.025879, train_acc: 1.000000, valid_acc: 1.000000
epoch: 34, step: 102, time: 39.3, train: 11794.141602, valid: 6563.526855, train_acc: 1.000000, valid_acc: 1.000000
epoch: 35, step: 105, time: 48.8, train: 12255.780273, valid: 6452.087402, train_acc: 1.000000, valid_acc: 1.000000
epoch: 36, step: 108, time: 43.8, train: 12012.344727, valid: 6338.794434, train_acc: 1.000000, valid_acc: 1.000000
epoch: 37, step: 111, time: 46.5, train: 11990.664062, valid: 6226.162109, train_acc: 1.000000, valid_acc: 1.000000
epoch: 38, step: 114, time: 42.8, train: 11770.660156, valid: 6104.711914, train_acc: 1.000000, valid_acc: 1.000000
epoch: 39, step: 117, time: 48.0, train: 10818.378906, valid: 5966.711426, train_acc: 1.000000, valid_acc: 1.000000
epoch: 40, step: 120, time: 44.0, train: 10591.827148, valid: 5847.650391, train_acc: 1.000000, valid_acc: 1.000000
epoch: 41, step: 122, time: 27.9, train: 10141.666016, valid: 5774.136230, train_acc: 1.000000, valid_acc: 1.000000
epoch: 42, step: 124, time: 28.0, train: 10000.441406, valid: 5693.164062, train_acc: 1.000000, valid_acc: 1.000000
epoch: 43, step: 127, time: 50.0, train: 10248.897461, valid: 7072.653809, train_acc: 1.000000, valid_acc: 1.000000
epoch: 44, step: 130, time: 44.5, train: 10012.682617, valid: 6916.335449, train_acc: 1.000000, valid_acc: 1.000000
epoch: 45, step: 133, time: 42.8, train: 8906.547852, valid: 5336.881348, train_acc: 1.000000, valid_acc: 1.000000
epoch: 46, step: 136, time: 37.4, train: 8706.811523, valid: 5211.812988, train_acc: 1.000000, valid_acc: 1.000000
epoch: 47, step: 139, time: 41.5, train: 9404.129883, valid: 5084.561523, train_acc: 1.000000, valid_acc: 1.000000
epoch: 48, step: 142, time: 39.6, train: 9182.725586, valid: 4964.963867, train_acc: 1.000000, valid_acc: 1.000000
epoch: 49, step: 145, time: 44.5, train: 8156.108398, valid: 6178.544434, train_acc: 1.000000, valid_acc: 1.000000
epoch: 50, step: 148, time: 41.6, train: 7957.084961, valid: 6026.170898, train_acc: 1.000000, valid_acc: 1.000000
epoch: 51, step: 151, time: 42.6, train: 7707.926270, valid: 4603.499512, train_acc: 1.000000, valid_acc: 1.000000
epoch: 52, step: 154, time: 37.3, train: 7512.039062, valid: 4486.005859, train_acc: 1.000000, valid_acc: 1.000000
epoch: 53, step: 157, time: 37.9, train: 6662.288086, valid: 4360.311523, train_acc: 1.000000, valid_acc: 1.000000
epoch: 54, step: 160, time: 34.7, train: 6509.015137, valid: 4243.911621, train_acc: 1.000000, valid_acc: 1.000000
epoch: 55, step: 163, time: 36.1, train: 5648.340820, valid: 5242.385742, train_acc: 1.000000, valid_acc: 1.000000
epoch: 56, step: 166, time: 34.1, train: 5496.889160, valid: 5107.207031, train_acc: 1.000000, valid_acc: 1.000000
epoch: 57, step: 169, time: 37.6, train: 6214.833496, valid: 3930.306885, train_acc: 1.000000, valid_acc: 1.000000
epoch: 58, step: 172, time: 36.3, train: 6053.607422, valid: 3830.092529, train_acc: 1.000000, valid_acc: 1.000000
epoch: 59, step: 175, time: 43.9, train: 6642.098633, valid: 3708.209717, train_acc: 1.000000, valid_acc: 1.000000
epoch: 60, step: 178, time: 40.4, train: 6437.567383, valid: 3595.811035, train_acc: 1.000000, valid_acc: 1.000000
epoch: 61, step: 181, time: 45.6, train: 6521.665527, valid: 4463.934570, train_acc: 1.000000, valid_acc: 1.000000
epoch: 62, step: 184, time: 40.9, train: 6327.047852, valid: 4312.466309, train_acc: 1.000000, valid_acc: 1.000000
epoch: 63, step: 187, time: 44.9, train: 5875.461914, valid: 3263.831787, train_acc: 1.000000, valid_acc: 1.000000
epoch: 64, step: 190, time: 38.6, train: 5681.397949, valid: 3154.148438, train_acc: 1.000000, valid_acc: 1.000000
epoch: 65, step: 194, time: 60.0, train: 4928.478027, valid: 4639.844727, train_acc: 1.000000, valid_acc: 1.000000
epoch: 66, step: 198, time: 53.4, train: 4740.216309, valid: 4423.340820, train_acc: 1.000000, valid_acc: 1.000000
epoch: 67, step: 201, time: 49.5, train: 5819.130859, valid: 3534.481689, train_acc: 1.000000, valid_acc: 1.000000
epoch: 68, step: 204, time: 45.0, train: 5600.983398, valid: 3399.801025, train_acc: 1.000000, valid_acc: 1.000000
epoch: 69, step: 207, time: 43.6, train: 4438.887207, valid: 2558.014160, train_acc: 1.000000, valid_acc: 1.000000
epoch: 70, step: 210, time: 39.2, train: 4266.757324, valid: 2457.825439, train_acc: 1.000000, valid_acc: 1.000000
epoch: 71, step: 213, time: 45.2, train: 4210.794922, valid: 3010.873291, train_acc: 1.000000, valid_acc: 1.000000
epoch: 72, step: 216, time: 41.8, train: 4039.482178, valid: 2893.143555, train_acc: 1.000000, valid_acc: 1.000000
epoch: 73, step: 219, time: 42.8, train: 3873.167725, valid: 2175.682617, train_acc: 1.000000, valid_acc: 1.000000
epoch: 74, step: 222, time: 38.0, train: 3716.588623, valid: 2088.802979, train_acc: 1.000000, valid_acc: 1.000000
epoch: 75, step: 225, time: 43.7, train: 3396.452637, valid: 2012.358154, train_acc: 1.000000, valid_acc: 1.000000
epoch: 76, step: 228, time: 39.8, train: 3254.582520, valid: 1932.206543, train_acc: 1.000000, valid_acc: 1.000000
epoch: 77, step: 231, time: 44.7, train: 3179.789795, valid: 2347.019775, train_acc: 1.000000, valid_acc: 1.000000
epoch: 78, step: 234, time: 41.3, train: 3045.357422, valid: 2254.503662, train_acc: 1.000000, valid_acc: 1.000000
epoch: 79, step: 237, time: 49.8, train: 3436.515625, valid: 2167.895264, train_acc: 1.000000, valid_acc: 1.000000
epoch: 80, step: 240, time: 45.4, train: 3282.206055, valid: 2076.555176, train_acc: 1.000000, valid_acc: 1.000000
epoch: 81, step: 243, time: 37.0, train: 2318.514404, valid: 1567.852295, train_acc: 1.000000, valid_acc: 1.000000
epoch: 82, step: 246, time: 34.0, train: 2216.018311, valid: 1504.691284, train_acc: 1.000000, valid_acc: 1.000000
epoch: 83, step: 249, time: 48.1, train: 2841.709961, valid: 1441.396362, train_acc: 1.000000, valid_acc: 1.000000
epoch: 84, step: 252, time: 44.4, train: 2719.057861, valid: 1382.999023, train_acc: 1.000000, valid_acc: 1.000000
epoch: 85, step: 255, time: 38.9, train: 2100.511230, valid: 1324.168457, train_acc: 1.000000, valid_acc: 1.000000
epoch: 86, step: 258, time: 35.0, train: 2003.271729, valid: 1269.117188, train_acc: 1.000000, valid_acc: 1.000000
epoch: 87, step: 261, time: 43.2, train: 2108.814453, valid: 1218.108276, train_acc: 1.000000, valid_acc: 1.000000
epoch: 88, step: 264, time: 38.0, train: 2018.360229, valid: 1168.614868, train_acc: 1.000000, valid_acc: 1.000000
epoch: 89, step: 267, time: 50.6, train: 2301.843750, valid: 1424.382202, train_acc: 1.000000, valid_acc: 1.000000
epoch: 90, step: 270, time: 46.9, train: 2204.473389, valid: 1361.921631, train_acc: 1.000000, valid_acc: 1.000000
epoch: 91, step: 273, time: 37.4, train: 1335.083008, valid: 1593.474854, train_acc: 1.000000, valid_acc: 1.000000
epoch: 92, step: 276, time: 33.8, train: 1270.745117, valid: 1532.941284, train_acc: 1.000000, valid_acc: 1.000000
epoch: 93, step: 279, time: 34.6, train: 1264.228516, valid: 944.158264, train_acc: 1.000000, valid_acc: 1.000000
epoch: 94, step: 282, time: 30.8, train: 1207.621460, valid: 909.620850, train_acc: 1.000000, valid_acc: 1.000000
epoch: 95, step: 285, time: 43.9, train: 1630.147095, valid: 876.144897, train_acc: 1.000000, valid_acc: 1.000000
epoch: 96, step: 288, time: 39.7, train: 1572.044434, valid: 841.099548, train_acc: 1.000000, valid_acc: 1.000000
epoch: 97, step: 291, time: 43.0, train: 1418.999634, valid: 805.567322, train_acc: 1.000000, valid_acc: 1.000000
epoch: 98, step: 294, time: 37.0, train: 1353.778320, valid: 771.776245, train_acc: 1.000000, valid_acc: 1.000000
epoch: 99, step: 296, time: 27.5, train: 1286.588135, valid: 750.372986, train_acc: 1.000000, valid_acc: 1.000000
epoch: 100, step: 298, time: 26.6, train: 1249.231445, valid: 729.207275, train_acc: 1.000000, valid_acc: 1.000000
epoch: 101, step: 301, time: 38.8, train: 1021.336182, valid: 896.186401, train_acc: 1.000000, valid_acc: 1.000000
epoch: 102, step: 304, time: 35.3, train: 973.116211, valid: 861.137634, train_acc: 1.000000, valid_acc: 1.000000
epoch: 103, step: 306, time: 39.2, train: 1304.176758, valid: 842.754395, train_acc: 1.000000, valid_acc: 1.000000
epoch: 104, step: 308, time: 34.8, train: 1267.092041, valid: 819.922974, train_acc: 1.000000, valid_acc: 1.000000
epoch: 105, step: 311, time: 51.9, train: 1011.630981, valid: 785.372314, train_acc: 1.000000, valid_acc: 1.000000
epoch: 106, step: 314, time: 47.4, train: 965.016052, valid: 752.997375, train_acc: 1.000000, valid_acc: 1.000000
epoch: 107, step: 316, time: 27.1, train: 854.393555, valid: 566.926819, train_acc: 1.000000, valid_acc: 1.000000
epoch: 108, step: 318, time: 24.7, train: 832.280640, valid: 551.779358, train_acc: 1.000000, valid_acc: 1.000000
epoch: 109, step: 321, time: 45.0, train: 911.611206, valid: 686.532776, train_acc: 1.000000, valid_acc: 1.000000
epoch: 110, step: 324, time: 40.0, train: 874.889648, valid: 659.862854, train_acc: 1.000000, valid_acc: 1.000000
epoch: 111, step: 327, time: 36.8, train: 699.028137, valid: 488.502380, train_acc: 1.000000, valid_acc: 1.000000
epoch: 112, step: 330, time: 33.4, train: 668.123047, valid: 469.754059, train_acc: 1.000000, valid_acc: 1.000000
epoch: 113, step: 332, time: 37.8, train: 903.168579, valid: 456.849426, train_acc: 1.000000, valid_acc: 1.000000
epoch: 114, step: 334, time: 32.5, train: 875.612671, valid: 445.092987, train_acc: 1.000000, valid_acc: 1.000000
epoch: 115, step: 336, time: 28.7, train: 756.249512, valid: 561.915039, train_acc: 1.000000, valid_acc: 1.000000
epoch: 116, step: 338, time: 26.4, train: 732.227173, valid: 546.180481, train_acc: 1.000000, valid_acc: 1.000000
epoch: 117, step: 340, time: 33.8, train: 771.669312, valid: 530.113647, train_acc: 1.000000, valid_acc: 1.000000
epoch: 118, step: 342, time: 29.6, train: 747.426392, valid: 515.323120, train_acc: 1.000000, valid_acc: 1.000000
epoch: 119, step: 345, time: 46.2, train: 657.404480, valid: 380.836243, train_acc: 1.000000, valid_acc: 1.000000
epoch: 120, step: 348, time: 40.1, train: 629.479919, valid: 365.530640, train_acc: 1.000000, valid_acc: 1.000000
epoch: 121, step: 351, time: 37.0, train: 543.511353, valid: 350.521027, train_acc: 1.000000, valid_acc: 1.000000
epoch: 122, step: 354, time: 30.9, train: 522.089722, valid: 336.243958, train_acc: 1.000000, valid_acc: 1.000000
epoch: 123, step: 357, time: 45.0, train: 584.517822, valid: 322.188629, train_acc: 1.000000, valid_acc: 1.000000
epoch: 124, step: 360, time: 42.0, train: 561.045349, valid: 308.669800, train_acc: 1.000000, valid_acc: 1.000000
epoch: 125, step: 363, time: 42.2, train: 501.493134, valid: 297.717041, train_acc: 1.000000, valid_acc: 1.000000
epoch: 126, step: 366, time: 38.6, train: 482.223694, valid: 285.395477, train_acc: 1.000000, valid_acc: 1.000000
epoch: 127, step: 368, time: 26.8, train: 440.412140, valid: 278.295197, train_acc: 1.000000, valid_acc: 1.000000
epoch: 128, step: 370, time: 24.8, train: 431.639282, valid: 270.836700, train_acc: 1.000000, valid_acc: 1.000000
epoch: 129, step: 373, time: 34.6, train: 336.303833, valid: 261.383698, train_acc: 1.000000, valid_acc: 1.000000
epoch: 130, step: 376, time: 31.5, train: 325.417206, valid: 251.837219, train_acc: 1.000000, valid_acc: 1.000000
epoch: 131, step: 379, time: 48.0, train: 461.025635, valid: 242.943359, train_acc: 1.000000, valid_acc: 1.000000
epoch: 132, step: 382, time: 43.3, train: 444.764526, valid: 233.749222, train_acc: 1.000000, valid_acc: 1.000000
epoch: 133, step: 385, time: 36.7, train: 316.343079, valid: 225.080307, train_acc: 1.000000, valid_acc: 1.000000
epoch: 134, step: 388, time: 32.4, train: 303.249298, valid: 216.899704, train_acc: 1.000000, valid_acc: 1.000000
epoch: 135, step: 391, time: 37.8, train: 288.430603, valid: 208.233185, train_acc: 1.000000, valid_acc: 1.000000
epoch: 136, step: 394, time: 33.3, train: 276.052216, valid: 201.300919, train_acc: 1.000000, valid_acc: 1.000000
epoch: 137, step: 397, time: 45.3, train: 308.886871, valid: 194.802765, train_acc: 1.000000, valid_acc: 1.000000
epoch: 138, step: 400, time: 39.4, train: 295.769348, valid: 188.072601, train_acc: 1.000000, valid_acc: 1.000000
epoch: 139, step: 403, time: 36.7, train: 285.840088, valid: 180.924850, train_acc: 1.000000, valid_acc: 1.000000
epoch: 140, step: 406, time: 32.8, train: 274.815308, valid: 174.399933, train_acc: 1.000000, valid_acc: 1.000000
epoch: 141, step: 409, time: 39.9, train: 251.818863, valid: 221.658127, train_acc: 1.000000, valid_acc: 1.000000
epoch: 142, step: 412, time: 36.5, train: 241.830154, valid: 213.956436, train_acc: 1.000000, valid_acc: 1.000000
epoch: 143, step: 415, time: 36.6, train: 217.579544, valid: 156.948212, train_acc: 1.000000, valid_acc: 1.000000
epoch: 144, step: 418, time: 34.1, train: 209.230377, valid: 151.590591, train_acc: 1.000000, valid_acc: 1.000000
epoch: 145, step: 420, time: 28.7, train: 232.159302, valid: 147.818390, train_acc: 1.000000, valid_acc: 1.000000
epoch: 146, step: 422, time: 25.0, train: 225.731949, valid: 144.349960, train_acc: 1.000000, valid_acc: 1.000000
epoch: 147, step: 425, time: 43.3, train: 241.284729, valid: 139.767776, train_acc: 1.000000, valid_acc: 1.000000
epoch: 148, step: 428, time: 40.0, train: 232.916092, valid: 134.676056, train_acc: 1.000000, valid_acc: 1.000000
epoch: 149, step: 431, time: 49.0, train: 252.877350, valid: 129.550735, train_acc: 1.000000, valid_acc: 1.000000
epoch: 150, step: 434, time: 43.1, train: 242.238907, valid: 124.327324, train_acc: 1.000000, valid_acc: 1.000000
epoch: 151, step: 437, time: 41.2, train: 176.435089, valid: 119.331146, train_acc: 1.000000, valid_acc: 1.000000
epoch: 152, step: 440, time: 35.0, train: 170.045303, valid: 114.771118, train_acc: 1.000000, valid_acc: 1.000000
epoch: 153, step: 442, time: 30.5, train: 194.212677, valid: 148.203049, train_acc: 1.000000, valid_acc: 1.000000
epoch: 154, step: 444, time: 28.0, train: 189.437042, valid: 144.620651, train_acc: 1.000000, valid_acc: 1.000000
epoch: 155, step: 446, time: 26.8, train: 163.282715, valid: 106.411247, train_acc: 1.000000, valid_acc: 1.000000
epoch: 156, step: 448, time: 25.1, train: 158.510254, valid: 103.763191, train_acc: 1.000000, valid_acc: 1.000000
epoch: 157, step: 451, time: 47.6, train: 176.118851, valid: 99.957443, train_acc: 1.000000, valid_acc: 1.000000
epoch: 158, step: 454, time: 41.9, train: 169.352524, valid: 96.300179, train_acc: 1.000000, valid_acc: 1.000000
epoch: 159, step: 456, time: 28.2, train: 133.144852, valid: 124.108955, train_acc: 1.000000, valid_acc: 1.000000
epoch: 160, step: 458, time: 25.8, train: 129.213226, valid: 121.251892, train_acc: 1.000000, valid_acc: 1.000000
epoch: 161, step: 461, time: 45.6, train: 147.988541, valid: 116.897629, train_acc: 1.000000, valid_acc: 1.000000
epoch: 162, step: 464, time: 40.8, train: 142.627426, valid: 112.736137, train_acc: 1.000000, valid_acc: 1.000000
epoch: 163, step: 466, time: 31.0, train: 146.642654, valid: 83.056381, train_acc: 1.000000, valid_acc: 1.000000
epoch: 164, step: 468, time: 28.4, train: 143.028046, valid: 80.985664, train_acc: 1.000000, valid_acc: 1.000000
epoch: 165, step: 470, time: 30.4, train: 110.866501, valid: 131.079956, train_acc: 1.000000, valid_acc: 1.000000
epoch: 166, step: 472, time: 29.3, train: 107.958076, valid: 128.057343, train_acc: 1.000000, valid_acc: 1.000000
epoch: 167, step: 475, time: 48.4, train: 132.811279, valid: 74.063637, train_acc: 1.000000, valid_acc: 1.000000
epoch: 168, step: 478, time: 43.7, train: 127.581917, valid: 71.475838, train_acc: 1.000000, valid_acc: 1.000000
epoch: 169, step: 481, time: 41.5, train: 105.630898, valid: 69.071480, train_acc: 1.000000, valid_acc: 1.000000
epoch: 170, step: 484, time: 36.3, train: 101.934998, valid: 66.643654, train_acc: 1.000000, valid_acc: 1.000000
epoch: 171, step: 487, time: 43.1, train: 102.252068, valid: 64.164558, train_acc: 1.000000, valid_acc: 1.000000
epoch: 172, step: 490, time: 38.0, train: 98.495461, valid: 61.891346, train_acc: 1.000000, valid_acc: 1.000000
epoch: 173, step: 493, time: 43.2, train: 98.761269, valid: 59.748917, train_acc: 1.000000, valid_acc: 1.000000
epoch: 174, step: 496, time: 40.0, train: 95.314285, valid: 57.544189, train_acc: 1.000000, valid_acc: 1.000000
epoch: 175, step: 499, time: 44.6, train: 88.729424, valid: 74.451393, train_acc: 1.000000, valid_acc: 1.000000
epoch: 176, step: 502, time: 39.6, train: 85.870659, valid: 71.904457, train_acc: 1.000000, valid_acc: 1.000000
epoch: 177, step: 505, time: 40.4, train: 65.000298, valid: 69.484703, train_acc: 1.000000, valid_acc: 1.000000
epoch: 178, step: 508, time: 36.4, train: 62.976456, valid: 67.337547, train_acc: 1.000000, valid_acc: 1.000000
epoch: 179, step: 510, time: 32.4, train: 81.403130, valid: 49.231853, train_acc: 1.000000, valid_acc: 1.000000
epoch: 180, step: 512, time: 31.6, train: 79.601639, valid: 48.123695, train_acc: 1.000000, valid_acc: 1.000000
epoch: 181, step: 514, time: 30.3, train: 80.158157, valid: 47.052097, train_acc: 1.000000, valid_acc: 1.000000
epoch: 182, step: 516, time: 27.4, train: 78.298042, valid: 45.968353, train_acc: 1.000000, valid_acc: 1.000000
epoch: 183, step: 519, time: 47.1, train: 78.818672, valid: 59.450962, train_acc: 1.000000, valid_acc: 1.000000
epoch: 184, step: 522, time: 44.6, train: 75.634178, valid: 57.395416, train_acc: 1.000000, valid_acc: 1.000000
epoch: 185, step: 525, time: 38.1, train: 56.000950, valid: 41.277908, train_acc: 1.000000, valid_acc: 1.000000
epoch: 186, step: 528, time: 35.2, train: 54.360001, valid: 39.837250, train_acc: 1.000000, valid_acc: 1.000000
epoch: 187, step: 531, time: 41.6, train: 51.882000, valid: 51.802460, train_acc: 1.000000, valid_acc: 1.000000
epoch: 188, step: 534, time: 36.7, train: 49.832394, valid: 50.214928, train_acc: 1.000000, valid_acc: 1.000000
epoch: 189, step: 537, time: 41.9, train: 51.967598, valid: 48.802971, train_acc: 1.000000, valid_acc: 1.000000
epoch: 190, step: 540, time: 36.8, train: 50.108456, valid: 47.296528, train_acc: 1.000000, valid_acc: 1.000000
epoch: 191, step: 543, time: 49.6, train: 63.512653, valid: 33.903912, train_acc: 1.000000, valid_acc: 1.000000
epoch: 192, step: 546, time: 44.4, train: 61.526207, valid: 32.709763, train_acc: 1.000000, valid_acc: 1.000000
epoch: 193, step: 548, time: 28.7, train: 46.088982, valid: 43.030205, train_acc: 1.000000, valid_acc: 1.000000
epoch: 194, step: 550, time: 26.0, train: 44.897934, valid: 42.069576, train_acc: 1.000000, valid_acc: 1.000000
epoch: 195, step: 553, time: 39.5, train: 43.355881, valid: 30.239189, train_acc: 1.000000, valid_acc: 1.000000
epoch: 196, step: 556, time: 33.1, train: 41.863243, valid: 29.235308, train_acc: 1.000000, valid_acc: 1.000000
epoch: 197, step: 559, time: 37.9, train: 40.306763, valid: 28.259571, train_acc: 1.000000, valid_acc: 1.000000
epoch: 198, step: 562, time: 34.4, train: 38.785088, valid: 27.327831, train_acc: 1.000000, valid_acc: 1.000000
epoch: 199, step: 564, time: 37.0, train: 47.996223, valid: 36.212772, train_acc: 1.000000, valid_acc: 1.000000
epoch: 200, step: 566, time: 32.3, train: 46.884228, valid: 35.405151, train_acc: 1.000000, valid_acc: 1.000000
"""  # Include all the data here

# Parse the data
epochs = []
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

for line in data.strip().split('\n'):
    parts = line.split(', ')
    epoch = int(parts[0].split(': ')[1])
    train_loss = float(parts[3].split(': ')[1])
    valid_loss = float(parts[4].split(': ')[1])
    train_acc = float(parts[5].split(': ')[1])
    valid_acc = float(parts[6].split(': ')[1])
    
    epochs.append(epoch)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)

# Plot train vs. validation loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, valid_losses, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss (L-Chiral Sample Data)')
plt.legend()
plt.grid(True)

# Plot train vs. validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
plt.plot(epochs, valid_accuracies, label='Validation Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy Run 2 (L-Chiral Sample Data)')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the plot to a file
plt.savefig('training_validation_plots_lchiral_sample_data2.png')
print("Plot saved as 'training_validation_plots_lchiral_sample_data2.png'")

