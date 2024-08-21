#!/usr/bin/env python

import matplotlib.pyplot as plt

# Sample data from the provided text
data = """epoch: 1, step: 3, time: 80.9, train: 0.682811, valid: 0.957416, train_acc: 0.731774, valid_acc: 0.940412
epoch: 2, step: 6, time: 73.0, train: 0.682633, valid: 0.956327, train_acc: 0.731076, valid_acc: 0.941308
epoch: 3, step: 9, time: 36.9, train: 0.647854, valid: 0.955433, train_acc: 0.730365, valid_acc: 0.941607
epoch: 4, step: 12, time: 32.6, train: 0.647570, valid: 0.955920, train_acc: 0.728955, valid_acc: 0.941308
epoch: 5, step: 15, time: 37.9, train: 0.668843, valid: 0.951718, train_acc: 0.728463, valid_acc: 0.925329
epoch: 6, step: 18, time: 36.5, train: 0.666067, valid: 0.953496, train_acc: 0.725875, valid_acc: 0.928116
epoch: 7, step: 21, time: 40.2, train: 0.674861, valid: 0.963748, train_acc: 0.723187, valid_acc: 0.929112
epoch: 8, step: 24, time: 35.8, train: 0.673112, valid: 0.965447, train_acc: 0.720357, valid_acc: 0.931302
epoch: 9, step: 27, time: 47.1, train: 0.658440, valid: 0.958335, train_acc: 0.716533, valid_acc: 0.932895
epoch: 10, step: 30, time: 39.8, train: 0.654360, valid: 0.958864, train_acc: 0.712853, valid_acc: 0.937077
epoch: 11, step: 33, time: 34.7, train: 0.653127, valid: 0.963477, train_acc: 0.708838, valid_acc: 0.936878
epoch: 12, step: 36, time: 30.0, train: 0.646266, valid: 0.964851, train_acc: 0.704369, valid_acc: 0.940263
epoch: 13, step: 39, time: 38.7, train: 0.639279, valid: 0.971655, train_acc: 0.706364, valid_acc: 0.956541
epoch: 14, step: 42, time: 34.6, train: 0.634149, valid: 0.972190, train_acc: 0.701144, valid_acc: 0.958035
epoch: 15, step: 45, time: 32.9, train: 0.618615, valid: 0.970739, train_acc: 0.687342, valid_acc: 0.951613
epoch: 16, step: 48, time: 28.4, train: 0.611902, valid: 0.972808, train_acc: 0.681532, valid_acc: 0.951414
epoch: 17, step: 51, time: 44.8, train: 0.622570, valid: 0.975862, train_acc: 0.676535, valid_acc: 0.952210
epoch: 18, step: 54, time: 46.2, train: 0.614743, valid: 0.979513, train_acc: 0.669774, valid_acc: 0.954998
epoch: 19, step: 57, time: 44.9, train: 0.627419, valid: 0.978772, train_acc: 0.662876, valid_acc: 0.958781
epoch: 20, step: 60, time: 41.6, train: 0.620727, valid: 0.980705, train_acc: 0.655872, valid_acc: 0.962166
epoch: 21, step: 63, time: 41.1, train: 0.609110, valid: 0.984866, train_acc: 0.647993, valid_acc: 0.974612
epoch: 22, step: 66, time: 37.5, train: 0.600429, valid: 0.986747, train_acc: 0.640386, valid_acc: 0.976852
epoch: 23, step: 70, time: 52.0, train: 0.572885, valid: 0.985935, train_acc: 0.636758, valid_acc: 0.978047
epoch: 24, step: 74, time: 49.1, train: 0.562373, valid: 0.990148, train_acc: 0.625761, valid_acc: 0.980436
epoch: 25, step: 77, time: 33.8, train: 0.551599, valid: 0.990761, train_acc: 0.609933, valid_acc: 0.978495
epoch: 26, step: 80, time: 30.4, train: 0.543224, valid: 0.993803, train_acc: 0.601023, valid_acc: 0.980486
epoch: 27, step: 84, time: 47.5, train: 0.533686, valid: 0.993953, train_acc: 0.589369, valid_acc: 0.981481
epoch: 28, step: 88, time: 40.3, train: 0.522112, valid: 0.995317, train_acc: 0.576666, valid_acc: 0.984070
epoch: 29, step: 91, time: 34.5, train: 0.522385, valid: 0.996288, train_acc: 0.566729, valid_acc: 0.984468
epoch: 30, step: 94, time: 33.8, train: 0.512915, valid: 0.997048, train_acc: 0.557077, valid_acc: 0.986061
epoch: 31, step: 97, time: 37.5, train: 0.489616, valid: 0.997827, train_acc: 0.555629, valid_acc: 0.992085
epoch: 32, step: 100, time: 29.5, train: 0.478812, valid: 0.998130, train_acc: 0.545675, valid_acc: 0.993280
epoch: 33, step: 103, time: 36.5, train: 0.474380, valid: 0.998295, train_acc: 0.536386, valid_acc: 0.994176
epoch: 34, step: 106, time: 31.4, train: 0.464872, valid: 0.998484, train_acc: 0.526266, valid_acc: 0.996117
epoch: 35, step: 109, time: 39.1, train: 0.454911, valid: 0.998669, train_acc: 0.506315, valid_acc: 0.997013
epoch: 36, step: 112, time: 32.7, train: 0.445672, valid: 0.999234, train_acc: 0.495866, valid_acc: 0.998805
epoch: 37, step: 115, time: 40.8, train: 0.450030, valid: 0.999539, train_acc: 0.495376, valid_acc: 0.999104
epoch: 38, step: 118, time: 36.0, train: 0.439331, valid: 0.999654, train_acc: 0.485181, valid_acc: 0.999104
epoch: 39, step: 120, time: 23.8, train: 0.414962, valid: 0.999668, train_acc: 0.468799, valid_acc: 0.999204
epoch: 40, step: 122, time: 22.2, train: 0.408045, valid: 0.999801, train_acc: 0.461746, valid_acc: 1.000000
epoch: 41, step: 124, time: 30.4, train: 0.414531, valid: 0.999790, train_acc: 0.463888, valid_acc: 1.000000
epoch: 42, step: 126, time: 27.3, train: 0.408269, valid: 0.999895, train_acc: 0.456993, valid_acc: 1.000000
epoch: 43, step: 129, time: 35.0, train: 0.383044, valid: 0.999859, train_acc: 0.437897, valid_acc: 1.000000
epoch: 44, step: 132, time: 29.9, train: 0.373318, valid: 0.999953, train_acc: 0.427455, valid_acc: 1.000000
epoch: 45, step: 135, time: 38.8, train: 0.356224, valid: 0.999829, train_acc: 0.416273, valid_acc: 1.000000
epoch: 46, step: 138, time: 32.4, train: 0.347178, valid: 0.999914, train_acc: 0.405930, valid_acc: 1.000000
epoch: 47, step: 141, time: 41.7, train: 0.358437, valid: 1.000000, train_acc: 0.395702, valid_acc: 1.000000
epoch: 48, step: 144, time: 35.6, train: 0.348541, valid: 1.000000, train_acc: 0.385599, valid_acc: 1.000000
epoch: 49, step: 147, time: 36.7, train: 0.331597, valid: 1.000000, train_acc: 0.384805, valid_acc: 1.000000
epoch: 50, step: 150, time: 32.1, train: 0.321032, valid: 0.999954, train_acc: 0.374630, valid_acc: 1.000000
epoch: 51, step: 153, time: 36.9, train: 0.315765, valid: 1.000000, train_acc: 0.364989, valid_acc: 1.000000
epoch: 52, step: 156, time: 32.2, train: 0.306207, valid: 1.000000, train_acc: 0.354994, valid_acc: 1.000000
epoch: 53, step: 158, time: 23.2, train: 0.306824, valid: 1.000000, train_acc: 0.339795, valid_acc: 1.000000
epoch: 54, step: 160, time: 22.1, train: 0.299676, valid: 1.000000, train_acc: 0.333234, valid_acc: 1.000000
epoch: 55, step: 162, time: 27.0, train: 0.302134, valid: 1.000000, train_acc: 0.326676, valid_acc: 1.000000
epoch: 56, step: 164, time: 25.6, train: 0.296675, valid: 0.999945, train_acc: 0.320357, valid_acc: 1.000000
epoch: 57, step: 166, time: 28.6, train: 0.280157, valid: 1.000000, train_acc: 0.322470, valid_acc: 1.000000
epoch: 58, step: 168, time: 24.2, train: 0.274577, valid: 1.000000, train_acc: 0.316183, valid_acc: 1.000000
epoch: 59, step: 170, time: 29.4, train: 0.279240, valid: 1.000000, train_acc: 0.301820, valid_acc: 1.000000
epoch: 60, step: 172, time: 26.9, train: 0.273310, valid: 1.000000, train_acc: 0.295706, valid_acc: 1.000000
epoch: 61, step: 176, time: 46.4, train: 0.249165, valid: 1.000000, train_acc: 0.283231, valid_acc: 1.000000
epoch: 62, step: 180, time: 42.1, train: 0.238112, valid: 1.000000, train_acc: 0.271404, valid_acc: 1.000000
epoch: 63, step: 183, time: 41.9, train: 0.231080, valid: 1.000000, train_acc: 0.271124, valid_acc: 1.000000
epoch: 64, step: 186, time: 34.7, train: 0.223524, valid: 1.000000, train_acc: 0.262544, valid_acc: 1.000000
epoch: 65, step: 189, time: 44.2, train: 0.214086, valid: 1.000000, train_acc: 0.255723, valid_acc: 1.000000
epoch: 66, step: 192, time: 36.6, train: 0.206120, valid: 1.000000, train_acc: 0.247639, valid_acc: 1.000000
epoch: 67, step: 195, time: 36.4, train: 0.194488, valid: 1.000000, train_acc: 0.230611, valid_acc: 1.000000
epoch: 68, step: 198, time: 29.0, train: 0.187836, valid: 1.000000, train_acc: 0.222977, valid_acc: 1.000000
epoch: 69, step: 201, time: 35.4, train: 0.190899, valid: 1.000000, train_acc: 0.215290, valid_acc: 1.000000
epoch: 70, step: 204, time: 30.0, train: 0.184269, valid: 1.000000, train_acc: 0.207934, valid_acc: 1.000000
epoch: 71, step: 206, time: 27.5, train: 0.182884, valid: 1.000000, train_acc: 0.207070, valid_acc: 1.000000
epoch: 72, step: 208, time: 24.6, train: 0.178040, valid: 1.000000, train_acc: 0.202297, valid_acc: 1.000000
epoch: 73, step: 211, time: 39.8, train: 0.171477, valid: 1.000000, train_acc: 0.191654, valid_acc: 1.000000
epoch: 74, step: 214, time: 34.6, train: 0.165569, valid: 1.000000, train_acc: 0.184918, valid_acc: 1.000000
epoch: 75, step: 217, time: 44.4, train: 0.161668, valid: 1.000000, train_acc: 0.178623, valid_acc: 1.000000
epoch: 76, step: 220, time: 39.6, train: 0.155877, valid: 1.000000, train_acc: 0.172245, valid_acc: 1.000000
epoch: 77, step: 223, time: 40.6, train: 0.134544, valid: 1.000000, train_acc: 0.166097, valid_acc: 1.000000
epoch: 78, step: 226, time: 34.8, train: 0.129290, valid: 1.000000, train_acc: 0.160206, valid_acc: 1.000000
epoch: 79, step: 229, time: 38.9, train: 0.129052, valid: 1.000000, train_acc: 0.154282, valid_acc: 1.000000
epoch: 80, step: 232, time: 36.7, train: 0.124547, valid: 1.000000, train_acc: 0.148825, valid_acc: 1.000000
epoch: 81, step: 235, time: 31.2, train: 0.110651, valid: 1.000000, train_acc: 0.143863, valid_acc: 1.000000
epoch: 82, step: 238, time: 27.5, train: 0.106399, valid: 1.000000, train_acc: 0.138796, valid_acc: 1.000000
epoch: 83, step: 241, time: 39.4, train: 0.110937, valid: 1.000000, train_acc: 0.137108, valid_acc: 1.000000
epoch: 84, step: 244, time: 33.6, train: 0.106621, valid: 1.000000, train_acc: 0.132372, valid_acc: 1.000000
epoch: 85, step: 247, time: 44.0, train: 0.108464, valid: 1.000000, train_acc: 0.127534, valid_acc: 1.000000
epoch: 86, step: 250, time: 39.2, train: 0.104369, valid: 1.000000, train_acc: 0.122903, valid_acc: 1.000000
epoch: 87, step: 253, time: 35.2, train: 0.097295, valid: 1.000000, train_acc: 0.115520, valid_acc: 1.000000
epoch: 88, step: 256, time: 30.4, train: 0.093423, valid: 1.000000, train_acc: 0.111341, valid_acc: 1.000000
epoch: 89, step: 259, time: 44.6, train: 0.092605, valid: 1.000000, train_acc: 0.107355, valid_acc: 1.000000
epoch: 90, step: 262, time: 40.1, train: 0.089130, valid: 1.000000, train_acc: 0.103337, valid_acc: 1.000000
epoch: 91, step: 265, time: 43.2, train: 0.091298, valid: 1.000000, train_acc: 0.099144, valid_acc: 1.000000
epoch: 92, step: 268, time: 35.8, train: 0.087740, valid: 1.000000, train_acc: 0.095409, valid_acc: 1.000000
epoch: 93, step: 271, time: 36.3, train: 0.075540, valid: 1.000000, train_acc: 0.092005, valid_acc: 1.000000
epoch: 94, step: 274, time: 31.6, train: 0.072757, valid: 1.000000, train_acc: 0.088551, valid_acc: 1.000000
epoch: 95, step: 277, time: 37.8, train: 0.071364, valid: 1.000000, train_acc: 0.089619, valid_acc: 1.000000
epoch: 96, step: 280, time: 33.6, train: 0.068561, valid: 1.000000, train_acc: 0.086317, valid_acc: 1.000000
epoch: 97, step: 282, time: 35.9, train: 0.068207, valid: 1.000000, train_acc: 0.082538, valid_acc: 1.000000
epoch: 98, step: 284, time: 32.3, train: 0.066318, valid: 1.000000, train_acc: 0.080535, valid_acc: 1.000000
epoch: 99, step: 287, time: 41.6, train: 0.056679, valid: 1.000000, train_acc: 0.075298, valid_acc: 1.000000
epoch: 100, step: 290, time: 36.3, train: 0.054594, valid: 1.000000, train_acc: 0.072545, valid_acc: 1.000000
epoch: 101, step: 293, time: 38.1, train: 0.055686, valid: 1.000000, train_acc: 0.069853, valid_acc: 1.000000
epoch: 102, step: 296, time: 32.6, train: 0.053544, valid: 1.000000, train_acc: 0.067351, valid_acc: 1.000000
epoch: 103, step: 299, time: 41.5, train: 0.058789, valid: 1.000000, train_acc: 0.068548, valid_acc: 1.000000
epoch: 104, step: 302, time: 35.1, train: 0.056649, valid: 1.000000, train_acc: 0.066034, valid_acc: 1.000000
epoch: 105, step: 305, time: 33.7, train: 0.045539, valid: 1.000000, train_acc: 0.063532, valid_acc: 1.000000
epoch: 106, step: 308, time: 29.2, train: 0.043960, valid: 1.000000, train_acc: 0.061208, valid_acc: 1.000000
epoch: 107, step: 311, time: 42.2, train: 0.045918, valid: 1.000000, train_acc: 0.060072, valid_acc: 1.000000
epoch: 108, step: 314, time: 37.6, train: 0.044110, valid: 1.000000, train_acc: 0.057922, valid_acc: 1.000000
epoch: 109, step: 317, time: 32.0, train: 0.037810, valid: 1.000000, train_acc: 0.051832, valid_acc: 1.000000
epoch: 110, step: 320, time: 28.9, train: 0.036233, valid: 1.000000, train_acc: 0.050002, valid_acc: 1.000000
epoch: 111, step: 323, time: 35.1, train: 0.039057, valid: 1.000000, train_acc: 0.048223, valid_acc: 1.000000
epoch: 112, step: 326, time: 30.5, train: 0.037547, valid: 1.000000, train_acc: 0.046504, valid_acc: 1.000000
epoch: 113, step: 329, time: 45.0, train: 0.038071, valid: 1.000000, train_acc: 0.044807, valid_acc: 1.000000
epoch: 114, step: 332, time: 39.7, train: 0.036586, valid: 1.000000, train_acc: 0.043195, valid_acc: 1.000000
epoch: 115, step: 335, time: 37.3, train: 0.031352, valid: 1.000000, train_acc: 0.044169, valid_acc: 1.000000
epoch: 116, step: 338, time: 32.8, train: 0.030257, valid: 1.000000, train_acc: 0.042592, valid_acc: 1.000000
epoch: 117, step: 342, time: 48.0, train: 0.031362, valid: 1.000000, train_acc: 0.038133, valid_acc: 1.000000
epoch: 118, step: 346, time: 45.5, train: 0.029799, valid: 1.000000, train_acc: 0.036295, valid_acc: 1.000000
epoch: 119, step: 349, time: 43.6, train: 0.026198, valid: 1.000000, train_acc: 0.037985, valid_acc: 1.000000
epoch: 120, step: 352, time: 39.5, train: 0.025210, valid: 1.000000, train_acc: 0.036677, valid_acc: 1.000000
epoch: 121, step: 355, time: 40.0, train: 0.024136, valid: 1.000000, train_acc: 0.032643, valid_acc: 1.000000
epoch: 122, step: 358, time: 35.5, train: 0.023156, valid: 1.000000, train_acc: 0.031531, valid_acc: 1.000000
epoch: 123, step: 361, time: 44.4, train: 0.025055, valid: 1.000000, train_acc: 0.030546, valid_acc: 1.000000
epoch: 124, step: 364, time: 41.0, train: 0.024206, valid: 1.000000, train_acc: 0.029446, valid_acc: 1.000000
epoch: 125, step: 366, time: 25.7, train: 0.021053, valid: 1.000000, train_acc: 0.028626, valid_acc: 1.000000
epoch: 126, step: 368, time: 23.1, train: 0.020572, valid: 1.000000, train_acc: 0.027939, valid_acc: 1.000000
epoch: 127, step: 370, time: 25.3, train: 0.023694, valid: 1.000000, train_acc: 0.027365, valid_acc: 1.000000
epoch: 128, step: 372, time: 23.0, train: 0.023142, valid: 1.000000, train_acc: 0.026704, valid_acc: 1.000000
epoch: 129, step: 374, time: 36.8, train: 0.023120, valid: 1.000000, train_acc: 0.026079, valid_acc: 1.000000
epoch: 130, step: 376, time: 32.3, train: 0.022595, valid: 1.000000, train_acc: 0.025427, valid_acc: 1.000000
epoch: 131, step: 378, time: 27.9, train: 0.020871, valid: 1.000000, train_acc: 0.024679, valid_acc: 1.000000
epoch: 132, step: 380, time: 23.9, train: 0.020339, valid: 1.000000, train_acc: 0.024058, valid_acc: 1.000000
epoch: 133, step: 382, time: 28.8, train: 0.020423, valid: 1.000000, train_acc: 0.023487, valid_acc: 1.000000
epoch: 134, step: 384, time: 28.0, train: 0.019912, valid: 1.000000, train_acc: 0.022897, valid_acc: 1.000000
epoch: 135, step: 387, time: 45.2, train: 0.018500, valid: 1.000000, train_acc: 0.023551, valid_acc: 1.000000
epoch: 136, step: 390, time: 42.7, train: 0.017849, valid: 1.000000, train_acc: 0.022682, valid_acc: 1.000000
epoch: 137, step: 392, time: 25.7, train: 0.017330, valid: 1.000000, train_acc: 0.020684, valid_acc: 1.000000
epoch: 138, step: 394, time: 23.3, train: 0.016835, valid: 1.000000, train_acc: 0.020145, valid_acc: 1.000000
epoch: 139, step: 397, time: 46.7, train: 0.016917, valid: 1.000000, train_acc: 0.020785, valid_acc: 1.000000
epoch: 140, step: 400, time: 39.7, train: 0.016298, valid: 1.000000, train_acc: 0.020016, valid_acc: 1.000000
epoch: 141, step: 403, time: 36.3, train: 0.013353, valid: 1.000000, train_acc: 0.018044, valid_acc: 1.000000
epoch: 142, step: 406, time: 32.1, train: 0.012926, valid: 1.000000, train_acc: 0.017380, valid_acc: 1.000000
epoch: 143, step: 409, time: 42.5, train: 0.014396, valid: 1.000000, train_acc: 0.016708, valid_acc: 1.000000
epoch: 144, step: 412, time: 35.8, train: 0.013910, valid: 1.000000, train_acc: 0.016104, valid_acc: 1.000000
epoch: 145, step: 415, time: 44.7, train: 0.013157, valid: 1.000000, train_acc: 0.015592, valid_acc: 1.000000
epoch: 146, step: 418, time: 42.1, train: 0.012694, valid: 1.000000, train_acc: 0.015024, valid_acc: 1.000000
epoch: 147, step: 422, time: 47.0, train: 0.010045, valid: 1.000000, train_acc: 0.014319, valid_acc: 1.000000
epoch: 148, step: 426, time: 41.7, train: 0.009593, valid: 1.000000, train_acc: 0.013664, valid_acc: 1.000000
epoch: 149, step: 429, time: 46.0, train: 0.010884, valid: 1.000000, train_acc: 0.013137, valid_acc: 1.000000
epoch: 150, step: 432, time: 41.0, train: 0.010524, valid: 1.000000, train_acc: 0.012681, valid_acc: 1.000000
epoch: 151, step: 435, time: 40.3, train: 0.009631, valid: 1.000000, train_acc: 0.012263, valid_acc: 1.000000
epoch: 152, step: 438, time: 35.6, train: 0.009299, valid: 1.000000, train_acc: 0.011833, valid_acc: 1.000000
epoch: 153, step: 441, time: 42.2, train: 0.009117, valid: 1.000000, train_acc: 0.011480, valid_acc: 1.000000
epoch: 154, step: 444, time: 35.2, train: 0.008794, valid: 1.000000, train_acc: 0.011079, valid_acc: 1.000000
epoch: 155, step: 447, time: 43.0, train: 0.008608, valid: 1.000000, train_acc: 0.010696, valid_acc: 1.000000
epoch: 156, step: 450, time: 36.2, train: 0.008295, valid: 1.000000, train_acc: 0.010325, valid_acc: 1.000000
epoch: 157, step: 452, time: 30.7, train: 0.008410, valid: 1.000000, train_acc: 0.010853, valid_acc: 1.000000
epoch: 158, step: 454, time: 26.7, train: 0.008211, valid: 1.000000, train_acc: 0.010599, valid_acc: 1.000000
epoch: 159, step: 457, time: 35.2, train: 0.006983, valid: 1.000000, train_acc: 0.009498, valid_acc: 1.000000
epoch: 160, step: 460, time: 26.9, train: 0.006729, valid: 1.000000, train_acc: 0.009174, valid_acc: 1.000000
epoch: 161, step: 462, time: 26.0, train: 0.007907, valid: 1.000000, train_acc: 0.008920, valid_acc: 1.000000
epoch: 162, step: 464, time: 24.5, train: 0.007711, valid: 1.000000, train_acc: 0.008711, valid_acc: 1.000000
epoch: 163, step: 467, time: 40.1, train: 0.005468, valid: 1.000000, train_acc: 0.009099, valid_acc: 1.000000
epoch: 164, step: 470, time: 33.6, train: 0.005302, valid: 1.000000, train_acc: 0.008800, valid_acc: 1.000000
epoch: 165, step: 473, time: 39.3, train: 0.006574, valid: 1.000000, train_acc: 0.007876, valid_acc: 1.000000
epoch: 166, step: 476, time: 35.4, train: 0.006366, valid: 1.000000, train_acc: 0.007609, valid_acc: 1.000000
epoch: 167, step: 479, time: 37.0, train: 0.005201, valid: 1.000000, train_acc: 0.007348, valid_acc: 1.000000
epoch: 168, step: 482, time: 32.5, train: 0.005027, valid: 1.000000, train_acc: 0.007104, valid_acc: 1.000000
epoch: 169, step: 485, time: 43.8, train: 0.005362, valid: 1.000000, train_acc: 0.006894, valid_acc: 1.000000
epoch: 170, step: 488, time: 38.2, train: 0.005159, valid: 1.000000, train_acc: 0.006669, valid_acc: 1.000000
epoch: 171, step: 491, time: 40.7, train: 0.004386, valid: 1.000000, train_acc: 0.006977, valid_acc: 1.000000
epoch: 172, step: 494, time: 37.9, train: 0.004248, valid: 1.000000, train_acc: 0.006749, valid_acc: 1.000000
epoch: 173, step: 497, time: 38.2, train: 0.004524, valid: 1.000000, train_acc: 0.006531, valid_acc: 1.000000
epoch: 174, step: 500, time: 34.5, train: 0.004370, valid: 1.000000, train_acc: 0.006321, valid_acc: 1.000000
epoch: 175, step: 503, time: 33.8, train: 0.003970, valid: 1.000000, train_acc: 0.005624, valid_acc: 1.000000
epoch: 176, step: 506, time: 32.0, train: 0.003848, valid: 1.000000, train_acc: 0.005446, valid_acc: 1.000000
epoch: 177, step: 509, time: 41.0, train: 0.004217, valid: 1.000000, train_acc: 0.005283, valid_acc: 1.000000
epoch: 178, step: 512, time: 37.4, train: 0.004078, valid: 1.000000, train_acc: 0.005108, valid_acc: 1.000000
epoch: 179, step: 515, time: 46.6, train: 0.003578, valid: 1.000000, train_acc: 0.005224, valid_acc: 1.000000
epoch: 180, step: 518, time: 41.0, train: 0.003446, valid: 1.000000, train_acc: 0.005055, valid_acc: 1.000000
epoch: 181, step: 521, time: 41.4, train: 0.003754, valid: 1.000000, train_acc: 0.004598, valid_acc: 1.000000
epoch: 182, step: 524, time: 32.1, train: 0.003613, valid: 1.000000, train_acc: 0.004445, valid_acc: 1.000000
epoch: 183, step: 527, time: 43.9, train: 0.002950, valid: 1.000000, train_acc: 0.004311, valid_acc: 1.000000
epoch: 184, step: 530, time: 36.4, train: 0.002853, valid: 1.000000, train_acc: 0.004167, valid_acc: 1.000000
epoch: 185, step: 534, time: 51.7, train: 0.002939, valid: 1.000000, train_acc: 0.003975, valid_acc: 1.000000
epoch: 186, step: 538, time: 44.5, train: 0.002805, valid: 1.000000, train_acc: 0.003801, valid_acc: 1.000000
epoch: 187, step: 540, time: 24.5, train: 0.002746, valid: 1.000000, train_acc: 0.003733, valid_acc: 1.000000
epoch: 188, step: 542, time: 21.7, train: 0.002687, valid: 1.000000, train_acc: 0.003652, valid_acc: 1.000000
epoch: 189, step: 544, time: 28.4, train: 0.003153, valid: 1.000000, train_acc: 0.003865, valid_acc: 1.000000
epoch: 190, step: 546, time: 29.8, train: 0.003083, valid: 1.000000, train_acc: 0.003776, valid_acc: 1.000000
epoch: 191, step: 549, time: 35.1, train: 0.002122, valid: 1.000000, train_acc: 0.003569, valid_acc: 1.000000
epoch: 192, step: 552, time: 30.5, train: 0.002059, valid: 1.000000, train_acc: 0.003452, valid_acc: 1.000000
epoch: 193, step: 555, time: 43.5, train: 0.002187, valid: 1.000000, train_acc: 0.003416, valid_acc: 1.000000
epoch: 194, step: 558, time: 39.6, train: 0.002111, valid: 1.000000, train_acc: 0.003310, valid_acc: 1.000000
epoch: 195, step: 561, time: 44.3, train: 0.002079, valid: 1.000000, train_acc: 0.003296, valid_acc: 1.000000
epoch: 196, step: 564, time: 39.3, train: 0.002000, valid: 1.000000, train_acc: 0.003191, valid_acc: 1.000000
epoch: 197, step: 567, time: 44.3, train: 0.002087, valid: 1.000000, train_acc: 0.002935, valid_acc: 1.000000
epoch: 198, step: 570, time: 40.3, train: 0.002019, valid: 1.000000, train_acc: 0.002840, valid_acc: 1.000000
epoch: 199, step: 573, time: 33.3, train: 0.001675, valid: 1.000000, train_acc: 0.002588, valid_acc: 1.000000
epoch: 200, step: 576, time: 29.9, train: 0.001621, valid: 1.000000, train_acc: 0.002508, valid_acc: 1.000000
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

