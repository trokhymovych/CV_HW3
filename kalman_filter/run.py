import cv2
import numpy as np
from kalman_filter import KalmanFilter

frame = np.ones((400, 400, 3), np.uint8) * 255

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result.avi', fourcc, 20.0, (400, 400), isColor=True)

def mousemove(event, x, y, s, p):
    global frame, current_mes, current_pre

    # getting current mouse position
    current_mes = np.array([[np.float32(x)], [np.float32(y)], [0], [0]])

    # getting Kalman correction
    current_pre = kalman.predict()

    # Cleaning the frame from output of previous action
    frame = np.ones((400, 400, 3), np.uint8) * 255

    # Getting coordinates
    cmx, cmy, cpx, cpy = int(current_mes[0]), int(current_mes[1]), int(current_pre[0]), int(current_pre[1])

    # Drawing current measurement and prediction
    cv2.circle(frame, (cmx, cmy), 5, (0, 255, 0), -1)
    cv2.circle(frame, (cpx, cpy), 5, (0, 0, 255), -1)

    # saving frame to video
    out.write(frame)

    # Making correction for next iteration
    kalman.update(current_mes)


cv2.namedWindow("Kalman filter HW")
cv2.setMouseCallback("Kalman filter HW", mousemove)

# Параметри початкового стану моделі
State = np.zeros((4, 1), np.float32)
Covariance = np.eye(State.shape[0])
Measurement = np.zeros((4, 1), np.float32)
Measurement_matrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], np.float32)
State_Transition = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
Action_Uncertainty = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]], np.float32)
Sensor_Noise = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]], np.float32)

# ignore control matrix and vector in our case (setting to zero)
kalman = KalmanFilter(X=State,
                      P=Covariance,
                      A=State_Transition,
                      Q=Action_Uncertainty,
                      Z=Measurement,
                      H=Measurement_matrix,
                      R=Sensor_Noise)

while True:
    cv2.imshow("Kalman filter HW", frame)
    ### ESC to exit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

out.release()
cv2.destroyAllWindows()
