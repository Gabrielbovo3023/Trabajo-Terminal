"""
Código que realiza el seguimiento corporal y del balón buscando apoyar al jugador en sus lanzamientos
GAFAS DE APOYO PARA ENTRENAMIENTO DE LANZAMIENTO DE TIRO DE LARGA DISTANCIA EN BALONCESTO
ELAB: GABRIEL BOVOPOULOS SALDAÑA
INSTITUTO POLITÉCNICO NACIONAL
INGENIERÍA EN MECATRÓNICA
"""

""" Librerías a utilizar """

import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
from scipy.spatial import distance as dist
#import math # Libreria para hacer operaciones para el tiro parabolico
import time
import requests
import sqlite3
import pandas as pd
import datetime
import keyboard # Para base de datos con control manual (Teclado)
import pyttsx3

""" Lectura de mensajes con audio"""
engine = pyttsx3.init()

""" Base de datos """

# Conectar a la base de datos (o crearla si no existe)
conn = sqlite3.connect('basededatos.db')
cursor = conn.cursor()

# Crear una tabla si no existe
conn.execute('DELETE FROM lanzamientos')
conn.commit()

""" Captura de video con cv2 con las tres diferentes cámaras propuestas dentro del proyecto """

captura = cv2.VideoCapture(0)
captura2 = cv2.VideoCapture(1)
url2 = 'http://192.168.137.154/640x480.jpg'  # Se cambia la direccion IP por la que genere la ESP32-CAM en automatico, no se pudo establecer una IP fija
captura3 = cv2.VideoCapture(url2)

""" Envío de mensajes con librería requests """

esp32ip = "192.168.137.154"  # Dirección IP de la ESP32-CAM
url = f"http://{esp32ip}/tu_ruta"

#  Mensaje a enviar
mensaje = ("Buen lanzamiento, realizalo nuevamente")
# Lectura de mensaje
engine.say(mensaje)
engine.runAndWait()
# Parámetros de la solicitud GET
params = {'message': mensaje}
response = requests.get(url, params=params) # Utilizar requests para enviar mensaje al url proporcionado
# Imprimir la respuesta del servidor
print(response.text)
time.sleep(8)  # Delay o retraso para que el jugador observe el mensaje con detenimiento

# Mensaje a enviar
mensaje = ("Realiza el lanzamiento")
# Lectura de mensaje
engine.say(mensaje)
engine.runAndWait()
# Parámetros de la solicitud GET
params = {'message': mensaje}
response = requests.get(url, params=params)  # Utilizar requests para enviar mensaje al url proporcionado
# Imprimir la respuesta del servidor
print(response.text)
time.sleep(8)  # Delay o retraso para que el jugador observe el mensaje con detenimiento

#  Mensaje a enviar
mensaje = ( "Debes formar una L con tu brazo y estirarlo por completo, intentalo de nuevo")
# Lectura de mensaje
engine.say(mensaje)
engine.runAndWait()
# Parámetros de la solicitud GET
params = {'message': mensaje}
response = requests.get(url, params=params) # Utilizar requests para enviar mensaje al url proporcionado
# Imprimir la respuesta del servidor
print(response.text)
time.sleep(8)  # Delay o retraso para que el jugador observe el mensaje con detenimiento

""" Variables para el seguimiento de cuerpo """

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# Contador y fase
counter = 0  # Contador para contar los tiros lanzados
stage = None  # Fase que se encuentra el jugador

""" Variables de seguimiento de balón """

# Balón
bal = 25  # Diametro del balón en cm
# Historial de coordenadas del balón
balon_trajectory = []
canasta_trajectory = []

""" Píxeles """

pixeles = None  # Variable de píxeles

""" Lanzamientos """

total_lanzamientos = 0
lanzamientos_exitosos = 0

""" Modelo y clases del modelo """

model = YOLO("C:/Users/Gabo/PycharmProjects/pythonProject2/modelofinal.pt")  # Modelo generado por roboflow y entrenado en colab con terminacion .pt
# Object classes / Clases de objeto
classNames = ["balon", "canasta"]  # Clases de objetos identificados del modelo

""" Ciclo y proceso de seguimiento de balón y cuerpo """

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:

        # Lectura de video
        ret, img = captura.read()  # Cámara lateral
        ret2, img2 = captura2.read()  # Cámara frontal
        captura3.open(url2)  # Cámara de la ESP
        ret3, img3 = captura3.read()  # Cámara de la ESP

        # Detección del balón y cuerpo cámara lateral
        results = pose.process(img)  # Cuerpo
        results2 = model(img, stream=True)  # Balon

        # Detección del balón cámara ESP32-CAM y frontal
        results3 = model(img2, stream=True)
        results4 = model(img3, stream=True) #ESP32-CAM

        """ Código realizado para el seguimiento del balón para la imagen lateral"""
        for r in results2:

            boxes = r.boxes

            for box in boxes:

                # Bounding box / Creación de "caja" de los objetos
                x1, y1, x2, y2 = box.xyxy[0]  ## esquina superior izquierda y esquina inferior derecha
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values / conversion a valores enteros

                #da = dist.euclidean((x1, y1), (x2, y2))
                #db = dist.euclidean((x1, y2), (x2, y1))

                #if pixeles is None:
                    # Medida del balon
                    #pixeles = db / bal

                # Medida del balon
                #dimA = da / pixeles
                #dimB = db / pixeles

                # Dibujar el tamaño del objeto en la imagen
                # cv2.putText(img, "{:.2f}cm".format(dimA), (int(x1 + 10), int(y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # cv2.putText(img, "{:.2f}cm".format(dimB), (int(x2 + 10), int(y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Put box in cam / Colocar caja
                cv2.rectangle(img, (x1, y1), (x2, y2), (225, 255, 0), 3)

                # Centroid Coordinates of detected object / Coordenadas del centroide del objeto detectado
                cx = int((x2 + x1) / 2.0)
                cy = int((y2 + y1) / 2.0)
                # print(cx,cy)

                # Dibujar centroide
                cv2.circle(img, (cx, cy), 5, (0, 255, 100), 2, cv2.FILLED)  # draw center dot on detected object

                # Confidence / Confianza del modelo
                # confidence = math.ceil((box.conf[0] * 100)) / 100
                confidence = 0.7
                print("Confidence --->", confidence)

                # Class name / Nombre de clase
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # Object details / Detalles de objeto
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                """ Dibujar la trayectoria del balón con un tiro parabólico """
                if classNames[cls] == "balon":
                    balon_trajectory.append((cx, cy))
                """
                if len(balon_trajectory) > 1:
                    for i in range(1, len(balon_trajectory)):
                        #cv2.line(img, balon_trajectory[i - 1], balon_trajectory[i], (0, 255, 0), 2)

                        # Calcular y dibujar la trayectoria parabólica
                        parabolic_trajectory = np.array(balon_trajectory, dtype=np.float32)
                        coeffs = np.polyfit(parabolic_trajectory[:, 0], parabolic_trajectory[:, 1], 2)
                        poly = np.poly1d(coeffs)
                        x_vals = np.linspace(min(parabolic_trajectory[:, 0]), max(parabolic_trajectory[:, 0]), 100)
                        y_vals = poly(x_vals)
                        parabolic_points = np.column_stack((x_vals, y_vals)).astype(np.int32)
                        #cv2.polylines(img, [parabolic_points], False, (0, 255, 0), 2)
        """
        """ Código realizado para el seguimiento del balón para img2 o cámara frontal """
        for r in results3:

            boxes = r.boxes

            for box in boxes:

                # Bounding box / Coordenadas de la "caja" de los objetos
                x1, y1, x2, y2 = box.xyxy[0]  ## esquina superior izquierda y esquina inferior derecha
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values / conversion a valores enteros

                #da = dist.euclidean((x1, y1), (x2, y2))
                #db = dist.euclidean((x1, y2), (x2, y1))
                """
                if pixeles is None:
                    # Medida del balon
                    pixeles = db / bal

                # Medida del balon
                dimA = da / pixeles
                dimB = db / pixeles

                # Dibujar el tamaño del objeto en la imagen
                cv2.putText(img2, "{:.2f}cm".format(dimA), (int(x1 + 10), int(y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
                cv2.putText(img2, "{:.2f}cm".format(dimB), (int(x2 + 10), int(y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
                """
                # Put box in cam / Colocar "caja" en objeto
                cv2.rectangle(img2, (x1, y1), (x2, y2), (225, 255, 0), 3)

                # Centroid Coordinates of detected object / Coordenadas del centroide del objeto detectado
                cx = int((x2 + x1) / 2.0)
                cy = int((y2 + y1) / 2.0)
                # print(cx,cy)

                # Dibujar centroide
                cv2.circle(img2, (cx, cy), 5, (0, 255, 100), 2, cv2.FILLED)  # draw center dot on detected object

                # Confidence / Confianza del modelo
                # confidence = math.ceil((box.conf[0] * 100)) / 100
                confidence = 0.7
                print("Confidence --->", confidence)

                # Class name / Nombres de clases
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # Object details / Detalles de objeto
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img2, classNames[cls], org, font, fontScale, color, thickness)

                # Agregar las coordenadas al historial del balón
                if classNames[cls] == "balon":
                    balon_trajectory.append((cx, cy))
                else:
                    canasta_trajectory.append((cx, cy))

                if len(balon_trajectory) > 0 and len(canasta_trajectory) > 0:

                    distancia_centros = dist.euclidean(balon_trajectory[-1], canasta_trajectory[-1])
                    umbral_entrada = 20

                    if distancia_centros < umbral_entrada:
                        resultado = "Canasto"
                        lanzamientos_exitosos += 1
                        total_lanzamientos += 1
                    else:
                        resultado = "Errado"
                        total_lanzamientos += 1


        #Código realizado para el seguimiento del balón para ESP32-CAM 
        for r in results4:

            boxes = r.boxes

            for box in boxes:

                # bounding box / coordenadas de la "caja" de los objetos
                x1, y1, x2, y2 = box.xyxy[0]  ## esquina superior izquierda y esquina inferior derecha
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values / conversion a valores enteros

                #da = dist.euclidean((x1, y1), (x2, y2))
                #db = dist.euclidean((x1, y2), (x2, y1))

                #if pixeles is None:
                    # Medida del balon
                    #pixeles = db / bal

                # Medida del balon
                #dimA = da / pixeles
                #dimB = db / pixeles

                # Dibujar el tamaño del objeto en la imagen
                #cv2.putText(img3, "{:.2f}cm".format(dimA), (int(x1 + 10), int(y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                 #           (255, 255, 255), 1)
                #cv2.putText(img3, "{:.2f}cm".format(dimB), (int(x2 + 10), int(y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                 #           (255, 255, 255), 1)

                # put box in cam
                cv2.rectangle(img3, (x1, y1), (x2, y2), (225, 255, 0), 3)

                # Centroid Coordinates of detected object
                cx = int((x2 + x1) / 2.0)
                cy = int((y2 + y1) / 2.0)
                # print(cx,cy)

                # Dibujar centroide
                cv2.circle(img3, (cx, cy), 5, (0, 255, 100), 2, cv2.FILLED)  # draw center dot on detected object

                # confidence
                # confidence = math.ceil((box.conf[0] * 100)) / 100
                confidence = 0.7
                print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img3, classNames[cls], org, font, fontScale, color, thickness)


        """ Codigo realizado para el seguimiento del cuerpo para cámara lateral"""
        # Extract landmarks / extraer marcas con mediapipe
        try:

            landmarks = results.pose_landmarks.landmark

            # Get coordinates / obtener coordenadas del hombro, brazo y muñeca
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle / Cálculo de ángulo
            def calculate_angle(a, b, c):
                a = np.array(a)  # First
                b = np.array(b)  # Mid
                c = np.array(c)  # End

                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)

                if angle > 180.0:
                    angle = 360 - angle

                return angle

            angle = calculate_angle(shoulder, elbow, wrist)

            # Visualizar el ángulo
            cv2.putText(img, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Curl counter logic / Fase del jugador en el lanzamiento
            if angle < 90:
                mensaje = ("Aún no estas en una buena posicion de tiro, sube un poco más tu brazo")
                engine.say(mensaje)
                engine.runAndWait()
            if 70 < angle < 90:
                stage = "Posicion de tiro"
                mensaje = ("Realiza el lanzamiento")
                engine.say(mensaje)
                engine.runAndWait()
            if 90 < angle < 110 and stage == 'Posicion de tiro':
                stage = "Lanzo"
                counter += 1
                print(counter)
                mensaje = ("debes formar una L con tu brazo y estirarlo por completo, intentalo de nuevo")
                # Lectura de mensaje
                engine.say(mensaje)
                engine.runAndWait()
                # Parámetros de la solicitud GET
                #params = {'message': mensaje}
                #response = requests.get(url, params=params)  # Utilizar requests para enviar mensaje al url proporcionado
                # Imprimir la respuesta del servidor
                print(response.text)
                #time.sleep(8)  # Delay o retraso para que el jugador observe el mensaje con detenimiento
            if 130 < angle and stage == 'Lanzo':
                stage = "Buen lanzamiento"
                counter += 1
                print(counter)
                mensaje = ("Buen lanzamiento, realizalo nuevamente")
                # Lectura de mensaje
                engine.say(mensaje)
                engine.runAndWait()
                # Parámetros de la solicitud GET
                #params = {'message': mensaje}
                #response = requests.get(url, params=params)  # Utilizar requests para enviar mensaje al url proporcionado
                # Imprimir la respuesta del servidor
                #print(response.text)
                #time.sleep(8)  # Delay o retraso para que el jugador observe el mensaje con detenimiento

            # Manejo manual de resultados
            if keyboard.is_pressed('f'):
                resultado = "canasto"
                lanzamientos_exitosos += 1
                total_lanzamientos += 1
            elif keyboard.is_pressed('b'):
                resultado = "fallo"
                total_lanzamientos += 1

            # Calcular el porcentaje de efectividad e impresión
            if total_lanzamientos > 0:
                porcentaje_efectividad = (lanzamientos_exitosos / total_lanzamientos) * 100
                print(f"Porcentaje de Efectividad: {porcentaje_efectividad:.2f}%")
            else:
                print("No se han registrado lanzamientos en la base de datos")

            # Fecha
            fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Insertar el resultado en la base de datos
            conn.execute('INSERT INTO lanzamientos (resultado, fecha) VALUES (?, ?)', (resultado, fecha,))
            conn.commit()

            # Resultados en la tabla
            # Leer el DataFrame desde la tabla 'lanzamientos'
            df = pd.read_sql_query('SELECT * FROM lanzamientos', conn)

            # Mostrar el DataFrame
            print(f"Datos de los lanzamientos de {fecha}:")
            print(df)

            # Obtener el total de lanzamientos y lanzamientos exitosos desde la base de datos
            total_lanzamientos = conn.execute('SELECT COUNT(*) FROM lanzamientos').fetchone()[0]
            lanzamientos_exitosos = \
            conn.execute('SELECT COUNT(*) FROM lanzamientos WHERE resultado="Encestó"').fetchone()[0]
            print(f"Porcentaje de Efectividad: {porcentaje_efectividad:.2f}%")

            """
                # Lógica de Resultados
                if distancia_aro_bal < umbral_entrada:
                    resultado = "Enceste"
                    lanzamientos_exitosos += 1
                else:
                    resultado = "Tiro errado"
            """

        except:
            pass

        # Render curl counter / Cajas para colocar textos
        cv2.rectangle(img, (0, 0), (320, 73), (155, 120, 100), -1)  # 255, 220, 200 azul claro, 155, 120, 100 azul.morado obscuro Cuadro para fase y repeticion
        cv2.rectangle(img2, (0, 430), (650, 480), (155, 120, 100), -1)  # 255, 220, 200 azul claro, 155, 120, 100 azul.morado obscuro Cuadro para mensaje

        # Rep data / Datos de repetición
        cv2.putText(img, 'REPETICION', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data / Datos de fase
        cv2.putText(img, 'FASE', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Mensaje mostrado al jugador
        cv2.putText(img2, 'INSTRUCCIONES:', (0, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img2, mensaje, (0, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Render detections / Dibujo de las lineas de seguimiento de cuerpo
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Resultados con cv2
        img3 = cv2.rotate(img3, cv2.ROTATE_180) # Rotación del video de las ESP32-CAM
        cv2.imshow('Camara frontal', cv2.resize(img2, (640, 480))) # img2 es cámara frontal
        cv2.imshow('Camara lateral', cv2.resize(img, (640, 480))) # img es cámara lateral
        cv2.imshow('Camara esp', cv2.resize(img3, (640, 480))) # img3 es cámara ESP32-CAM

        # Esperar "q" para terminar el proceso
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

""" Finalización de código """

# Mensaje de despedida
#mensaje = ("Ha finalizado el entrenamiento, retira las gafas con cuidado")
# Parámetros de la solicitud GET
#params = {'message': mensaje}
#response = requests.get(url, params=params)  # Utilizar requests para enviar mensaje al url proporcionado
# Imprimir la respuesta del servidor
#print(response.text)
#time.sleep(5)  # Delay o retraso para que el jugador observe el mensaje con detenimiento

"""
LA IMG2 DEBERÁ SER LA CÁMARA FRONTAL O DEBAJO DEL ARO
LA IMG SERÁ LA LATERAL
IM3 LA ESP32-CAM
"""

# Cerrar la conexión a la base de datos
conn.close()
captura.release()
captura2.release()
captura3.release()
cv2.destroyAllWindows()