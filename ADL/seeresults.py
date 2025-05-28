import cv2
import torch
import json
import colorsys

# --- 1. Función para generar N colores brillantes en BGR ---
def get_bright_colors(n):
    colors = []
    for i in range(n):
        h = i / n
        s, v = 0.9, 0.9
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(b*255), int(g*255), int(r*255)))
    return colors

# --- 2. Parámetros y carga de datos ---
IMAGE_PATH = '/dataset/rgb_frames/P_11/000030.jpg'
MODEL_PATH = '../../resultados_test/new_dist/pre_fine_tunning/Version_entrenada_con_1_8M/results-0.pkl'
LABEL_MAP_PATH = '../config/label_map_ADL.json'

# Solo nos interesan estas dos clases
ALLOWED = {'tv', 'monitor'}

image = cv2.imread(IMAGE_PATH)
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)

obj = torch.load(MODEL_PATH)
bboxes = obj['res_info'][4238]

# Averigua qué índices de label corresponden a tv y monitor
label2name = {int(k):v for k,v in label_map.items()}
name2label = {v:int(k) for k,v in label_map.items()}
keep_labels = { name2label[name] for name in ALLOWED }

# Paleta de colores (dos colores brillantes: uno para tv, otro para monitor)
palette = get_bright_colors(len(keep_labels))
label_colors = {lab: palette[i] for i, lab in enumerate(keep_labels)}

# Para llevar registro de rects de texto ya ocupadas
occupied = []

monitors = []
tvs      = []
for x1,y1,x2,y2,conf,lab in bboxes:
    lab = int(lab)
    if conf<0.35: continue
    name = label_map[str(lab)]
    if name=='monitor':
        monitors.append((x1,y1,x2,y2,lab))
    elif name=='tv':
        tvs.append((x1,y1,x2,y2,lab))

# 1) Dibuja MONITOR primero
if monitors:
    x1,y1,x2,y2,lab = monitors[0]   # asumimos 1 monitor; si hay varios elige uno
    color = label_colors[lab]
    cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)

    # Texto para "monitor"
    txt = 'monitor'
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th_text), base = cv2.getTextSize(txt, font, scale, th)
    tx_mon = int(x1)
    ty_mon = int(y1) - 5

    # Fondo y texto
    cv2.rectangle(image,
                  (tx_mon,   ty_mon - th_text - base),
                  (tx_mon+tw, ty_mon + base),
                  color, cv2.FILLED)
    cv2.putText(image, txt, (tx_mon, ty_mon), font, scale, (255,255,255), th, cv2.LINE_AA)

    # Guarda la región ocupada por el texto de monitor
    mon_text_rect = (tx_mon, ty_mon - th_text - base, tw, th_text + base)
else:
    mon_text_rect = None

# 2) Dibuja TV, pero coloca su texto a la derecha del monitor
for x1,y1,x2,y2,lab in tvs:
    color = label_colors[lab]
    cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)

    txt = 'tv'
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th_text), base = cv2.getTextSize(txt, font, scale, th)

    if mon_text_rect:
        # posición X justo a la derecha del texto de monitor + 10 px de margen
        tx_tv = mon_text_rect[0] + mon_text_rect[2] + 10
        # misma línea vertical que monitor
        ty_tv = mon_text_rect[1] + th_text + base
    else:
        # fallback: sobre su propia caja
        tx_tv = int(x1)
        ty_tv = int(y1) - 5

    # Fondo y texto TV
    cv2.rectangle(image,
                  (tx_tv,   ty_tv - th_text - base),
                  (tx_tv+tw, ty_tv + base),
                  color, cv2.FILLED)
    cv2.putText(image, txt, (tx_tv, ty_tv), font, scale, (255,255,255), th, cv2.LINE_AA)

# Guardar resultado
cv2.imwrite('../../resultados_test/new_dist/pre_fine_tunning/Version_entrenada_con_1_8M/version_GR_oficial.jpg', image)
