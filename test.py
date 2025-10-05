import cv2
import mediapipe as mp

# Use your test image path
img_path = "face.jpg"

mp_face_mesh = mp.solutions.face_mesh

# --- Use accurate lip indices from FACEMESH_LIPS ---
FACEMESH_LIPS = frozenset([
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375),
    (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
    (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324),
    (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
    (82, 13), (13, 312), (312, 311), (311, 310),
    (310, 415), (415, 308)
])

# Get the unique landmark indices for lips
lip_indices = sorted(set(idx for pair in FACEMESH_LIPS for idx in pair))

# Read image
img = cv2.imread(img_path)
if img is None:
    print("Image not found:", img_path)
    exit()

h, w, _ = img.shape

with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        print("No face detected!")
    else:
        for face_landmarks in results.multi_face_landmarks:
            for idx in lip_indices:
                p = face_landmarks.landmark[idx]
                px, py = int(p.x * w), int(p.y * h)
                cv2.circle(img, (px, py), 2, (0, 0, 255), -1)

        cv2.imshow("Lips landmarks", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()