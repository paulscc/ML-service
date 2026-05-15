from fastai.vision.all import *
import requests
from PIL import Image
import io
import os

# Cargar el modelo
learn = load_learner('multipetsmodel.pkl')
print("Modelo cargado exitosamente")
print("Clases:", learn.dls.vocab)
print()

# URLs de imágenes de prueba (imágenes gratuitas de Unsplash)
test_images = [
    ("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400", "cat"),
    ("https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400", "dog"),
    ("https://images.unsplash.com/photo-1520302630591-fd1c66edc19d?w=400", "goldfish"),
    ("https://images.unsplash.com/photo-1585110396067-1f2cc9ab2b55?w=400", "guinea_pig"),
    ("https://images.unsplash.com/photo-1425082661705-1834bfd09dca?w=400", "hamster"),
    ("https://images.unsplash.com/photo-1506765515384-028b60a970df?w=400", "lizard"),
    ("https://images.unsplash.com/photo-1552728089-57bdde30beb3?w=400", "parrot"),
    ("https://images.unsplash.com/photo-1585110396000-c9ffd4e4b308?w=400", "rabbit"),
    ("https://images.unsplash.com/photo-1531386151447-fd76ad50012f?w=400", "snake"),
    ("https://images.unsplash.com/photo-1437622368342-7a3d73a34c8f?w=400", "turtle")
]

# Crear directorio para imágenes
os.makedirs('test_images', exist_ok=True)

print("Descargando imágenes de prueba...")
for url, label in test_images:
    try:
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content))
        filename = f"test_images/{label}.jpg"
        img.save(filename)
        print(f"  Descargado: {label}.jpg")
    except Exception as e:
        print(f"  Error descargando {label}: {e}")

print("\nProbando el modelo con las imágenes...")
print("-" * 60)

# Probar cada imagen
for url, true_label in test_images:
    try:
        filename = f"test_images/{true_label}.jpg"
        if os.path.exists(filename):
            # Cargar y predecir
            img = PILImage.create(filename)
            pred_class, pred_idx, outputs = learn.predict(img)
            
            # Obtener probabilidades
            probs = outputs
            top_probs, top_indices = torch.topk(probs, 3)
            
            print(f"\nImagen: {true_label}.jpg")
            print(f"Etiqueta real: {true_label}")
            print(f"Predicción: {pred_class}")
            print(f"Top 3 predicciones:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                class_name = learn.dls.vocab[idx]
                print(f"  {i+1}. {class_name}: {prob:.4f}")
    except Exception as e:
        print(f"\nError procesando {true_label}: {e}")

print("\n" + "-" * 60)
print("Prueba completada")
