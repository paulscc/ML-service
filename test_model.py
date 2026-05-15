import torch
import pickle

print("Intentando cargar con PyTorch (weights_only=False)...")
try:
    model = torch.load('multipetsmodel.pkl', map_location='cpu', weights_only=False)
    print("Éxito con PyTorch")
    print("Tipo del modelo:", type(model))
    print("\nAtributos del modelo:")
    if hasattr(model, '__dict__'):
        for attr, value in model.__dict__.items():
            print(f"  {attr}: {type(value)}")
except Exception as e:
    print(f"Error con PyTorch: {e}")

print("\nIntentando cargar con fastai...")
try:
    from fastai.learner import load_learner
    learn = load_learner('multipetsmodel.pkl')
    print("Éxito con fastai")
    print("Tipo del modelo:", type(learn))
    print("\nClases del modelo:")
    print(learn.dls.vocab)
except Exception as e:
    print(f"Error con fastai: {e}")
