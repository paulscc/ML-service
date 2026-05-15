PS C:\Users\pcstu\OneDrive\Documentos\pkl> python test_model.py
Intentando cargar con PyTorch (weights_only=False)...
Éxito con PyTorch
Tipo del modelo: <class 'fastai.learner.Learner'>

Atributos del modelo:
  dls: <class 'fastai.data.core.DataLoaders'>
  model: <class 'torch.nn.modules.container.Sequential'>
  __stored_args__: <class 'dict'>
  loss_func: <class 'fastai.losses.CrossEntropyLossFlat'>
  opt_func: <class 'function'>
  lr: <class 'float'>
  splitter: <class 'function'>
  _metrics: <class 'fastcore.foundation.L'>
  path: <class 'pathlib.WindowsPath'>
  model_dir: <class 'str'>
  wd: <class 'NoneType'>
  wd_bn_bias: <class 'bool'>
  train_bn: <class 'bool'>
  moms: <class 'tuple'>
  default_cbs: <class 'bool'>
  training: <class 'bool'>
  create_mbar: <class 'bool'>
  logger: <class 'builtin_function_or_method'>
  opt: <class 'NoneType'>
  cbs: <class 'fastcore.foundation.L'>
  train_eval: <class 'fastai.callback.core.TrainEvalCallback'>
  recorder: <class 'fastai.learner.Recorder'>
  cast_to_tensor: <class 'fastai.learner.CastToTensor'>
  progress: <class 'fastai.callback.progress.ProgressCallback'>
  n_epoch: <class 'int'>
  arch: <class 'str'>
  normalize: <class 'bool'>
  n_out: <class 'int'>
  pretrained: <class 'bool'>
  epoch: <class 'int'>
  loss: <class 'NoneType'>
  train_iter: <class 'int'>
  pct_train: <class 'float'>
  dl: <class 'NoneType'>
  n_iter: <class 'int'>
  iter: <class 'int'>
  xb: <class 'tuple'>
  yb: <class 'tuple'>
  pred: <class 'NoneType'>
  loss_grad: <class 'fastai.torch_core.TensorBase'>
  final_record: <class 'fastcore.foundation.L'>
  smooth_loss: <class 'fastai.torch_core.TensorBase'>
  lock: <class '_thread.lock'>

Intentando cargar con fastai...
Éxito con fastai
Tipo del modelo: <class 'fastai.learner.Learner'>

Clases del modelo:
['cat', 'dog', 'goldfish', 'guinea pig', 'hamster', 'lizard', 'parrot', 'rabbit', 'snake', 'turtle']