from pathlib import Path
from argparse import a
import torch
from train_classifier import HorizonRollClassifier


# Function to Convert to ONNX
def convert_ONNX(model, onnx_filename, input_size):
    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(*input_size, requires_grad=True)

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      onnx_filename,  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'])  # the model's output names
    # dynamic_axes={'modelInput': {1: 'batch_size'},  # variable length axes
    #               'modelOutput': {1: 'batch_size'}})
    print("*************************************")
    print(f'Saving onnx model to {onnx_filename}')
    print("*************************************")


model = HorizonRollClassifier.load_from_checkpoint(args.predict_checkpoint, no_mask=True, no_resize=True,
                                                   no_normalize=True, no_rotate=True,
                                                   return_orig=True)

checkpt_path = Path(args.predict_checkpoint)
onnx_path = checkpt_path.parent / Path(checkpt_path.stem + '.onnx')
# if not onnx_path.exists():
model.eval()