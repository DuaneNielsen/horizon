import tensorrt as trt
import torch
import timm

model = timm.create_model('efficientnet_b0', pretrained=True)
input_size = (1, 3, 224, 224)

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


# Function to Convert to ONNX
def Convert_ONNX():
    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(*input_size, requires_grad=True)

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "ImageClassifier.onnx",  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')


def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True

        # generate TensorRT engine optimized for the target platform
        print('Building an engine...')
        engine = builder.build_cuda_engine(network)
        context = engine.create_execution_context()
        print("Completed creating Engine")

        return engine, context


if __name__ == '__main__':
    Convert_ONNX()
    # initialize TensorRT engine and parse ONNX model
    engine, context = build_engine('ImageClassifier.onnx')

