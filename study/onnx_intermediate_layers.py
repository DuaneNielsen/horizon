import onnx

model = onnx.load('/home/tornado/horizon/checkpoints/classifier/northern-breeze-53/epoch=196-step=8864.onnx')
inter_layers = ['647', '709'] # output tensor names
value_info_protos = []
shape_info = onnx.shape_inference.infer_shapes(model)
for idx, node in enumerate(shape_info.graph.value_info):
    if node.name in inter_layers:
        print(idx, node)
        value_info_protos.append(node)
assert len(value_info_protos) == len(inter_layers)
model.graph.output.extend(value_info_protos)  #  in inference stage, these tensor will be added to output dict.
onnx.checker.check_model(model)
onnx.save(model, './test.onnx')